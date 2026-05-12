#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/mlir-support/tostring.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
using namespace mlir;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};

namespace {
using namespace lingodb::compiler::dialect;

struct TupleTypeManagedModel
   : public db::ManagedType::ExternalModel<TupleTypeManagedModel, mlir::TupleType> {
   bool needsManagement(mlir::Type t) const {
      auto tuple = mlir::cast<mlir::TupleType>(t);
      for (auto element : tuple.getTypes()) {
         if (auto managed = mlir::dyn_cast<db::ManagedType>(element)) {
            if (managed.needsManagement()) return true;
         }
      }
      return false;
   }
};

struct PackOpRefCountedModel
   : public db::RefCountedOp::ExternalModel<PackOpRefCountedModel, util::PackOp> {
   void getOwnedOperands(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& result) const {
      for (auto v : mlir::cast<util::PackOp>(op).getVals()) result.push_back(v);
   }
   void getBorrowedResults(mlir::Operation*, llvm::SmallVectorImpl<mlir::Value>&) const {}
   mlir::Operation* rewriteForRefCount(mlir::Operation*, mlir::OpBuilder&, llvm::DenseSet<mlir::Value>&) const { return nullptr; }
};

struct UnPackOpRefCountedModel
   : public db::RefCountedOp::ExternalModel<UnPackOpRefCountedModel, util::UnPackOp> {
   void getOwnedOperands(mlir::Operation*, llvm::SmallVectorImpl<mlir::Value>&) const {}
   void getBorrowedResults(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& result) const {
      for (auto v : mlir::cast<util::UnPackOp>(op).getVals()) result.push_back(v);
   }
   mlir::Operation* rewriteForRefCount(mlir::Operation*, mlir::OpBuilder&, llvm::DenseSet<mlir::Value>&) const { return nullptr; }
};

struct GetTupleOpRefCountedModel
   : public db::RefCountedOp::ExternalModel<GetTupleOpRefCountedModel, util::GetTupleOp> {
   void getOwnedOperands(mlir::Operation*, llvm::SmallVectorImpl<mlir::Value>&) const {}
   void getBorrowedResults(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& result) const {
      result.push_back(mlir::cast<util::GetTupleOp>(op).getVal());
   }
   mlir::Operation* rewriteForRefCount(mlir::Operation*, mlir::OpBuilder&, llvm::DenseSet<mlir::Value>&) const { return nullptr; }
};

struct ForOpRefCountedModel
   : public db::RefCountedOp::ExternalModel<ForOpRefCountedModel, mlir::scf::ForOp> {
   void getOwnedOperands(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& result) const {
      for (auto v : mlir::cast<mlir::scf::ForOp>(op).getInitArgs()) result.push_back(v);
   }
   void getBorrowedResults(mlir::Operation*, llvm::SmallVectorImpl<mlir::Value>&) const {}
   mlir::Operation* rewriteForRefCount(mlir::Operation*, mlir::OpBuilder&, llvm::DenseSet<mlir::Value>&) const { return nullptr; }
};

struct WhileOpRefCountedModel
   : public db::RefCountedOp::ExternalModel<WhileOpRefCountedModel, mlir::scf::WhileOp> {
   void getOwnedOperands(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& result) const {
      for (auto v : mlir::cast<mlir::scf::WhileOp>(op).getInits()) result.push_back(v);
   }
   void getBorrowedResults(mlir::Operation*, llvm::SmallVectorImpl<mlir::Value>&) const {}
   mlir::Operation* rewriteForRefCount(mlir::Operation*, mlir::OpBuilder&, llvm::DenseSet<mlir::Value>&) const { return nullptr; }
};

// arith.select can't grow ref counts in place; replace with scf.if and emit
// add_use in each arm so the chosen value owns its reference.
struct SelectOpRefCountedModel
   : public db::RefCountedOp::ExternalModel<SelectOpRefCountedModel, mlir::arith::SelectOp> {
   void getOwnedOperands(mlir::Operation*, llvm::SmallVectorImpl<mlir::Value>&) const {}
   void getBorrowedResults(mlir::Operation*, llvm::SmallVectorImpl<mlir::Value>&) const {}
   mlir::Operation* rewriteForRefCount(mlir::Operation* op, mlir::OpBuilder& builder, llvm::DenseSet<mlir::Value>& returnedValues) const {
      auto selectOp = mlir::cast<mlir::arith::SelectOp>(op);
      auto managed = mlir::dyn_cast<db::ManagedType>(selectOp.getType());
      if (!managed || !managed.needsManagement()) return nullptr;
      auto ifOp = builder.create<mlir::scf::IfOp>(selectOp->getLoc(), selectOp.getType(), selectOp.getCondition(), true);
      {
         mlir::OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
         thenBuilder.create<db::MemoryAddUse>(selectOp->getLoc(), selectOp.getTrueValue());
         thenBuilder.create<mlir::scf::YieldOp>(selectOp->getLoc(), selectOp.getTrueValue());
      }
      {
         mlir::OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
         elseBuilder.create<db::MemoryAddUse>(selectOp->getLoc(), selectOp.getFalseValue());
         elseBuilder.create<mlir::scf::YieldOp>(selectOp->getLoc(), selectOp.getFalseValue());
      }
      selectOp.replaceAllUsesWith(ifOp.getResult(0));
      if (returnedValues.contains(selectOp.getResult())) {
         returnedValues.erase(selectOp.getResult());
         returnedValues.insert(ifOp.getResult(0));
      }
      selectOp.erase();
      return ifOp;
   }
};
} // namespace

void lingodb::compiler::dialect::db::DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/DB/IR/DBOps.cpp.inc"

      >();
   addInterfaces<DBInlinerInterface>();
   registerTypes();
   mlir::TupleType::attachInterface<TupleTypeManagedModel>(*getContext());
   // External ops attach lazily — ensure their dialects are loaded first.
   getContext()->getOrLoadDialect<util::UtilDialect>();
   getContext()->getOrLoadDialect<mlir::scf::SCFDialect>();
   getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
   util::PackOp::attachInterface<PackOpRefCountedModel>(*getContext());
   util::UnPackOp::attachInterface<UnPackOpRefCountedModel>(*getContext());
   util::GetTupleOp::attachInterface<GetTupleOpRefCountedModel>(*getContext());
   mlir::scf::ForOp::attachInterface<ForOpRefCountedModel>(*getContext());
   mlir::scf::WhileOp::attachInterface<WhileOpRefCountedModel>(*getContext());
   mlir::arith::SelectOp::attachInterface<SelectOpRefCountedModel>(*getContext());
   runtimeFunctionRegistry = db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext());
}

::mlir::Operation* lingodb::compiler::dialect::db::DBDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value, ::mlir::Type type, ::mlir::Location loc) {
   if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
      if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(value)) {
         return builder.create<db::ConstantOp>(loc, type, builder.getStringAttr(support::decimalToString(intAttr.getValue().getLoBits(64).getLimitedValue(), intAttr.getValue().getHiBits(64).getLimitedValue(), decimalType.getS())));
      }
   }
   if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
      if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(value)) {
         return builder.create<db::ConstantOp>(loc, type, builder.getStringAttr(support::dateToString(intAttr.getInt())));
      }
   }
   if (mlir::isa<db::StringType, mlir::IntegerType, mlir::FloatType>(type)) {
      return builder.create<db::ConstantOp>(loc, type, value);
   }
   return nullptr;
}
#include "lingodb/compiler/Dialect/DB/IR/DBOpsDialect.cpp.inc"
