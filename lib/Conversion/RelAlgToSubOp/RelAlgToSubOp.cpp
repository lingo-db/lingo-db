#include "mlir-support/parsing.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/util/FunctionHelper.h>
using namespace mlir;

namespace {
struct RelalgToSubOpLoweringPass
   : public PassWrapper<RelalgToSubOpLoweringPass, OperationPass<ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "to-subop"; }

   RelalgToSubOpLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, mlir::db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithmeticDialect, mlir::relalg::RelAlgDialect, mlir::subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};
static std::string getUniqueMember(std::string name) {
   static std::unordered_map<std::string, size_t> counts;
   return name + std::to_string(counts[name]++);
}

class BaseTableLowering : public OpConversionPattern<mlir::relalg::BaseTableOp> {
   public:
   using OpConversionPattern<mlir::relalg::BaseTableOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::relalg::BaseTableOp baseTableOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<mlir::Type> types;
      std::vector<Attribute> colNames;
      std::vector<Attribute> colTypes;
      std::vector<NamedAttribute> mapping;
      std::string tableName = baseTableOp->getAttr("table_identifier").cast<mlir::StringAttr>().str();
      std::string scanDescription = R"({ "table": ")" + tableName + R"(", "columns": [ )";
      bool first = true;
      for (auto namedAttr : baseTableOp.columnsAttr().getValue()) {
         auto identifier = namedAttr.getName();
         auto attr = namedAttr.getValue();
         auto attrDef = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
         if (!first) {
            scanDescription += ",";
         } else {
            first = false;
         }
         scanDescription += "\"" + identifier.str() + "\"";
         auto memberName = getUniqueMember(identifier.str());
         colNames.push_back(rewriter.getStringAttr(memberName));
         colTypes.push_back(mlir::TypeAttr::get(attrDef.getColumn().type));
         mapping.push_back(rewriter.getNamedAttr(memberName, attrDef));
      }
      scanDescription += "] }";
      auto tableRefType = mlir::subop::TableRefType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(colNames), rewriter.getArrayAttr(colTypes)));
      mlir::Value tableRef = rewriter.create<mlir::subop::GetReferenceOp>(baseTableOp->getLoc(), tableRefType, rewriter.getStringAttr(scanDescription));
      rewriter.replaceOpWithNewOp<mlir::subop::ScanOp>(baseTableOp, tableRef, rewriter.getDictionaryAttr(mapping));
      return success();
   }
};
static mlir::LogicalResult safelyMoveRegion(ConversionPatternRewriter& rewriter, mlir::Region& source, mlir::Region& target) {
   rewriter.inlineRegionBefore(source, target, target.end());
   {
      if (!target.empty()) {
         source.push_back(new Block);
         std::vector<mlir::Location> locs;
         for (size_t i = 0; i < target.front().getArgumentTypes().size(); i++) {
            locs.push_back(target.front().getArgument(i).getLoc());
         }
         source.front().addArguments(target.front().getArgumentTypes(), locs);
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(&source.front());
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc());
      }
   }
   return success();
}
static mlir::Value translateSelection(mlir::Value stream, mlir::Region& predicate, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto& columnManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope("map");
   std::string attributeName = "predicate";
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = rewriter.getI1Type(); //todo: make sure it is really i1 (otherwise: nullable<i1> -> i1)
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(markAttrDef));
   auto terminator = mlir::cast<mlir::tuples::ReturnOp>(predicate.front().getTerminator());
   if (terminator.results().empty()) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto constTrueBlock=new Block;
      mapOp.fn().push_back(constTrueBlock);
      rewriter.setInsertionPointToStart(constTrueBlock);
      mlir::Value constTrue = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      rewriter.create<mlir::tuples::ReturnOp>(loc, constTrue);
   }else{
      assert(safelyMoveRegion(rewriter, predicate, mapOp.fn()).succeeded());

   }
   return rewriter.create<mlir::subop::FilterOp>(loc, mapOp.result(), mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(columnManager.createRef(&ra)));
}
class SelectionLowering : public OpConversionPattern<mlir::relalg::SelectionOp> {
   public:
   using OpConversionPattern<mlir::relalg::SelectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::SelectionOp selectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(selectionOp, translateSelection(adaptor.rel(), selectionOp.predicate(), rewriter, selectionOp->getLoc()));
      return success();
   }
};
class MapLowering : public OpConversionPattern<mlir::relalg::MapOp> {
   public:
   using OpConversionPattern<mlir::relalg::MapOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::MapOp mapOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto mapOp2 = rewriter.replaceOpWithNewOp<mlir::subop::MapOp>(mapOp, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.rel(), mapOp.computed_cols());
      assert(safelyMoveRegion(rewriter, mapOp.predicate(), mapOp2.fn()).succeeded());

      return success();
   }
};
class RenamingLowering : public OpConversionPattern<mlir::relalg::RenamingOp> {
   public:
   using OpConversionPattern<mlir::relalg::RenamingOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::RenamingOp renamingOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::subop::RenamingOp>(renamingOp, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.rel(), renamingOp.columns());
      return success();
   }
};
class ProjectionAllLowering : public OpConversionPattern<mlir::relalg::ProjectionOp> {
   public:
   using OpConversionPattern<mlir::relalg::ProjectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::ProjectionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.set_semantic() == mlir::relalg::SetSemantic::distinct) return failure();
      rewriter.replaceOp(projectionOp, adaptor.rel());
      return success();
   }
};
class ProjectionDistinctLowering : public OpConversionPattern<mlir::relalg::ProjectionOp> {
   public:
   using OpConversionPattern<mlir::relalg::ProjectionOp>::OpConversionPattern;
   mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::ValueRange leftUnpacked, mlir::ValueRange rightUnpacked, mlir::Location loc) const {
      mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      for (size_t i = 0; i < leftUnpacked.size(); i++) {
         mlir::Value compared;
         auto currLeftType = leftUnpacked[i].getType();
         auto currRightType = rightUnpacked[i].getType();
         auto currLeftNullableType = currLeftType.dyn_cast_or_null<mlir::db::NullableType>();
         auto currRightNullableType = currRightType.dyn_cast_or_null<mlir::db::NullableType>();
         if (currLeftNullableType || currRightNullableType) {
            mlir::Value isNull1 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), leftUnpacked[i]);
            mlir::Value isNull2 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), rightUnpacked[i]);
            mlir::Value anyNull = rewriter.create<mlir::arith::OrIOp>(loc, isNull1, isNull2);
            mlir::Value bothNull = rewriter.create<mlir::arith::AndIOp>(loc, isNull1, isNull2);
            compared = rewriter.create<mlir::scf::IfOp>(
                                  loc, rewriter.getI1Type(), anyNull, [&](mlir::OpBuilder& b, mlir::Location loc) { b.create<mlir::scf::YieldOp>(loc, bothNull); },
                                  [&](mlir::OpBuilder& b, mlir::Location loc) {
                                     mlir::Value left = rewriter.create<mlir::db::NullableGetVal>(loc, leftUnpacked[i]);
                                     mlir::Value right = rewriter.create<mlir::db::NullableGetVal>(loc, rightUnpacked[i]);
                                     mlir::Value cmpRes = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, left, right);
                                     b.create<mlir::scf::YieldOp>(loc, cmpRes);
                                  })
                          .getResult(0);
         } else {
            compared = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, leftUnpacked[i], rightUnpacked[i]);
         }
         mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), mlir::ValueRange({equal, compared}));
         equal = localEqual;
      }
      return equal;
   }
   LogicalResult matchAndRewrite(mlir::relalg::ProjectionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.set_semantic() != mlir::relalg::SetSemantic::distinct) return failure();
      auto* context = getContext();
      auto loc = projectionOp->getLoc();

      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<mlir::Location> locations;
      std::vector<NamedAttribute> defMapping;
      for (auto x : projectionOp.cols()) {
         auto ref = x.cast<mlir::tuples::ColumnRefAttr>();
         auto memberName = getUniqueMember("keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
         locations.push_back(projectionOp->getLoc());
      }
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {}), mlir::ArrayAttr::get(context, {}));
      mlir::Value state;
      auto stateType = mlir::subop::HashMapType::get(rewriter.getContext(), keyMembers, stateMembers);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
         auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{});
         state = createOp.res();
      }
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      referenceDefAttr.getColumn().type = mlir::subop::EntryRefType::get(context, stateType);
      auto lookupOp = rewriter.create<mlir::subop::LookupOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), adaptor.rel(), state, projectionOp.cols(), referenceDefAttr);
      auto* initialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initialValueBlock);
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc());
      }
      lookupOp.initFn().push_back(initialValueBlock);
      mlir::Block* equalBlock = new Block;
      lookupOp.eqFn().push_back(equalBlock);
      equalBlock->addArguments(keyTypes, locations);
      equalBlock->addArguments(keyTypes, locations);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(equalBlock);
         mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc(), compared);
      }
      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));

      rewriter.replaceOp(projectionOp, scan);
      return success();
   }
};
class MaterializationHelper {
   std::vector<NamedAttribute> defMapping;
   std::vector<NamedAttribute> refMapping;
   std::vector<Attribute> types;
   std::vector<Attribute> names;
   std::unordered_map<const mlir::tuples::Column*, size_t> colToMemberPos;
   mlir::MLIRContext* context;

   public:
   MaterializationHelper(const mlir::relalg::ColumnSet& columns, mlir::MLIRContext* context) : context(context) {
      size_t i = 0;
      for (auto* x : columns) {
         types.push_back(mlir::TypeAttr::get(x->type));
         colToMemberPos[x] = i++;
         std::string name = getUniqueMember("member");
         auto nameAttr = mlir::StringAttr::get(context, name);
         names.push_back(nameAttr);
         defMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(x)));
         refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(x)));
      }
   }
   std::pair<const mlir::tuples::Column*, std::string> addFlag() {
      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto colDef = colManager.createDef(colManager.getUniqueScope("flag"), "flag");
      auto i1Type = mlir::IntegerType::get(context, 1);
      colDef.getColumn().type = i1Type;
      auto* x = &colDef.getColumn();
      types.push_back(mlir::TypeAttr::get(i1Type));
      colToMemberPos[x] = names.size();
      std::string name = getUniqueMember("flag");
      auto nameAttr = mlir::StringAttr::get(context, name);
      names.push_back(nameAttr);
      defMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(x)));
      refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(x)));
      return {&colDef.getColumn(), name};
   }
   mlir::subop::StateMembersAttr createStateMembersAttr() {
      return mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, names), mlir::ArrayAttr::get(context, types));
   }

   mlir::DictionaryAttr createStateColumnMapping() {
      return mlir::DictionaryAttr::get(context, defMapping);
   }
   mlir::DictionaryAttr createColumnstateMapping() {
      return mlir::DictionaryAttr::get(context, refMapping);
   }
   mlir::StringAttr lookupStateMemberForMaterializedColumn(const mlir::tuples::Column* column) {
      return names.at(colToMemberPos.at(column)).cast<mlir::StringAttr>();
   }
};
static mlir::Value translateNLJoin(mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::OpBuilder& rewriter, mlir::Location loc) {
   MaterializationHelper helper(columns, rewriter.getContext());
   auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
      vector = rewriter.create<mlir::subop::CreateOp>(loc, vectorType, mlir::Attribute());
   }
   rewriter.create<mlir::subop::MaterializeOp>(loc, left, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), right);
   auto* b = new Block;
   b->addArguments(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.region().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto scan = rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping());
      rewriter.create<mlir::tuples::ReturnOp>(loc, scan.res());
   }
   return nestedMapOp.res();
}
static mlir::Value mapBool(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, bool value, const mlir::tuples::Column* column) {
   Block* mapBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      mlir::Value val = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
      rewriter.create<mlir::tuples::ReturnOp>(loc, val);
   }

   auto& columnManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(column);
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(markAttrDef));
   mapOp.fn().push_back(mapBlock);
   return mapOp.result();
}
static std::pair<mlir::Value, const mlir::tuples::Column*> mapBool(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, bool value) {
   Block* mapBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      mlir::Value val = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
      rewriter.create<mlir::tuples::ReturnOp>(loc, val);
   }

   auto& columnManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope("map");
   std::string attributeName = "boolval";
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = rewriter.getI1Type();
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(markAttrDef));
   mapOp.fn().push_back(mapBlock);
   return {mapOp.result(), &markAttrDef.getColumn()};
}
static mlir::Value mapColsToNull(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> defAttrs;
   Block* mapBlock = new Block;
   auto tupleArg = mapBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      std::vector<mlir::Value> res;
      for (mlir::Attribute attr : mapping) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         mlir::Value nullValue = rewriter.create<mlir::db::NullOp>(loc, defAttr->type);
         res.push_back(nullValue);
         defAttrs.push_back(colManager.createDef(defAttr));
      }
      rewriter.create<mlir::tuples::ReturnOp>(loc, res);
   }
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs));
   mapOp.fn().push_back(mapBlock);
   return mapOp.result();
}
static mlir::Value mapColsToNullable(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> defAttrs;
   Block* mapBlock = new Block;
   auto tupleArg = mapBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      std::vector<mlir::Value> res;
      for (mlir::Attribute attr : mapping) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
         const auto* refAttr = *mlir::relalg::ColumnSet::fromArrayAttr(fromExisting).begin();
         mlir::Value value = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), colManager.createRef(refAttr), tupleArg);
         if (refAttr->type != defAttr->type) {
            mlir::Value tmp = rewriter.create<mlir::db::AsNullableOp>(loc, defAttr->type, value);
            value = tmp;
         }
         res.push_back(value);
         defAttrs.push_back(colManager.createDef(defAttr));
      }
      rewriter.create<mlir::tuples::ReturnOp>(loc, res);
   }
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs));
   mapOp.fn().push_back(mapBlock);
   return mapOp.result();
}
static std::pair<mlir::Value, std::string> createMarkerState(mlir::OpBuilder& rewriter, mlir::Location loc) {
   auto memberName = getUniqueMember("marker");
   mlir::Type stateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(memberName)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getI1Type())})));
   Block* initialValueBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(initialValueBlock);
      mlir::Value val = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.create<mlir::tuples::ReturnOp>(loc, val);
   }
   auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute());
   createOp.initFn().push_back(initialValueBlock);

   return {createOp.res(), memberName};
}
static std::pair<mlir::Value, std::string> createCounterState(mlir::OpBuilder& rewriter, mlir::Location loc) {
   auto memberName = getUniqueMember("counter");
   mlir::Type stateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(memberName)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getI64Type())})));
   Block* initialValueBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(initialValueBlock);
      mlir::Value val = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      rewriter.create<mlir::tuples::ReturnOp>(loc, val);
   }
   auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute());
   createOp.initFn().push_back(initialValueBlock);

   return {createOp.res(), memberName};
}
static mlir::Value translateSemiJoin(mlir::Region& predicate, mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, bool anti = false) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   MaterializationHelper helper(columns, rewriter.getContext());
   auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
      vector = rewriter.create<mlir::subop::CreateOp>(loc, vectorType, mlir::Attribute());
   }
   rewriter.create<mlir::subop::MaterializeOp>(loc, right, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), left);
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.region().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping());
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, scan, tuple);
      auto filtered = translateSelection(combined, predicate, rewriter, loc);
      auto [mapped, boolColumn] = mapBool(filtered, rewriter, loc, true);
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      referenceDefAttr.getColumn().type = mlir::subop::EntryRefType::get(rewriter.getContext(), markerState.getType());
      auto afterLookup = rewriter.create<mlir::subop::LookupOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(rewriter.getContext()), mapped, markerState, rewriter.getArrayAttr({}), referenceDefAttr);
      rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterLookup, colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, colManager.createRef(boolColumn))));
      auto markerDefAttr = colManager.createDef(colManager.getUniqueScope("marker"), "marker");
      markerDefAttr.getColumn().type = rewriter.getI1Type();
      Value scanState = rewriter.create<mlir::subop::ScanOp>(loc, markerState, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, markerDefAttr)));
      Value filtered2 = rewriter.create<mlir::subop::FilterOp>(loc, scanState, anti ? mlir::subop::FilterSemantic::none_true : mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr({colManager.createRef(&markerDefAttr.getColumn())}));
      rewriter.create<mlir::tuples::ReturnOp>(loc, filtered2);
   }
   return nestedMapOp.res();
}
static mlir::Value translateOuterJoin(mlir::ArrayAttr mapping, mlir::Region& predicate, mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   MaterializationHelper helper(columns, rewriter.getContext());
   auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
      vector = rewriter.create<mlir::subop::CreateOp>(loc, vectorType, mlir::Attribute());
   }
   rewriter.create<mlir::subop::MaterializeOp>(loc, right, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), left);
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.region().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping());
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, scan, tuple);
      auto filtered = translateSelection(combined, predicate, rewriter, loc);
      auto [mapped, boolColumn] = mapBool(filtered, rewriter, loc, true);
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      referenceDefAttr.getColumn().type = mlir::subop::EntryRefType::get(rewriter.getContext(), markerState.getType());
      auto afterLookup = rewriter.create<mlir::subop::LookupOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(rewriter.getContext()), mapped, markerState, rewriter.getArrayAttr({}), referenceDefAttr);
      rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterLookup, colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, colManager.createRef(boolColumn))));
      auto markerDefAttr = colManager.createDef(colManager.getUniqueScope("marker"), "marker");
      markerDefAttr.getColumn().type = rewriter.getI1Type();
      Value scanState = rewriter.create<mlir::subop::ScanOp>(loc, markerState, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, markerDefAttr)));
      Value filteredNoMatch = rewriter.create<mlir::subop::FilterOp>(loc, scanState, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({colManager.createRef(&markerDefAttr.getColumn())}));
      auto mappedNull = mapColsToNull(filteredNoMatch, rewriter, loc, mapping);
      auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, mapping);
      Value unioned = rewriter.create<mlir::subop::UnionOp>(loc, mappedNullable, mappedNull);
      rewriter.create<mlir::tuples::ReturnOp>(loc, unioned);
   }
   return nestedMapOp.res();
}
static mlir::Value translateReverseSemiJoin(mlir::Region& predicate, mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, bool anti = false) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   MaterializationHelper helper(columns, rewriter.getContext());
   auto [flagColumn, flagMember] = helper.addFlag();
   auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
      vector = rewriter.create<mlir::subop::CreateOp>(loc, vectorType, mlir::Attribute());
   }
   left = mapBool(left, rewriter, loc, false, flagColumn);
   rewriter.create<mlir::subop::MaterializeOp>(loc, left, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), right);
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.region().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("scan"), "ref");
      referenceDefAttr.getColumn().type = mlir::subop::EntryRefType::get(rewriter.getContext(), vectorType);
      mlir::Value scan = rewriter.create<mlir::subop::ScanRefsOp>(loc, vector, referenceDefAttr);

      mlir::Value gathered = rewriter.create<mlir::subop::GatherOp>(loc, scan, colManager.createRef(&referenceDefAttr.getColumn()), helper.createStateColumnMapping());
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, gathered, tuple);
      auto filtered = translateSelection(combined, predicate, rewriter, loc);
      auto markerDefAttr = colManager.createDef(colManager.getUniqueScope("marker"), "marker");
      markerDefAttr.getColumn().type = rewriter.getI1Type();
      auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
      rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterBool, colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, colManager.createRef(&markerDefAttr.getColumn()))));
      rewriter.create<mlir::tuples::ReturnOp>(loc);
   }
   mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping());
   auto filtered = rewriter.create<mlir::subop::FilterOp>(loc, scan, anti ? mlir::subop::FilterSemantic::none_true : mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr({colManager.createRef(flagColumn)}));

   return filtered.res();
}
class CrossProductLowering : public OpConversionPattern<mlir::relalg::CrossProductOp> {
   public:
   using OpConversionPattern<mlir::relalg::CrossProductOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::CrossProductOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(crossProductOp, translateNLJoin(adaptor.left(), adaptor.right(), mlir::cast<Operator>(crossProductOp.left().getDefiningOp()).getAvailableColumns(), rewriter, crossProductOp->getLoc()));
      return success();
   }
};
class SemiJoinLowering : public OpConversionPattern<mlir::relalg::SemiJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::SemiJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::SemiJoinOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(crossProductOp, translateReverseSemiJoin(crossProductOp.predicate(), adaptor.left(), adaptor.right(), mlir::cast<Operator>(crossProductOp.left().getDefiningOp()).getAvailableColumns(), rewriter, crossProductOp->getLoc()));
      return success();
   }
};
class AntiSemiJoinLowering : public OpConversionPattern<mlir::relalg::AntiSemiJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::AntiSemiJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::AntiSemiJoinOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(crossProductOp, translateReverseSemiJoin(crossProductOp.predicate(), adaptor.left(), adaptor.right(), mlir::cast<Operator>(crossProductOp.left().getDefiningOp()).getAvailableColumns(), rewriter, crossProductOp->getLoc(), true));
      return success();
   }
};
class OuterJoinLowering : public OpConversionPattern<mlir::relalg::OuterJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::OuterJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::OuterJoinOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(crossProductOp, translateOuterJoin(crossProductOp.mapping(), crossProductOp.predicate(), adaptor.left(), adaptor.right(), mlir::cast<Operator>(crossProductOp.right().getDefiningOp()).getAvailableColumns(), rewriter, crossProductOp->getLoc()));
      return success();
   }
};
class SingleJoinLowering : public OpConversionPattern<mlir::relalg::SingleJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::SingleJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::SingleJoinOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(crossProductOp, translateOuterJoin(crossProductOp.mapping(), crossProductOp.predicate(), adaptor.left(), adaptor.right(), mlir::cast<Operator>(crossProductOp.right().getDefiningOp()).getAvailableColumns(), rewriter, crossProductOp->getLoc()));
      return success();
   }
};
class LimitLowering : public OpConversionPattern<mlir::relalg::LimitOp> {
   public:
   using OpConversionPattern<mlir::relalg::LimitOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::LimitOp limitOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto [counterState, counterName] = createCounterState(rewriter, limitOp->getLoc());
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      referenceDefAttr.getColumn().type = mlir::subop::EntryRefType::get(rewriter.getContext(), counterState.getType());
      auto counterDefAttr = colManager.createDef(colManager.getUniqueScope("limit"), "counter");
      counterDefAttr.getColumn().type = rewriter.getI64Type();
      auto afterLookup = rewriter.create<mlir::subop::LookupOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.rel(), counterState, rewriter.getArrayAttr({}), referenceDefAttr);
      auto gathered = rewriter.create<mlir::subop::GatherOp>(rewriter.getUnknownLoc(), afterLookup, colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getDictionaryAttr(rewriter.getNamedAttr(counterName, counterDefAttr)));

      tuples::ColumnDefAttr markAttrDef = colManager.createDef(colManager.getUniqueScope("map"), "predicate");
      markAttrDef.getColumn().type = rewriter.getI1Type();
      tuples::ColumnDefAttr updatedCounterDef = colManager.createDef(colManager.getUniqueScope("map"), "counter");
      updatedCounterDef.getColumn().type = rewriter.getI64Type();
      auto mapOp = rewriter.create<mlir::subop::MapOp>(limitOp->getLoc(), mlir::tuples::TupleStreamType::get(rewriter.getContext()), gathered, rewriter.getArrayAttr({markAttrDef, updatedCounterDef}));
      auto* mapBlock = new mlir::Block;
      mlir::Value tuple = mapBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), limitOp->getLoc());
      mapOp.fn().push_back(mapBlock);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         mlir::Value currCounter = rewriter.create<mlir::tuples::GetColumnOp>(limitOp->getLoc(), rewriter.getI64Type(), colManager.createRef(&counterDefAttr.getColumn()), tuple);
         mlir::Value limitVal = rewriter.create<mlir::db::ConstantOp>(limitOp->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(limitOp.rows()));
         mlir::Value one = rewriter.create<mlir::db::ConstantOp>(limitOp->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
         mlir::Value ltLimit = rewriter.create<mlir::db::CmpOp>(limitOp->getLoc(), mlir::db::DBCmpPredicate::lt, currCounter, limitVal);
         mlir::Value updatedCounter = rewriter.create<mlir::db::AddOp>(limitOp->getLoc(), currCounter, one);
         rewriter.create<mlir::tuples::ReturnOp>(limitOp->getLoc(), mlir::ValueRange{ltLimit, updatedCounter});
      }

      rewriter.replaceOpWithNewOp<mlir::subop::FilterOp>(limitOp, mapOp.result(), mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(colManager.createRef(&markAttrDef.getColumn())));
      rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), mapOp.result(), colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getDictionaryAttr(rewriter.getNamedAttr(counterName, colManager.createRef(&updatedCounterDef.getColumn()))));
      return success();
   }
};
class InnerJoinNLLowering : public OpConversionPattern<mlir::relalg::InnerJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::InnerJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::InnerJoinOp innerJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto cp = translateNLJoin(adaptor.left(), adaptor.right(), mlir::cast<Operator>(innerJoinOp.left().getDefiningOp()).getAvailableColumns(), rewriter, innerJoinOp->getLoc());
      rewriter.replaceOp(innerJoinOp, translateSelection(cp, innerJoinOp.predicate(), rewriter, innerJoinOp->getLoc()));
      return success();
   }
};
class SortLowering : public OpConversionPattern<mlir::relalg::SortOp> {
   public:
   using OpConversionPattern<mlir::relalg::SortOp>::OpConversionPattern;
   mlir::Value createSortPredicate(mlir::OpBuilder& builder, std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria, mlir::Value trueVal, mlir::Value falseVal, size_t pos, mlir::Location loc) const {
      if (pos < sortCriteria.size()) {
         mlir::Value lt = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, sortCriteria[pos].first, sortCriteria[pos].second);
         lt = builder.create<mlir::db::DeriveTruth>(loc, lt);
         auto ifOp = builder.create<mlir::scf::IfOp>(
            loc, builder.getI1Type(), lt, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, trueVal); }, [&](mlir::OpBuilder& builder, mlir::Location loc) {
               mlir::Value eq = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, sortCriteria[pos].first, sortCriteria[pos].second);
               eq=builder.create<mlir::db::DeriveTruth>(loc,eq);
               auto ifOp2 = builder.create<mlir::scf::IfOp>(loc, builder.getI1Type(), eq,[&](mlir::OpBuilder& builder, mlir::Location loc) {
                     builder.create<mlir::scf::YieldOp>(loc, createSortPredicate(builder, sortCriteria, trueVal, falseVal, pos + 1,loc));
                  },[&](mlir::OpBuilder& builder, mlir::Location loc) {
                     builder.create<mlir::scf::YieldOp>(loc, falseVal);
                  });
               builder.create<mlir::scf::YieldOp>(loc, ifOp2.getResult(0)); });
         return ifOp.getResult(0);
      } else {
         return falseVal;
      }
   }
   LogicalResult matchAndRewrite(mlir::relalg::SortOp sortOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      MaterializationHelper helper(sortOp.getAvailableColumns(), rewriter.getContext());
      auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
         vector = rewriter.create<mlir::subop::CreateOp>(sortOp->getLoc(), vectorType, mlir::Attribute{});
      }
      rewriter.create<mlir::subop::MaterializeOp>(sortOp->getLoc(), adaptor.rel(), vector, helper.createColumnstateMapping());
      auto* block = new Block;
      std::vector<Attribute> sortByMembers;
      std::vector<Type> argumentTypes;
      std::vector<Location> locs;
      for (auto attr : sortOp.sortspecs()) {
         auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
         argumentTypes.push_back(sortspecAttr.getAttr().getColumn().type);
         locs.push_back(sortOp->getLoc());
         sortByMembers.push_back(helper.lookupStateMemberForMaterializedColumn(&sortspecAttr.getAttr().getColumn()));
      }
      block->addArguments(argumentTypes, locs);
      block->addArguments(argumentTypes, locs);
      std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
      for (auto attr : sortOp.sortspecs()) {
         auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
         mlir::Value left = block->getArgument(sortCriteria.size());
         mlir::Value right = block->getArgument(sortCriteria.size() + sortOp.sortspecs().size());
         if (sortspecAttr.getSortSpec() == mlir::relalg::SortSpec::desc) {
            std::swap(left, right);
         }
         sortCriteria.push_back({left, right});
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto trueVal = rewriter.create<mlir::db::ConstantOp>(sortOp->getLoc(), rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         auto falseVal = rewriter.create<mlir::db::ConstantOp>(sortOp->getLoc(), rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         rewriter.create<mlir::tuples::ReturnOp>(sortOp->getLoc(), createSortPredicate(rewriter, sortCriteria, trueVal, falseVal, 0, sortOp->getLoc()));
      }
      auto subOpSort = rewriter.create<mlir::subop::SortOp>(sortOp->getLoc(), vector, rewriter.getArrayAttr(sortByMembers));
      subOpSort.region().getBlocks().push_back(block);
      rewriter.replaceOpWithNewOp<mlir::subop::ScanOp>(sortOp, vector, helper.createStateColumnMapping());
      return success();
   }
};
class TmpLowering : public OpConversionPattern<mlir::relalg::TmpOp> {
   public:
   using OpConversionPattern<mlir::relalg::TmpOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::relalg::TmpOp tmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      MaterializationHelper helper(tmpOp.getAvailableColumns(), rewriter.getContext());

      auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
         vector = rewriter.create<mlir::subop::CreateOp>(tmpOp->getLoc(), vectorType, mlir::Attribute{});
      }
      rewriter.create<mlir::subop::MaterializeOp>(tmpOp->getLoc(), adaptor.rel(), vector, helper.createColumnstateMapping());
      std::vector<mlir::Value> results;
      for (size_t i = 0; i < tmpOp.getNumResults(); i++) {
         results.push_back(rewriter.create<mlir::subop::ScanOp>(tmpOp->getLoc(), vector, helper.createStateColumnMapping()));
      }
      rewriter.replaceOp(tmpOp, results);
      return success();
   }
};
class MaterializeLowering : public OpConversionPattern<mlir::relalg::MaterializeOp> {
   public:
   using OpConversionPattern<mlir::relalg::MaterializeOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Attribute> colNames;
      std::vector<Attribute> colMemberNames;
      std::vector<Attribute> colTypes;
      std::vector<NamedAttribute> mapping;
      for (size_t i = 0; i < materializeOp.columns().size(); i++) {
         auto columnName = materializeOp.columns()[i].cast<mlir::StringAttr>().str();
         auto colMemberName = getUniqueMember(columnName);
         auto columnAttr = materializeOp.cols()[i].cast<mlir::tuples::ColumnRefAttr>();
         auto columnType = columnAttr.getColumn().type;
         colNames.push_back(rewriter.getStringAttr(columnName));
         colMemberNames.push_back(rewriter.getStringAttr(colMemberName));
         colTypes.push_back(mlir::TypeAttr::get(columnType));
         mapping.push_back(rewriter.getNamedAttr(colMemberName, columnAttr));
      }
      auto tableRefType = mlir::subop::TableType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(colMemberNames), rewriter.getArrayAttr(colTypes)));
      mlir::Value table;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
         table = rewriter.create<mlir::subop::CreateOp>(materializeOp->getLoc(), tableRefType, rewriter.getArrayAttr(colNames));
      }
      rewriter.create<mlir::subop::MaterializeOp>(materializeOp->getLoc(), adaptor.rel(), table, rewriter.getDictionaryAttr(mapping));
      rewriter.replaceOpWithNewOp<mlir::subop::ConvertToExplicit>(materializeOp, mlir::dsa::TableType::get(rewriter.getContext()), table);

      return success();
   }
};
class AggregationLowering : public OpConversionPattern<mlir::relalg::AggregationOp> {
   public:
   using OpConversionPattern<mlir::relalg::AggregationOp>::OpConversionPattern;
   mlir::Attribute getMaxValueAttr(mlir::Type type) const {
      auto* context = type.getContext();
      mlir::OpBuilder builder(context);
      mlir::Attribute maxValAttr = ::llvm::TypeSwitch<::mlir::Type, mlir::Attribute>(type)

                                      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                                         if (t.getP() < 19) {
                                            return (mlir::Attribute) builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                         }
                                         std::vector<uint64_t> parts = {0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF};
                                         return (mlir::Attribute) builder.getIntegerAttr(mlir::IntegerType::get(context, 128), mlir::APInt(128, parts));
                                      })
                                      .Case<::mlir::IntegerType>([&](::mlir::IntegerType) {
                                         return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                      })
                                      .Case<::mlir::FloatType>([&](::mlir::FloatType t) {
                                         if (t.getWidth() == 32) {
                                            return (mlir::Attribute) builder.getF32FloatAttr(std::numeric_limits<float>::max());
                                         } else if (t.getWidth() == 64) {
                                            return (mlir::Attribute) builder.getF64FloatAttr(std::numeric_limits<double>::max());
                                         } else {
                                            assert(false && "should not happen");
                                            return mlir::Attribute();
                                         }
                                      })
                                      .Default([&](::mlir::Type) { return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max()); });
      return maxValAttr;
   }
   using finalizeFnT = std::function<std::pair<const mlir::tuples::Column*, mlir::Value>(mlir::ValueRange, mlir::OpBuilder& builder)>;
   using aggregationFnT = std::function<std::vector<mlir::Value>(mlir::ValueRange, mlir::ValueRange, mlir::OpBuilder& builder)>;
   struct AnalyzedAggregation {
      mlir::TupleType keyTupleType;
      mlir::TupleType valTupleType;

      mlir::relalg::OrderedAttributes key;
      mlir::relalg::OrderedAttributes val;

      std::vector<std::function<std::pair<const mlir::tuples::Column*, mlir::Value>(mlir::ValueRange, mlir::OpBuilder& builder)>> finalizeFunctions;
      std::vector<std::function<std::vector<mlir::Value>(mlir::ValueRange, mlir::ValueRange, mlir::OpBuilder& builder)>> aggregationFunctions;
      std::vector<mlir::Value> defaultValues;
      std::vector<mlir::Type> aggrTypes;
   };
   mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::ValueRange leftUnpacked, mlir::ValueRange rightUnpacked, mlir::Location loc) const {
      mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      for (size_t i = 0; i < leftUnpacked.size(); i++) {
         mlir::Value compared;
         auto currLeftType = leftUnpacked[i].getType();
         auto currRightType = rightUnpacked[i].getType();
         auto currLeftNullableType = currLeftType.dyn_cast_or_null<mlir::db::NullableType>();
         auto currRightNullableType = currRightType.dyn_cast_or_null<mlir::db::NullableType>();
         if (currLeftNullableType || currRightNullableType) {
            mlir::Value isNull1 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), leftUnpacked[i]);
            mlir::Value isNull2 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), rightUnpacked[i]);
            mlir::Value anyNull = rewriter.create<mlir::arith::OrIOp>(loc, isNull1, isNull2);
            mlir::Value bothNull = rewriter.create<mlir::arith::AndIOp>(loc, isNull1, isNull2);
            compared = rewriter.create<mlir::scf::IfOp>(
                                  loc, rewriter.getI1Type(), anyNull, [&](mlir::OpBuilder& b, mlir::Location loc) { b.create<mlir::scf::YieldOp>(loc, bothNull); },
                                  [&](mlir::OpBuilder& b, mlir::Location loc) {
                                     mlir::Value left = rewriter.create<mlir::db::NullableGetVal>(loc, leftUnpacked[i]);
                                     mlir::Value right = rewriter.create<mlir::db::NullableGetVal>(loc, rightUnpacked[i]);
                                     mlir::Value cmpRes = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, left, right);
                                     b.create<mlir::scf::YieldOp>(loc, cmpRes);
                                  })
                          .getResult(0);
         } else {
            compared = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, leftUnpacked[i], rightUnpacked[i]);
         }
         mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), mlir::ValueRange({equal, compared}));
         equal = localEqual;
      }
      return equal;
   }
   void analyze(mlir::relalg::AggregationOp aggregationOp, mlir::OpBuilder& builder, AnalyzedAggregation& analyzedAggregation) const {
      analyzedAggregation.key = mlir::relalg::OrderedAttributes::fromRefArr(aggregationOp.group_by_colsAttr());

      auto counterType = builder.getI64Type();
      mlir::tuples::ReturnOp terminator = mlir::cast<mlir::tuples::ReturnOp>(aggregationOp.aggr_func().front().getTerminator());

      for (size_t i = 0; i < aggregationOp.computed_cols().size(); i++) {
         auto* destAttr = &aggregationOp.computed_cols()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         mlir::Value computedVal = terminator.results()[i];
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            auto loc = aggrFn->getLoc();
            auto* attr = &aggrFn.attr().getColumn();
            auto attrIsNullable = attr->type.isa<mlir::db::NullableType>();
            size_t currValIdx = analyzedAggregation.val.insert(attr);
            mlir::Type resultingType = destAttr->type;
            size_t currDestIdx = analyzedAggregation.aggrTypes.size();

            if (aggrFn.fn() == mlir::relalg::AggrFunc::sum) {
               analyzedAggregation.finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               analyzedAggregation.aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(loc, resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(loc, getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               analyzedAggregation.defaultValues.push_back(initVal);
               analyzedAggregation.aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value added = builder.create<mlir::db::AddOp>(loc, resultingType, currVal, newVal);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), newVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, currVal, added);
                  }
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value casted = newVal;
                     if (currVal.getType() != newVal.getType()) {
                        casted = builder.create<mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                     }
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), currVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, updatedVal));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::min) {
               analyzedAggregation.finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               analyzedAggregation.aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(loc, resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(loc, getBaseType(resultingType), getMaxValueAttr(resultingType));
               }
               analyzedAggregation.defaultValues.push_back(initVal);
               analyzedAggregation.aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value newLtCurr = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, newVal, currVal);
                  mlir::Value casted = newVal;
                  if (newVal.getType() != currVal.getType()) {
                     casted = builder.create<mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                  }
                  mlir::Value newLtCurrT = builder.create<mlir::db::DeriveTruth>(loc, newLtCurr);

                  mlir::Value added = builder.create<mlir::arith::SelectOp>(loc, newLtCurrT, casted, currVal);
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), currVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, added));
                  } else {
                     res.push_back(added);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::max) {
               analyzedAggregation.finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               analyzedAggregation.aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(loc, resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(loc, getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               analyzedAggregation.defaultValues.push_back(initVal);
               analyzedAggregation.aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value currGtNew = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::gt, currVal, newVal);
                  mlir::Value casted = newVal;
                  if (newVal.getType() != currVal.getType()) {
                     casted = builder.create<mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                  }
                  mlir::Value currGTNewT = builder.create<mlir::db::DeriveTruth>(loc, currGtNew);
                  mlir::Value added = builder.create<mlir::arith::SelectOp>(loc, currGTNewT, currVal, casted);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), newVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, updatedVal));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::avg) {
               analyzedAggregation.aggrTypes.push_back(resultingType);
               analyzedAggregation.aggrTypes.push_back(counterType);
               mlir::Value initVal = builder.create<mlir::db::ConstantOp>(loc, getBaseType(resultingType), builder.getI64IntegerAttr(0));
               mlir::Value initCounterVal = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(0));
               mlir::Value defaultVal = resultingType.isa<mlir::db::NullableType>() ? builder.create<mlir::db::AsNullableOp>(loc, resultingType, initVal) : initVal;
               analyzedAggregation.defaultValues.push_back(defaultVal);
               analyzedAggregation.defaultValues.push_back(initCounterVal);
               analyzedAggregation.finalizeFunctions.push_back([loc, currDestIdx = currDestIdx, destAttr = destAttr, resultingType = resultingType](mlir::ValueRange range, mlir::OpBuilder builder) {
                  mlir::Value casted=builder.create<mlir::db::CastOp>(loc, getBaseType(resultingType), range[currDestIdx+1]);
                  if(resultingType.isa<mlir::db::NullableType>()&&casted.getType()!=resultingType){
                     casted=builder.create<mlir::db::AsNullableOp>(loc, resultingType, casted);
                  }
                  mlir::Value average=builder.create<mlir::db::DivOp>(loc, resultingType, range[currDestIdx], casted);
                  return std::make_pair(destAttr, average); });
               analyzedAggregation.aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  mlir::Value added1 = builder.create<mlir::db::AddOp>(loc, resultingType, aggr[currDestIdx], val[currValIdx]);
                  mlir::Value added2 = builder.create<mlir::db::AddOp>(loc, counterType, aggr[currDestIdx + 1], one);
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), val[currValIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added1));
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx + 1], added2));
                  } else {
                     res.push_back(added1);
                     res.push_back(added2);
                  }

                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::count) {
               size_t currDestIdx = analyzedAggregation.aggrTypes.size();
               auto initCounterVal = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(0));
               analyzedAggregation.defaultValues.push_back(initCounterVal);
               analyzedAggregation.aggrTypes.push_back(resultingType);
               analyzedAggregation.finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               analyzedAggregation.aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, currValIdx = currValIdx, counterType = counterType, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  mlir::Value value = builder.create<mlir::db::AddOp>(loc, resultingType, aggr[currDestIdx], one);
                  if (attrIsNullable) {
                     mlir::Value isNull2 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), val[currValIdx]);
                     mlir::Value tmp = builder.create<mlir::arith::SelectOp>(loc, isNull2, aggr[currDestIdx], value);
                     value = tmp;
                  }

                  res.push_back(value);
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::any) {
               size_t currDestIdx = analyzedAggregation.aggrTypes.size();
               auto initVal = builder.create<mlir::util::UndefOp>(loc, resultingType);
               analyzedAggregation.defaultValues.push_back(initVal);
               analyzedAggregation.aggrTypes.push_back(resultingType);
               analyzedAggregation.finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               analyzedAggregation.aggregationFunctions.push_back([currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  res.push_back(val[currValIdx]);
                  return res;
               });
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            auto loc = countOp->getLoc();

            size_t currDestIdx = analyzedAggregation.aggrTypes.size();
            analyzedAggregation.aggrTypes.push_back(counterType);
            auto initCounterVal = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(0));
            analyzedAggregation.defaultValues.push_back(initCounterVal);
            analyzedAggregation.finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

            analyzedAggregation.aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
               std::vector<mlir::Value> res;
               auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
               mlir::Value added2 = builder.create<mlir::db::AddOp>(loc, counterType, aggr[currDestIdx], one);
               res.push_back(added2);
               return res;
            });
         }
      };
      analyzedAggregation.keyTupleType = analyzedAggregation.key.getTupleType(builder.getContext());
      analyzedAggregation.valTupleType = analyzedAggregation.val.getTupleType(builder.getContext());
   }
   LogicalResult matchAndRewrite(mlir::relalg::AggregationOp aggregationOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = aggregationOp->getLoc();
      auto* context = rewriter.getContext();
      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      mlir::Value state;
      AnalyzedAggregation analyzedAggregation;
      Block* initialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initialValueBlock);
         analyze(aggregationOp, rewriter, analyzedAggregation);
         rewriter.create<mlir::tuples::ReturnOp>(loc, analyzedAggregation.defaultValues);
      }
      std::vector<mlir::Attribute> names;
      std::vector<mlir::Attribute> types;
      std::vector<mlir::tuples::ColumnRefAttr> stateColumnsRef;
      std::vector<mlir::Attribute> stateColumnsDef;
      std::vector<NamedAttribute> defMapping;

      for (auto t : analyzedAggregation.aggrTypes) {
         auto memberName = getUniqueMember("aggrval");
         names.push_back(rewriter.getStringAttr(memberName));
         types.push_back(mlir::TypeAttr::get(t));
         auto def = colManager.createDef(colManager.getUniqueScope("aggrval"), "aggrval");
         def.getColumn().type = t;
         stateColumnsRef.push_back(colManager.createRef(&def.getColumn()));
         stateColumnsDef.push_back(def);
         defMapping.push_back(rewriter.getNamedAttr(memberName, def));
      }
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, names), mlir::ArrayAttr::get(context, types));
      mlir::Type stateType;

      mlir::Value afterLookup;
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");

      if (aggregationOp.group_by_cols().empty()) {
         stateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), stateMembers);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
            auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{});
            createOp.initFn().push_back(initialValueBlock);
            state = createOp.res();
         }
         afterLookup = rewriter.create<mlir::subop::LookupOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), adaptor.rel(), state, rewriter.getArrayAttr({}), referenceDefAttr);

      } else {
         std::vector<mlir::Attribute> keyNames;
         std::vector<mlir::Attribute> keyTypesAttr;
         std::vector<mlir::Type> keyTypes;
         std::vector<mlir::Location> locations;
         for (auto x : aggregationOp.group_by_cols()) {
            auto ref = x.cast<mlir::tuples::ColumnRefAttr>();
            auto memberName = getUniqueMember("keyval");
            keyNames.push_back(rewriter.getStringAttr(memberName));
            keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
            keyTypes.push_back((ref.getColumn().type));
            defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
            locations.push_back(aggregationOp->getLoc());
         }
         auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
         stateType = mlir::subop::HashMapType::get(rewriter.getContext(), keyMembers, stateMembers);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
            auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{});
            state = createOp.res();
         }
         auto lookupOp = rewriter.create<mlir::subop::LookupOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), adaptor.rel(), state, aggregationOp.group_by_cols(), referenceDefAttr);
         afterLookup = lookupOp;
         lookupOp.initFn().push_back(initialValueBlock);
         mlir::Block* equalBlock = new Block;
         lookupOp.eqFn().push_back(equalBlock);
         equalBlock->addArguments(keyTypes, locations);
         equalBlock->addArguments(keyTypes, locations);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(equalBlock);
            mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), aggregationOp->getLoc());
            rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc(), compared);
         }
      }
      referenceDefAttr.getColumn().type = mlir::subop::EntryRefType::get(context, stateType);
      auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(rewriter.getUnknownLoc(), afterLookup, colManager.createRef(&referenceDefAttr.getColumn()), analyzedAggregation.val.getArrayAttr(context), rewriter.getArrayAttr(names));
      mlir::Block* reduceBlock = new Block;
      size_t numColumns = analyzedAggregation.val.getAttrs().size();
      size_t numMembers = types.size();
      for (auto* c : analyzedAggregation.val.getAttrs()) {
         reduceBlock->addArgument(c->type, loc);
      }
      for (auto t : analyzedAggregation.aggrTypes) {
         reduceBlock->addArgument(t, loc);
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(reduceBlock);
         std::vector<mlir::Value> valuesx;
         for (auto aggrFn : analyzedAggregation.aggregationFunctions) {
            auto vec = aggrFn(reduceBlock->getArguments().drop_front(numColumns), reduceBlock->getArguments().drop_back(numMembers), rewriter);
            valuesx.insert(valuesx.end(), vec.begin(), vec.end());
         }
         rewriter.create<mlir::tuples::ReturnOp>(loc, valuesx);
      }
      reduceOp.region().push_back(reduceBlock);

      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));
      auto mapOp = rewriter.replaceOpWithNewOp<mlir::subop::MapOp>(aggregationOp, mlir::tuples::TupleStreamType::get(context), scan, aggregationOp.computed_cols());
      mlir::Block* mapBlock = new Block;
      mapBlock->addArgument(mlir::tuples::TupleType::get(context), loc);
      mapOp.fn().push_back(mapBlock);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         std::vector<mlir::Value> unpackedAggr;
         for (auto ref : stateColumnsRef) {
            unpackedAggr.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, ref.getColumn().type, ref, mapBlock->getArgument(0)));
         }
         std::vector<mlir::Value> res;
         for (auto fn : analyzedAggregation.finalizeFunctions) {
            auto [attr, val] = fn(unpackedAggr, rewriter);
            res.push_back(val);
         }
         rewriter.create<mlir::tuples::ReturnOp>(loc, res);
      }

      return success();
   }
};

void RelalgToSubOpLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper().setParentModule(module);

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();
   target.addIllegalDialect<relalg::RelAlgDialect>();
   target.addLegalDialect<subop::SubOperatorDialect>();
   target.addLegalDialect<db::DBDialect>();
   target.addLegalDialect<dsa::DSADialect>();

   target.addLegalDialect<tuples::TupleStreamDialect>();
   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   target.addLegalDialect<arith::ArithmeticDialect>();
   target.addLegalDialect<cf::ControlFlowDialect>();
   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();

   TypeConverter typeConverter;
   typeConverter.addConversion([](mlir::tuples::TupleStreamType t) { return t; });
   auto* ctxt = &getContext();

   RewritePatternSet patterns(&getContext());

   patterns.insert<BaseTableLowering>(typeConverter, ctxt);
   patterns.insert<SelectionLowering>(typeConverter, ctxt);
   patterns.insert<MapLowering>(typeConverter, ctxt);
   patterns.insert<SortLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeLowering>(typeConverter, ctxt);
   patterns.insert<RenamingLowering>(typeConverter, ctxt);
   patterns.insert<ProjectionAllLowering>(typeConverter, ctxt);
   patterns.insert<ProjectionDistinctLowering>(typeConverter, ctxt);
   patterns.insert<TmpLowering>(typeConverter, ctxt);
   patterns.insert<CrossProductLowering>(typeConverter, ctxt);
   patterns.insert<InnerJoinNLLowering>(typeConverter, ctxt);
   patterns.insert<AggregationLowering>(typeConverter, ctxt);
   patterns.insert<SemiJoinLowering>(typeConverter, ctxt);
   patterns.insert<AntiSemiJoinLowering>(typeConverter, ctxt);
   patterns.insert<OuterJoinLowering>(typeConverter, ctxt);
   patterns.insert<SingleJoinLowering>(typeConverter, ctxt);
   patterns.insert<LimitLowering>(typeConverter, ctxt);

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}
} //namespace
std::unique_ptr<mlir::Pass>
mlir::relalg::createLowerToSubOpPass() {
   return std::make_unique<RelalgToSubOpLoweringPass>();
}
void mlir::relalg::createLowerRelAlgToSubOpPipeline(mlir::OpPassManager& pm) {
   pm.addPass(mlir::relalg::createLowerToSubOpPass());
}
void mlir::relalg::registerRelAlgToSubOpConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createLowerToSubOpPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-relalg-to-subop",
      "",
      mlir::relalg::createLowerRelAlgToSubOpPipeline);
}