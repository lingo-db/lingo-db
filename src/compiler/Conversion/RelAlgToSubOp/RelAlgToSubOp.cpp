#include "lingodb/compiler/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"

#include "lingodb/compiler/Conversion/RelAlgToSubOp/OrderedAttributes.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"

#include <iostream>

using namespace mlir;

namespace {
using namespace lingodb::compiler::dialect;
struct RelalgToSubOpLoweringPass
   : public PassWrapper<RelalgToSubOpLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelalgToSubOpLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "to-subop"; }

   RelalgToSubOpLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect, relalg::RelAlgDialect, subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};
static std::string getUniqueMember(MLIRContext* context, std::string name) {
   auto& memberManager = context->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   return memberManager.getUniqueMember(name);
}
static relalg::ColumnSet getRequired(Operator op) {
   auto available = op.getAvailableColumns();

   relalg::ColumnSet required;
   for (auto* user : op->getUsers()) {
      if (auto consumingOp = mlir::dyn_cast_or_null<Operator>(user)) {
         required.insert(getRequired(consumingOp));
         required.insert(consumingOp.getUsedColumns());
      }
      if (auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(user)) {
         required.insert(relalg::ColumnSet::fromArrayAttr(materializeOp.getCols()));
      }
   }
   return available.intersect(required);
}
class BaseTableLowering : public OpConversionPattern<relalg::BaseTableOp> {
   public:
   using OpConversionPattern<relalg::BaseTableOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(relalg::BaseTableOp baseTableOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto required = getRequired(baseTableOp);
      std::vector<mlir::Type> types;
      std::vector<Attribute> colNames;
      std::vector<Attribute> colTypes;
      std::vector<NamedAttribute> mapping;
      std::string tableName = mlir::cast<mlir::StringAttr>(baseTableOp->getAttr("table_identifier")).str();
      std::string scanDescription = R"({ "table": ")" + tableName + R"(", "mapping": { )";
      bool first = true;
      for (auto namedAttr : baseTableOp.getColumns().getValue()) {
         auto identifier = namedAttr.getName();
         auto attr = namedAttr.getValue();
         auto attrDef = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         if (!first) {
            scanDescription += ",";
         } else {
            first = false;
         }
         auto memberName = getUniqueMember(getContext(), identifier.str());
         scanDescription += "\"" + memberName + "\" :\"" + identifier.str() + "\"";

         colNames.push_back(rewriter.getStringAttr(memberName));
         colTypes.push_back(mlir::TypeAttr::get(attrDef.getColumn().type));
         if (required.contains(&attrDef.getColumn())) {
            mapping.push_back(rewriter.getNamedAttr(memberName, attrDef));
         }
      }
      scanDescription += "} }";
      auto tableRefType = subop::TableType::get(rewriter.getContext(), subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(colNames), rewriter.getArrayAttr(colTypes)));
      mlir::Value tableRef = rewriter.create<subop::GetExternalOp>(baseTableOp->getLoc(), tableRefType, rewriter.getStringAttr(scanDescription));
      rewriter.replaceOpWithNewOp<subop::ScanOp>(baseTableOp, tableRef, rewriter.getDictionaryAttr(mapping));
      return success();
   }
};

static mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::ValueRange leftUnpacked, mlir::ValueRange rightUnpacked, mlir::Location loc) {
   mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
   for (size_t i = 0; i < leftUnpacked.size(); i++) {
      mlir::Value compared = rewriter.create<db::CmpOp>(loc, db::DBCmpPredicate::isa, leftUnpacked[i], rightUnpacked[i]);
      mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), mlir::ValueRange({equal, compared}));
      equal = localEqual;
   }
   return equal;
}
static std::pair<tuples::ColumnDefAttr, tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}

static mlir::Value map(mlir::Value stream, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::ArrayAttr createdColumns, std::function<std::vector<mlir::Value>(mlir::ConversionPatternRewriter&, subop::MapCreationHelper& helper, mlir::Location)> fn) {
   subop::MapCreationHelper helper(rewriter.getContext());
   helper.buildBlock(rewriter, [&](mlir::ConversionPatternRewriter& rewriter) {
      rewriter.create<tuples::ReturnOp>(loc, fn(rewriter, helper, loc));
   });
   auto mapOp = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, createdColumns, helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   return mapOp.getResult();
}
static mlir::Value translateSelection(mlir::Value stream, mlir::Region& predicate, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto terminator = mlir::cast<tuples::ReturnOp>(predicate.front().getTerminator());
   bool isTrivialSel = false;
   if (terminator->getNumOperands() == 1) {
      if (auto constOp = mlir::dyn_cast_or_null<arith::ConstantOp>(terminator->getOperand(0).getDefiningOp())) {
         if (auto boolAttr = mlir::dyn_cast_or_null<mlir::BoolAttr>(constOp.getValue())) {
            if (boolAttr.getValue()) {
               isTrivialSel = true;
            }
         }
      }
   }
   if (terminator.getResults().empty() || isTrivialSel) {
      return stream;
   } else {
      auto& predicateBlock = predicate.front();
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predicateBlock.getTerminator())) {
         mlir::Value matched = returnOp.getResults()[0];
         std::vector<std::pair<int, mlir::Value>> conditions;
         if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(matched.getDefiningOp())) {
            for (auto c : andOp.getVals()) {
               int p = 1000;
               if (auto* defOp = c.getDefiningOp()) {
                  if (auto betweenOp = mlir::dyn_cast_or_null<db::BetweenOp>(defOp)) {
                     auto t = betweenOp.getVal().getType();
                     p = ::llvm::TypeSwitch<mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::db::DateType>([&](::db::DateType t) { return 2; })
                            .Case<::db::DecimalType>([&](::db::DecimalType t) { return 3; })
                            .Case<::db::CharType, ::db::TimestampType, ::db::IntervalType, ::mlir::FloatType>([&](mlir::Type t) { return 2; })
                            .Case<::db::StringType>([&](::db::StringType t) { return 10; })
                            .Default([](::mlir::Type) { return 100; });
                     p -= 1;
                  } else if (auto cmpOp = mlir::dyn_cast_or_null<relalg::CmpOpInterface>(defOp)) {
                     auto t = cmpOp.getLeft().getType();
                     p = ::llvm::TypeSwitch<mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::db::DateType>([&](::db::DateType t) { return 2; })
                            .Case<::db::DecimalType>([&](::db::DecimalType t) { return 3; })
                            .Case<::db::CharType, ::db::TimestampType, ::db::IntervalType, ::mlir::FloatType>([&](mlir::Type t) { return 2; })
                            .Case<::db::StringType>([&](::db::StringType t) { return 10; })
                            .Default([](::mlir::Type) { return 100; });
                  }
                  conditions.push_back({p, c});
               }
            }
         } else {
            conditions.push_back({0, matched});
         }
         std::sort(conditions.begin(), conditions.end(), [](auto a, auto b) { return a.first < b.first; });
         for (auto c : conditions) {
            auto [predDef, predRef] = createColumn(rewriter.getI1Type(), "map", "pred");
            stream = map(stream, rewriter, loc, rewriter.getArrayAttr(predDef), [&](mlir::ConversionPatternRewriter& b, subop::MapCreationHelper& helper, mlir::Location loc) -> std::vector<mlir::Value> {
               mlir::IRMapping mapping;
               auto helperOp = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(0));
               relalg::detail::inlineOpIntoBlock(c.second.getDefiningOp(), c.second.getDefiningOp()->getParentOp(), b.getInsertionBlock(), mapping, helperOp);
               b.eraseOp(helperOp);
               mlir::Value predVal = mapping.lookupOrNull(c.second);
               if (mlir::isa<db::NullableType>(predVal.getType())) {
                  predVal = b.create<db::DeriveTruth>(loc, predVal);
               }
               std::vector<mlir::Operation*> toErase;
               b.getInsertionBlock()->walk([&](tuples::GetColumnOp getColumnOp) {
                  getColumnOp.replaceAllUsesWith(helper.access(getColumnOp.getAttr(), getColumnOp->getLoc()));
                  toErase.push_back(getColumnOp);
               });
               for (auto* op : toErase) {
                  op->erase();
               }
               return {predVal};
            });
            stream = rewriter.create<subop::FilterOp>(loc, stream, subop::FilterSemantic::all_true, rewriter.getArrayAttr(predRef));
         }
         return stream;
      } else {
         assert(false && "invalid");
         return Value();
      }
   }
}
class SelectionLowering : public OpConversionPattern<relalg::SelectionOp> {
   public:
   using OpConversionPattern<relalg::SelectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::SelectionOp selectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto repl = translateSelection(adaptor.getRel(), selectionOp.getPredicate(), rewriter, selectionOp->getLoc());
      if (auto* definingOp = repl.getDefiningOp()) {
         if (selectionOp->hasAttr("selectivity")) {
            definingOp->setAttr("selectivity", selectionOp->getAttr("selectivity"));
         }
      }
      rewriter.replaceOp(selectionOp, repl);

      return success();
   }
};
class MapLowering : public OpConversionPattern<relalg::MapOp> {
   public:
   using OpConversionPattern<relalg::MapOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::MapOp mapOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      subop::MapCreationHelper helper(rewriter.getContext());
      std::vector<mlir::Operation*> toMove;
      for (auto& op : mapOp.getRegion().front()) {
         toMove.push_back(&op);
      }
      for (auto* op : toMove) {
         op->remove();
         helper.getMapBlock()->push_back(op);
      }
      std::vector<mlir::Operation*> toErase;
      helper.getMapBlock()->walk([&](tuples::GetColumnOp getColumnOp) {
         getColumnOp.replaceAllUsesWith(helper.access(getColumnOp.getAttr(), getColumnOp->getLoc()));
         toErase.push_back(getColumnOp);
      });
      for (auto* op : toErase) {
         rewriter.eraseOp(op);
      }
      auto mapOp2 = rewriter.replaceOpWithNewOp<subop::MapOp>(mapOp, tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRel(), mapOp.getComputedCols(), helper.getColRefs());
      mapOp2.getFn().push_back(helper.getMapBlock());
      return success();
   }
};
class RenamingLowering : public OpConversionPattern<relalg::RenamingOp> {
   public:
   using OpConversionPattern<relalg::RenamingOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::RenamingOp renamingOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<subop::RenamingOp>(renamingOp, tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRel(), renamingOp.getColumns());
      return success();
   }
};
class ProjectionAllLowering : public OpConversionPattern<relalg::ProjectionOp> {
   public:
   using OpConversionPattern<relalg::ProjectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::ProjectionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.getSetSemantic() == relalg::SetSemantic::distinct) return failure();
      rewriter.replaceOp(projectionOp, adaptor.getRel());
      return success();
   }
};

static mlir::Block* createCompareBlock(std::vector<mlir::Type> keyTypes, ConversionPatternRewriter& rewriter, mlir::Location loc) {
   mlir::Block* equalBlock = new Block;
   std::vector<mlir::Location> locations(keyTypes.size(), loc);
   equalBlock->addArguments(keyTypes, locations);
   equalBlock->addArguments(keyTypes, locations);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(equalBlock);
      mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
      rewriter.create<tuples::ReturnOp>(loc, compared);
   }
   return equalBlock;
}

class ProjectionDistinctLowering : public OpConversionPattern<relalg::ProjectionOp> {
   public:
   using OpConversionPattern<relalg::ProjectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::ProjectionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.getSetSemantic() != relalg::SetSemantic::distinct) return failure();
      auto* context = getContext();
      auto loc = projectionOp->getLoc();

      auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      for (auto x : projectionOp.getCols()) {
         auto ref = mlir::cast<tuples::ColumnRefAttr>(x);
         auto memberName = getUniqueMember(getContext(), "keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      auto keyMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {}), mlir::ArrayAttr::get(context, {}));

      auto stateType = subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers, false);
      mlir::Value state = rewriter.create<subop::GenericCreateOp>(loc, stateType);
      auto [referenceDef, referenceRef] = createColumn(subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto lookupOp = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(getContext()), adaptor.getRel(), state, projectionOp.getCols(), referenceDef);
      auto* initialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initialValueBlock);
         rewriter.create<tuples::ReturnOp>(loc);
      }
      lookupOp.getInitFn().push_back(initialValueBlock);
      lookupOp.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));
      auto reduceOp = rewriter.create<subop::ReduceOp>(loc, lookupOp, referenceRef, rewriter.getArrayAttr({}), rewriter.getArrayAttr({}));

      {
         mlir::Block* reduceBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(reduceBlock);
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({}));
         reduceOp.getRegion().push_back(reduceBlock);
      }
      {
         mlir::Block* combineBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(combineBlock);
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({}));
         reduceOp.getCombine().push_back(combineBlock);
      }
      mlir::Value scan = rewriter.create<subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));

      rewriter.replaceOp(projectionOp, scan);
      return success();
   }
};
class MaterializationHelper {
   std::vector<NamedAttribute> defMapping;
   std::vector<NamedAttribute> refMapping;
   std::vector<Attribute> types;
   std::vector<Attribute> names;
   std::unordered_map<const tuples::Column*, size_t> colToMemberPos;
   mlir::MLIRContext* context;

   public:
   MaterializationHelper(const relalg::ColumnSet& columns, mlir::MLIRContext* context) : context(context) {
      size_t i = 0;
      for (auto* x : columns) {
         types.push_back(mlir::TypeAttr::get(x->type));
         colToMemberPos[x] = i++;
         std::string name = getUniqueMember(context, "member");
         auto nameAttr = mlir::StringAttr::get(context, name);
         names.push_back(nameAttr);
         defMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createDef(x)));
         refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createRef(x)));
      }
   }
   MaterializationHelper(mlir::ArrayAttr columnAttrs, mlir::MLIRContext* context) : context(context) {
      size_t i = 0;
      for (auto columnAttr : columnAttrs) {
         std::string name = getUniqueMember(context, "member");
         auto nameAttr = mlir::StringAttr::get(context, name);
         names.push_back(nameAttr);
         if (auto columnDef = mlir::dyn_cast<tuples::ColumnDefAttr>(columnAttr)) {
            auto* x = &columnDef.getColumn();
            types.push_back(mlir::TypeAttr::get(x->type));
            colToMemberPos[x] = i++;
            defMapping.push_back(mlir::NamedAttribute(nameAttr, columnDef));
            refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createRef(x)));
         } else if (auto columnRef = mlir::dyn_cast<tuples::ColumnRefAttr>(columnAttr)) {
            auto* x = &columnRef.getColumn();
            types.push_back(mlir::TypeAttr::get(x->type));
            colToMemberPos[x] = i++;
            refMapping.push_back(mlir::NamedAttribute(nameAttr, columnRef));
            defMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createDef(x)));
         }
      }
   }
   mlir::Type getType(size_t i) {
      return mlir::cast<mlir::TypeAttr>(types.at(i)).getValue();
   }

   std::string addFlag(tuples::ColumnDefAttr flagAttrDef) {
      auto i1Type = mlir::IntegerType::get(context, 1);
      types.push_back(mlir::TypeAttr::get(i1Type));
      colToMemberPos[&flagAttrDef.getColumn()] = names.size();
      std::string name = getUniqueMember(context, "flag");
      auto nameAttr = mlir::StringAttr::get(context, name);
      names.push_back(nameAttr);
      defMapping.push_back(mlir::NamedAttribute(nameAttr, flagAttrDef));
      refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createRef(&flagAttrDef.getColumn())));
      return name;
   }
   subop::StateMembersAttr createStateMembersAttr(std::vector<mlir::Attribute> localNames = {}, std::vector<mlir::Attribute> localTypes = {}) {
      localNames.insert(localNames.end(), names.begin(), names.end());
      localTypes.insert(localTypes.end(), types.begin(), types.end());
      return subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, localNames), mlir::ArrayAttr::get(context, localTypes));
   }

   mlir::DictionaryAttr createStateColumnMapping(std::vector<mlir::NamedAttribute> additional = {}, std::unordered_set<std::string> excludedMembers = {}) {
      for (auto x : defMapping) {
         if (!excludedMembers.contains(x.getName().str())) {
            additional.push_back(x);
         }
      }
      return mlir::DictionaryAttr::get(context, additional);
   }
   mlir::DictionaryAttr createColumnstateMapping(std::vector<mlir::NamedAttribute> additional = {}) {
      additional.insert(additional.end(), refMapping.begin(), refMapping.end());
      return mlir::DictionaryAttr::get(context, additional);
   }
   mlir::StringAttr lookupStateMemberForMaterializedColumn(const tuples::Column* column) {
      return mlir::cast<mlir::StringAttr>(names.at(colToMemberPos.at(column)));
   }
};
class ConstRelationLowering : public OpConversionPattern<relalg::ConstRelationOp> {
   public:
   using OpConversionPattern<relalg::ConstRelationOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(relalg::ConstRelationOp constRelationOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = constRelationOp->getLoc();
      std::vector<mlir::Type> returnTypes{tuples::TupleStreamType::get(rewriter.getContext())};
      for (auto i = 0ull; i < constRelationOp.getValues().size(); i++) {
         returnTypes.push_back(tuples::TupleStreamType::get(rewriter.getContext()));
      }
      auto generateOp = rewriter.create<subop::GenerateOp>(constRelationOp.getLoc(), returnTypes, constRelationOp.getColumns());
      {
         auto* generateBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard2(rewriter);
         rewriter.setInsertionPointToStart(generateBlock);
         generateOp.getRegion().push_back(generateBlock);
         for (auto rowAttr : constRelationOp.getValues()) {
            auto row = mlir::cast<ArrayAttr>(rowAttr);
            std::vector<Value> values;
            size_t i = 0;
            for (auto entryAttr : row.getValue()) {
               auto type = mlir::cast<tuples::ColumnDefAttr>(constRelationOp.getColumns()[i]).getColumn().type;
               if (mlir::isa<db::NullableType>(type) && mlir::isa<mlir::UnitAttr>(entryAttr)) {
                  auto entryVal = rewriter.create<db::NullOp>(constRelationOp->getLoc(), type);
                  values.push_back(entryVal);
                  i++;
               } else {
                  mlir::Value entryVal = rewriter.create<db::ConstantOp>(constRelationOp->getLoc(), getBaseType(type), entryAttr);
                  if (mlir::isa<db::NullableType>(type)) {
                     entryVal = rewriter.create<db::AsNullableOp>(constRelationOp->getLoc(), type, entryVal);
                  }
                  values.push_back(entryVal);
                  i++;
               }
            }
            rewriter.create<subop::GenerateEmitOp>(constRelationOp->getLoc(), values);
         }
         rewriter.create<tuples::ReturnOp>(loc);
      }
      rewriter.replaceOp(constRelationOp, generateOp.getRes());
      return success();
   }
};

static mlir::Value mapBool(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, bool value, const tuples::Column* column) {
   Block* mapBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      mlir::Value val = rewriter.create<db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
      rewriter.create<tuples::ReturnOp>(loc, val);
   }

   auto& columnManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(column);
   auto mapOp = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(markAttrDef), rewriter.getArrayAttr({}));
   mapOp.getFn().push_back(mapBlock);
   return mapOp.getResult();
}

static std::pair<mlir::Value, const tuples::Column*> mapBool(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, bool value) {
   Block* mapBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      mlir::Value val = rewriter.create<db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
      rewriter.create<tuples::ReturnOp>(loc, val);
   }

   auto& columnManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope("map");
   std::string attributeName = "boolval";
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = rewriter.getI1Type();
   auto mapOp = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(markAttrDef), rewriter.getArrayAttr({}));
   mapOp.getFn().push_back(mapBlock);
   return {mapOp.getResult(), &markAttrDef.getColumn()};
}
static std::pair<mlir::Value, const tuples::Column*> mapIndex(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, size_t value) {
   Block* mapBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      mlir::Value val = rewriter.create<mlir::arith::ConstantIndexOp>(loc, value);
      rewriter.create<tuples::ReturnOp>(loc, val);
   }

   auto& columnManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope("map");
   std::string attributeName = "ival";
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = rewriter.getI1Type();
   auto mapOp = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(markAttrDef), rewriter.getArrayAttr({}));
   mapOp.getFn().push_back(mapBlock);
   return {mapOp.getResult(), &markAttrDef.getColumn()};
}
static mlir::Value mapColsToNull(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, relalg::ColumnSet excluded = {}) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> defAttrs;
   Block* mapBlock = new Block;
   mapBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      std::vector<mlir::Value> res;
      for (mlir::Attribute attr : mapping) {
         auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         auto* defAttr = &relationDefAttr.getColumn();
         auto fromExisting = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[0]);
         if (excluded.contains(&fromExisting.getColumn())) continue;
         mlir::Value nullValue = rewriter.create<db::NullOp>(loc, defAttr->type);
         res.push_back(nullValue);
         defAttrs.push_back(colManager.createDef(defAttr));
      }
      rewriter.create<tuples::ReturnOp>(loc, res);
   }
   auto mapOp = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs), rewriter.getArrayAttr({}));
   mapOp.getFn().push_back(mapBlock);
   return mapOp.getResult();
}
static mlir::Value mapColsToNullable(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, size_t exisingOffset = 0, relalg::ColumnSet excluded = {}) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> defAttrs;
   subop::MapCreationHelper helper(rewriter.getContext());
   helper.buildBlock(rewriter, [&](mlir::OpBuilder& rewriter) {
      std::vector<mlir::Value> res;
      for (mlir::Attribute attr : mapping) {
         auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         auto* defAttr = &relationDefAttr.getColumn();
         auto fromExisting = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[exisingOffset]);
         if (excluded.contains(&fromExisting.getColumn())) continue;
         mlir::Value value = helper.access(fromExisting, loc);
         if (fromExisting.getColumn().type != defAttr->type) {
            mlir::Value tmp = rewriter.create<db::AsNullableOp>(loc, defAttr->type, value);
            value = tmp;
         }
         res.push_back(value);
         defAttrs.push_back(colManager.createDef(defAttr));
      }
      rewriter.create<tuples::ReturnOp>(loc, res);
   });
   auto mapOp = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs), helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   return mapOp.getResult();
}
class UnionAllLowering : public OpConversionPattern<relalg::UnionOp> {
   public:
   using OpConversionPattern<relalg::UnionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::UnionOp unionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (unionOp.getSetSemantic() != relalg::SetSemantic::all) return failure();
      auto loc = unionOp->getLoc();
      mlir::Value left = mapColsToNullable(adaptor.getLeft(), rewriter, loc, unionOp.getMapping(), 0);
      mlir::Value right = mapColsToNullable(adaptor.getRight(), rewriter, loc, unionOp.getMapping(), 1);
      rewriter.replaceOpWithNewOp<subop::UnionOp>(unionOp, mlir::ValueRange({left, right}));
      return success();
   }
};

class UnionDistinctLowering : public OpConversionPattern<relalg::UnionOp> {
   public:
   using OpConversionPattern<relalg::UnionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::UnionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.getSetSemantic() != relalg::SetSemantic::distinct) return failure();
      auto* context = getContext();
      auto loc = projectionOp->getLoc();

      auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      std::vector<mlir::Attribute> refs;
      for (auto x : projectionOp.getMapping()) {
         auto ref = mlir::cast<tuples::ColumnDefAttr>(x);
         refs.push_back(colManager.createRef(&ref.getColumn()));
         auto memberName = getUniqueMember(getContext(), "keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      auto keyMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {}), mlir::ArrayAttr::get(context, {}));

      auto stateType = subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers, false);
      mlir::Value state = rewriter.create<subop::GenericCreateOp>(loc, stateType);
      auto [referenceDefLeft, referenceRefLeft] = createColumn(subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto [referenceDefRight, referenceRefRight] = createColumn(subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      mlir::Value left = mapColsToNullable(adaptor.getLeft(), rewriter, loc, projectionOp.getMapping(), 0);
      mlir::Value right = mapColsToNullable(adaptor.getRight(), rewriter, loc, projectionOp.getMapping(), 1);
      auto lookupOpLeft = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(getContext()), left, state, rewriter.getArrayAttr(refs), referenceDefLeft);
      auto* leftInitialValueBlock = new Block;

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(leftInitialValueBlock);
         rewriter.create<tuples::ReturnOp>(loc);
      }
      lookupOpLeft.getInitFn().push_back(leftInitialValueBlock);
      lookupOpLeft.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

      auto reduceOpLeft = rewriter.create<subop::ReduceOp>(loc, lookupOpLeft, referenceRefLeft, rewriter.getArrayAttr({}), rewriter.getArrayAttr({}));

      {
         mlir::Block* reduceBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(reduceBlock);
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({}));
         reduceOpLeft.getRegion().push_back(reduceBlock);
      }
      {
         mlir::Block* combineBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(combineBlock);
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({}));
         reduceOpLeft.getCombine().push_back(combineBlock);
      }

      auto lookupOpRight = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(getContext()), right, state, rewriter.getArrayAttr(refs), referenceDefRight);
      auto* rightInitialValueBlock = new Block;

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rightInitialValueBlock);
         rewriter.create<tuples::ReturnOp>(loc);
      }
      lookupOpRight.getInitFn().push_back(rightInitialValueBlock);
      lookupOpRight.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));
      auto reduceOpRight = rewriter.create<subop::ReduceOp>(loc, lookupOpRight, referenceRefRight, rewriter.getArrayAttr({}), rewriter.getArrayAttr({}));

      {
         mlir::Block* reduceBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(reduceBlock);
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({}));
         reduceOpRight.getRegion().push_back(reduceBlock);
      }
      {
         mlir::Block* combineBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(combineBlock);
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({}));
         reduceOpRight.getCombine().push_back(combineBlock);
      }

      mlir::Value scan = rewriter.create<subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));

      rewriter.replaceOp(projectionOp, scan);
      return success();
   }
};
class CountingSetOperationLowering : public ConversionPattern {
   public:
   CountingSetOperationLowering(mlir::MLIRContext* context)
      : ConversionPattern(MatchAnyOpTypeTag(), 1, context) {}
   LogicalResult match(mlir::Operation* op) const override {
      return mlir::success(mlir::isa<relalg::ExceptOp, relalg::IntersectOp>(op));
   }
   void rewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      bool distinct = op->getAttrOfType<relalg::SetSemanticAttr>("set_semantic").getValue() == relalg::SetSemantic::distinct;
      bool except = mlir::isa<relalg::ExceptOp>(op);
      auto* context = getContext();
      auto loc = op->getLoc();

      auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      std::vector<mlir::Attribute> refs;
      auto mapping = op->getAttrOfType<mlir::ArrayAttr>("mapping");
      for (auto x : mapping) {
         auto ref = mlir::cast<tuples::ColumnDefAttr>(x);
         refs.push_back(colManager.createRef(&ref.getColumn()));
         auto memberName = getUniqueMember(getContext(), "keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      std::string counterName1 = getUniqueMember(getContext(), "counter");
      std::string counterName2 = getUniqueMember(getContext(), "counter");
      auto [counter1Def, counter1Ref] = createColumn(rewriter.getI64Type(), "set", "counter");
      auto [counter2Def, counter2Ref] = createColumn(rewriter.getI64Type(), "set", "counter");
      defMapping.push_back(rewriter.getNamedAttr(counterName1, counter1Def));
      defMapping.push_back(rewriter.getNamedAttr(counterName2, counter2Def));
      std::vector<mlir::Attribute> counterNames = {rewriter.getStringAttr(counterName1), rewriter.getStringAttr(counterName2)};
      std::vector<mlir::Attribute> counterTypes = {mlir::TypeAttr::get(rewriter.getI64Type()), mlir::TypeAttr::get(rewriter.getI64Type())};
      auto keyMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, counterNames), mlir::ArrayAttr::get(context, counterTypes));

      auto stateType = subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers, false);
      mlir::Value state = rewriter.create<subop::GenericCreateOp>(loc, stateType);
      auto [referenceDefLeft, referenceRefLeft] = createColumn(subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto [referenceDefRight, referenceRefRight] = createColumn(subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      mlir::Value left = mapColsToNullable(operands[0], rewriter, loc, mapping, 0);
      mlir::Value right = mapColsToNullable(operands[1], rewriter, loc, mapping, 1);
      auto lookupOpLeft = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(getContext()), left, state, rewriter.getArrayAttr(refs), referenceDefLeft);
      auto* leftInitialValueBlock = new Block;
      auto* rightInitialValueBlock = new Block;

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(leftInitialValueBlock);
         mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({zeroI64, zeroI64}));
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rightInitialValueBlock);
         mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({zeroI64, zeroI64}));
      }
      lookupOpLeft.getInitFn().push_back(leftInitialValueBlock);
      lookupOpLeft.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));
      {
         auto reduceOp = rewriter.create<subop::ReduceOp>(loc, lookupOpLeft, referenceRefLeft, rewriter.getArrayAttr({}), rewriter.getArrayAttr(counterNames));

         {
            mlir::Block* reduceBlock = new Block;
            mlir::Value currCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value otherCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceBlock);
            mlir::Value constOne = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
            currCounter = rewriter.create<mlir::arith::AddIOp>(loc, currCounter, constOne);
            rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({currCounter, otherCounter}));
            reduceOp.getRegion().push_back(reduceBlock);
         }
         {
            mlir::Block* combineBlock = new Block;
            mlir::Value counter1A = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value counter2A = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value counter1B = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value counter2B = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(combineBlock);
            mlir::Value counter1 = rewriter.create<mlir::arith::AddIOp>(loc, counter1A, counter1B);
            mlir::Value counter2 = rewriter.create<mlir::arith::AddIOp>(loc, counter2A, counter2B);
            rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({counter1, counter2}));
            reduceOp.getCombine().push_back(combineBlock);
         }
      }
      auto lookupOpRight = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(getContext()), right, state, rewriter.getArrayAttr(refs), referenceDefRight);
      lookupOpRight.getInitFn().push_back(rightInitialValueBlock);
      lookupOpRight.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

      {
         auto reduceOp = rewriter.create<subop::ReduceOp>(loc, lookupOpRight, referenceRefRight, rewriter.getArrayAttr({}), rewriter.getArrayAttr(counterNames));
         {
            mlir::Block* reduceBlock = new Block;
            mlir::Value otherCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value currCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceBlock);
            mlir::Value constOne = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
            currCounter = rewriter.create<mlir::arith::AddIOp>(loc, currCounter, constOne);
            rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({otherCounter, currCounter}));
            reduceOp.getRegion().push_back(reduceBlock);
         }
         {
            mlir::Block* combineBlock = new Block;
            mlir::Value counter1A = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value counter2A = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value counter1B = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value counter2B = combineBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(combineBlock);
            mlir::Value counter1 = rewriter.create<mlir::arith::AddIOp>(loc, counter1A, counter1B);
            mlir::Value counter2 = rewriter.create<mlir::arith::AddIOp>(loc, counter2A, counter2B);
            rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange({counter1, counter2}));
            reduceOp.getCombine().push_back(combineBlock);
         }
      }
      mlir::Value scan = rewriter.create<subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));
      if (distinct) {
         auto [predicateDef, predicateRef] = createColumn(rewriter.getI64Type(), "set", "predicate");

         scan = map(scan, rewriter, loc, rewriter.getArrayAttr(predicateDef), [&, counter1Ref = counter1Ref, counter2Ref = counter2Ref](mlir::OpBuilder& rewriter, subop::MapCreationHelper& helper, mlir::Location loc) {
            mlir::Value leftVal = helper.access(counter1Ref, loc);
            mlir::Value rightVal = helper.access(counter2Ref, loc);
            mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
            mlir::Value leftNonZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, leftVal, zeroI64);
            mlir::Value outputTuple;
            if (except) {
               mlir::Value rightZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, rightVal, zeroI64);
               outputTuple = rewriter.create<mlir::arith::AndIOp>(loc, leftNonZero, rightZero);
            } else {
               mlir::Value rightNonZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, rightVal, zeroI64);
               outputTuple = rewriter.create<mlir::arith::AndIOp>(loc, leftNonZero, rightNonZero);
            }
            return std::vector<mlir::Value>({outputTuple});
         });
         scan = rewriter.create<subop::FilterOp>(loc, scan, subop::FilterSemantic::all_true, rewriter.getArrayAttr(predicateRef));
      } else {
         auto [repeatDef, repeatRef] = createColumn(rewriter.getIndexType(), "set", "repeat");

         scan = map(scan, rewriter, loc, rewriter.getArrayAttr(repeatDef), [&, counter1Ref = counter1Ref, counter2Ref = counter2Ref](mlir::OpBuilder& rewriter, subop::MapCreationHelper& helper, mlir::Location loc) {
            mlir::Value leftVal = helper.access(counter1Ref, loc);
            mlir::Value rightVal = helper.access(counter2Ref, loc);
            mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
            mlir::Value repeatNum;
            if (except) {
               mlir::Value remaining = rewriter.create<mlir::arith::SubIOp>(loc, leftVal, rightVal);
               mlir::Value ltZ = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, remaining, zeroI64);
               remaining = rewriter.create<mlir::arith::SelectOp>(loc, ltZ, zeroI64, remaining);
               repeatNum = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), remaining);
            } else {
               mlir::Value lGtR = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, leftVal, rightVal);
               mlir::Value remaining = rewriter.create<mlir::arith::SelectOp>(loc, lGtR, rightVal, leftVal);
               repeatNum = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), remaining);
            }
            return std::vector<mlir::Value>({repeatNum});
         });
         auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), scan, rewriter.getArrayAttr({repeatRef}));
         auto* b = new Block;
         b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
         mlir::Value repeatNumber = b->addArgument(rewriter.getIndexType(), loc);
         nestedMapOp.getRegion().push_back(b);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(b);
            auto generateOp = rewriter.create<subop::GenerateOp>(loc, std::vector<mlir::Type>{tuples::TupleStreamType::get(rewriter.getContext()), tuples::TupleStreamType::get(rewriter.getContext())}, rewriter.getArrayAttr({}));
            {
               auto* generateBlock = new Block;
               mlir::OpBuilder::InsertionGuard guard2(rewriter);
               rewriter.setInsertionPointToStart(generateBlock);
               generateOp.getRegion().push_back(generateBlock);
               mlir::Value zeroIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
               mlir::Value oneIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
               rewriter.create<mlir::scf::ForOp>(loc, zeroIdx, repeatNumber, oneIdx, mlir::ValueRange{}, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange vr) {
                  b.create<subop::GenerateEmitOp>(loc, mlir::ValueRange{});
                  b.create<mlir::scf::YieldOp>(loc);
               });
               rewriter.create<tuples::ReturnOp>(loc);
            }
            rewriter.create<tuples::ReturnOp>(loc, generateOp.getRes());
         }
         scan = nestedMapOp.getRes();
      }
      rewriter.replaceOp(op, scan);
   }
};
static std::pair<mlir::Value, std::string> createMarkerState(mlir::OpBuilder& rewriter, mlir::Location loc) {
   auto memberName = getUniqueMember(rewriter.getContext(), "marker");
   mlir::Type stateType = subop::SimpleStateType::get(rewriter.getContext(), subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(memberName)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getI1Type())})));
   Block* initialValueBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(initialValueBlock);
      mlir::Value val = rewriter.create<db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.create<tuples::ReturnOp>(loc, val);
   }
   auto createOp = rewriter.create<subop::CreateSimpleStateOp>(loc, stateType);
   createOp.getInitFn().push_back(initialValueBlock);

   return {createOp.getRes(), memberName};
}
static std::pair<mlir::Value, std::string> createCounterState(mlir::OpBuilder& rewriter, mlir::Location loc) {
   auto memberName = getUniqueMember(rewriter.getContext(), "counter");
   mlir::Type stateType = subop::SimpleStateType::get(rewriter.getContext(), subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(memberName)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getI64Type())})));
   Block* initialValueBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(initialValueBlock);
      mlir::Value val = rewriter.create<db::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      rewriter.create<tuples::ReturnOp>(loc, val);
   }
   auto createOp = rewriter.create<subop::CreateSimpleStateOp>(loc, stateType);
   createOp.getInitFn().push_back(initialValueBlock);

   return {createOp.getRes(), memberName};
}

static mlir::Value translateNLJ(mlir::Value left, mlir::Value right, relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   MaterializationHelper helper(columns, rewriter.getContext());
   if (columns.empty()) {
      auto [counterState, counterName] = createCounterState(rewriter, loc);
      auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      // for right side: increment counter:
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      referenceDefAttr.getColumn().type = subop::LookupEntryRefType::get(rewriter.getContext(), mlir::cast<subop::LookupAbleState>(counterState.getType()));
      auto lookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), right, counterState, rewriter.getArrayAttr({}), referenceDefAttr);

      // Create reduce operation that increases counter for each seen tuple
      auto reduceOp = rewriter.create<subop::ReduceOp>(loc, lookup, colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getArrayAttr({}), rewriter.getArrayAttr({rewriter.getStringAttr(counterName)}));
      mlir::Block* reduceBlock = new Block;
      auto counter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(reduceBlock);
         auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
         mlir::Value updatedCounter = rewriter.create<mlir::arith::AddIOp>(loc, counter, one);
         rewriter.create<tuples::ReturnOp>(loc, updatedCounter);
      }
      reduceOp.getRegion().push_back(reduceBlock);
      mlir::Block* combineBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(combineBlock);
         auto counter1 = combineBlock->addArgument(rewriter.getI64Type(), loc);
         auto counter2 = combineBlock->addArgument(rewriter.getI64Type(), loc);
         mlir::Value sum = rewriter.create<mlir::arith::AddIOp>(loc, counter1, counter2);
         rewriter.create<tuples::ReturnOp>(loc, sum);
      }
      reduceOp.getCombine().push_back(combineBlock);

      auto referenceDefAttr2 = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      auto counterValDefAttr = colManager.createDef(colManager.getUniqueScope("counter"), "val");
      counterValDefAttr.getColumn().type = rewriter.getI64Type();
      referenceDefAttr2.getColumn().type = subop::LookupEntryRefType::get(rewriter.getContext(), mlir::cast<subop::LookupAbleState>(counterState.getType()));
      auto lookup2 = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), left, counterState, rewriter.getArrayAttr({}), referenceDefAttr2);
      auto gathered = rewriter.create<subop::GatherOp>(loc, lookup2.getRes(), colManager.createRef(&referenceDefAttr2.getColumn()), rewriter.getDictionaryAttr(rewriter.getNamedAttr(counterName, counterValDefAttr)));
      auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), gathered.getRes(), rewriter.getArrayAttr({colManager.createRef(&counterValDefAttr.getColumn())}));

      // for left side: loop
      auto* b = new Block;
      mlir::Value tuple = b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
      mlir::Value counterVal = b->addArgument(rewriter.getI64Type(), loc);
      nestedMapOp.getRegion().push_back(b);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(b);
         auto [markerState, markerName] = createMarkerState(rewriter, loc);
         auto generateOp = rewriter.create<subop::GenerateOp>(loc, std::vector<mlir::Type>{tuples::TupleStreamType::get(rewriter.getContext()), tuples::TupleStreamType::get(rewriter.getContext())}, rewriter.getArrayAttr({}));
         {
            auto* generateBlock = new Block;
            mlir::OpBuilder::InsertionGuard guard2(rewriter);
            rewriter.setInsertionPointToStart(generateBlock);
            generateOp.getRegion().push_back(generateBlock);
            counterVal = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), counterVal);
            mlir::Value zeroIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value oneIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
            rewriter.create<mlir::scf::ForOp>(loc, zeroIdx, counterVal, oneIdx, mlir::ValueRange{}, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange vr) {
               b.create<subop::GenerateEmitOp>(loc, mlir::ValueRange{});
               b.create<mlir::scf::YieldOp>(loc);
            });
            rewriter.create<tuples::ReturnOp>(loc);
         }

         mlir::Value combined = rewriter.create<subop::CombineTupleOp>(loc, generateOp.getRes(), tuple);
         rewriter.create<tuples::ReturnOp>(loc, fn(combined, rewriter));
      }
      return nestedMapOp.getRes();
   } else {
      auto vectorType = subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector = rewriter.create<subop::GenericCreateOp>(loc, vectorType);
      rewriter.create<subop::MaterializeOp>(loc, right, vector, helper.createColumnstateMapping());
      auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), left, rewriter.getArrayAttr({}));
      auto* b = new Block;
      mlir::Value tuple = b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
      nestedMapOp.getRegion().push_back(b);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(b);
         auto [markerState, markerName] = createMarkerState(rewriter, loc);
         mlir::Value scan = rewriter.create<subop::ScanOp>(loc, vector, helper.createStateColumnMapping());
         mlir::Value combined = rewriter.create<subop::CombineTupleOp>(loc, scan, tuple);
         rewriter.create<tuples::ReturnOp>(loc, fn(combined, rewriter));
      }
      return nestedMapOp.getRes();
   }
}
mlir::Block* createEqFn(mlir::ConversionPatternRewriter& rewriter, mlir::ArrayAttr leftColumns, mlir::ArrayAttr rightColumns, mlir::ArrayAttr nullsEqual, mlir::Location loc) {
   mlir::OpBuilder::InsertionGuard guard(rewriter);
   auto* eqFnBlock = new mlir::Block;
   rewriter.setInsertionPointToStart(eqFnBlock);
   std::vector<mlir::Value> leftArgs;
   std::vector<mlir::Value> rightArgs;
   for (auto i = 0ull; i < leftColumns.size(); i++) {
      leftArgs.push_back(eqFnBlock->addArgument(mlir::cast<tuples::ColumnRefAttr>(leftColumns[i]).getColumn().type, loc));
   }
   for (auto i = 0ull; i < rightColumns.size(); i++) {
      rightArgs.push_back(eqFnBlock->addArgument(mlir::cast<tuples::ColumnRefAttr>(rightColumns[i]).getColumn().type, loc));
   }
   std::vector<mlir::Value> cmps;
   for (auto z : llvm::zip(leftArgs, rightArgs, nullsEqual)) {
      auto [l, r, nE] = z;
      bool useIsa = mlir::cast<mlir::IntegerAttr>(nE).getInt();
      mlir::Value compared = rewriter.create<db::CmpOp>(loc, useIsa ? db::DBCmpPredicate::isa : db::DBCmpPredicate::eq, l, r);
      cmps.push_back(compared);
   }
   mlir::Value anded;
   if (cmps.size() == 1) {
      anded = cmps[0];
   } else {
      // If we have more than one comparison, we need to combine them
      anded = rewriter.create<db::AndOp>(loc, cmps);
   }
   if (mlir::isa<db::NullableType>(anded.getType())) {
      anded = rewriter.create<db::DeriveTruth>(loc, anded);
   }
   rewriter.create<tuples::ReturnOp>(loc, anded);
   return eqFnBlock;
}
std::pair<mlir::Block*, mlir::ArrayAttr> createVerifyEqFnForTuple(mlir::ConversionPatternRewriter& rewriter, mlir::ArrayAttr leftColumns, mlir::ArrayAttr rightColumns, mlir::ArrayAttr nullsEqual, mlir::Location loc) {
   subop::MapCreationHelper helper(rewriter.getContext());
   helper.buildBlock(rewriter, [&](mlir::ConversionPatternRewriter& rewriter) {
      std::vector<mlir::Value> leftArgs;
      std::vector<mlir::Value> rightArgs;
      for (auto i = 0ull; i < leftColumns.size(); i++) {
         auto leftCol = mlir::cast<tuples::ColumnRefAttr>(leftColumns[i]);
         leftArgs.push_back(helper.access(leftCol, loc));
      }
      for (auto i = 0ull; i < rightColumns.size(); i++) {
         auto rightCol = mlir::cast<tuples::ColumnRefAttr>(rightColumns[i]);
         rightArgs.push_back(helper.access(rightCol, loc));
      }
      std::vector<mlir::Value> cmps;
      for (auto z : llvm::zip(leftArgs, rightArgs, nullsEqual)) {
         auto [l, r, nE] = z;
         bool useIsa = mlir::cast<mlir::IntegerAttr>(nE).getInt();
         mlir::Value compared = rewriter.create<db::CmpOp>(loc, useIsa ? db::DBCmpPredicate::isa : db::DBCmpPredicate::eq, l, r);
         cmps.push_back(compared);
      }
      mlir::Value anded = rewriter.create<db::AndOp>(loc, cmps);
      if (mlir::isa<db::NullableType>(anded.getType())) {
         anded = rewriter.create<db::DeriveTruth>(loc, anded);
      }
      rewriter.create<tuples::ReturnOp>(loc, anded);
   });

   return {helper.getMapBlock(), helper.getColRefs()};
}

static mlir::Value translateHJ(mlir::Value left, mlir::Value right, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   auto keyColumns = relalg::ColumnSet::fromArrayAttr(hashRight);
   MaterializationHelper keyHelper(hashRight, rewriter.getContext());
   auto valueColumns = columns;
   valueColumns.remove(keyColumns);
   MaterializationHelper valueHelper(valueColumns, rewriter.getContext());
   auto multiMapType = subop::MultiMapType::get(rewriter.getContext(), keyHelper.createStateMembersAttr(), valueHelper.createStateMembersAttr());
   mlir::Value multiMap = rewriter.create<subop::GenericCreateOp>(loc, multiMapType);
   auto insertOp = rewriter.create<subop::InsertOp>(loc, right, multiMap, keyHelper.createColumnstateMapping(valueHelper.createColumnstateMapping().getValue()));
   insertOp.getEqFn().push_back(createEqFn(rewriter, hashRight, hashRight, nullsEqual, loc));

   auto entryRefType = subop::MultiMapEntryRefType::get(rewriter.getContext(), multiMapType);
   auto entryRefListType = subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   auto afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), left, multiMap, hashLeft, listDef);
   afterLookup.getEqFn().push_back(createEqFn(rewriter, hashRight, hashLeft, nullsEqual, loc));
   auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);
   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      mlir::Value scan = rewriter.create<subop::ScanListOp>(loc, list, entryDef);
      mlir::Value gathered = rewriter.create<subop::GatherOp>(loc, scan, entryRef, keyHelper.createStateColumnMapping(valueHelper.createStateColumnMapping().getValue()));
      mlir::Value combined = rewriter.create<subop::CombineTupleOp>(loc, gathered, tuple);
      rewriter.create<tuples::ReturnOp>(loc, fn(combined, rewriter));
   }
   return nestedMapOp.getRes();
}
static mlir::Value translateINLJ(mlir::Value left, mlir::Value right, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   auto loc = op->getLoc();
   auto rightScan = mlir::cast<subop::ScanOp>(right.getDefiningOp());
   auto* ctxt = rewriter.getContext();
   auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string tableName = op->getAttrOfType<mlir::StringAttr>("table").str();
   mlir::ArrayAttr primaryKeyHashValue = rewriter.getArrayAttr({colManager.createRef(tableName, "primaryKeyHashValue")});
   auto keyColumns = relalg::ColumnSet::fromArrayAttr(primaryKeyHashValue);
   auto valueColumns = columns;
   valueColumns.remove(keyColumns);

   bool first = true;
   std::vector<Attribute> keyColNames, keyColTypes, valColNames, valColTypes;
   std::vector<NamedAttribute> mapping;

   // Create description for external index get operation
   std::string externalIndexDescription = R"({"type": "hash", "index": ")" + op->getAttrOfType<mlir::StringAttr>("index").str() + R"(", "relation": ")" + tableName + R"(", "mapping": { )";
   for (auto namedAttr : rightScan.getMapping()) {
      auto identifier = namedAttr.getName();
      auto attr = namedAttr.getValue();
      auto attrDef = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
      if (!first) {
         externalIndexDescription += ",";
      } else {
         first = false;
      }
      auto memberName = getUniqueMember(ctxt, identifier.str());
      externalIndexDescription += "\"" + memberName + "\" :\"" + colManager.getName(&attrDef.getColumn()).second + "\"";

      if (keyColumns.contains(&attrDef.getColumn())) {
         keyColNames.push_back((rewriter.getStringAttr(memberName)));
         keyColTypes.push_back(mlir::TypeAttr::get(attrDef.getColumn().type));
         mapping.push_back(rewriter.getNamedAttr(memberName, attrDef));
      }
      if (valueColumns.contains(&attrDef.getColumn())) {
         keyColNames.push_back((rewriter.getStringAttr(memberName)));
         keyColTypes.push_back(mlir::TypeAttr::get(attrDef.getColumn().type));
         mapping.push_back(rewriter.getNamedAttr(memberName, attrDef));
      }
   }
   externalIndexDescription += "} }";

   auto keyStateMembers = subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(keyColNames), rewriter.getArrayAttr(keyColTypes));
   auto valueStateMembers = subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(valColNames), rewriter.getArrayAttr(valColTypes));

   auto externalHashIndexType = subop::ExternalHashIndexType::get(rewriter.getContext(), keyStateMembers, valueStateMembers);
   mlir::Value externalHashIndex = rewriter.create<subop::GetExternalOp>(loc, externalHashIndexType, externalIndexDescription);
   // Erase table scan
   rewriter.eraseOp(rightScan->getOperand(0).getDefiningOp());
   rewriter.eraseOp(rightScan);

   auto entryRefType = subop::ExternalHashIndexEntryRefType::get(rewriter.getContext(), externalHashIndexType);
   auto entryRefListType = subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   auto afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), left, externalHashIndex, hashLeft, listDef);

   auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);
   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      auto scan = rewriter.create<subop::ScanListOp>(loc, list, entryDef);
      auto gathered = rewriter.create<subop::GatherOp>(loc, scan, entryRef, rewriter.getDictionaryAttr(mapping));
      auto combined = rewriter.create<subop::CombineTupleOp>(loc, gathered, tuple);

      // eliminate hash collisions
      auto [markerAttrDef, markerAttrRef] = createColumn(rewriter.getI1Type(), "map", "predicate");
      auto [block, refCols] = createVerifyEqFnForTuple(rewriter, hashLeft, hashRight, nullsEqual, loc);
      subop::MapOp keep = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), combined, rewriter.getArrayAttr(markerAttrDef), refCols);

      keep.getFn().push_back(block);
      mlir::Value filtered = rewriter.create<subop::FilterOp>(loc, keep, subop::FilterSemantic::all_true, rewriter.getArrayAttr(markerAttrRef));

      rewriter.create<tuples::ReturnOp>(loc, fn(filtered, rewriter));
   }
   return nestedMapOp.getRes();
}
static mlir::Value translateNL(mlir::Value left, mlir::Value right, bool useHash, bool useIndexNestedLoop, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   if (useHash) {
      return translateHJ(left, right, nullsEqual, hashLeft, hashRight, columns, rewriter, op->getLoc(), fn);
   } else if (useIndexNestedLoop) {
      return translateINLJ(left, right, nullsEqual, hashLeft, hashRight, columns, rewriter, op, fn);
   } else {
      return translateNLJ(left, right, columns, rewriter, op->getLoc(), fn);
   }
}

static std::pair<mlir::Value, mlir::Value> translateNLJWithMarker(mlir::Value left, mlir::Value right, relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr, std::string markerName)> fn) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   MaterializationHelper helper(columns, rewriter.getContext());
   auto flagMember = helper.addFlag(markerDefAttr);
   auto vectorType = subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector = rewriter.create<subop::GenericCreateOp>(loc, vectorType);
   left = mapBool(left, rewriter, loc, false, &markerDefAttr.getColumn());
   rewriter.create<subop::MaterializeOp>(loc, left, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), right, rewriter.getArrayAttr({}));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("scan"), "ref");
      referenceDefAttr.getColumn().type = subop::EntryRefType::get(rewriter.getContext(), vectorType);
      mlir::Value scan = rewriter.create<subop::ScanRefsOp>(loc, vector, referenceDefAttr);

      mlir::Value gathered = rewriter.create<subop::GatherOp>(loc, scan, colManager.createRef(&referenceDefAttr.getColumn()), helper.createStateColumnMapping({}, {flagMember}));
      mlir::Value combined = rewriter.create<subop::CombineTupleOp>(loc, gathered, tuple);
      auto res = fn(combined, tuple, rewriter, colManager.createRef(&referenceDefAttr.getColumn()), flagMember);
      if (res) {
         rewriter.create<tuples::ReturnOp>(loc, res);
      } else {
         rewriter.create<tuples::ReturnOp>(loc);
      }
   }
   return {nestedMapOp.getRes(), rewriter.create<subop::ScanOp>(loc, vector, helper.createStateColumnMapping())};
}

static std::pair<mlir::Value, mlir::Value> translateHJWithMarker(mlir::Value left, mlir::Value right, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr, std::string markerName)> fn) {
   auto keyColumns = relalg::ColumnSet::fromArrayAttr(hashLeft);
   MaterializationHelper keyHelper(hashLeft, rewriter.getContext());
   auto valueColumns = columns;
   valueColumns.remove(keyColumns);
   MaterializationHelper valueHelper(valueColumns, rewriter.getContext());
   auto flagMember = valueHelper.addFlag(markerDefAttr);
   auto multiMapType = subop::MultiMapType::get(rewriter.getContext(), keyHelper.createStateMembersAttr(), valueHelper.createStateMembersAttr());
   mlir::Value multiMap = rewriter.create<subop::GenericCreateOp>(loc, multiMapType);
   left = mapBool(left, rewriter, loc, false, &markerDefAttr.getColumn());
   auto insertOp = rewriter.create<subop::InsertOp>(loc, left, multiMap, keyHelper.createColumnstateMapping(valueHelper.createColumnstateMapping().getValue()));
   insertOp.getEqFn().push_back(createEqFn(rewriter, hashLeft, hashLeft, nullsEqual, loc));
   auto entryRefType = subop::MultiMapEntryRefType::get(rewriter.getContext(), multiMapType);
   auto entryRefListType = subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   auto afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), right, multiMap, hashRight, listDef);
   afterLookup.getEqFn().push_back(createEqFn(rewriter, hashLeft, hashRight, nullsEqual, loc));

   auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);

   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      mlir::Value scan = rewriter.create<subop::ScanListOp>(loc, list, entryDef);
      mlir::Value gathered = rewriter.create<subop::GatherOp>(loc, scan, entryRef, keyHelper.createStateColumnMapping(valueHelper.createStateColumnMapping({}, {flagMember}).getValue()));
      mlir::Value combined = rewriter.create<subop::CombineTupleOp>(loc, gathered, tuple);
      auto res = fn(combined, tuple, rewriter, entryRef, flagMember);
      if (res) {
         rewriter.create<tuples::ReturnOp>(loc, res);
      } else {
         rewriter.create<tuples::ReturnOp>(loc);
      }
   }
   return {nestedMapOp.getRes(), rewriter.create<subop::ScanOp>(loc, multiMap, keyHelper.createStateColumnMapping(valueHelper.createStateColumnMapping().getValue()))};
}
static std::pair<mlir::Value, mlir::Value> translateNLWithMarker(mlir::Value left, mlir::Value right, bool useHash, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr, std::string markerName)> fn) {
   if (useHash) {
      return translateHJWithMarker(left, right, nullsEqual, hashLeft, hashRight, columns, rewriter, loc, markerDefAttr, fn);
   } else {
      return translateNLJWithMarker(left, right, columns, rewriter, loc, markerDefAttr, fn);
   }
}

static mlir::Value anyTuple(mlir::Value stream, tuples::ColumnDefAttr markerDefAttr, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto [markerState, markerName] = createMarkerState(rewriter, loc);
   auto [mapped, boolColumn] = mapBool(stream, rewriter, loc, true);
   auto [referenceDefAttr, referenceRefAttr] = createColumn(subop::LookupEntryRefType::get(rewriter.getContext(), mlir::cast<subop::LookupAbleState>(markerState.getType())), "lookup", "ref");
   auto afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), mapped, markerState, rewriter.getArrayAttr({}), referenceDefAttr);
   rewriter.create<subop::ScatterOp>(loc, afterLookup, referenceRefAttr, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, colManager.createRef(boolColumn))));
   return rewriter.create<subop::ScanOp>(loc, markerState, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, markerDefAttr)));
}

class CrossProductLowering : public OpConversionPattern<relalg::CrossProductOp> {
   public:
   using OpConversionPattern<relalg::CrossProductOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::CrossProductOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(crossProductOp, translateNL(adaptor.getRight(), adaptor.getLeft(), false, false, mlir::ArrayAttr(), mlir::ArrayAttr(), mlir::ArrayAttr(), getRequired(mlir::cast<Operator>(crossProductOp.getLeft().getDefiningOp())), rewriter, crossProductOp, [](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                            return v;
                         }));
      return success();
   }
};
class InnerJoinNLLowering : public OpConversionPattern<relalg::InnerJoinOp> {
   public:
   using OpConversionPattern<relalg::InnerJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::InnerJoinOp innerJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = innerJoinOp->getLoc();
      bool useHash = innerJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = innerJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");
      rewriter.replaceOp(innerJoinOp, translateNL(adaptor.getRight(), adaptor.getLeft(), useHash, useIndexNestedLoop, nullsEqual, rightHash, leftHash, getRequired(mlir::cast<Operator>(innerJoinOp.getLeft().getDefiningOp())), rewriter, innerJoinOp, [loc, &innerJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                            return translateSelection(v, innerJoinOp.getPredicate(), rewriter, loc);
                         }));
      return success();
   }
};
class SemiJoinLowering : public OpConversionPattern<relalg::SemiJoinOp> {
   public:
   using OpConversionPattern<relalg::SemiJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::SemiJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = semiJoinOp->getLoc();
      bool reverse = semiJoinOp->hasAttr("reverseSides");
      bool useHash = semiJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = semiJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp())), rewriter, semiJoinOp, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               return rewriter.create<subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), subop::FilterSemantic::all_true, rewriter.getArrayAttr({markerRefAttr}));
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [_, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            return {};
         });
         rewriter.replaceOpWithNewOp<subop::FilterOp>(semiJoinOp, scan, subop::FilterSemantic::all_true, rewriter.getArrayAttr({flagAttrRef}));
      }
      return success();
   }
};
class MarkJoinLowering : public OpConversionPattern<relalg::MarkJoinOp> {
   public:
   using OpConversionPattern<relalg::MarkJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::MarkJoinOp markJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = markJoinOp->getLoc();
      bool reverse = markJoinOp->hasAttr("reverseSides");
      bool useHash = markJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = markJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = markJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = markJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = markJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(markJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(markJoinOp.getRight().getDefiningOp())), rewriter, markJoinOp, [loc, &markJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, markJoinOp.getPredicate(), rewriter, loc);
                               return anyTuple(filtered, markJoinOp.getMarkattr(), rewriter, loc);
                            }));
      } else {
         auto [_, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(markJoinOp.getLeft().getDefiningOp())), rewriter, loc, markJoinOp.getMarkattr(), [loc, &markJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, markJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            return {};
         });
         rewriter.replaceOp(markJoinOp, scan);
      }
      return success();
   }
};
class AntiSemiJoinLowering : public OpConversionPattern<relalg::AntiSemiJoinOp> {
   public:
   using OpConversionPattern<relalg::AntiSemiJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::AntiSemiJoinOp antiSemiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = antiSemiJoinOp->getLoc();
      bool reverse = antiSemiJoinOp->hasAttr("reverseSides");
      bool useHash = antiSemiJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = antiSemiJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(antiSemiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(antiSemiJoinOp.getRight().getDefiningOp())), rewriter, antiSemiJoinOp, [loc, &antiSemiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, antiSemiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               return rewriter.create<subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [_, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(antiSemiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &antiSemiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, antiSemiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            return {};
         });
         rewriter.replaceOpWithNewOp<subop::FilterOp>(antiSemiJoinOp, scan, subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
      }
      return success();
   }
};
class FullOuterJoinLowering : public OpConversionPattern<relalg::FullOuterJoinOp> {
   public:
   using OpConversionPattern<relalg::FullOuterJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::FullOuterJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = semiJoinOp->getLoc();
      bool useHash = semiJoinOp->hasAttr("useHashJoin");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto leftColumns = getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp()));
      auto rightColumns = getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp()));
      auto nullsEqual = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
      auto [stream, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp, leftColumns, rightColumns](mlir::Value v, mlir::Value tuple, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
         auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
         auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
         auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
         rewriter.create<subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
         auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());
         auto [markerDefAttr2, markerRefAttr2] = createColumn(rewriter.getI1Type(), "marker", "marker");
         Value filteredNoMatch = rewriter.create<subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr2, rewriter, loc), subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr2}));
         mlir::Value combined = rewriter.create<subop::CombineTupleOp>(loc, filteredNoMatch, tuple);

         auto mappedNullable2 = mapColsToNullable(combined, rewriter, loc, semiJoinOp.getMapping(), 0, leftColumns);

         auto mappedNull = mapColsToNull(mappedNullable2, rewriter, loc, semiJoinOp.getMapping(), rightColumns);
         return rewriter.create<subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
      });
      auto noMatches = rewriter.create<subop::FilterOp>(loc, scan, subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
      auto mappedNullable = mapColsToNullable(noMatches, rewriter, loc, semiJoinOp.getMapping(), 0, rightColumns);
      auto mappedNull = mapColsToNull(mappedNullable, rewriter, loc, semiJoinOp.getMapping(), leftColumns);
      rewriter.replaceOpWithNewOp<subop::UnionOp>(semiJoinOp, mlir::ValueRange{stream, mappedNull});

      return success();
   }
};
class OuterJoinLowering : public OpConversionPattern<relalg::OuterJoinOp> {
   public:
   using OpConversionPattern<relalg::OuterJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::OuterJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = semiJoinOp->getLoc();
      bool reverse = semiJoinOp->hasAttr("reverseSides");
      bool useHash = semiJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = semiJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp())), rewriter, semiJoinOp, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               Value filteredNoMatch = rewriter.create<subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                               auto mappedNull = mapColsToNull(filteredNoMatch, rewriter, loc, semiJoinOp.getMapping());
                               auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());
                               return rewriter.create<subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [stream, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());

            return mappedNullable;
         });
         auto noMatches = rewriter.create<subop::FilterOp>(loc, scan, subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
         auto mappedNull = mapColsToNull(noMatches, rewriter, loc, semiJoinOp.getMapping());
         rewriter.replaceOpWithNewOp<subop::UnionOp>(semiJoinOp, mlir::ValueRange{stream, mappedNull});
      }
      return success();
   }
};
class SingleJoinLowering : public OpConversionPattern<relalg::SingleJoinOp> {
   public:
   using OpConversionPattern<relalg::SingleJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::SingleJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = semiJoinOp->getLoc();
      bool reverse = semiJoinOp->hasAttr("reverseSides");
      bool useHash = semiJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = semiJoinOp->hasAttr("useIndexNestedLoop");
      bool isConstantJoin = semiJoinOp->hasAttr("constantJoin");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (isConstantJoin) {
         auto columnsToMaterialize = getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp()));
         MaterializationHelper helper(columnsToMaterialize, rewriter.getContext());
         auto constantStateType = subop::SimpleStateType::get(rewriter.getContext(), helper.createStateMembersAttr());
         mlir::Value constantState = rewriter.create<subop::CreateSimpleStateOp>(loc, constantStateType);
         auto entryRefType = subop::LookupEntryRefType::get(rewriter.getContext(), constantStateType);
         auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
         auto afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRight(), constantState, rewriter.getArrayAttr({}), entryDef);
         rewriter.create<subop::ScatterOp>(loc, afterLookup, entryRef, helper.createColumnstateMapping());
         auto [entryDefLeft, entryRefLeft] = createColumn(entryRefType, "lookup", "entryref");

         auto afterLookupLeft = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getLeft(), constantState, rewriter.getArrayAttr({}), entryDefLeft);
         auto gathered = rewriter.create<subop::GatherOp>(loc, afterLookupLeft, entryRefLeft, helper.createStateColumnMapping());
         auto mappedNullable = mapColsToNullable(gathered.getRes(), rewriter, loc, semiJoinOp.getMapping());
         rewriter.replaceOp(semiJoinOp, mappedNullable);
      } else if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp())), rewriter, semiJoinOp, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               Value filteredNoMatch = rewriter.create<subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                               auto mappedNull = mapColsToNull(filteredNoMatch, rewriter, loc, semiJoinOp.getMapping());
                               auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());
                               return rewriter.create<subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [stream, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());

            return mappedNullable;
         });
         auto noMatches = rewriter.create<subop::FilterOp>(loc, scan, subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
         auto mappedNull = mapColsToNull(noMatches, rewriter, loc, semiJoinOp.getMapping());
         rewriter.replaceOpWithNewOp<subop::UnionOp>(semiJoinOp, mlir::ValueRange{stream, mappedNull});
      }
      return success();
   }
};

class LimitLowering : public OpConversionPattern<relalg::LimitOp> {
   public:
   using OpConversionPattern<relalg::LimitOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::LimitOp limitOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = limitOp->getLoc();
      relalg::ColumnSet requiredColumns = getRequired(limitOp);
      MaterializationHelper helper(requiredColumns, rewriter.getContext());

      auto* block = new Block;
      std::vector<Attribute> sortByMembers;
      std::vector<Location> locs;
      std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         mlir::Value isLt = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
         rewriter.create<tuples::ReturnOp>(loc, isLt);
      }
      auto heapType = subop::HeapType::get(getContext(), helper.createStateMembersAttr(), limitOp.getMaxRows());
      auto createHeapOp = rewriter.create<subop::CreateHeapOp>(loc, heapType, rewriter.getArrayAttr(sortByMembers));
      createHeapOp.getRegion().getBlocks().push_back(block);
      rewriter.create<subop::MaterializeOp>(loc, adaptor.getRel(), createHeapOp.getRes(), helper.createColumnstateMapping());
      rewriter.replaceOpWithNewOp<subop::ScanOp>(limitOp, createHeapOp.getRes(), helper.createStateColumnMapping());
      return success();
   }
};
static mlir::Value spaceShipCompare(mlir::OpBuilder& builder, std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria, size_t pos, mlir::Location loc) {
   mlir::Value compareRes = builder.create<db::SortCompare>(loc, sortCriteria.at(pos).first, sortCriteria.at(pos).second);
   auto zero = builder.create<db::ConstantOp>(loc, builder.getI8Type(), builder.getIntegerAttr(builder.getI8Type(), 0));
   auto isZero = builder.create<db::CmpOp>(loc, db::DBCmpPredicate::eq, compareRes, zero);
   if (pos + 1 < sortCriteria.size()) {
      auto ifOp = builder.create<mlir::scf::IfOp>(
         loc, isZero, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, spaceShipCompare(builder, sortCriteria, pos + 1, loc)); }, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, compareRes); });
      return ifOp.getResult(0);
   } else {
      return compareRes;
   }
}
static mlir::Value createSortedView(ConversionPatternRewriter& rewriter, mlir::Value buffer, mlir::ArrayAttr sortSpecs, mlir::Location loc, MaterializationHelper& helper) {
   auto* block = new Block;
   std::vector<Attribute> sortByMembers;
   std::vector<Type> argumentTypes;
   std::vector<Location> locs;
   for (auto attr : sortSpecs) {
      auto sortspecAttr = mlir::cast<relalg::SortSpecificationAttr>(attr);
      argumentTypes.push_back(sortspecAttr.getAttr().getColumn().type);
      locs.push_back(loc);
      sortByMembers.push_back(helper.lookupStateMemberForMaterializedColumn(&sortspecAttr.getAttr().getColumn()));
   }
   block->addArguments(argumentTypes, locs);
   block->addArguments(argumentTypes, locs);
   std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
   for (auto attr : sortSpecs) {
      auto sortspecAttr = mlir::cast<relalg::SortSpecificationAttr>(attr);
      mlir::Value left = block->getArgument(sortCriteria.size());
      mlir::Value right = block->getArgument(sortCriteria.size() + sortSpecs.size());
      if (sortspecAttr.getSortSpec() == relalg::SortSpec::desc) {
         std::swap(left, right);
      }
      sortCriteria.push_back({left, right});
   }
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      auto zero = rewriter.create<db::ConstantOp>(loc, rewriter.getI8Type(), rewriter.getIntegerAttr(rewriter.getI8Type(), 0));
      auto spaceShipResult = spaceShipCompare(rewriter, sortCriteria, 0, loc);
      mlir::Value isLt = rewriter.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, spaceShipResult, zero);
      rewriter.create<tuples::ReturnOp>(loc, isLt);
   }

   auto subOpSort = rewriter.create<subop::CreateSortedViewOp>(loc, subop::SortedViewType::get(rewriter.getContext(), mlir::cast<subop::State>(buffer.getType())), buffer, rewriter.getArrayAttr(sortByMembers));
   subOpSort.getRegion().getBlocks().push_back(block);
   return subOpSort.getResult();
}
class SortLowering : public OpConversionPattern<relalg::SortOp> {
   public:
   using OpConversionPattern<relalg::SortOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::SortOp sortOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = sortOp->getLoc();
      relalg::ColumnSet requiredColumns = getRequired(sortOp);
      requiredColumns.insert(sortOp.getUsedColumns());
      MaterializationHelper helper(requiredColumns, rewriter.getContext());
      auto vectorType = subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector = rewriter.create<subop::GenericCreateOp>(sortOp->getLoc(), vectorType);
      rewriter.create<subop::MaterializeOp>(sortOp->getLoc(), adaptor.getRel(), vector, helper.createColumnstateMapping());
      auto sortedView = createSortedView(rewriter, vector, sortOp.getSortspecs(), loc, helper);
      auto scanOp = rewriter.replaceOpWithNewOp<subop::ScanOp>(sortOp, sortedView, helper.createStateColumnMapping());
      scanOp->setAttr("sequential", rewriter.getUnitAttr());
      return success();
   }
};

class TopKLowering : public OpConversionPattern<relalg::TopKOp> {
   public:
   using OpConversionPattern<relalg::TopKOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::TopKOp topk, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = topk->getLoc();
      relalg::ColumnSet requiredColumns = getRequired(topk);
      requiredColumns.insert(topk.getUsedColumns());
      MaterializationHelper helper(requiredColumns, rewriter.getContext());

      auto* block = new Block;
      std::vector<Attribute> sortByMembers;
      std::vector<Type> argumentTypes;
      std::vector<Location> locs;
      for (auto attr : topk.getSortspecs()) {
         auto sortspecAttr = mlir::cast<relalg::SortSpecificationAttr>(attr);
         argumentTypes.push_back(sortspecAttr.getAttr().getColumn().type);
         locs.push_back(loc);
         sortByMembers.push_back(helper.lookupStateMemberForMaterializedColumn(&sortspecAttr.getAttr().getColumn()));
      }
      block->addArguments(argumentTypes, locs);
      block->addArguments(argumentTypes, locs);
      std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
      for (auto attr : topk.getSortspecs()) {
         auto sortspecAttr = mlir::cast<relalg::SortSpecificationAttr>(attr);
         mlir::Value left = block->getArgument(sortCriteria.size());
         mlir::Value right = block->getArgument(sortCriteria.size() + topk.getSortspecs().size());
         if (sortspecAttr.getSortSpec() == relalg::SortSpec::desc) {
            std::swap(left, right);
         }
         sortCriteria.push_back({left, right});
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto zero = rewriter.create<db::ConstantOp>(loc, rewriter.getI8Type(), rewriter.getIntegerAttr(rewriter.getI8Type(), 0));
         auto spaceShipResult = spaceShipCompare(rewriter, sortCriteria, 0, loc);
         mlir::Value isLt = rewriter.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, spaceShipResult, zero);
         rewriter.create<tuples::ReturnOp>(loc, isLt);
      }
      auto heapType = subop::HeapType::get(getContext(), helper.createStateMembersAttr(), topk.getMaxRows());
      auto createHeapOp = rewriter.create<subop::CreateHeapOp>(loc, heapType, rewriter.getArrayAttr(sortByMembers));
      createHeapOp.getRegion().getBlocks().push_back(block);
      rewriter.create<subop::MaterializeOp>(loc, adaptor.getRel(), createHeapOp.getRes(), helper.createColumnstateMapping());
      auto scanOp = rewriter.replaceOpWithNewOp<subop::ScanOp>(topk, createHeapOp.getRes(), helper.createStateColumnMapping());
      scanOp->setAttr("sequential", rewriter.getUnitAttr());
      return success();
   }
};
class TmpLowering : public OpConversionPattern<relalg::TmpOp> {
   public:
   using OpConversionPattern<relalg::TmpOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(relalg::TmpOp tmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      MaterializationHelper helper(getRequired(tmpOp), rewriter.getContext());

      auto vectorType = subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector = rewriter.create<subop::GenericCreateOp>(tmpOp->getLoc(), vectorType);
      rewriter.create<subop::MaterializeOp>(tmpOp->getLoc(), adaptor.getRel(), vector, helper.createColumnstateMapping());
      std::vector<mlir::Value> results;
      for (size_t i = 0; i < tmpOp.getNumResults(); i++) {
         results.push_back(rewriter.create<subop::ScanOp>(tmpOp->getLoc(), vector, helper.createStateColumnMapping()));
      }
      rewriter.replaceOp(tmpOp, results);
      return success();
   }
};
class MaterializeLowering : public OpConversionPattern<relalg::MaterializeOp> {
   public:
   using OpConversionPattern<relalg::MaterializeOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto localTableType = mlir::cast<subop::LocalTableType>(materializeOp.getResult().getType());
      std::vector<Attribute> colNames;
      std::vector<NamedAttribute> mapping;
      for (size_t i = 0; i < materializeOp.getColumns().size(); i++) {
         auto columnName = mlir::cast<mlir::StringAttr>(materializeOp.getColumns()[i]);
         auto colMemberName = mlir::cast<mlir::StringAttr>(localTableType.getMembers().getNames()[i]).str();
         auto columnAttr = mlir::cast<tuples::ColumnRefAttr>(materializeOp.getCols()[i]);
         mapping.push_back(rewriter.getNamedAttr(colMemberName, columnAttr));
         colNames.push_back(columnName);
      }
      //todo: think about if this is really okay: two states have the same members...
      mlir::Value resultTable = rewriter.create<subop::GenericCreateOp>(materializeOp->getLoc(), subop::ResultTableType::get(getContext(), localTableType.getMembers()));
      rewriter.create<subop::MaterializeOp>(materializeOp->getLoc(), adaptor.getRel(), resultTable, rewriter.getDictionaryAttr(mapping));
      mlir::Value localTable = rewriter.create<subop::CreateFrom>(materializeOp.getLoc(), localTableType, rewriter.getArrayAttr(colNames), resultTable);
      rewriter.replaceOp(materializeOp, localTable);

      return success();
   }
};
class DistAggrFunc {
   protected:
   Type stateType;
   tuples::ColumnDefAttr destAttribute;
   tuples::ColumnRefAttr sourceAttribute;

   public:
   DistAggrFunc(const tuples::ColumnDefAttr& destAttribute, const tuples::ColumnRefAttr& sourceAttribute) : stateType(mlir::cast<tuples::ColumnDefAttr>(destAttribute).getColumn().type), destAttribute(destAttribute), sourceAttribute(sourceAttribute) {}
   virtual mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) = 0;
   virtual mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) = 0;
   virtual mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) = 0;
   const tuples::ColumnDefAttr& getDestAttribute() const {
      return destAttribute;
   }
   const tuples::ColumnRefAttr& getSourceAttribute() const {
      return sourceAttribute;
   }
   const Type& getStateType() const {
      return stateType;
   }
   virtual ~DistAggrFunc() {}
};

class CountStarAggrFunc : public DistAggrFunc {
   public:
   explicit CountStarAggrFunc(const tuples::ColumnDefAttr& destAttribute) : DistAggrFunc(destAttribute, tuples::ColumnRefAttr()) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      return builder.create<db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      auto one = builder.create<db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(1));
      return builder.create<db::AddOp>(loc, stateType, state, one);
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      return builder.create<db::AddOp>(loc, left, right);
   }
};
class CountAggrFunc : public DistAggrFunc {
   public:
   explicit CountAggrFunc(const tuples::ColumnDefAttr& destAttribute, const tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      return builder.create<db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      auto one = builder.create<db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(1));
      auto added = builder.create<db::AddOp>(loc, stateType, state, one);
      if (mlir::isa<db::NullableType>(args[0].getType())) {
         auto isNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), args[0]);
         return builder.create<mlir::arith::SelectOp>(loc, isNull, state, added);
      } else {
         return added;
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      return builder.create<db::AddOp>(loc, left, right);
   }
};
class AnyAggrFunc : public DistAggrFunc {
   public:
   explicit AnyAggrFunc(const tuples::ColumnDefAttr& destAttribute, const tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      return builder.create<util::UndefOp>(loc, stateType);
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      return args[0];
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      return left;
   }
};
class MaxAggrFunc : public DistAggrFunc {
   public:
   explicit MaxAggrFunc(const tuples::ColumnDefAttr& destAttribute, const tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      if (mlir::isa<db::NullableType>(stateType)) {
         return builder.create<db::NullOp>(loc, stateType);
      } else {
         return builder.create<db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
      }
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      mlir::Value stateLtArg = builder.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, state, args[0]);
      mlir::Value stateLtArgTruth = builder.create<db::DeriveTruth>(loc, stateLtArg);
      if (mlir::isa<db::NullableType>(stateType) && mlir::isa<db::NullableType>(args[0].getType())) {
         // state nullable, arg nullable
         mlir::Value isNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateLtArgTruth, isNull);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, args[0], state);
      } else if (mlir::isa<db::NullableType>(stateType)) {
         // state nullable, arg not nullable
         mlir::Value isNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateLtArgTruth, isNull);
         mlir::Value casted = builder.create<db::AsNullableOp>(loc, stateType, args[0]);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, casted, state);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<mlir::arith::SelectOp>(loc, stateLtArg, args[0], state);
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      mlir::Value leftLtRight = builder.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, left, right);
      mlir::Value leftLtRightTruth = builder.create<db::DeriveTruth>(loc, leftLtRight);
      if (mlir::isa<db::NullableType>(stateType)) {
         mlir::Value isLeftNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), left);
         mlir::Value setToRight = builder.create<mlir::arith::OrIOp>(loc, leftLtRightTruth, isLeftNull);
         return builder.create<mlir::arith::SelectOp>(loc, setToRight, right, left);
      } else {
         return builder.create<mlir::arith::SelectOp>(loc, leftLtRight, right, left);
      }
   }
};
class MinAggrFunc : public DistAggrFunc {
   mlir::Attribute getMaxValueAttr(mlir::Type type) const {
      auto* context = type.getContext();
      mlir::OpBuilder builder(context);
      mlir::Attribute maxValAttr = ::llvm::TypeSwitch<::mlir::Type, mlir::Attribute>(type)
                                      .Case<::db::DecimalType>([&](::db::DecimalType t) {
                                         if (t.getP() < 19) {
                                            return (mlir::Attribute) builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                         }
                                         std::vector<uint64_t> parts = {0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF};
                                         return (mlir::Attribute) builder.getIntegerAttr(mlir::IntegerType::get(context, 128), mlir::APInt(128, parts));
                                      })
                                      .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) {
                                         if (t.getWidth() == 32) {
                                            return (mlir::Attribute) builder.getI32IntegerAttr(std::numeric_limits<int32_t>::max());
                                         } else if (t.getWidth() == 64) {
                                            return (mlir::Attribute) builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                         } else {
                                            assert(false && "should not happen");
                                            return mlir::Attribute();
                                         }
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

   public:
   explicit MinAggrFunc(const tuples::ColumnDefAttr& destAttribute, const tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      if (mlir::isa<db::NullableType>(stateType)) {
         return builder.create<db::NullOp>(loc, stateType);
      } else {
         return builder.create<db::ConstantOp>(loc, stateType, getMaxValueAttr(stateType));
      }
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      mlir::Value stateGtArg = builder.create<db::CmpOp>(loc, db::DBCmpPredicate::gt, state, args[0]);
      mlir::Value stateGtArgTruth = builder.create<db::DeriveTruth>(loc, stateGtArg);
      if (mlir::isa<db::NullableType>(stateType) && mlir::isa<db::NullableType>(args[0].getType())) {
         // state nullable, arg nullable
         mlir::Value isNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateGtArgTruth, isNull);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, args[0], state);
      } else if (mlir::isa<db::NullableType>(stateType)) {
         // state nullable, arg not nullable
         mlir::Value casted = builder.create<db::AsNullableOp>(loc, stateType, args[0]);
         mlir::Value isNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateGtArgTruth, isNull);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, casted, state);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<mlir::arith::SelectOp>(loc, stateGtArg, args[0], state);
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      mlir::Value leftLtRight = builder.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, left, right);
      mlir::Value leftLtRightTruth = builder.create<db::DeriveTruth>(loc, leftLtRight);
      if (mlir::isa<db::NullableType>(stateType)) {
         mlir::Value isRightNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), right);
         mlir::Value setToLeft = builder.create<mlir::arith::OrIOp>(loc, leftLtRightTruth, isRightNull);
         return builder.create<mlir::arith::SelectOp>(loc, setToLeft, left, right);
      } else {
         return builder.create<mlir::arith::SelectOp>(loc, leftLtRight, left, right);
      }
   }
};
class SumAggrFunc : public DistAggrFunc {
   public:
   explicit SumAggrFunc(const tuples::ColumnDefAttr& destAttribute, const tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      if (mlir::isa<db::NullableType>(stateType)) {
         return builder.create<db::NullOp>(loc, stateType);
      } else {
         return builder.create<db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
      }
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      if (mlir::isa<db::NullableType>(stateType) && mlir::isa<db::NullableType>(args[0].getType())) {
         // state nullable, arg nullable
         mlir::Value isStateNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value isArgNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), args[0]);
         mlir::Value sum = builder.create<db::AddOp>(loc, state, args[0]);
         sum = builder.create<mlir::arith::SelectOp>(loc, isArgNull, state, sum);
         return builder.create<mlir::arith::SelectOp>(loc, isStateNull, args[0], sum);
      } else if (mlir::isa<db::NullableType>(stateType)) {
         // state nullable, arg not nullable
         mlir::Value isStateNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value zero = builder.create<db::ConstantOp>(loc, getBaseType(stateType), builder.getI64IntegerAttr(0));
         zero = builder.create<db::AsNullableOp>(loc, stateType, zero);
         state = builder.create<mlir::arith::SelectOp>(loc, isStateNull, zero, state);
         return builder.create<db::AddOp>(loc, state, args[0]);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<db::AddOp>(loc, state, args[0]);
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      if (mlir::isa<db::NullableType>(stateType)) {
         // state nullable, arg not nullable
         mlir::Value isLeftNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), left);
         mlir::Value isRightNull = builder.create<db::IsNullOp>(loc, builder.getI1Type(), right);
         mlir::Value zero = builder.create<db::ConstantOp>(loc, getBaseType(stateType), builder.getI64IntegerAttr(0));
         zero = builder.create<db::AsNullableOp>(loc, stateType, zero);
         mlir::Value newLeft = builder.create<mlir::arith::SelectOp>(loc, isLeftNull, zero, left);
         mlir::Value newRight = builder.create<mlir::arith::SelectOp>(loc, isRightNull, zero, right);
         mlir::Value sum = builder.create<db::AddOp>(loc, newLeft, newRight);
         mlir::Value bothNull = builder.create<mlir::arith::AndIOp>(loc, isLeftNull, isRightNull);
         return builder.create<mlir::arith::SelectOp>(loc, bothNull, left, sum);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<db::AddOp>(loc, left, right);
      }
   }
};
class OrderedWindowFunc {
   protected:
   tuples::ColumnDefAttr destAttribute;
   tuples::ColumnRefAttr sourceAttribute;

   public:
   OrderedWindowFunc(const tuples::ColumnDefAttr& destAttribute, const tuples::ColumnRefAttr& sourceAttribute) : destAttribute(destAttribute), sourceAttribute(sourceAttribute) {}
   virtual mlir::Value evaluate(mlir::ConversionPatternRewriter& builder, mlir::Location loc, mlir::Value stream, tuples::ColumnRefAttr beginReference, tuples::ColumnRefAttr endReference, tuples::ColumnRefAttr currReference) = 0;
   const tuples::ColumnDefAttr& getDestAttribute() const {
      return destAttribute;
   }
   const tuples::ColumnRefAttr& getSourceAttribute() const {
      return sourceAttribute;
   }
   virtual ~OrderedWindowFunc() {}
};
class RankWindowFunc : public OrderedWindowFunc {
   public:
   explicit RankWindowFunc(const tuples::ColumnDefAttr& destAttribute) : OrderedWindowFunc(destAttribute, tuples::ColumnRefAttr()) {}
   mlir::Value evaluate(mlir::ConversionPatternRewriter& builder, mlir::Location loc, mlir::Value stream, tuples::ColumnRefAttr beginReference, tuples::ColumnRefAttr endReference, tuples::ColumnRefAttr currReference) override {
      auto [entriesBetweenDef, entriesBetweenRef] = createColumn(builder.getIndexType(), "window", "entries_between");
      auto entriesBetweenRef2 = entriesBetweenRef;
      mlir::Value afterEntriesBetween = builder.create<subop::EntriesBetweenOp>(loc, stream, beginReference, currReference, entriesBetweenDef);
      return map(afterEntriesBetween, builder, loc, builder.getArrayAttr(destAttribute), [&](mlir::ConversionPatternRewriter& rewriter, subop::MapCreationHelper& helper, mlir::Location loc) {
         mlir::Value between = helper.access(entriesBetweenRef2, loc);
         mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         mlir::Value rankIndex = builder.create<mlir::arith::AddIOp>(loc, between, one);
         mlir::Value rank = builder.create<mlir::arith::IndexCastOp>(loc, destAttribute.getColumn().type, rankIndex);
         return std::vector<mlir::Value>{rank};
      });
   }
};
static Block* createAggrFuncInitialValueBlock(mlir::Location loc, mlir::OpBuilder& rewriter, std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs) {
   Block* initialValueBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(initialValueBlock);
      std::vector<mlir::Value> defaultValues;
      for (auto aggrFn : distAggrFuncs) {
         defaultValues.push_back(aggrFn->createDefaultValue(rewriter, loc));
      }
      rewriter.create<tuples::ReturnOp>(loc, defaultValues);
   }
   return initialValueBlock;
}
void performAggrFuncReduce(mlir::Location loc, mlir::OpBuilder& rewriter, std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs, tuples::ColumnRefAttr reference, mlir::Value stream, std::vector<mlir::Attribute> names, std::vector<NamedAttribute> defMapping) {
   mlir::Block* reduceBlock = new Block;
   mlir::Block* combineBlock = new Block;
   std::vector<mlir::Attribute> relevantColumns;
   std::unordered_map<tuples::Column*, mlir::Value> stateMap;
   std::unordered_map<tuples::Column*, mlir::Value> argMap;
   for (auto aggrFn : distAggrFuncs) {
      auto sourceColumn = aggrFn->getSourceAttribute();
      if (sourceColumn) {
         relevantColumns.push_back(sourceColumn);
         mlir::Value arg = reduceBlock->addArgument(sourceColumn.getColumn().type, loc);
         argMap.insert({&sourceColumn.getColumn(), arg});
      }
   }
   for (auto aggrFn : distAggrFuncs) {
      mlir::Value state = reduceBlock->addArgument(aggrFn->getStateType(), loc);
      stateMap.insert({&aggrFn->getDestAttribute().getColumn(), state});
   }
   auto reduceOp = rewriter.create<subop::ReduceOp>(loc, stream, reference, rewriter.getArrayAttr(relevantColumns), rewriter.getArrayAttr(names));
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(reduceBlock);
      std::vector<mlir::Value> newStateValues;
      for (auto aggrFn : distAggrFuncs) {
         std::vector<mlir::Value> arguments;
         if (aggrFn->getSourceAttribute()) {
            arguments.push_back(argMap.at(&aggrFn->getSourceAttribute().getColumn()));
         }
         mlir::Value result = aggrFn->aggregate(rewriter, loc, stateMap.at(&aggrFn->getDestAttribute().getColumn()), arguments);
         newStateValues.push_back(result);
      }
      rewriter.create<tuples::ReturnOp>(loc, newStateValues);
   }

   {
      std::vector<mlir::Value> combinedValues;
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(combineBlock);
      std::vector<mlir::Value> leftArgs;
      std::vector<mlir::Value> rightArgs;
      for (auto aggrFn : distAggrFuncs) {
         auto stateType = aggrFn->getStateType();
         leftArgs.push_back(combineBlock->addArgument(stateType, loc));
      }
      for (auto aggrFn : distAggrFuncs) {
         auto stateType = aggrFn->getStateType();
         rightArgs.push_back(combineBlock->addArgument(stateType, loc));
      }
      for (size_t i = 0; i < distAggrFuncs.size(); i++) {
         combinedValues.push_back(distAggrFuncs.at(i)->combine(rewriter, loc, leftArgs.at(i), rightArgs.at(i)));
      }
      rewriter.create<tuples::ReturnOp>(loc, combinedValues);
   }
   reduceOp.getRegion().push_back(reduceBlock);
   reduceOp.getCombine().push_back(combineBlock);
}
static std::tuple<mlir::Value, mlir::DictionaryAttr, mlir::DictionaryAttr> performAggregation(mlir::Location loc, mlir::OpBuilder& rewriter, std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs, relalg::OrderedAttributes keyAttributes, mlir::Value stream, std::function<void(mlir::Location, mlir::OpBuilder&, std::vector<std::shared_ptr<DistAggrFunc>>, tuples::ColumnRefAttr, mlir::Value, std::vector<mlir::Attribute>, std::vector<NamedAttribute>)> createReduceFn) {
   auto* context = rewriter.getContext();
   auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   mlir::Value state;
   auto* initialValueBlock = createAggrFuncInitialValueBlock(loc, rewriter, distAggrFuncs);
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   std::vector<NamedAttribute> defMapping;
   std::vector<NamedAttribute> computedDefMapping;

   for (auto aggrFn : distAggrFuncs) {
      auto memberName = getUniqueMember(rewriter.getContext(), "aggrval");
      names.push_back(rewriter.getStringAttr(memberName));
      types.push_back(mlir::TypeAttr::get(aggrFn->getStateType()));
      auto def = aggrFn->getDestAttribute();
      defMapping.push_back(rewriter.getNamedAttr(memberName, def));
      computedDefMapping.push_back(rewriter.getNamedAttr(memberName, def));
   }
   auto stateMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, names), mlir::ArrayAttr::get(context, types));
   mlir::Type stateType;

   mlir::Value afterLookup;
   auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");

   if (keyAttributes.getAttrs().empty()) {
      stateType = subop::SimpleStateType::get(rewriter.getContext(), stateMembers);
      auto createOp = rewriter.create<subop::CreateSimpleStateOp>(loc, stateType);
      createOp.getInitFn().push_back(initialValueBlock);
      state = createOp.getRes();
      afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, state, rewriter.getArrayAttr({}), referenceDefAttr);
   } else {
      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<mlir::Location> locations;
      for (auto* x : keyAttributes.getAttrs()) {
         auto memberName = getUniqueMember(rewriter.getContext(), "keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(x->type));
         keyTypes.push_back((x->type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(x)));
         locations.push_back(loc);
      }
      auto keyMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      stateType = subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers, false);

      auto createOp = rewriter.create<subop::GenericCreateOp>(loc, stateType);
      state = createOp.getRes();
      auto lookupOp = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, state, keyAttributes.getArrayAttr(rewriter.getContext()), referenceDefAttr);
      afterLookup = lookupOp;
      lookupOp.getInitFn().push_back(initialValueBlock);
      mlir::Block* equalBlock = new Block;
      lookupOp.getEqFn().push_back(equalBlock);
      // equal functions compare two values (e.g. attr1 and attr2) both of the same type
      equalBlock->addArguments(keyTypes, locations); // types of attr1
      equalBlock->addArguments(keyTypes, locations); // types of attr2
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(equalBlock);
         mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
         rewriter.create<tuples::ReturnOp>(loc, compared);
      }
   }
   referenceDefAttr.getColumn().type = subop::LookupEntryRefType::get(context, mlir::cast<subop::LookupAbleState>(stateType));

   auto referenceRefAttr = colManager.createRef(&referenceDefAttr.getColumn());
   createReduceFn(loc, rewriter, distAggrFuncs, referenceRefAttr, afterLookup, names, defMapping);
   return {state, rewriter.getDictionaryAttr(defMapping), rewriter.getDictionaryAttr(computedDefMapping)};
}
class WindowLowering : public OpConversionPattern<relalg::WindowOp> {
   public:
   using OpConversionPattern<relalg::WindowOp>::OpConversionPattern;
   struct AnalyzedWindow {
      std::vector<std::pair<relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distAggrFuncs;
      std::vector<std::shared_ptr<OrderedWindowFunc>> orderedWindowFunctions;
   };

   void analyze(relalg::WindowOp windowOp, AnalyzedWindow& analyzedWindow) const {
      tuples::ReturnOp terminator = mlir::cast<tuples::ReturnOp>(windowOp.getAggrFunc().front().getTerminator());
      std::unordered_map<mlir::Operation*, std::pair<relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distinct;
      distinct.insert({nullptr, {relalg::OrderedAttributes::fromVec({}), {}}});
      for (size_t i = 0; i < windowOp.getComputedCols().size(); i++) {
         auto destColumnAttr = mlir::cast<tuples::ColumnDefAttr>(windowOp.getComputedCols()[i]);
         mlir::Value computedVal = terminator.getResults()[i];
         mlir::Value tupleStream;
         std::shared_ptr<DistAggrFunc> distAggrFunc;
         if (auto aggrFn = mlir::dyn_cast_or_null<relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            tupleStream = aggrFn.getRel();
            auto sourceColumnAttr = aggrFn.getAttr();
            if (aggrFn.getFn() == relalg::AggrFunc::sum) {
               distAggrFunc = std::make_shared<SumAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::min) {
               distAggrFunc = std::make_shared<MinAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::max) {
               distAggrFunc = std::make_shared<MaxAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::any) {
               distAggrFunc = std::make_shared<AnyAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::count) {
               distAggrFunc = std::make_shared<CountAggrFunc>(destColumnAttr, sourceColumnAttr);
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            tupleStream = countOp.getRel();
            distAggrFunc = std::make_shared<CountStarAggrFunc>(destColumnAttr);
         }
         if (auto rankOp = mlir::dyn_cast_or_null<relalg::RankOp>(computedVal.getDefiningOp())) {
            analyzedWindow.orderedWindowFunctions.push_back(std::make_shared<RankWindowFunc>(destColumnAttr));
         }
         if (distAggrFunc) {
            if (!distinct.count(tupleStream.getDefiningOp())) {
               if (auto projectionOp = mlir::dyn_cast_or_null<relalg::ProjectionOp>(tupleStream.getDefiningOp())) {
                  distinct[tupleStream.getDefiningOp()] = {relalg::OrderedAttributes::fromRefArr(projectionOp.getCols()), {}};
               }
            }
            distinct.at(tupleStream.getDefiningOp()).second.push_back(distAggrFunc);
         }
      };
      for (auto d : distinct) {
         analyzedWindow.distAggrFuncs.push_back({d.second.first, d.second.second});
      }
   }
   void performWindowOp(relalg::WindowOp windowOp, mlir::Value inputStream, ConversionPatternRewriter& rewriter, std::function<mlir::Value(ConversionPatternRewriter&, mlir::Value, mlir::DictionaryAttr, mlir::Location)> evaluate) const {
      relalg::ColumnSet requiredColumns = getRequired(windowOp);
      auto& colManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      requiredColumns.insert(windowOp.getUsedColumns());
      requiredColumns.remove(windowOp.getCreatedColumns());
      auto loc = windowOp->getLoc();
      if (windowOp.getPartitionBy().empty()) {
         MaterializationHelper helper(requiredColumns, rewriter.getContext());

         auto vectorType = subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
         mlir::Value vector = rewriter.create<subop::GenericCreateOp>(loc, vectorType);
         rewriter.create<subop::MaterializeOp>(loc, inputStream, vector, helper.createColumnstateMapping());
         mlir::Value continuousView;
         if (windowOp.getOrderBy().empty()) {
            auto continuousViewType = subop::ContinuousViewType::get(rewriter.getContext(), vectorType);
            continuousView = rewriter.create<subop::CreateContinuousView>(loc, continuousViewType, vector);
         } else {
            auto sortedView = createSortedView(rewriter, vector, windowOp.getOrderBy(), loc, helper);
            auto continuousViewType = subop::ContinuousViewType::get(rewriter.getContext(), mlir::cast<subop::State>(sortedView.getType()));
            continuousView = rewriter.create<subop::CreateContinuousView>(loc, continuousViewType, sortedView);
         }
         rewriter.replaceOp(windowOp, evaluate(rewriter, continuousView, helper.createStateColumnMapping(), loc));
      } else {
         auto keyAttributes = relalg::OrderedAttributes::fromRefArr(windowOp.getPartitionBy());
         auto valueColumns = requiredColumns;
         valueColumns.remove(relalg::ColumnSet::fromArrayAttr(windowOp.getPartitionBy()));
         MaterializationHelper helper(valueColumns, rewriter.getContext());
         auto bufferType = subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());

         std::vector<NamedAttribute> defMapping;

         std::vector<mlir::Attribute> keyNames;
         auto* context = getContext();
         std::vector<mlir::Attribute> keyTypesAttr;
         std::vector<mlir::Type> keyTypes;
         std::vector<mlir::Location> locations;
         for (auto* x : keyAttributes.getAttrs()) {
            auto memberName = getUniqueMember(getContext(), "keyval");
            keyNames.push_back(rewriter.getStringAttr(memberName));
            keyTypesAttr.push_back(mlir::TypeAttr::get(x->type));
            keyTypes.push_back((x->type));
            defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(x)));
            locations.push_back(loc);
         }
         auto bufferMember = getUniqueMember(getContext(), "buffer");
         auto [bufferDef, bufferRef] = createColumn(bufferType, "window", "buffer");
         defMapping.push_back(rewriter.getNamedAttr(bufferMember, bufferDef));

         auto stateMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {rewriter.getStringAttr(bufferMember)}), mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(bufferType)}));
         auto keyMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
         auto hashMapType = subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers, false);

         auto createOp = rewriter.create<subop::GenericCreateOp>(loc, hashMapType);
         mlir::Value hashMap = createOp.getRes();

         auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
         referenceDefAttr.getColumn().type = subop::LookupEntryRefType::get(context, hashMapType);

         auto lookupOp = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(getContext()), inputStream, hashMap, keyAttributes.getArrayAttr(rewriter.getContext()), referenceDefAttr);
         mlir::Value afterLookup = lookupOp;
         Block* initialValueBlock = new Block;
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(initialValueBlock);
            std::vector<mlir::Value> defaultValues;
            mlir::Value buffer = rewriter.create<subop::GenericCreateOp>(loc, bufferType);
            buffer.getDefiningOp()->setAttr("initial_capacity", rewriter.getI64IntegerAttr(1));
            buffer.getDefiningOp()->setAttr("group", rewriter.getI64IntegerAttr(0));
            rewriter.create<tuples::ReturnOp>(loc, buffer);
         }
         lookupOp.getInitFn().push_back(initialValueBlock);
         mlir::Block* equalBlock = new Block;
         lookupOp.getEqFn().push_back(equalBlock);
         equalBlock->addArguments(keyTypes, locations);
         equalBlock->addArguments(keyTypes, locations);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(equalBlock);
            mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
            rewriter.create<tuples::ReturnOp>(loc, compared);
         }
         auto referenceRefAttr = colManager.createRef(&referenceDefAttr.getColumn());
         auto orderedValueColumns = relalg::OrderedAttributes::fromColumns(valueColumns);

         auto reduceOp = rewriter.create<subop::ReduceOp>(loc, afterLookup, referenceRefAttr, orderedValueColumns.getArrayAttr(context), rewriter.getArrayAttr({rewriter.getStringAttr(bufferMember)}));
         {
            mlir::Block* reduceBlock = new Block;

            std::vector<mlir::Value> columnValues;
            for (auto* c : orderedValueColumns.getAttrs()) {
               columnValues.push_back(reduceBlock->addArgument(c->type, loc));
            }
            mlir::Value currentBuffer = reduceBlock->addArgument(bufferType, loc);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceBlock);
            std::vector<mlir::Value> newStateValues;
            mlir::Value stream = rewriter.create<subop::InFlightOp>(loc, columnValues, orderedValueColumns.getDefArrayAttr(context));
            rewriter.create<subop::MaterializeOp>(loc, stream, currentBuffer, helper.createColumnstateMapping());
            rewriter.create<tuples::ReturnOp>(loc, currentBuffer);
            reduceOp.getRegion().push_back(reduceBlock);
         }
         {
            mlir::Block* combineBlock = new Block;
            mlir::Value currentBuffer = combineBlock->addArgument(bufferType, loc);
            mlir::Value otherBuffer = combineBlock->addArgument(bufferType, loc);

            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(combineBlock);
            std::vector<mlir::Value> newStateValues;
            auto nestedExecutionGroup = rewriter.create<subop::NestedExecutionGroupOp>(loc, mlir::TypeRange{}, mlir::ValueRange{currentBuffer, otherBuffer});
            auto* nestedBlock = new Block;
            nestedExecutionGroup.getSubOps().push_back(nestedBlock);
            auto leftGroup = nestedBlock->addArgument(currentBuffer.getType(), loc);
            auto rightGroup = nestedBlock->addArgument(otherBuffer.getType(), loc);
            {
               mlir::OpBuilder::InsertionGuard guard(rewriter);
               rewriter.setInsertionPointToStart(nestedBlock);
               {
                  mlir::OpBuilder::InsertionGuard guard(rewriter);
                  auto step = rewriter.create<subop::ExecutionStepOp>(loc, mlir::TypeRange{}, mlir::ValueRange{leftGroup, rightGroup}, rewriter.getBoolArrayAttr({false, false}));
                  auto* stepBlock = new Block;
                  step.getRegion().push_back(stepBlock);
                  rewriter.setInsertionPointToStart(stepBlock);
                  auto leftStep = stepBlock->addArgument(leftGroup.getType(), loc);
                  auto rightStep = stepBlock->addArgument(rightGroup.getType(), loc);
                  auto scan = rewriter.create<subop::ScanOp>(loc, rightStep, helper.createStateColumnMapping());
                  rewriter.create<subop::MaterializeOp>(loc, scan.getRes(), leftStep, helper.createColumnstateMapping());
                  rewriter.create<subop::ExecutionStepReturnOp>(loc, mlir::ValueRange{});
               }
               rewriter.create<subop::NestedExecutionGroupReturnOp>(loc, mlir::ValueRange{});
            }
            rewriter.create<tuples::ReturnOp>(loc, currentBuffer);
            reduceOp.getCombine().push_back(combineBlock);
         }
         mlir::Value newStream = rewriter.create<subop::ScanOp>(loc, hashMap, rewriter.getDictionaryAttr(defMapping));
         auto nestedMapOp = rewriter.create<subop::NestedMapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), newStream, rewriter.getArrayAttr({bufferRef}));
         auto* b = new Block;
         b->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
         mlir::Value buffer = b->addArgument(bufferType, loc);
         nestedMapOp.getRegion().push_back(b);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(b);
            mlir::Value continuousView;
            if (windowOp.getOrderBy().empty()) {
               auto continuousViewType = subop::ContinuousViewType::get(rewriter.getContext(), mlir::cast<subop::State>(buffer.getType()));
               continuousView = rewriter.create<subop::CreateContinuousView>(loc, continuousViewType, buffer);
            } else {
               auto sortedView = createSortedView(rewriter, buffer, windowOp.getOrderBy(), loc, helper);
               auto continuousViewType = subop::ContinuousViewType::get(rewriter.getContext(), mlir::cast<subop::State>(sortedView.getType()));
               continuousView = rewriter.create<subop::CreateContinuousView>(loc, continuousViewType, sortedView);
            }
            rewriter.create<tuples::ReturnOp>(loc, evaluate(rewriter, continuousView, helper.createStateColumnMapping(), loc));
         }

         rewriter.replaceOp(windowOp, nestedMapOp.getRes());
      }
   }

   std::tuple<Value, DictionaryAttr> buildSegmentTree(Location loc, ConversionPatternRewriter& rewriter, std::vector<std::shared_ptr<DistAggrFunc>> aggrFuncs, Value continuousView, mlir::DictionaryAttr continuousViewMapping) const {
      std::vector<mlir::Attribute> relevantMembers;
      std::unordered_map<tuples::Column*, mlir::Value> stateMap;
      auto* initBlock = new Block;
      {
         std::vector<mlir::Value> initialValues;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initBlock);
         for (auto aggrFn : aggrFuncs) {
            auto sourceColumn = aggrFn->getSourceAttribute();
            mlir::Value initValue = aggrFn->createDefaultValue(rewriter, loc);
            if (sourceColumn) {
               for (auto x : continuousViewMapping) {
                  if (&mlir::cast<tuples::ColumnDefAttr>(x.getValue()).getColumn() == &sourceColumn.getColumn()) {
                     relevantMembers.push_back(x.getName());
                  }
               }
               mlir::Value arg = initBlock->addArgument(sourceColumn.getColumn().type, loc);
               initValue = aggrFn->aggregate(rewriter, loc, initValue, arg);
            } else {
               initValue = aggrFn->aggregate(rewriter, loc, initValue, {});
            }
            initialValues.push_back(initValue);
         }
         rewriter.create<tuples::ReturnOp>(loc, initialValues);
      }
      auto* combineBlock = new Block;

      {
         std::vector<mlir::Value> combinedValues;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(combineBlock);
         std::vector<mlir::Value> leftArgs;
         std::vector<mlir::Value> rightArgs;
         for (auto aggrFn : aggrFuncs) {
            auto stateType = aggrFn->getStateType();
            leftArgs.push_back(combineBlock->addArgument(stateType, loc));
         }
         for (auto aggrFn : aggrFuncs) {
            auto stateType = aggrFn->getStateType();
            rightArgs.push_back(combineBlock->addArgument(stateType, loc));
         }
         for (size_t i = 0; i < aggrFuncs.size(); i++) {
            combinedValues.push_back(aggrFuncs.at(i)->combine(rewriter, loc, leftArgs.at(i), rightArgs.at(i)));
         }
         rewriter.create<tuples::ReturnOp>(loc, combinedValues);
      }
      std::vector<NamedAttribute> defMapping;
      std::vector<mlir::Attribute> names;
      std::vector<mlir::Attribute> types;
      for (auto aggrFn : aggrFuncs) {
         auto memberName = getUniqueMember(getContext(), "aggrval");
         names.push_back(rewriter.getStringAttr(memberName));
         types.push_back(mlir::TypeAttr::get(aggrFn->getStateType()));
         auto def = aggrFn->getDestAttribute();
         defMapping.push_back(rewriter.getNamedAttr(memberName, def));
      }
      auto fromMemberName = getUniqueMember(getContext(), "from");
      auto toMemberName = getUniqueMember(getContext(), "to");
      auto continuousViewRefType = subop::ContinuousEntryRefType::get(rewriter.getContext(), mlir::cast<subop::ContinuousViewType>(continuousView.getType()));
      auto cVRTAttr = mlir::TypeAttr::get(continuousViewRefType);
      auto keyStateMembers = subop::StateMembersAttr::get(rewriter.getContext(), mlir::ArrayAttr::get(rewriter.getContext(), {rewriter.getStringAttr(fromMemberName), rewriter.getStringAttr(toMemberName)}), mlir::ArrayAttr::get(rewriter.getContext(), {cVRTAttr, cVRTAttr}));
      auto valueStateMembers = subop::StateMembersAttr::get(rewriter.getContext(), mlir::ArrayAttr::get(rewriter.getContext(), names), mlir::ArrayAttr::get(rewriter.getContext(), types));
      auto segmentTreeViewType = subop::SegmentTreeViewType::get(rewriter.getContext(), keyStateMembers, valueStateMembers);
      auto createOp = rewriter.create<subop::CreateSegmentTreeView>(loc, segmentTreeViewType, continuousView, rewriter.getArrayAttr(relevantMembers));
      createOp.getInitialFn().push_back(initBlock);
      createOp.getCombineFn().push_back(combineBlock);
      return {createOp.getResult(), rewriter.getDictionaryAttr(defMapping)};
   }
   LogicalResult matchAndRewrite(relalg::WindowOp windowOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      AnalyzedWindow analyzedWindow;
      analyze(windowOp, analyzedWindow);
      //don't handle distinct aggregate functions for window functions
      if (analyzedWindow.distAggrFuncs.size() > 1) return failure();
      std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs;
      if (analyzedWindow.distAggrFuncs.size() == 1) {
         if (!analyzedWindow.distAggrFuncs[0].first.getAttrs().empty()) return failure();
         distAggrFuncs = analyzedWindow.distAggrFuncs[0].second;
      }
      auto from = static_cast<int64_t>(windowOp.getFrom());
      auto to = static_cast<int64_t>(windowOp.getTo());
      auto fromBegin = from == std::numeric_limits<int64_t>().min();
      auto fromEnd = to == std::numeric_limits<int64_t>().max();

      auto evaluate = [&](ConversionPatternRewriter& rewriter, mlir::Value continuousView, mlir::DictionaryAttr columnMapping, mlir::Location loc) {
         auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         auto continuousViewRefType = subop::ContinuousEntryRefType::get(rewriter.getContext(), mlir::cast<subop::ContinuousViewType>(continuousView.getType()));
         auto [beginReferenceDefAttr, beginReferenceRefAttr] = createColumn(continuousViewRefType, "view", "begin");
         auto [endReferenceDefAttr, endReferenceRefAttr] = createColumn(continuousViewRefType, "view", "end");
         auto [referenceDefAttr, referenceRefAttr] = createColumn(continuousViewRefType, "scan", "ref");

         std::tuple<mlir::Value, mlir::DictionaryAttr, mlir::DictionaryAttr> staticAggregateResults;
         std::tuple<mlir::Value, mlir::DictionaryAttr> segmentTreeViewResult;
         if (!distAggrFuncs.empty() && fromBegin && fromEnd) {
            mlir::Value scan = rewriter.create<subop::ScanOp>(loc, continuousView, columnMapping);
            staticAggregateResults = performAggregation(loc, rewriter, distAggrFuncs, relalg::OrderedAttributes::fromVec({}), scan, performAggrFuncReduce);
         } else if (!distAggrFuncs.empty()) {
            segmentTreeViewResult = buildSegmentTree(loc, rewriter, distAggrFuncs, continuousView, columnMapping);
         }
         mlir::Value scan = rewriter.create<subop::ScanRefsOp>(loc, continuousView, referenceDefAttr);
         mlir::Value afterGather = rewriter.create<subop::GatherOp>(loc, scan, referenceRefAttr, columnMapping);
         mlir::Value afterBegin = rewriter.create<subop::GetBeginReferenceOp>(loc, afterGather, continuousView, beginReferenceDefAttr);
         mlir::Value afterEnd = rewriter.create<subop::GetEndReferenceOp>(loc, afterBegin, continuousView, endReferenceDefAttr);
         mlir::Value current = afterEnd;
         tuples::ColumnRefAttr rangeBegin;
         tuples::ColumnRefAttr rangeEnd;
         if (fromBegin) {
            rangeBegin = beginReferenceRefAttr;
         }
         if (fromEnd) {
            rangeEnd = endReferenceRefAttr;
         }
         if (from == 0) {
            rangeBegin = referenceRefAttr;
         } else {
            auto [fromDefAttr, fromRefAttr] = createColumn(continuousViewRefType, "frame", "from");
            auto [withConst, constCol] = mapIndex(current, rewriter, loc, from);
            current = rewriter.create<subop::OffsetReferenceBy>(loc, withConst, referenceRefAttr, colManager.createRef(constCol), fromDefAttr);
            rangeBegin = fromRefAttr;
         }
         if (to == 0) {
            rangeEnd = referenceRefAttr;
         } else {
            auto [toDefAttr, toRefAttr] = createColumn(continuousViewRefType, "frame", "to");
            auto [withConst, constCol] = mapIndex(current, rewriter, loc, to);
            current = rewriter.create<subop::OffsetReferenceBy>(loc, withConst, referenceRefAttr, colManager.createRef(constCol), toDefAttr);
            rangeEnd = toRefAttr;
         }
         assert(rangeBegin && rangeEnd);
         for (auto orderedWindowFn : analyzedWindow.orderedWindowFunctions) {
            current = orderedWindowFn->evaluate(rewriter, loc, current, rangeBegin, rangeEnd, colManager.createRef(&referenceDefAttr.getColumn()));
         }
         if (!distAggrFuncs.empty() && fromBegin && fromEnd) {
            mlir::Value state = std::get<0>(staticAggregateResults);
            mlir::DictionaryAttr stateColumnMapping = std::get<2>(staticAggregateResults);
            auto [referenceDef, referenceRef] = createColumn(subop::LookupEntryRefType::get(getContext(), mlir::cast<subop::LookupAbleState>(state.getType())), "lookup", "ref");
            mlir::Value afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(getContext()), current, state, rewriter.getArrayAttr({}), referenceDef);
            current = rewriter.create<subop::GatherOp>(loc, afterLookup, referenceRef, stateColumnMapping);
         } else if (!distAggrFuncs.empty()) {
            mlir::Value segmentTreeView = std::get<0>(segmentTreeViewResult);
            mlir::DictionaryAttr stateColumnMapping = std::get<1>(segmentTreeViewResult);
            auto [referenceDef, referenceRef] = createColumn(subop::LookupEntryRefType::get(getContext(), mlir::cast<subop::LookupAbleState>(segmentTreeView.getType())), "lookup", "ref");
            mlir::Value afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(getContext()), current, segmentTreeView, rewriter.getArrayAttr({rangeBegin, rangeEnd}), referenceDef);
            current = rewriter.create<subop::GatherOp>(loc, afterLookup, referenceRef, stateColumnMapping);
         }
         return current;
      };
      performWindowOp(windowOp, adaptor.getRel(), rewriter, evaluate);
      return success();
   }
};
class AggregationLowering : public OpConversionPattern<relalg::AggregationOp> {
   public:
   using OpConversionPattern<relalg::AggregationOp>::OpConversionPattern;
   struct AnalyzedAggregation {
      std::vector<std::pair<relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distAggrFuncs;
   };

   void analyze(relalg::AggregationOp aggregationOp, AnalyzedAggregation& analyzedAggregation) const {
      tuples::ReturnOp terminator = mlir::cast<tuples::ReturnOp>(aggregationOp.getAggrFunc().front().getTerminator());
      std::unordered_map<mlir::Operation*, std::pair<relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distinct;
      distinct.insert({nullptr, {relalg::OrderedAttributes::fromVec({}), {}}});
      for (size_t i = 0; i < aggregationOp.getComputedCols().size(); i++) {
         auto destColumnAttr = mlir::cast<tuples::ColumnDefAttr>(aggregationOp.getComputedCols()[i]);
         mlir::Value computedVal = terminator.getResults()[i];
         mlir::Value tupleStream;
         std::shared_ptr<DistAggrFunc> distAggrFunc;
         if (auto aggrFn = mlir::dyn_cast_or_null<relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            tupleStream = aggrFn.getRel();
            auto sourceColumnAttr = aggrFn.getAttr();
            switch (aggrFn.getFn()) {
               case relalg::AggrFunc::sum:
                  distAggrFunc = std::make_shared<SumAggrFunc>(destColumnAttr, sourceColumnAttr);
                  break;
               case relalg::AggrFunc::min:
                  distAggrFunc = std::make_shared<MinAggrFunc>(destColumnAttr, sourceColumnAttr);
                  break;
               case relalg::AggrFunc::max:
                  distAggrFunc = std::make_shared<MaxAggrFunc>(destColumnAttr, sourceColumnAttr);
                  break;
               case relalg::AggrFunc::any:
                  distAggrFunc = std::make_shared<AnyAggrFunc>(destColumnAttr, sourceColumnAttr);
                  break;
               case relalg::AggrFunc::count:
                  distAggrFunc = std::make_shared<CountAggrFunc>(destColumnAttr, sourceColumnAttr);
                  break;
               default:
                  assert(false && "Aggregation Function is not implemented");
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            tupleStream = countOp.getRel();
            distAggrFunc = std::make_shared<CountStarAggrFunc>(destColumnAttr);
         }

         if (!distinct.count(tupleStream.getDefiningOp())) {
            if (auto projectionOp = mlir::dyn_cast_or_null<relalg::ProjectionOp>(tupleStream.getDefiningOp())) {
               distinct[tupleStream.getDefiningOp()] = {relalg::OrderedAttributes::fromRefArr(projectionOp.getCols()), {}};
            }
         }
         distinct.at(tupleStream.getDefiningOp()).second.push_back(distAggrFunc);
      };
      for (auto d : distinct) {
         analyzedAggregation.distAggrFuncs.push_back({d.second.first, d.second.second});
      }
   }

   LogicalResult matchAndRewrite(relalg::AggregationOp aggregationOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      AnalyzedAggregation analyzedAggregation;
      analyze(aggregationOp, analyzedAggregation);

      auto loc = aggregationOp->getLoc();

      auto keyAttributes = relalg::OrderedAttributes::fromRefArr(aggregationOp.getGroupByColsAttr());
      std::vector<std::tuple<mlir::Value, mlir::DictionaryAttr, mlir::DictionaryAttr>> subResults;
      for (auto x : analyzedAggregation.distAggrFuncs) {
         auto distinctBy = x.first;
         auto& aggrFuncs = x.second;
         if (aggrFuncs.empty()) continue;
         mlir::Value tree = adaptor.getRel();
         if (distinctBy.getAttrs().size() != 0) {
            auto projectionAttrs = keyAttributes.getAttrs();
            auto distinctAttrs = distinctBy.getAttrs();
            projectionAttrs.insert(projectionAttrs.end(), distinctAttrs.begin(), distinctAttrs.end());
            tree = rewriter.create<relalg::ProjectionOp>(loc, relalg::SetSemantic::distinct, tree, relalg::OrderedAttributes::fromVec(projectionAttrs).getArrayAttr(rewriter.getContext()));
         }
         auto partialResult = performAggregation(loc, rewriter, x.second, keyAttributes, tree, performAggrFuncReduce);
         subResults.push_back(partialResult);
      }
      if (subResults.empty()) {
         // handle the case that aggregation is only used for distinct projection
         rewriter.replaceOpWithNewOp<relalg::ProjectionOp>(aggregationOp, relalg::SetSemantic::distinct, adaptor.getRel(), aggregationOp.getGroupByCols());
         return success();
      }

      // rejoin the aggregation functions
      mlir::Value newStream = rewriter.create<subop::ScanOp>(loc, std::get<0>(subResults.at(0)), std::get<1>(subResults.at(0)));
      ; //= scan %state of subresult 0
      for (size_t i = 1; i < subResults.size(); i++) {
         mlir::Value state = std::get<0>(subResults.at(i));
         mlir::DictionaryAttr stateColumnMapping = std::get<2>(subResults.at(i));

         auto [optionalReferenceDef, optionalReferenceRef] = createColumn(subop::OptionalType::get(getContext(), subop::LookupEntryRefType::get(getContext(), mlir::cast<subop::LookupAbleState>(state.getType()))), "lookup", "ref");
         auto [finalReferenceDef, finalReferenceRef] = createColumn(subop::LookupEntryRefType::get(getContext(), mlir::cast<subop::LookupAbleState>(state.getType())), "lookup", "ref");

         mlir::Value afterLookup;
         if (keyAttributes.getAttrs().empty()) {
            afterLookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(getContext()), newStream, state, aggregationOp.getGroupByCols(), finalReferenceDef);
         } else {
            std::vector<mlir::Type> keyTypes;
            std::vector<mlir::Location> locations;
            for (auto* x : keyAttributes.getAttrs()) {
               keyTypes.push_back((x->type));
               locations.push_back(loc);
            }

            auto lookupOp = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), newStream, state, keyAttributes.getArrayAttr(rewriter.getContext()), optionalReferenceDef);

            mlir::Block* equalBlock = new Block;
            lookupOp.getEqFn().push_back(equalBlock);
            equalBlock->addArguments(keyTypes, locations);
            equalBlock->addArguments(keyTypes, locations);
            {
               mlir::OpBuilder::InsertionGuard guard(rewriter);
               rewriter.setInsertionPointToStart(equalBlock);
               mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
               rewriter.create<tuples::ReturnOp>(loc, compared);
            }
            afterLookup = rewriter.create<subop::UnwrapOptionalRefOp>(loc, lookupOp.getRes(), optionalReferenceRef, finalReferenceDef);
         }
         newStream = rewriter.create<subop::GatherOp>(loc, afterLookup, finalReferenceRef, stateColumnMapping);
      }

      rewriter.replaceOp(aggregationOp, newStream);

      return success();
   }
};

class GroupJoinLowering : public OpConversionPattern<relalg::GroupJoinOp> {
   public:
   using OpConversionPattern<relalg::GroupJoinOp>::OpConversionPattern;
   struct AnalyzedAggregation {
      std::vector<std::pair<relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distAggrFuncs;
   };

   void analyze(relalg::GroupJoinOp groupJoinOp, AnalyzedAggregation& analyzedAggregation) const {
      tuples::ReturnOp terminator = mlir::cast<tuples::ReturnOp>(groupJoinOp.getAggrFunc().front().getTerminator());
      std::unordered_map<mlir::Operation*, std::pair<relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distinct;
      distinct.insert({nullptr, {relalg::OrderedAttributes::fromVec({}), {}}});
      for (size_t i = 0; i < groupJoinOp.getComputedCols().size(); i++) {
         auto destColumnAttr = mlir::cast<tuples::ColumnDefAttr>(groupJoinOp.getComputedCols()[i]);
         mlir::Value computedVal = terminator.getResults()[i];
         mlir::Value tupleStream;
         std::shared_ptr<DistAggrFunc> distAggrFunc;
         if (auto aggrFn = mlir::dyn_cast_or_null<relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            tupleStream = aggrFn.getRel();
            auto sourceColumnAttr = aggrFn.getAttr();
            if (aggrFn.getFn() == relalg::AggrFunc::sum) {
               distAggrFunc = std::make_shared<SumAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::min) {
               distAggrFunc = std::make_shared<MinAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::max) {
               distAggrFunc = std::make_shared<MaxAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::any) {
               distAggrFunc = std::make_shared<AnyAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == relalg::AggrFunc::count) {
               distAggrFunc = std::make_shared<CountAggrFunc>(destColumnAttr, sourceColumnAttr);
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            tupleStream = countOp.getRel();
            distAggrFunc = std::make_shared<CountStarAggrFunc>(destColumnAttr);
         }

         if (!distinct.count(tupleStream.getDefiningOp())) {
            if (auto projectionOp = mlir::dyn_cast_or_null<relalg::ProjectionOp>(tupleStream.getDefiningOp())) {
               distinct[tupleStream.getDefiningOp()] = {relalg::OrderedAttributes::fromRefArr(projectionOp.getCols()), {}};
            }
         }
         distinct.at(tupleStream.getDefiningOp()).second.push_back(distAggrFunc);
      };
      for (auto d : distinct) {
         analyzedAggregation.distAggrFuncs.push_back({d.second.first, d.second.second});
      }
   }

   LogicalResult matchAndRewrite(relalg::GroupJoinOp groupJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      AnalyzedAggregation analyzedAggregation;
      analyze(groupJoinOp, analyzedAggregation);
      if (analyzedAggregation.distAggrFuncs.size() != 1) return failure();
      if (!analyzedAggregation.distAggrFuncs[0].first.getAttrs().empty()) return failure();
      std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs = analyzedAggregation.distAggrFuncs[0].second;

      auto* context = rewriter.getContext();
      auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto loc = groupJoinOp->getLoc();
      auto storedColumns = groupJoinOp.getUsedColumns().intersect(groupJoinOp.getChildren()[0].getAvailableColumns());
      for (auto z : llvm::zip(groupJoinOp.getLeftCols(), groupJoinOp.getRightCols())) {
         auto leftType = mlir::cast<tuples::ColumnRefAttr>(std::get<0>(z)).getColumn().type;
         auto rightType = mlir::cast<tuples::ColumnRefAttr>(std::get<1>(z)).getColumn().type;
         if (leftType == rightType) {
            storedColumns.remove(relalg::ColumnSet::from(&mlir::cast<tuples::ColumnRefAttr>(std::get<0>(z)).getColumn()));
         }
      }
      auto additionalColumns = relalg::OrderedAttributes::fromColumns(storedColumns);
      Block* initialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initialValueBlock);
         std::vector<mlir::Value> defaultValues;
         if (groupJoinOp.getBehavior() == relalg::GroupJoinBehavior::inner) {
            defaultValues.push_back(rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1));
         }
         for (auto* c : additionalColumns.getAttrs()) {
            defaultValues.push_back(rewriter.create<util::UndefOp>(loc, c->type));
         }
         for (auto aggrFn : distAggrFuncs) {
            defaultValues.push_back(aggrFn->createDefaultValue(rewriter, loc));
         }
         rewriter.create<tuples::ReturnOp>(loc, defaultValues);
      }
      std::vector<mlir::Attribute> names;
      std::vector<mlir::Attribute> reduceNames;
      std::vector<mlir::Attribute> types;
      std::vector<mlir::Type> valueTypes;
      std::vector<mlir::Location> valueLocations;
      std::vector<NamedAttribute> defMapping;
      std::vector<NamedAttribute> additionalColsDefMapping;
      std::vector<size_t> additionalColsMemberIdx;
      std::vector<Attribute> additionalColsRefs;
      std::vector<mlir::Type> additionalColsTypes;
      std::vector<mlir::Location> additionalColsLocations;
      tuples::ColumnRefAttr marker;
      std::string markerMember;
      if (groupJoinOp.getBehavior() == relalg::GroupJoinBehavior::inner) {
         markerMember = getUniqueMember(getContext(), "gjvalmarker");
         names.push_back(rewriter.getStringAttr(markerMember));
         types.push_back(mlir::TypeAttr::get(rewriter.getI1Type()));
         valueTypes.push_back(rewriter.getI1Type());
         valueLocations.push_back(loc);
         auto [def, ref] = createColumn(rewriter.getI1Type(), "groupjoin", "marker");
         marker = ref;
         defMapping.push_back(rewriter.getNamedAttr(markerMember, def));
      }
      for (auto* c : additionalColumns.getAttrs()) {
         auto memberName = getUniqueMember(getContext(), "gjval");
         names.push_back(rewriter.getStringAttr(memberName));
         types.push_back(mlir::TypeAttr::get(c->type));
         auto def = colManager.createDef(c);
         defMapping.push_back(rewriter.getNamedAttr(memberName, def));
         additionalColsDefMapping.push_back(rewriter.getNamedAttr(memberName, def));
         additionalColsMemberIdx.push_back(names.size() - 1);
         additionalColsRefs.push_back(colManager.createRef(c));
         valueTypes.push_back(c->type);
         valueLocations.push_back(loc);
         additionalColsTypes.push_back(c->type);
         additionalColsLocations.push_back(loc);
      }

      for (auto aggrFn : distAggrFuncs) {
         auto memberName = getUniqueMember(getContext(), "aggrval");
         names.push_back(rewriter.getStringAttr(memberName));
         reduceNames.push_back(rewriter.getStringAttr(memberName));
         types.push_back(mlir::TypeAttr::get(aggrFn->getStateType()));
         auto def = aggrFn->getDestAttribute();
         defMapping.push_back(rewriter.getNamedAttr(memberName, def));
         valueTypes.push_back(aggrFn->getStateType());
         valueLocations.push_back(loc);
      }
      auto stateMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, names), mlir::ArrayAttr::get(context, types));
      auto leftKeys = relalg::OrderedAttributes::fromRefArr(groupJoinOp.getLeftCols());
      auto rightKeys = relalg::OrderedAttributes::fromRefArr(groupJoinOp.getRightCols());

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<mlir::Type> otherKeyTypes;
      std::vector<mlir::Location> locations;
      for (auto* x : leftKeys.getAttrs()) {
         auto memberName = getUniqueMember(getContext(), "gjkeyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(x->type));
         keyTypes.push_back((x->type));
         locations.push_back(loc);
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(x)));
      }
      for (auto* x : rightKeys.getAttrs()) {
         otherKeyTypes.push_back((x->type));
      }
      auto keyMembers = subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateType = subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers, false);

      auto createOp = rewriter.create<subop::GenericCreateOp>(loc, stateType);
      auto state = createOp.getRes();
      auto [referenceDefAttr, referenceRefAttr] = createColumn(subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto lookupOp = rewriter.create<subop::LookupOrInsertOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getLeft(), state, groupJoinOp.getLeftCols(), referenceDefAttr);
      lookupOp.getInitFn().push_back(initialValueBlock);
      {
         mlir::Block* equalBlock = new Block;
         lookupOp.getEqFn().push_back(equalBlock);
         equalBlock->addArguments(keyTypes, locations);
         equalBlock->addArguments(keyTypes, locations);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(equalBlock);
            mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
            rewriter.create<tuples::ReturnOp>(loc, compared);
         }
      }
      auto reduceOp = rewriter.create<subop::ReduceOp>(loc, lookupOp, referenceRefAttr, rewriter.getArrayAttr(additionalColsRefs), rewriter.getArrayAttr(names));

      {
         mlir::Block* reduceBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         reduceBlock->addArguments(additionalColsTypes, additionalColsLocations);
         reduceBlock->addArguments(valueTypes, valueLocations);
         rewriter.setInsertionPointToStart(reduceBlock);
         std::vector<mlir::Value> storeVals;
         for (size_t i = 0; i < valueTypes.size(); i++) {
            auto colVal = reduceBlock->getArgument(i + additionalColsRefs.size());
            storeVals.push_back(colVal);
         }
         for (size_t i = 0; i < additionalColsMemberIdx.size(); i++) {
            auto colVal = reduceBlock->getArgument(i);
            storeVals[additionalColsMemberIdx[i]] = colVal;
         }
         rewriter.create<tuples::ReturnOp>(loc, storeVals);
         reduceOp.getRegion().push_back(reduceBlock);
      }
      {
         mlir::Block* combineBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         combineBlock->addArguments(valueTypes, valueLocations);
         combineBlock->addArguments(valueTypes, valueLocations);
         rewriter.setInsertionPointToStart(combineBlock);
         std::vector<mlir::Value> leftValsVec;
         for (size_t i = 0; i < valueTypes.size(); i++) {
            auto colVal = combineBlock->getArgument(i);
            leftValsVec.push_back(colVal);
         }
         rewriter.create<tuples::ReturnOp>(loc, leftValsVec);
         reduceOp.getCombine().push_back(combineBlock);
      }

      auto [aggrDef, aggrRef] = createColumn(subop::OptionalType::get(getContext(), subop::LookupEntryRefType::get(context, stateType)), "lookup", "ref");
      auto [unwrappedAggrDef, unwrappedAggrRef] = createColumn(subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto aggrLookupOp = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRight(), state, groupJoinOp.getRightCols(), aggrDef);
      {
         mlir::Block* equalBlock = new Block;
         aggrLookupOp.getEqFn().push_back(equalBlock);
         equalBlock->addArguments(keyTypes, locations);
         equalBlock->addArguments(otherKeyTypes, locations);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(equalBlock);
            mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
            rewriter.create<tuples::ReturnOp>(loc, compared);
         }
      }
      mlir::Value unwrap = rewriter.create<subop::UnwrapOptionalRefOp>(loc, aggrLookupOp.getRes(), aggrRef, unwrappedAggrDef);
      std::vector<mlir::Attribute> renameLeftDefs;
      std::vector<mlir::Attribute> renameRightDefs;
      for (auto z : llvm::zip(groupJoinOp.getLeftCols(), groupJoinOp.getRightCols())) {
         auto rightAttr = mlir::cast<tuples::ColumnRefAttr>(std::get<1>(z));
         auto leftAttr = mlir::cast<tuples::ColumnRefAttr>(std::get<0>(z));
         if (!storedColumns.contains(&leftAttr.getColumn())) {
            renameLeftDefs.push_back(tuples::ColumnDefAttr::get(getContext(), leftAttr.getName(), leftAttr.getColumnPtr(), rewriter.getArrayAttr(rightAttr)));
         }
         renameRightDefs.push_back(tuples::ColumnDefAttr::get(getContext(), rightAttr.getName(), rightAttr.getColumnPtr(), rewriter.getArrayAttr(leftAttr)));
      }
      if (!additionalColsDefMapping.empty()) {
         unwrap = rewriter.create<subop::GatherOp>(loc, unwrap, unwrappedAggrRef, rewriter.getDictionaryAttr(additionalColsDefMapping));
      }
      unwrap = rewriter.create<subop::RenamingOp>(loc, unwrap, rewriter.getArrayAttr(renameLeftDefs));
      auto filtered = translateSelection(unwrap, groupJoinOp.getPredicate(), rewriter, loc);
      if (groupJoinOp.getBehavior() == relalg::GroupJoinBehavior::inner) {
         auto [withTrueCol, trueCol] = mapBool(filtered, rewriter, loc, true);
         rewriter.create<subop::ScatterOp>(loc, withTrueCol, unwrappedAggrRef, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerMember, colManager.createRef(trueCol))));
      }
      if (!groupJoinOp.getMappedCols().empty()) {
         subop::MapCreationHelper helper(rewriter.getContext());
         std::vector<mlir::Operation*> toMove;
         for (auto& op : groupJoinOp.getMapFunc().front()) {
            toMove.push_back(&op);
         }
         for (auto* op : toMove) {
            op->remove();
            helper.getMapBlock()->push_back(op);
         }
         std::vector<mlir::Operation*> toErase;
         helper.getMapBlock()->walk([&](tuples::GetColumnOp getColumnOp) {
            getColumnOp.replaceAllUsesWith(helper.access(getColumnOp.getAttr(), getColumnOp->getLoc()));
            toErase.push_back(getColumnOp);
         });
         for (auto* op : toErase) {
            rewriter.eraseOp(op);
         }
         auto mapOp2 = rewriter.create<subop::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), filtered, groupJoinOp.getMappedCols(), helper.getColRefs());
         mapOp2.getFn().push_back(helper.getMapBlock());
         filtered = mapOp2;
      }
      performAggrFuncReduce(loc, rewriter, distAggrFuncs, unwrappedAggrRef, filtered, reduceNames, defMapping);

      mlir::Value scan = rewriter.create<subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));
      if (groupJoinOp.getBehavior() == relalg::GroupJoinBehavior::inner) {
         scan = rewriter.create<subop::FilterOp>(loc, scan, subop::FilterSemantic::all_true, rewriter.getArrayAttr(marker));
      }
      rewriter.replaceOpWithNewOp<subop::RenamingOp>(groupJoinOp, scan, rewriter.getArrayAttr(renameRightDefs));
      return success();
   }
};

class NestedLowering : public OpConversionPattern<relalg::NestedOp> {
   public:
   using OpConversionPattern<relalg::NestedOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::NestedOp nestedOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto* b = &nestedOp.getNestedFn().front();
      auto* terminator = b->getTerminator();
      auto returnOp = mlir::cast<tuples::ReturnOp>(terminator);
      rewriter.inlineBlockBefore(b, &*rewriter.getInsertionPoint(), adaptor.getInputs());
      {
         auto* b2 = new mlir::Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(b2);
         rewriter.create<tuples::ReturnOp>(rewriter.getUnknownLoc());
      }
      std::vector<mlir::Value> res;
      for (auto val : returnOp.getResults()) {
         res.push_back(val);
      }
      rewriter.replaceOp(nestedOp, res);
      rewriter.eraseOp(terminator);
      return success();
   }
};

class TrackTuplesLowering : public OpConversionPattern<relalg::TrackTuplesOP> {
   public:
   using OpConversionPattern<relalg::TrackTuplesOP>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::TrackTuplesOP trackTuplesOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = trackTuplesOp->getLoc();

      // Create counter as single i64 state initialized to 0
      auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto [counterState, counterName] = createCounterState(rewriter, loc);
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      referenceDefAttr.getColumn().type = subop::LookupEntryRefType::get(rewriter.getContext(), mlir::cast<subop::LookupAbleState>(counterState.getType()));
      auto lookup = rewriter.create<subop::LookupOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRel(), counterState, rewriter.getArrayAttr({}), referenceDefAttr);

      // Create reduce operation that increases counter for each seen tuple
      auto reduceOp = rewriter.create<subop::ReduceOp>(loc, lookup, colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getArrayAttr({}), rewriter.getArrayAttr({rewriter.getStringAttr(counterName)}));
      mlir::Block* reduceBlock = new Block;
      auto counter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(reduceBlock);
         auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
         mlir::Value updatedCounter = rewriter.create<mlir::arith::AddIOp>(loc, counter, one);
         rewriter.create<tuples::ReturnOp>(loc, updatedCounter);
      }
      reduceOp.getRegion().push_back(reduceBlock);
      mlir::Block* combineBlock = new Block;
      auto counter1 = combineBlock->addArgument(rewriter.getI64Type(), loc);
      auto counter2 = combineBlock->addArgument(rewriter.getI64Type(), loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(combineBlock);
         mlir::Value sum = rewriter.create<mlir::arith::AddIOp>(loc, counter1, counter2);
         rewriter.create<tuples::ReturnOp>(loc, sum);
      }
      reduceOp.getCombine().push_back(combineBlock);

      // Saves counter state to execution context
      rewriter.create<subop::SetTrackedCountOp>(loc, counterState, adaptor.getResultId(), counterName);

      rewriter.eraseOp(trackTuplesOp);
      return mlir::success();
   }
};
class QueryOpLowering : public OpConversionPattern<relalg::QueryOp> {
   public:
   using OpConversionPattern<relalg::QueryOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::QueryOp queryOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto executionGroup = rewriter.create<subop::ExecutionGroupOp>(queryOp->getLoc(), queryOp->getResultTypes(), adaptor.getInputs());
      executionGroup.getSubOps().getBlocks().clear();

      rewriter.inlineRegionBefore(queryOp.getQueryOps(), executionGroup.getSubOps(), executionGroup.getSubOps().end());
      rewriter.replaceOp(queryOp, executionGroup);
      return mlir::success();
   }
};
class QueryReturnOpLowering : public OpConversionPattern<relalg::QueryReturnOp> {
   public:
   using OpConversionPattern<relalg::QueryReturnOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(relalg::QueryReturnOp queryReturnOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<subop::ExecutionGroupReturnOp>(queryReturnOp, adaptor.getInputs());

      return mlir::success();
   }
};

void RelalgToSubOpLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<util::UtilDialect>()->getFunctionHelper().setParentModule(module);

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalDialect<gpu::GPUDialect>();
   target.addLegalDialect<async::AsyncDialect>();
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();
   target.addIllegalDialect<relalg::RelAlgDialect>();
   target.addLegalDialect<subop::SubOperatorDialect>();
   target.addLegalDialect<db::DBDialect>();
   target.addLegalDialect<lingodb::compiler::dialect::arrow::ArrowDialect>();

   target.addLegalDialect<tuples::TupleStreamDialect>();
   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   target.addLegalDialect<arith::ArithDialect>();
   target.addLegalDialect<cf::ControlFlowDialect>();
   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();

   TypeConverter typeConverter;
   typeConverter.addConversion([](tuples::TupleStreamType t) { return t; });
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
   patterns.insert<ConstRelationLowering>(typeConverter, ctxt);
   patterns.insert<MarkJoinLowering>(typeConverter, ctxt);
   patterns.insert<CrossProductLowering>(typeConverter, ctxt);
   patterns.insert<InnerJoinNLLowering>(typeConverter, ctxt);
   patterns.insert<AggregationLowering>(typeConverter, ctxt);
   patterns.insert<WindowLowering>(typeConverter, ctxt);
   patterns.insert<SemiJoinLowering>(typeConverter, ctxt);
   patterns.insert<AntiSemiJoinLowering>(typeConverter, ctxt);
   patterns.insert<OuterJoinLowering>(typeConverter, ctxt);
   patterns.insert<FullOuterJoinLowering>(typeConverter, ctxt);
   patterns.insert<SingleJoinLowering>(typeConverter, ctxt);
   patterns.insert<LimitLowering>(typeConverter, ctxt);
   patterns.insert<TopKLowering>(typeConverter, ctxt);
   patterns.insert<UnionAllLowering>(typeConverter, ctxt);
   patterns.insert<UnionDistinctLowering>(typeConverter, ctxt);
   patterns.insert<CountingSetOperationLowering>(ctxt);
   patterns.insert<GroupJoinLowering>(ctxt);
   patterns.insert<NestedLowering>(ctxt);
   patterns.insert<TrackTuplesLowering>(ctxt);
   patterns.insert<QueryOpLowering>(ctxt);
   patterns.insert<QueryReturnOpLowering>(ctxt);

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}
} //namespace
std::unique_ptr<mlir::Pass>
relalg::createLowerToSubOpPass() {
   return std::make_unique<RelalgToSubOpLoweringPass>();
}
void relalg::createLowerRelAlgToSubOpPipeline(mlir::OpPassManager& pm) {
   pm.addPass(relalg::createLowerToSubOpPass());
}
void relalg::registerRelAlgToSubOpConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createLowerToSubOpPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-relalg-to-subop",
      "",
      relalg::createLowerRelAlgToSubOpPipeline);
}
