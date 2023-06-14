#include "mlir-support/parsing.h"
#include "mlir/Conversion/RelAlgToSubOp/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelalgToSubOpLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "to-subop"; }

   RelalgToSubOpLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, mlir::db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect, mlir::relalg::RelAlgDialect, mlir::subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};
static std::string getUniqueMember(MLIRContext* context, std::string name) {
   auto& memberManager = context->getLoadedDialect<mlir::subop::SubOperatorDialect>()->getMemberManager();
   return memberManager.getUniqueMember(name);
}
static mlir::relalg::ColumnSet getRequired(Operator op) {
   auto available = op.getAvailableColumns();

   mlir::relalg::ColumnSet required;
   for (auto* user : op->getUsers()) {
      if (auto consumingOp = mlir::dyn_cast_or_null<Operator>(user)) {
         required.insert(getRequired(consumingOp));
         required.insert(consumingOp.getUsedColumns());
      }
      if (auto materializeOp = mlir::dyn_cast_or_null<mlir::relalg::MaterializeOp>(user)) {
         required.insert(mlir::relalg::ColumnSet::fromArrayAttr(materializeOp.getCols()));
      }
   }
   return available.intersect(required);
}
class BaseTableLowering : public OpConversionPattern<mlir::relalg::BaseTableOp> {
   public:
   using OpConversionPattern<mlir::relalg::BaseTableOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::relalg::BaseTableOp baseTableOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto required = getRequired(baseTableOp);
      std::vector<mlir::Type> types;
      std::vector<Attribute> colNames;
      std::vector<Attribute> colTypes;
      std::vector<NamedAttribute> mapping;
      std::string tableName = baseTableOp->getAttr("table_identifier").cast<mlir::StringAttr>().str();
      std::string scanDescription = R"({ "table": ")" + tableName + R"(", "mapping": { )";
      bool first = true;
      for (auto namedAttr : baseTableOp.getColumns().getValue()) {
         auto identifier = namedAttr.getName();
         auto attr = namedAttr.getValue();
         auto attrDef = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
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
      auto tableRefType = mlir::subop::TableType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(colNames), rewriter.getArrayAttr(colTypes)));
      mlir::Value tableRef = rewriter.create<mlir::subop::GetExternalOp>(baseTableOp->getLoc(), tableRefType, rewriter.getStringAttr(scanDescription));
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
static mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::ValueRange leftUnpacked, mlir::ValueRange rightUnpacked, mlir::Location loc) {
   mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
   for (size_t i = 0; i < leftUnpacked.size(); i++) {
      mlir::Value compared = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::isa, leftUnpacked[i], rightUnpacked[i]);
      mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), mlir::ValueRange({equal, compared}));
      equal = localEqual;
   }
   return equal;
}
static std::pair<mlir::tuples::ColumnDefAttr, mlir::tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}
static mlir::Value map(mlir::Value stream, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::ArrayAttr createdColumns, std::function<std::vector<mlir::Value>(mlir::ConversionPatternRewriter&, mlir::Value, mlir::Location)> fn) {
   Block* mapBlock = new Block;
   auto tupleArg = mapBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      rewriter.create<mlir::tuples::ReturnOp>(loc, fn(rewriter, tupleArg, loc));
   }
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, createdColumns);
   mapOp.getFn().push_back(mapBlock);
   return mapOp.getResult();
}
static mlir::Value translateSelection(mlir::Value stream, mlir::Region& predicate, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto terminator = mlir::cast<mlir::tuples::ReturnOp>(predicate.front().getTerminator());
   if (terminator.getResults().empty()) {
      auto [markAttrDef, markAttrRef] = createColumn(rewriter.getI1Type(), "map", "predicate");
      auto mapped = map(stream, rewriter, loc, rewriter.getArrayAttr(markAttrDef), [](mlir::ConversionPatternRewriter& rewriter, mlir::Value, mlir::Location loc) {
         return std::vector<mlir::Value>{rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1))};
      });
      return rewriter.create<mlir::subop::FilterOp>(loc, mapped, mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(markAttrRef));

   } else {
      auto& predicateBlock = predicate.front();
      if (auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(predicateBlock.getTerminator())) {
         mlir::Value matched = returnOp.getResults()[0];
         std::vector<std::pair<int, mlir::Value>> conditions;
         if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(matched.getDefiningOp())) {
            for (auto c : andOp.getVals()) {
               int p = 1000;
               if (auto* defOp = c.getDefiningOp()) {
                  if (auto betweenOp = mlir::dyn_cast_or_null<mlir::db::BetweenOp>(defOp)) {
                     auto t = betweenOp.getVal().getType();
                     p = ::llvm::TypeSwitch<mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) { return 2; })
                            .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) { return 3; })
                            .Case<::mlir::db::CharType, ::mlir::db::TimestampType, ::mlir::db::IntervalType, ::mlir::FloatType>([&](mlir::Type t) { return 2; })
                            .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) { return 10; })
                            .Default([](::mlir::Type) { return 100; });
                     p -= 1;
                  } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(defOp)) {
                     auto t = cmpOp.getLeft().getType();
                     p = ::llvm::TypeSwitch<mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) { return 2; })
                            .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) { return 3; })
                            .Case<::mlir::db::CharType, ::mlir::db::TimestampType, ::mlir::db::IntervalType, ::mlir::FloatType>([&](mlir::Type t) { return 2; })
                            .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) { return 10; })
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
            stream = map(stream, rewriter, loc, rewriter.getArrayAttr(predDef), [&](mlir::ConversionPatternRewriter& b, mlir::Value tuple, mlir::Location loc) -> std::vector<mlir::Value> {
               mlir::IRMapping mapping;
               mapping.map(predicateBlock.getArgument(0), tuple);
               auto helperOp = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(0));
               mlir::relalg::detail::inlineOpIntoBlock(c.second.getDefiningOp(), c.second.getDefiningOp()->getParentOp(), b.getInsertionBlock(), mapping, helperOp);
               b.eraseOp(helperOp);
               mlir::Value predVal = mapping.lookupOrNull(c.second);
               if (predVal.getType().isa<mlir::db::NullableType>()) {
                  predVal = b.create<mlir::db::DeriveTruth>(loc, predVal);
               }
               return {predVal};
            });
            stream = rewriter.create<mlir::subop::FilterOp>(loc, stream, mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(predRef));
         }
         return stream;
      } else {
         assert(false && "invalid");
         return Value();
      }
   }
}
class SelectionLowering : public OpConversionPattern<mlir::relalg::SelectionOp> {
   public:
   using OpConversionPattern<mlir::relalg::SelectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::SelectionOp selectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
class MapLowering : public OpConversionPattern<mlir::relalg::MapOp> {
   public:
   using OpConversionPattern<mlir::relalg::MapOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::MapOp mapOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto mapOp2 = rewriter.replaceOpWithNewOp<mlir::subop::MapOp>(mapOp, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRel(), mapOp.getComputedCols());
      assert(safelyMoveRegion(rewriter, mapOp.getPredicate(), mapOp2.getFn()).succeeded());

      return success();
   }
};
class RenamingLowering : public OpConversionPattern<mlir::relalg::RenamingOp> {
   public:
   using OpConversionPattern<mlir::relalg::RenamingOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::RenamingOp renamingOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::subop::RenamingOp>(renamingOp, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRel(), renamingOp.getColumns());
      return success();
   }
};
class ProjectionAllLowering : public OpConversionPattern<mlir::relalg::ProjectionOp> {
   public:
   using OpConversionPattern<mlir::relalg::ProjectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::ProjectionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.getSetSemantic() == mlir::relalg::SetSemantic::distinct) return failure();
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
      rewriter.create<mlir::tuples::ReturnOp>(loc, compared);
   }
   return equalBlock;
}

class ProjectionDistinctLowering : public OpConversionPattern<mlir::relalg::ProjectionOp> {
   public:
   using OpConversionPattern<mlir::relalg::ProjectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::ProjectionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.getSetSemantic() != mlir::relalg::SetSemantic::distinct) return failure();
      auto* context = getContext();
      auto loc = projectionOp->getLoc();

      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      for (auto x : projectionOp.getCols()) {
         auto ref = x.cast<mlir::tuples::ColumnRefAttr>();
         auto memberName = getUniqueMember(getContext(), "keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {}), mlir::ArrayAttr::get(context, {}));

      auto stateType = mlir::subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers);
      mlir::Value state = rewriter.create<mlir::subop::GenericCreateOp>(loc, stateType);
      auto [referenceDef, referenceRef] = createColumn(mlir::subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto lookupOp = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), adaptor.getRel(), state, projectionOp.getCols(), referenceDef);
      auto* initialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initialValueBlock);
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
      lookupOp.getInitFn().push_back(initialValueBlock);
      lookupOp.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

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
         std::string name = getUniqueMember(context, "member");
         auto nameAttr = mlir::StringAttr::get(context, name);
         names.push_back(nameAttr);
         defMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(x)));
         refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(x)));
      }
   }
   MaterializationHelper(mlir::ArrayAttr columnAttrs, mlir::MLIRContext* context) : context(context) {
      size_t i = 0;
      for (auto columnAttr : columnAttrs) {
         std::string name = getUniqueMember(context, "member");
         auto nameAttr = mlir::StringAttr::get(context, name);
         names.push_back(nameAttr);
         if (auto columnDef = columnAttr.dyn_cast<mlir::tuples::ColumnDefAttr>()) {
            auto* x = &columnDef.getColumn();
            types.push_back(mlir::TypeAttr::get(x->type));
            colToMemberPos[x] = i++;
            defMapping.push_back(mlir::NamedAttribute(nameAttr, columnDef));
            refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(x)));
         } else if (auto columnRef = columnAttr.dyn_cast<mlir::tuples::ColumnRefAttr>()) {
            auto* x = &columnRef.getColumn();
            types.push_back(mlir::TypeAttr::get(x->type));
            colToMemberPos[x] = i++;
            refMapping.push_back(mlir::NamedAttribute(nameAttr, columnRef));
            defMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(x)));
         }
      }
   }
   mlir::Type getType(size_t i) {
      return types.at(i).cast<mlir::TypeAttr>().getValue();
   }

   std::string addFlag(mlir::tuples::ColumnDefAttr flagAttrDef) {
      auto i1Type = mlir::IntegerType::get(context, 1);
      types.push_back(mlir::TypeAttr::get(i1Type));
      colToMemberPos[&flagAttrDef.getColumn()] = names.size();
      std::string name = getUniqueMember(context, "flag");
      auto nameAttr = mlir::StringAttr::get(context, name);
      names.push_back(nameAttr);
      defMapping.push_back(mlir::NamedAttribute(nameAttr, flagAttrDef));
      refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(&flagAttrDef.getColumn())));
      return name;
   }
   mlir::subop::StateMembersAttr createStateMembersAttr(std::vector<mlir::Attribute> localNames = {}, std::vector<mlir::Attribute> localTypes = {}) {
      localNames.insert(localNames.end(), names.begin(), names.end());
      localTypes.insert(localTypes.end(), types.begin(), types.end());
      return mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, localNames), mlir::ArrayAttr::get(context, localTypes));
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
   mlir::StringAttr lookupStateMemberForMaterializedColumn(const mlir::tuples::Column* column) {
      return names.at(colToMemberPos.at(column)).cast<mlir::StringAttr>();
   }
};
class ConstRelationLowering : public OpConversionPattern<mlir::relalg::ConstRelationOp> {
   public:
   using OpConversionPattern<mlir::relalg::ConstRelationOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::relalg::ConstRelationOp constRelationOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = constRelationOp->getLoc();

      auto generateOp = rewriter.replaceOpWithNewOp<mlir::subop::GenerateOp>(constRelationOp, mlir::tuples::TupleStreamType::get(rewriter.getContext()), constRelationOp.getColumns());
      {
         auto* generateBlock = new Block;
         mlir::OpBuilder::InsertionGuard guard2(rewriter);
         rewriter.setInsertionPointToStart(generateBlock);
         generateOp.getRegion().push_back(generateBlock);
         for (auto rowAttr : constRelationOp.getValues()) {
            auto row = rowAttr.cast<ArrayAttr>();
            std::vector<Value> values;
            size_t i = 0;
            for (auto entryAttr : row.getValue()) {
               auto type = constRelationOp.getColumns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn().type;
               if (type.isa<mlir::db::NullableType>() && entryAttr.isa<mlir::UnitAttr>()) {
                  auto entryVal = rewriter.create<mlir::db::NullOp>(constRelationOp->getLoc(), type);
                  values.push_back(entryVal);
                  i++;
               } else {
                  mlir::Value entryVal = rewriter.create<mlir::db::ConstantOp>(constRelationOp->getLoc(), getBaseType(type), entryAttr);
                  if (type.isa<mlir::db::NullableType>()) {
                     entryVal = rewriter.create<mlir::db::AsNullableOp>(constRelationOp->getLoc(), type, entryVal);
                  }
                  values.push_back(entryVal);
                  i++;
               }
            }
            rewriter.create<mlir::subop::GenerateEmitOp>(constRelationOp->getLoc(), values);
         }
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
      return success();
   }
};

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
   mapOp.getFn().push_back(mapBlock);
   return mapOp.getResult();
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
   mapOp.getFn().push_back(mapBlock);
   return {mapOp.getResult(), &markAttrDef.getColumn()};
}
static std::pair<mlir::Value, const mlir::tuples::Column*> mapIndex(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, size_t value) {
   Block* mapBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      mlir::Value val = rewriter.create<mlir::arith::ConstantIndexOp>(loc, value);
      rewriter.create<mlir::tuples::ReturnOp>(loc, val);
   }

   auto& columnManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope("map");
   std::string attributeName = "ival";
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = rewriter.getI1Type();
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(markAttrDef));
   mapOp.getFn().push_back(mapBlock);
   return {mapOp.getResult(), &markAttrDef.getColumn()};
}
static mlir::Value mapColsToNull(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, mlir::relalg::ColumnSet excluded = {}) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> defAttrs;
   Block* mapBlock = new Block;
   mapBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);
      std::vector<mlir::Value> res;
      for (mlir::Attribute attr : mapping) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0].cast<mlir::tuples::ColumnRefAttr>();
         if (excluded.contains(&fromExisting.getColumn())) continue;
         mlir::Value nullValue = rewriter.create<mlir::db::NullOp>(loc, defAttr->type);
         res.push_back(nullValue);
         defAttrs.push_back(colManager.createDef(defAttr));
      }
      rewriter.create<mlir::tuples::ReturnOp>(loc, res);
   }
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs));
   mapOp.getFn().push_back(mapBlock);
   return mapOp.getResult();
}
static mlir::Value mapColsToNullable(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, size_t exisingOffset = 0, mlir::relalg::ColumnSet excluded = {}) {
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
         auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[exisingOffset].cast<mlir::tuples::ColumnRefAttr>();
         if (excluded.contains(&fromExisting.getColumn())) continue;
         mlir::Value value = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), fromExisting, tupleArg);
         if (fromExisting.getColumn().type != defAttr->type) {
            mlir::Value tmp = rewriter.create<mlir::db::AsNullableOp>(loc, defAttr->type, value);
            value = tmp;
         }
         res.push_back(value);
         defAttrs.push_back(colManager.createDef(defAttr));
      }
      rewriter.create<mlir::tuples::ReturnOp>(loc, res);
   }
   auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs));
   mapOp.getFn().push_back(mapBlock);
   return mapOp.getResult();
}
class UnionAllLowering : public OpConversionPattern<mlir::relalg::UnionOp> {
   public:
   using OpConversionPattern<mlir::relalg::UnionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::UnionOp unionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (unionOp.getSetSemantic() != mlir::relalg::SetSemantic::all) return failure();
      auto loc = unionOp->getLoc();
      mlir::Value left = mapColsToNullable(adaptor.getLeft(), rewriter, loc, unionOp.getMapping(), 0);
      mlir::Value right = mapColsToNullable(adaptor.getRight(), rewriter, loc, unionOp.getMapping(), 1);
      rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(unionOp, mlir::ValueRange({left, right}));
      return success();
   }
};

class UnionDistinctLowering : public OpConversionPattern<mlir::relalg::UnionOp> {
   public:
   using OpConversionPattern<mlir::relalg::UnionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::UnionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.getSetSemantic() != mlir::relalg::SetSemantic::distinct) return failure();
      auto* context = getContext();
      auto loc = projectionOp->getLoc();

      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      std::vector<mlir::Attribute> refs;
      for (auto x : projectionOp.getMapping()) {
         auto ref = x.cast<mlir::tuples::ColumnDefAttr>();
         refs.push_back(colManager.createRef(&ref.getColumn()));
         auto memberName = getUniqueMember(getContext(), "keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {}), mlir::ArrayAttr::get(context, {}));

      auto stateType = mlir::subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers);
      mlir::Value state = rewriter.create<mlir::subop::GenericCreateOp>(loc, stateType);
      auto [referenceDef, referenceRef] = createColumn(mlir::subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      mlir::Value left = mapColsToNullable(adaptor.getLeft(), rewriter, loc, projectionOp.getMapping(), 0);
      mlir::Value right = mapColsToNullable(adaptor.getRight(), rewriter, loc, projectionOp.getMapping(), 1);
      auto lookupOpLeft = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), left, state, rewriter.getArrayAttr(refs), referenceDef);
      auto lookupOpRight = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), right, state, rewriter.getArrayAttr(refs), referenceDef);
      auto* leftInitialValueBlock = new Block;
      auto* rightInitialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(leftInitialValueBlock);
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rightInitialValueBlock);
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
      lookupOpLeft.getInitFn().push_back(leftInitialValueBlock);
      lookupOpRight.getInitFn().push_back(rightInitialValueBlock);
      lookupOpLeft.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));
      lookupOpRight.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));

      rewriter.replaceOp(projectionOp, scan);
      return success();
   }
};
class CountingSetOperationLowering : public ConversionPattern {
   public:
   CountingSetOperationLowering(mlir::MLIRContext* context)
      : ConversionPattern(MatchAnyOpTypeTag(), 1, context) {}
   LogicalResult match(mlir::Operation* op) const override {
      return mlir::success(mlir::isa<mlir::relalg::ExceptOp, mlir::relalg::IntersectOp>(op));
   }
   void rewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      bool distinct = op->getAttrOfType<mlir::relalg::SetSemanticAttr>("set_semantic").getValue() == mlir::relalg::SetSemantic::distinct;
      bool except = mlir::isa<mlir::relalg::ExceptOp>(op);
      auto* context = getContext();
      auto loc = op->getLoc();

      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      std::vector<mlir::Attribute> refs;
      auto mapping = op->getAttrOfType<mlir::ArrayAttr>("mapping");
      for (auto x : mapping) {
         auto ref = x.cast<mlir::tuples::ColumnDefAttr>();
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
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, counterNames), mlir::ArrayAttr::get(context, counterTypes));

      auto stateType = mlir::subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers);
      mlir::Value state = rewriter.create<mlir::subop::GenericCreateOp>(loc, stateType);
      auto [referenceDefLeft, referenceRefLeft] = createColumn(mlir::subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto [referenceDefRight, referenceRefRight] = createColumn(mlir::subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      mlir::Value left = mapColsToNullable(operands[0], rewriter, loc, mapping, 0);
      mlir::Value right = mapColsToNullable(operands[1], rewriter, loc, mapping, 1);
      auto lookupOpLeft = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), left, state, rewriter.getArrayAttr(refs), referenceDefLeft);
      auto* leftInitialValueBlock = new Block;
      auto* rightInitialValueBlock = new Block;

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(leftInitialValueBlock);
         mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
         rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({zeroI64, zeroI64}));
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rightInitialValueBlock);
         mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
         rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({zeroI64, zeroI64}));
      }
      lookupOpLeft.getInitFn().push_back(leftInitialValueBlock);
      lookupOpLeft.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));
      {
         auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(loc, lookupOpLeft, referenceRefLeft, rewriter.getArrayAttr({}), rewriter.getArrayAttr(counterNames));

         {
            mlir::Block* reduceBlock = new Block;
            mlir::Value currCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value otherCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceBlock);
            mlir::Value constOne = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
            currCounter = rewriter.create<mlir::arith::AddIOp>(loc, currCounter, constOne);
            rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({currCounter, otherCounter}));
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
            rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({counter1, counter2}));
            reduceOp.getCombine().push_back(combineBlock);
         }
      }
      auto lookupOpRight = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), right, state, rewriter.getArrayAttr(refs), referenceDefRight);
      lookupOpRight.getInitFn().push_back(rightInitialValueBlock);
      lookupOpRight.getEqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

      {
         auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(loc, lookupOpRight, referenceRefRight, rewriter.getArrayAttr({}), rewriter.getArrayAttr(counterNames));
         {
            mlir::Block* reduceBlock = new Block;
            mlir::Value otherCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::Value currCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceBlock);
            mlir::Value constOne = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
            currCounter = rewriter.create<mlir::arith::AddIOp>(loc, currCounter, constOne);
            rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({otherCounter, currCounter}));
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
            rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({counter1, counter2}));
            reduceOp.getCombine().push_back(combineBlock);
         }
      }
      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));
      if (distinct) {
         auto [predicateDef, predicateRef] = createColumn(rewriter.getI64Type(), "set", "predicate");

         scan = map(scan, rewriter, loc, rewriter.getArrayAttr(predicateDef), [&, counter1Ref = counter1Ref, counter2Ref = counter2Ref](mlir::OpBuilder& rewriter, mlir::Value tuple, mlir::Location loc) {
            mlir::Value leftVal = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), counter1Ref, tuple);
            mlir::Value rightVal = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), counter2Ref, tuple);
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
         scan = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(predicateRef));
      } else {
         auto [repeatDef, repeatRef] = createColumn(rewriter.getIndexType(), "set", "repeat");

         scan = map(scan, rewriter, loc, rewriter.getArrayAttr(repeatDef), [&, counter1Ref = counter1Ref, counter2Ref = counter2Ref](mlir::OpBuilder& rewriter, mlir::Value tuple, mlir::Location loc) {
            mlir::Value leftVal = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), counter1Ref, tuple);
            mlir::Value rightVal = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), counter2Ref, tuple);
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
         auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), scan, rewriter.getArrayAttr({repeatRef}));
         auto* b = new Block;
         b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
         mlir::Value repeatNumber = b->addArgument(rewriter.getIndexType(), loc);
         nestedMapOp.getRegion().push_back(b);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(b);
            auto generateOp = rewriter.create<mlir::subop::GenerateOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), rewriter.getArrayAttr({}));
            {
               auto* generateBlock = new Block;
               mlir::OpBuilder::InsertionGuard guard2(rewriter);
               rewriter.setInsertionPointToStart(generateBlock);
               generateOp.getRegion().push_back(generateBlock);
               mlir::Value zeroIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
               mlir::Value oneIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
               rewriter.create<mlir::scf::ForOp>(loc, zeroIdx, repeatNumber, oneIdx, mlir::ValueRange{}, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange vr) {
                  b.create<mlir::subop::GenerateEmitOp>(loc, mlir::ValueRange{});
                  b.create<mlir::scf::YieldOp>(loc);
               });
               rewriter.create<mlir::tuples::ReturnOp>(loc);
            }
            rewriter.create<mlir::tuples::ReturnOp>(loc, generateOp.getRes());
         }
         scan = nestedMapOp.getRes();
      }
      rewriter.replaceOp(op, scan);
   }
};
static std::pair<mlir::Value, std::string> createMarkerState(mlir::OpBuilder& rewriter, mlir::Location loc) {
   auto memberName = getUniqueMember(rewriter.getContext(), "marker");
   mlir::Type stateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(memberName)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getI1Type())})));
   Block* initialValueBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(initialValueBlock);
      mlir::Value val = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.create<mlir::tuples::ReturnOp>(loc, val);
   }
   auto createOp = rewriter.create<mlir::subop::CreateSimpleStateOp>(loc, stateType);
   createOp.getInitFn().push_back(initialValueBlock);

   return {createOp.getRes(), memberName};
}
static std::pair<mlir::Value, std::string> createCounterState(mlir::OpBuilder& rewriter, mlir::Location loc) {
   auto memberName = getUniqueMember(rewriter.getContext(), "counter");
   mlir::Type stateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(memberName)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getI64Type())})));
   Block* initialValueBlock = new Block;
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(initialValueBlock);
      mlir::Value val = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      rewriter.create<mlir::tuples::ReturnOp>(loc, val);
   }
   auto createOp = rewriter.create<mlir::subop::CreateSimpleStateOp>(loc, stateType);
   createOp.getInitFn().push_back(initialValueBlock);

   return {createOp.getRes(), memberName};
}

static mlir::Value translateNLJ(mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   MaterializationHelper helper(columns, rewriter.getContext());
   auto vectorType = mlir::subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector = rewriter.create<mlir::subop::GenericCreateOp>(loc, vectorType);
   rewriter.create<mlir::subop::MaterializeOp>(loc, right, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), left, rewriter.getArrayAttr({}));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping());
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, scan, tuple);
      rewriter.create<mlir::tuples::ReturnOp>(loc, fn(combined, rewriter));
   }
   return nestedMapOp.getRes();
}
mlir::Block* createEqFn(mlir::ConversionPatternRewriter& rewriter, mlir::ArrayAttr leftColumns, mlir::ArrayAttr rightColumns, mlir::ArrayAttr nullsEqual, mlir::Location loc) {
   mlir::OpBuilder::InsertionGuard guard(rewriter);
   auto* eqFnBlock = new mlir::Block;
   rewriter.setInsertionPointToStart(eqFnBlock);
   std::vector<mlir::Value> leftArgs;
   std::vector<mlir::Value> rightArgs;
   for (auto i = 0ull; i < leftColumns.size(); i++) {
      leftArgs.push_back(eqFnBlock->addArgument(leftColumns[i].cast<mlir::tuples::ColumnRefAttr>().getColumn().type, loc));
   }
   for (auto i = 0ull; i < rightColumns.size(); i++) {
      rightArgs.push_back(eqFnBlock->addArgument(rightColumns[i].cast<mlir::tuples::ColumnRefAttr>().getColumn().type, loc));
   }
   std::vector<mlir::Value> cmps;
   for (auto z : llvm::zip(leftArgs, rightArgs, nullsEqual)) {
      auto [l, r, nE] = z;
      bool useIsa = nE.cast<mlir::IntegerAttr>().getInt();
      mlir::Value compared = rewriter.create<mlir::db::CmpOp>(loc, useIsa ? mlir::db::DBCmpPredicate::isa : mlir::db::DBCmpPredicate::eq, l, r);
      cmps.push_back(compared);
   }
   mlir::Value anded = rewriter.create<mlir::db::AndOp>(loc, cmps);
   if (anded.getType().isa<mlir::db::NullableType>()) {
      anded = rewriter.create<mlir::db::DeriveTruth>(loc, anded);
   }
   rewriter.create<mlir::tuples::ReturnOp>(loc, anded);
   return eqFnBlock;
}
mlir::Block* createVerifyEqFnForTuple(mlir::ConversionPatternRewriter& rewriter, mlir::ArrayAttr leftColumns, mlir::ArrayAttr rightColumns, mlir::ArrayAttr nullsEqual, mlir::Location loc) {
   mlir::OpBuilder::InsertionGuard guard(rewriter);
   auto* eqFnBlock = new mlir::Block;
   eqFnBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   rewriter.setInsertionPointToStart(eqFnBlock);
   std::vector<mlir::Value> leftArgs;
   std::vector<mlir::Value> rightArgs;
   for (auto i = 0ull; i < leftColumns.size(); i++) {
      auto leftCol = leftColumns[i].cast<mlir::tuples::ColumnRefAttr>();
      leftArgs.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, leftCol.getColumn().type, leftCol, eqFnBlock->getArgument(0)));
   }
   for (auto i = 1ull; i < rightColumns.size(); i++) {
      auto rightCol = rightColumns[i].cast<mlir::tuples::ColumnRefAttr>();
      rightArgs.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, rightCol.getColumn().type, rightCol, eqFnBlock->getArgument(0)));
   }
   std::vector<mlir::Value> cmps;
   for (auto z : llvm::zip(leftArgs, rightArgs, nullsEqual)) {
      auto [l, r, nE] = z;
      bool useIsa = nE.cast<mlir::IntegerAttr>().getInt();
      mlir::Value compared = rewriter.create<mlir::db::CmpOp>(loc, useIsa ? mlir::db::DBCmpPredicate::isa : mlir::db::DBCmpPredicate::eq, l, r);
      cmps.push_back(compared);
   }
   mlir::Value anded = rewriter.create<mlir::db::AndOp>(loc, cmps);
   if (anded.getType().isa<mlir::db::NullableType>()) {
      anded = rewriter.create<mlir::db::DeriveTruth>(loc, anded);
   }
   rewriter.create<mlir::tuples::ReturnOp>(loc, anded);
   return eqFnBlock;
}

static mlir::Value translateHJ(mlir::Value left, mlir::Value right, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   auto keyColumns = mlir::relalg::ColumnSet::fromArrayAttr(hashRight);
   MaterializationHelper keyHelper(hashRight, rewriter.getContext());
   auto valueColumns = columns;
   valueColumns.remove(keyColumns);
   MaterializationHelper valueHelper(valueColumns, rewriter.getContext());
   auto multiMapType = mlir::subop::MultiMapType::get(rewriter.getContext(), keyHelper.createStateMembersAttr(), valueHelper.createStateMembersAttr());
   mlir::Value multiMap = rewriter.create<mlir::subop::GenericCreateOp>(loc, multiMapType);
   auto insertOp = rewriter.create<mlir::subop::InsertOp>(loc, right, multiMap, keyHelper.createColumnstateMapping(valueHelper.createColumnstateMapping().getValue()));
   insertOp.getEqFn().push_back(createEqFn(rewriter, hashRight, hashRight, nullsEqual, loc));

   auto entryRefType = mlir::subop::MultiMapEntryRefType::get(rewriter.getContext(), multiMapType);
   auto entryRefListType = mlir::subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), left, multiMap, hashLeft, listDef);
   afterLookup.getEqFn().push_back(createEqFn(rewriter, hashRight, hashLeft, nullsEqual, loc));
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);
   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      mlir::Value scan = rewriter.create<mlir::subop::ScanListOp>(loc, list, entryDef);
      mlir::Value gathered = rewriter.create<mlir::subop::GatherOp>(loc, scan, entryRef, keyHelper.createStateColumnMapping(valueHelper.createStateColumnMapping().getValue()));
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, gathered, tuple);
      rewriter.create<mlir::tuples::ReturnOp>(loc, fn(combined, rewriter));
   }
   return nestedMapOp.getRes();
}
static mlir::Value translateINLJ(mlir::Value left, mlir::Value right, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   auto rightScan = mlir::cast<mlir::subop::ScanOp>(right.getDefiningOp());
   auto* ctxt = rewriter.getContext();
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string tableName = mlir::cast<mlir::StringAttr>(hashRight[0]).str();
   mlir::ArrayAttr primaryKeyHashValue = rewriter.getArrayAttr({colManager.createRef(tableName, "primaryKeyHashValue")});
   auto keyColumns = mlir::relalg::ColumnSet::fromArrayAttr(primaryKeyHashValue);
   auto valueColumns = columns;
   valueColumns.remove(keyColumns);

   bool first = true;
   std::vector<Attribute> keyColNames, keyColTypes, valColNames, valColTypes;
   std::vector<NamedAttribute> mapping;

   // Create description for external index get operation
   std::string externalIndexDescription = R"({ "externalHashIndex": ")" + tableName + R"(", "mapping": { )";
   for (auto namedAttr : rightScan.getMapping()) {
      auto identifier = namedAttr.getName();
      auto attr = namedAttr.getValue();
      auto attrDef = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
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

   auto keyStateMembers = mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(keyColNames), rewriter.getArrayAttr(keyColTypes));
   auto valueStateMembers = mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr(valColNames), rewriter.getArrayAttr(valColTypes));

   auto externalHashIndexType = mlir::subop::ExternalHashIndexType::get(rewriter.getContext(), keyStateMembers, valueStateMembers);
   mlir::Value externalHashIndex = rewriter.create<mlir::subop::GetExternalOp>(loc, externalHashIndexType, externalIndexDescription);
   // Erase table scan
   rewriter.eraseOp(rightScan->getOperand(0).getDefiningOp());
   rewriter.eraseOp(rightScan);

   auto entryRefType = mlir::subop::ExternalHashIndexEntryRefType::get(rewriter.getContext(), externalHashIndexType);
   auto entryRefListType = mlir::subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), left, externalHashIndex, hashLeft, listDef);

   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);
   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      auto scan = rewriter.create<mlir::subop::ScanListOp>(loc, list, entryDef);
      auto gathered = rewriter.create<mlir::subop::GatherOp>(loc, scan, entryRef, rewriter.getDictionaryAttr(mapping));
      auto combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, gathered, tuple);

      // eliminate hash collisions
      auto [markerAttrDef, markerAttrRef] = createColumn(rewriter.getI1Type(), "map", "predicate");
      mlir::subop::MapOp keep = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), combined, rewriter.getArrayAttr(markerAttrDef));
      keep.getFn().push_back(createVerifyEqFnForTuple(rewriter, hashLeft, hashRight, nullsEqual, loc));
      mlir::Value filtered = rewriter.create<mlir::subop::FilterOp>(loc, keep, mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(markerAttrRef));

      rewriter.create<mlir::tuples::ReturnOp>(loc, fn(filtered, rewriter));
   }
   return nestedMapOp.getRes();
}
static mlir::Value translateNL(mlir::Value left, mlir::Value right, bool useHash, bool useIndexNestedLoop, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   if (useHash) {
      return translateHJ(left, right, nullsEqual, hashLeft, hashRight, columns, rewriter, loc, fn);
   } else if (useIndexNestedLoop) {
      return translateINLJ(left, right, nullsEqual, hashLeft, hashRight, columns, rewriter, loc, fn);
   } else {
      return translateNLJ(left, right, columns, rewriter, loc, fn);
   }
}

static std::pair<mlir::Value, mlir::Value> translateNLJWithMarker(mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr, std::string markerName)> fn) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   MaterializationHelper helper(columns, rewriter.getContext());
   auto flagMember = helper.addFlag(markerDefAttr);
   auto vectorType = mlir::subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector = rewriter.create<mlir::subop::GenericCreateOp>(loc, vectorType);
   left = mapBool(left, rewriter, loc, false, &markerDefAttr.getColumn());
   rewriter.create<mlir::subop::MaterializeOp>(loc, left, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), right, rewriter.getArrayAttr({}));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("scan"), "ref");
      referenceDefAttr.getColumn().type = mlir::subop::EntryRefType::get(rewriter.getContext(), vectorType);
      mlir::Value scan = rewriter.create<mlir::subop::ScanRefsOp>(loc, vector, referenceDefAttr);

      mlir::Value gathered = rewriter.create<mlir::subop::GatherOp>(loc, scan, colManager.createRef(&referenceDefAttr.getColumn()), helper.createStateColumnMapping({}, {flagMember}));
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, gathered, tuple);
      auto res = fn(combined, tuple, rewriter, colManager.createRef(&referenceDefAttr.getColumn()), flagMember);
      if (res) {
         rewriter.create<mlir::tuples::ReturnOp>(loc, res);
      } else {
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
   }
   return {nestedMapOp.getRes(), rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping())};
}

static std::pair<mlir::Value, mlir::Value> translateHJWithMarker(mlir::Value left, mlir::Value right, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr, std::string markerName)> fn) {
   auto keyColumns = mlir::relalg::ColumnSet::fromArrayAttr(hashLeft);
   MaterializationHelper keyHelper(hashLeft, rewriter.getContext());
   auto valueColumns = columns;
   valueColumns.remove(keyColumns);
   MaterializationHelper valueHelper(valueColumns, rewriter.getContext());
   auto flagMember = valueHelper.addFlag(markerDefAttr);
   auto multiMapType = mlir::subop::MultiMapType::get(rewriter.getContext(), keyHelper.createStateMembersAttr(), valueHelper.createStateMembersAttr());
   mlir::Value multiMap = rewriter.create<mlir::subop::GenericCreateOp>(loc, multiMapType);
   left = mapBool(left, rewriter, loc, false, &markerDefAttr.getColumn());
   auto insertOp = rewriter.create<mlir::subop::InsertOp>(loc, left, multiMap, keyHelper.createColumnstateMapping(valueHelper.createColumnstateMapping().getValue()));
   insertOp.getEqFn().push_back(createEqFn(rewriter, hashLeft, hashLeft, nullsEqual, loc));
   auto entryRefType = mlir::subop::MultiMapEntryRefType::get(rewriter.getContext(), multiMapType);
   auto entryRefListType = mlir::subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), right, multiMap, hashRight, listDef);
   afterLookup.getEqFn().push_back(createEqFn(rewriter, hashLeft, hashRight, nullsEqual, loc));

   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);

   nestedMapOp.getRegion().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      mlir::Value scan = rewriter.create<mlir::subop::ScanListOp>(loc, list, entryDef);
      mlir::Value gathered = rewriter.create<mlir::subop::GatherOp>(loc, scan, entryRef, keyHelper.createStateColumnMapping(valueHelper.createStateColumnMapping({}, {flagMember}).getValue()));
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, gathered, tuple);
      auto res = fn(combined, tuple, rewriter, entryRef, flagMember);
      if (res) {
         rewriter.create<mlir::tuples::ReturnOp>(loc, res);
      } else {
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
   }
   return {nestedMapOp.getRes(), rewriter.create<mlir::subop::ScanOp>(loc, multiMap, keyHelper.createStateColumnMapping(valueHelper.createStateColumnMapping().getValue()))};
}
static std::pair<mlir::Value, mlir::Value> translateNLWithMarker(mlir::Value left, mlir::Value right, bool useHash, mlir::ArrayAttr nullsEqual, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr, std::string markerName)> fn) {
   if (useHash) {
      return translateHJWithMarker(left, right, nullsEqual, hashLeft, hashRight, columns, rewriter, loc, markerDefAttr, fn);
   } else {
      return translateNLJWithMarker(left, right, columns, rewriter, loc, markerDefAttr, fn);
   }
}

static mlir::Value anyTuple(mlir::Value stream, mlir::tuples::ColumnDefAttr markerDefAttr, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   auto [markerState, markerName] = createMarkerState(rewriter, loc);
   auto [mapped, boolColumn] = mapBool(stream, rewriter, loc, true);
   auto [referenceDefAttr, referenceRefAttr] = createColumn(mlir::subop::LookupEntryRefType::get(rewriter.getContext(), markerState.getType().cast<mlir::subop::LookupAbleState>()), "lookup", "ref");
   auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), mapped, markerState, rewriter.getArrayAttr({}), referenceDefAttr);
   rewriter.create<mlir::subop::ScatterOp>(loc, afterLookup, referenceRefAttr, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, colManager.createRef(boolColumn))));
   return rewriter.create<mlir::subop::ScanOp>(loc, markerState, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, markerDefAttr)));
}

class CrossProductLowering : public OpConversionPattern<mlir::relalg::CrossProductOp> {
   public:
   using OpConversionPattern<mlir::relalg::CrossProductOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::CrossProductOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = crossProductOp->getLoc();
      rewriter.replaceOp(crossProductOp, translateNL(adaptor.getRight(), adaptor.getLeft(), false, false, mlir::ArrayAttr(), mlir::ArrayAttr(), mlir::ArrayAttr(), getRequired(mlir::cast<Operator>(crossProductOp.getLeft().getDefiningOp())), rewriter, loc, [](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                            return v;
                         }));
      return success();
   }
};
class InnerJoinNLLowering : public OpConversionPattern<mlir::relalg::InnerJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::InnerJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::InnerJoinOp innerJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = innerJoinOp->getLoc();
      bool useHash = innerJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = innerJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");
      rewriter.replaceOp(innerJoinOp, translateNL(adaptor.getRight(), adaptor.getLeft(), useHash, useIndexNestedLoop, nullsEqual, rightHash, leftHash, getRequired(mlir::cast<Operator>(innerJoinOp.getLeft().getDefiningOp())), rewriter, loc, [loc, &innerJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                            return translateSelection(v, innerJoinOp.getPredicate(), rewriter, loc);
                         }));
      return success();
   }
};
class SemiJoinLowering : public OpConversionPattern<mlir::relalg::SemiJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::SemiJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::SemiJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = semiJoinOp->getLoc();
      bool reverse = semiJoinOp->hasAttr("reverseSides");
      bool useHash = semiJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = semiJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp())), rewriter, loc, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               return rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr({markerRefAttr}));
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [_, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            return {};
         });
         rewriter.replaceOpWithNewOp<mlir::subop::FilterOp>(semiJoinOp, scan, mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr({flagAttrRef}));
      }
      return success();
   }
};
class MarkJoinLowering : public OpConversionPattern<mlir::relalg::MarkJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::MarkJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::MarkJoinOp markJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = markJoinOp->getLoc();
      bool reverse = markJoinOp->hasAttr("reverseSides");
      bool useHash = markJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = markJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = markJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = markJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = markJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(markJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(markJoinOp.getRight().getDefiningOp())), rewriter, loc, [loc, &markJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, markJoinOp.getPredicate(), rewriter, loc);
                               return anyTuple(filtered, markJoinOp.getMarkattr(), rewriter, loc);
                            }));
      } else {
         auto [_, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(markJoinOp.getLeft().getDefiningOp())), rewriter, loc, markJoinOp.getMarkattr(), [loc, &markJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, markJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            return {};
         });
         rewriter.replaceOp(markJoinOp, scan);
      }
      return success();
   }
};
class AntiSemiJoinLowering : public OpConversionPattern<mlir::relalg::AntiSemiJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::AntiSemiJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::AntiSemiJoinOp antiSemiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = antiSemiJoinOp->getLoc();
      bool reverse = antiSemiJoinOp->hasAttr("reverseSides");
      bool useHash = antiSemiJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = antiSemiJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(antiSemiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(antiSemiJoinOp.getRight().getDefiningOp())), rewriter, loc, [loc, &antiSemiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, antiSemiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               return rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [_, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(antiSemiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &antiSemiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, antiSemiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            return {};
         });
         rewriter.replaceOpWithNewOp<mlir::subop::FilterOp>(antiSemiJoinOp, scan, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
      }
      return success();
   }
};
class FullOuterJoinLowering : public OpConversionPattern<mlir::relalg::FullOuterJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::FullOuterJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::FullOuterJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = semiJoinOp->getLoc();
      bool useHash = semiJoinOp->hasAttr("useHashJoin");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto leftColumns = getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp()));
      auto rightColumns = getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp()));
      auto nullsEqual = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
      auto [stream, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp, leftColumns, rightColumns](mlir::Value v, mlir::Value tuple, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
         auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
         auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
         auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
         rewriter.create<mlir::subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
         auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());
         auto [markerDefAttr2, markerRefAttr2] = createColumn(rewriter.getI1Type(), "marker", "marker");
         Value filteredNoMatch = rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr2, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr2}));
         mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, filteredNoMatch, tuple);

         auto mappedNullable2 = mapColsToNullable(combined, rewriter, loc, semiJoinOp.getMapping(), 0, leftColumns);

         auto mappedNull = mapColsToNull(mappedNullable2, rewriter, loc, semiJoinOp.getMapping(), rightColumns);
         return rewriter.create<mlir::subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
      });
      auto noMatches = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
      auto mappedNullable = mapColsToNullable(noMatches, rewriter, loc, semiJoinOp.getMapping(), 0, rightColumns);
      auto mappedNull = mapColsToNull(mappedNullable, rewriter, loc, semiJoinOp.getMapping(), leftColumns);
      rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(semiJoinOp, mlir::ValueRange{stream, mappedNull});

      return success();
   }
};
class OuterJoinLowering : public OpConversionPattern<mlir::relalg::OuterJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::OuterJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::OuterJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = semiJoinOp->getLoc();
      bool reverse = semiJoinOp->hasAttr("reverseSides");
      bool useHash = semiJoinOp->hasAttr("useHashJoin");
      bool useIndexNestedLoop = semiJoinOp->hasAttr("useIndexNestedLoop");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto nullsEqual = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");

      if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp())), rewriter, loc, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               Value filteredNoMatch = rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                               auto mappedNull = mapColsToNull(filteredNoMatch, rewriter, loc, semiJoinOp.getMapping());
                               auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());
                               return rewriter.create<mlir::subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [stream, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());

            return mappedNullable;
         });
         auto noMatches = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
         auto mappedNull = mapColsToNull(noMatches, rewriter, loc, semiJoinOp.getMapping());
         rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(semiJoinOp, mlir::ValueRange{stream, mappedNull});
      }
      return success();
   }
};
class SingleJoinLowering : public OpConversionPattern<mlir::relalg::SingleJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::SingleJoinOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::SingleJoinOp semiJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
         auto constantStateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), helper.createStateMembersAttr());
         mlir::Value constantState = rewriter.create<mlir::subop::CreateSimpleStateOp>(loc, constantStateType);
         auto entryRefType = mlir::subop::LookupEntryRefType::get(rewriter.getContext(), constantStateType);
         auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
         auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRight(), constantState, rewriter.getArrayAttr({}), entryDef);
         rewriter.create<mlir::subop::ScatterOp>(loc, afterLookup, entryRef, helper.createColumnstateMapping());
         auto [entryDefLeft, entryRefLeft] = createColumn(entryRefType, "lookup", "entryref");

         auto afterLookupLeft = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getLeft(), constantState, rewriter.getArrayAttr({}), entryDefLeft);
         auto gathered = rewriter.create<mlir::subop::GatherOp>(loc, afterLookupLeft, entryRefLeft, helper.createStateColumnMapping());
         auto mappedNullable = mapColsToNullable(gathered.getRes(), rewriter, loc, semiJoinOp.getMapping());
         rewriter.replaceOp(semiJoinOp, mappedNullable);
      } else if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.getLeft(), adaptor.getRight(), useHash, useIndexNestedLoop, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getRight().getDefiningOp())), rewriter, loc, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               Value filteredNoMatch = rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                               auto mappedNull = mapColsToNull(filteredNoMatch, rewriter, loc, semiJoinOp.getMapping());
                               auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());
                               return rewriter.create<mlir::subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [stream, scan] = translateNLWithMarker(adaptor.getLeft(), adaptor.getRight(), useHash, nullsEqual, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.getLeft().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v, mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.getPredicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(loc, afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.getMapping());

            return mappedNullable;
         });
         auto noMatches = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
         auto mappedNull = mapColsToNull(noMatches, rewriter, loc, semiJoinOp.getMapping());
         rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(semiJoinOp, mlir::ValueRange{stream, mappedNull});
      }
      return success();
   }
};

class LimitLowering : public OpConversionPattern<mlir::relalg::LimitOp> {
   public:
   using OpConversionPattern<mlir::relalg::LimitOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::LimitOp limitOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = limitOp->getLoc();
      mlir::relalg::ColumnSet requiredColumns = getRequired(limitOp);
      MaterializationHelper helper(requiredColumns, rewriter.getContext());

      auto* block = new Block;
      std::vector<Attribute> sortByMembers;
      std::vector<Location> locs;
      std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         mlir::Value isLt = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
         rewriter.create<mlir::tuples::ReturnOp>(loc, isLt);
      }
      auto heapType = mlir::subop::HeapType::get(getContext(), helper.createStateMembersAttr(), limitOp.getMaxRows());
      auto createHeapOp = rewriter.create<mlir::subop::CreateHeapOp>(loc, heapType, rewriter.getArrayAttr(sortByMembers));
      createHeapOp.getRegion().getBlocks().push_back(block);
      rewriter.create<mlir::subop::MaterializeOp>(loc, adaptor.getRel(), createHeapOp.getRes(), helper.createColumnstateMapping());
      rewriter.replaceOpWithNewOp<mlir::subop::ScanOp>(limitOp, createHeapOp.getRes(), helper.createStateColumnMapping());
      return success();
   }
};
static mlir::Value spaceShipCompare(mlir::OpBuilder& builder, std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria, size_t pos, mlir::Location loc) {
   mlir::Value compareRes = builder.create<mlir::db::SortCompare>(loc, sortCriteria.at(pos).first, sortCriteria.at(pos).second);
   auto zero = builder.create<mlir::db::ConstantOp>(loc, builder.getI8Type(), builder.getIntegerAttr(builder.getI8Type(), 0));
   auto isZero = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, compareRes, zero);
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
      auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
      argumentTypes.push_back(sortspecAttr.getAttr().getColumn().type);
      locs.push_back(loc);
      sortByMembers.push_back(helper.lookupStateMemberForMaterializedColumn(&sortspecAttr.getAttr().getColumn()));
   }
   block->addArguments(argumentTypes, locs);
   block->addArguments(argumentTypes, locs);
   std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
   for (auto attr : sortSpecs) {
      auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
      mlir::Value left = block->getArgument(sortCriteria.size());
      mlir::Value right = block->getArgument(sortCriteria.size() + sortSpecs.size());
      if (sortspecAttr.getSortSpec() == mlir::relalg::SortSpec::desc) {
         std::swap(left, right);
      }
      sortCriteria.push_back({left, right});
   }
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      auto zero = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI8Type(), rewriter.getIntegerAttr(rewriter.getI8Type(), 0));
      auto spaceShipResult = spaceShipCompare(rewriter, sortCriteria, 0, loc);
      mlir::Value isLt = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, spaceShipResult, zero);
      rewriter.create<mlir::tuples::ReturnOp>(loc, isLt);
   }

   auto subOpSort = rewriter.create<mlir::subop::CreateSortedViewOp>(loc, mlir::subop::SortedViewType::get(rewriter.getContext(), buffer.getType().cast<mlir::subop::State>()), buffer, rewriter.getArrayAttr(sortByMembers));
   subOpSort.getRegion().getBlocks().push_back(block);
   return subOpSort.getResult();
}
class SortLowering : public OpConversionPattern<mlir::relalg::SortOp> {
   public:
   using OpConversionPattern<mlir::relalg::SortOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::SortOp sortOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = sortOp->getLoc();
      mlir::relalg::ColumnSet requiredColumns = getRequired(sortOp);
      requiredColumns.insert(sortOp.getUsedColumns());
      MaterializationHelper helper(requiredColumns, rewriter.getContext());
      auto vectorType = mlir::subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector = rewriter.create<mlir::subop::GenericCreateOp>(sortOp->getLoc(), vectorType);
      rewriter.create<mlir::subop::MaterializeOp>(sortOp->getLoc(), adaptor.getRel(), vector, helper.createColumnstateMapping());
      auto sortedView = createSortedView(rewriter, vector, sortOp.getSortspecs(), loc, helper);
      auto scanOp = rewriter.replaceOpWithNewOp<mlir::subop::ScanOp>(sortOp, sortedView, helper.createStateColumnMapping());
      scanOp->setAttr("sequential", rewriter.getUnitAttr());
      return success();
   }
};

class TopKLowering : public OpConversionPattern<mlir::relalg::TopKOp> {
   public:
   using OpConversionPattern<mlir::relalg::TopKOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::TopKOp topk, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = topk->getLoc();
      mlir::relalg::ColumnSet requiredColumns = getRequired(topk);
      requiredColumns.insert(topk.getUsedColumns());
      MaterializationHelper helper(requiredColumns, rewriter.getContext());

      auto* block = new Block;
      std::vector<Attribute> sortByMembers;
      std::vector<Type> argumentTypes;
      std::vector<Location> locs;
      for (auto attr : topk.getSortspecs()) {
         auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
         argumentTypes.push_back(sortspecAttr.getAttr().getColumn().type);
         locs.push_back(loc);
         sortByMembers.push_back(helper.lookupStateMemberForMaterializedColumn(&sortspecAttr.getAttr().getColumn()));
      }
      block->addArguments(argumentTypes, locs);
      block->addArguments(argumentTypes, locs);
      std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
      for (auto attr : topk.getSortspecs()) {
         auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
         mlir::Value left = block->getArgument(sortCriteria.size());
         mlir::Value right = block->getArgument(sortCriteria.size() + topk.getSortspecs().size());
         if (sortspecAttr.getSortSpec() == mlir::relalg::SortSpec::desc) {
            std::swap(left, right);
         }
         sortCriteria.push_back({left, right});
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto zero = rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI8Type(), rewriter.getIntegerAttr(rewriter.getI8Type(), 0));
         auto spaceShipResult = spaceShipCompare(rewriter, sortCriteria, 0, loc);
         mlir::Value isLt = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, spaceShipResult, zero);
         rewriter.create<mlir::tuples::ReturnOp>(loc, isLt);
      }
      auto heapType = mlir::subop::HeapType::get(getContext(), helper.createStateMembersAttr(), topk.getMaxRows());
      auto createHeapOp = rewriter.create<mlir::subop::CreateHeapOp>(loc, heapType, rewriter.getArrayAttr(sortByMembers));
      createHeapOp.getRegion().getBlocks().push_back(block);
      rewriter.create<mlir::subop::MaterializeOp>(loc, adaptor.getRel(), createHeapOp.getRes(), helper.createColumnstateMapping());
      auto scanOp = rewriter.replaceOpWithNewOp<mlir::subop::ScanOp>(topk, createHeapOp.getRes(), helper.createStateColumnMapping());
      scanOp->setAttr("sequential", rewriter.getUnitAttr());
      return success();
   }
};
class TmpLowering : public OpConversionPattern<mlir::relalg::TmpOp> {
   public:
   using OpConversionPattern<mlir::relalg::TmpOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::relalg::TmpOp tmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      MaterializationHelper helper(getRequired(tmpOp), rewriter.getContext());

      auto vectorType = mlir::subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector = rewriter.create<mlir::subop::GenericCreateOp>(tmpOp->getLoc(), vectorType);
      rewriter.create<mlir::subop::MaterializeOp>(tmpOp->getLoc(), adaptor.getRel(), vector, helper.createColumnstateMapping());
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
      auto resultTableType = materializeOp.getResult().getType().cast<mlir::subop::ResultTableType>();
      std::vector<Attribute> colNames;
      std::vector<NamedAttribute> mapping;
      for (size_t i = 0; i < materializeOp.getColumns().size(); i++) {
         auto columnName = materializeOp.getColumns()[i].cast<mlir::StringAttr>();
         auto colMemberName = resultTableType.getMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto columnAttr = materializeOp.getCols()[i].cast<mlir::tuples::ColumnRefAttr>();
         mapping.push_back(rewriter.getNamedAttr(colMemberName, columnAttr));
         colNames.push_back(columnName);
      }
      mlir::Value table = rewriter.create<mlir::subop::CreateResultTableOp>(materializeOp->getLoc(), resultTableType, rewriter.getArrayAttr(colNames));
      rewriter.create<mlir::subop::MaterializeOp>(materializeOp->getLoc(), adaptor.getRel(), table, rewriter.getDictionaryAttr(mapping));
      rewriter.replaceOp(materializeOp, table);

      return success();
   }
};
class DistAggrFunc {
   protected:
   Type stateType;
   mlir::tuples::ColumnDefAttr destAttribute;
   mlir::tuples::ColumnRefAttr sourceAttribute;

   public:
   DistAggrFunc(const mlir::tuples::ColumnDefAttr& destAttribute, const mlir::tuples::ColumnRefAttr& sourceAttribute) : stateType(destAttribute.cast<mlir::tuples::ColumnDefAttr>().getColumn().type), destAttribute(destAttribute), sourceAttribute(sourceAttribute) {}
   virtual mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) = 0;
   virtual mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) = 0;
   virtual mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) = 0;
   const mlir::tuples::ColumnDefAttr& getDestAttribute() const {
      return destAttribute;
   }
   const mlir::tuples::ColumnRefAttr& getSourceAttribute() const {
      return sourceAttribute;
   }
   const Type& getStateType() const {
      return stateType;
   }
   virtual ~DistAggrFunc() {}
};

class CountStarAggrFunc : public DistAggrFunc {
   public:
   explicit CountStarAggrFunc(const mlir::tuples::ColumnDefAttr& destAttribute) : DistAggrFunc(destAttribute, mlir::tuples::ColumnRefAttr()) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      return builder.create<mlir::db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      auto one = builder.create<mlir::db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(1));
      return builder.create<mlir::db::AddOp>(loc, stateType, state, one);
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      return builder.create<mlir::db::AddOp>(loc, left, right);
   }
};
class CountAggrFunc : public DistAggrFunc {
   public:
   explicit CountAggrFunc(const mlir::tuples::ColumnDefAttr& destAttribute, const mlir::tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      return builder.create<mlir::db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      auto one = builder.create<mlir::db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(1));
      auto added = builder.create<mlir::db::AddOp>(loc, stateType, state, one);
      if (args[0].getType().isa<mlir::db::NullableType>()) {
         auto isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), args[0]);
         return builder.create<mlir::arith::SelectOp>(loc, isNull, state, added);
      } else {
         return added;
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      return builder.create<mlir::db::AddOp>(loc, left, right);
   }
};
class AnyAggrFunc : public DistAggrFunc {
   public:
   explicit AnyAggrFunc(const mlir::tuples::ColumnDefAttr& destAttribute, const mlir::tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      return builder.create<mlir::util::UndefOp>(loc, stateType);
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
   explicit MaxAggrFunc(const mlir::tuples::ColumnDefAttr& destAttribute, const mlir::tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      if (stateType.isa<mlir::db::NullableType>()) {
         return builder.create<mlir::db::NullOp>(loc, stateType);
      } else {
         return builder.create<mlir::db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
      }
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      mlir::Value stateLtArg = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, state, args[0]);
      mlir::Value stateLtArgTruth = builder.create<mlir::db::DeriveTruth>(loc, stateLtArg);
      if (stateType.isa<mlir::db::NullableType>() && args[0].getType().isa<mlir::db::NullableType>()) {
         // state nullable, arg nullable
         mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateLtArgTruth, isNull);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, args[0], state);
      } else if (stateType.isa<mlir::db::NullableType>()) {
         // state nullable, arg not nullable
         mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateLtArgTruth, isNull);
         mlir::Value casted = builder.create<mlir::db::AsNullableOp>(loc, stateType, args[0]);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, casted, state);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<mlir::arith::SelectOp>(loc, stateLtArg, args[0], state);
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      mlir::Value leftLtRight = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, left, right);
      mlir::Value leftLtRightTruth = builder.create<mlir::db::DeriveTruth>(loc, leftLtRight);
      if (stateType.isa<mlir::db::NullableType>()) {
         mlir::Value isLeftNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), left);
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
                                      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
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
   explicit MinAggrFunc(const mlir::tuples::ColumnDefAttr& destAttribute, const mlir::tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      if (stateType.isa<mlir::db::NullableType>()) {
         return builder.create<mlir::db::NullOp>(loc, stateType);
      } else {
         return builder.create<mlir::db::ConstantOp>(loc, stateType, getMaxValueAttr(stateType));
      }
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      mlir::Value stateGtArg = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::gt, state, args[0]);
      mlir::Value stateGtArgTruth = builder.create<mlir::db::DeriveTruth>(loc, stateGtArg);
      if (stateType.isa<mlir::db::NullableType>() && args[0].getType().isa<mlir::db::NullableType>()) {
         // state nullable, arg nullable
         mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateGtArgTruth, isNull);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, args[0], state);
      } else if (stateType.isa<mlir::db::NullableType>()) {
         // state nullable, arg not nullable
         mlir::Value casted = builder.create<mlir::db::AsNullableOp>(loc, stateType, args[0]);
         mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value overwriteState = builder.create<mlir::arith::OrIOp>(loc, stateGtArgTruth, isNull);
         return builder.create<mlir::arith::SelectOp>(loc, overwriteState, casted, state);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<mlir::arith::SelectOp>(loc, stateGtArg, args[0], state);
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      mlir::Value leftLtRight = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, left, right);
      mlir::Value leftLtRightTruth = builder.create<mlir::db::DeriveTruth>(loc, leftLtRight);
      if (stateType.isa<mlir::db::NullableType>()) {
         mlir::Value isRightNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), right);
         mlir::Value setToLeft = builder.create<mlir::arith::OrIOp>(loc, leftLtRightTruth, isRightNull);
         return builder.create<mlir::arith::SelectOp>(loc, setToLeft, left, right);
      } else {
         return builder.create<mlir::arith::SelectOp>(loc, leftLtRight, left, right);
      }
   }
};
class SumAggrFunc : public DistAggrFunc {
   public:
   explicit SumAggrFunc(const mlir::tuples::ColumnDefAttr& destAttribute, const mlir::tuples::ColumnRefAttr& sourceColumn) : DistAggrFunc(destAttribute, sourceColumn) {}
   mlir::Value createDefaultValue(mlir::OpBuilder& builder, mlir::Location loc) override {
      if (stateType.isa<mlir::db::NullableType>()) {
         return builder.create<mlir::db::NullOp>(loc, stateType);
      } else {
         return builder.create<mlir::db::ConstantOp>(loc, stateType, builder.getI64IntegerAttr(0));
      }
   }
   mlir::Value aggregate(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value state, mlir::ValueRange args) override {
      if (stateType.isa<mlir::db::NullableType>() && args[0].getType().isa<mlir::db::NullableType>()) {
         // state nullable, arg nullable
         mlir::Value isStateNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value isArgNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), args[0]);
         mlir::Value sum = builder.create<mlir::db::AddOp>(loc, state, args[0]);
         sum = builder.create<mlir::arith::SelectOp>(loc, isArgNull, state, sum);
         return builder.create<mlir::arith::SelectOp>(loc, isStateNull, args[0], sum);
      } else if (stateType.isa<mlir::db::NullableType>()) {
         // state nullable, arg not nullable
         mlir::Value isStateNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), state);
         mlir::Value zero = builder.create<mlir::db::ConstantOp>(loc, getBaseType(stateType), builder.getI64IntegerAttr(0));
         zero = builder.create<mlir::db::AsNullableOp>(loc, stateType, zero);
         state = builder.create<mlir::arith::SelectOp>(loc, isStateNull, zero, state);
         return builder.create<mlir::db::AddOp>(loc, state, args[0]);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<mlir::db::AddOp>(loc, state, args[0]);
      }
   }
   mlir::Value combine(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value left, mlir::Value right) override {
      if (stateType.isa<mlir::db::NullableType>()) {
         // state nullable, arg not nullable
         mlir::Value isLeftNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), left);
         mlir::Value isRightNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), right);
         mlir::Value zero = builder.create<mlir::db::ConstantOp>(loc, getBaseType(stateType), builder.getI64IntegerAttr(0));
         zero = builder.create<mlir::db::AsNullableOp>(loc, stateType, zero);
         mlir::Value newLeft = builder.create<mlir::arith::SelectOp>(loc, isLeftNull, zero, left);
         mlir::Value newRight = builder.create<mlir::arith::SelectOp>(loc, isRightNull, zero, right);
         mlir::Value sum = builder.create<mlir::db::AddOp>(loc, newLeft, newRight);
         mlir::Value bothNull = builder.create<mlir::arith::AndIOp>(loc, isLeftNull, isRightNull);
         return builder.create<mlir::arith::SelectOp>(loc, bothNull, left, sum);
      } else {
         //state non-nullable, arg not nullable
         return builder.create<mlir::db::AddOp>(loc, left, right);
      }
   }
};
class OrderedWindowFunc {
   protected:
   mlir::tuples::ColumnDefAttr destAttribute;
   mlir::tuples::ColumnRefAttr sourceAttribute;

   public:
   OrderedWindowFunc(const mlir::tuples::ColumnDefAttr& destAttribute, const mlir::tuples::ColumnRefAttr& sourceAttribute) : destAttribute(destAttribute), sourceAttribute(sourceAttribute) {}
   virtual mlir::Value evaluate(mlir::ConversionPatternRewriter& builder, mlir::Location loc, mlir::Value stream, mlir::tuples::ColumnRefAttr beginReference, mlir::tuples::ColumnRefAttr endReference, mlir::tuples::ColumnRefAttr currReference) = 0;
   const mlir::tuples::ColumnDefAttr& getDestAttribute() const {
      return destAttribute;
   }
   const mlir::tuples::ColumnRefAttr& getSourceAttribute() const {
      return sourceAttribute;
   }
   virtual ~OrderedWindowFunc() {}
};
class RankWindowFunc : public OrderedWindowFunc {
   public:
   explicit RankWindowFunc(const mlir::tuples::ColumnDefAttr& destAttribute) : OrderedWindowFunc(destAttribute, mlir::tuples::ColumnRefAttr()) {}
   mlir::Value evaluate(mlir::ConversionPatternRewriter& builder, mlir::Location loc, mlir::Value stream, mlir::tuples::ColumnRefAttr beginReference, mlir::tuples::ColumnRefAttr endReference, mlir::tuples::ColumnRefAttr currReference) override {
      auto [entriesBetweenDef, entriesBetweenRef] = createColumn(builder.getIndexType(), "window", "entries_between");
      auto entriesBetweenRef2 = entriesBetweenRef;
      mlir::Value afterEntriesBetween = builder.create<mlir::subop::EntriesBetweenOp>(loc, stream, beginReference, currReference, entriesBetweenDef);
      return map(afterEntriesBetween, builder, loc, builder.getArrayAttr(destAttribute), [&](mlir::ConversionPatternRewriter& rewriter, mlir::Value tuple, mlir::Location loc) {
         mlir::Value between = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getIndexType(), entriesBetweenRef2, tuple);
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
      rewriter.create<mlir::tuples::ReturnOp>(loc, defaultValues);
   }
   return initialValueBlock;
}
void performAggrFuncReduce(mlir::Location loc, mlir::OpBuilder& rewriter, std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs, mlir::tuples::ColumnRefAttr reference, mlir::Value stream, std::vector<mlir::Attribute> names, std::vector<NamedAttribute> defMapping) {
   mlir::Block* reduceBlock = new Block;
   mlir::Block* combineBlock = new Block;
   std::vector<mlir::Attribute> relevantColumns;
   std::unordered_map<mlir::tuples::Column*, mlir::Value> stateMap;
   std::unordered_map<mlir::tuples::Column*, mlir::Value> argMap;
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
   auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(loc, stream, reference, rewriter.getArrayAttr(relevantColumns), rewriter.getArrayAttr(names));
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
      rewriter.create<mlir::tuples::ReturnOp>(loc, newStateValues);
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
      rewriter.create<mlir::tuples::ReturnOp>(loc, combinedValues);
   }
   reduceOp.getRegion().push_back(reduceBlock);
   reduceOp.getCombine().push_back(combineBlock);
}
static std::tuple<mlir::Value, mlir::DictionaryAttr, mlir::DictionaryAttr> performAggregation(mlir::Location loc, mlir::OpBuilder& rewriter, std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs, mlir::relalg::OrderedAttributes keyAttributes, mlir::Value stream, std::function<void(mlir::Location, mlir::OpBuilder&, std::vector<std::shared_ptr<DistAggrFunc>>, mlir::tuples::ColumnRefAttr, mlir::Value, std::vector<mlir::Attribute>, std::vector<NamedAttribute>)> createReduceFn) {
   auto* context = rewriter.getContext();
   auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
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
   auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, names), mlir::ArrayAttr::get(context, types));
   mlir::Type stateType;

   mlir::Value afterLookup;
   auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");

   if (keyAttributes.getAttrs().empty()) {
      stateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), stateMembers);
      auto createOp = rewriter.create<mlir::subop::CreateSimpleStateOp>(loc, stateType);
      createOp.getInitFn().push_back(initialValueBlock);
      state = createOp.getRes();
      afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, state, rewriter.getArrayAttr({}), referenceDefAttr);

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
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      stateType = mlir::subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers);

      auto createOp = rewriter.create<mlir::subop::GenericCreateOp>(loc, stateType);
      state = createOp.getRes();
      auto lookupOp = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, state, keyAttributes.getArrayAttr(rewriter.getContext()), referenceDefAttr);
      afterLookup = lookupOp;
      lookupOp.getInitFn().push_back(initialValueBlock);
      mlir::Block* equalBlock = new Block;
      lookupOp.getEqFn().push_back(equalBlock);
      equalBlock->addArguments(keyTypes, locations);
      equalBlock->addArguments(keyTypes, locations);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(equalBlock);
         mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
         rewriter.create<mlir::tuples::ReturnOp>(loc, compared);
      }
   }
   referenceDefAttr.getColumn().type = mlir::subop::LookupEntryRefType::get(context, stateType.cast<mlir::subop::LookupAbleState>());

   auto referenceRefAttr = colManager.createRef(&referenceDefAttr.getColumn());
   createReduceFn(loc, rewriter, distAggrFuncs, referenceRefAttr, afterLookup, names, defMapping);
   return {state, rewriter.getDictionaryAttr(defMapping), rewriter.getDictionaryAttr(computedDefMapping)};
}
class WindowLowering : public OpConversionPattern<mlir::relalg::WindowOp> {
   public:
   using OpConversionPattern<mlir::relalg::WindowOp>::OpConversionPattern;
   struct AnalyzedWindow {
      std::vector<std::pair<mlir::relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distAggrFuncs;
      std::vector<std::shared_ptr<OrderedWindowFunc>> orderedWindowFunctions;
   };

   void analyze(mlir::relalg::WindowOp windowOp, AnalyzedWindow& analyzedWindow) const {
      mlir::tuples::ReturnOp terminator = mlir::cast<mlir::tuples::ReturnOp>(windowOp.getAggrFunc().front().getTerminator());
      std::unordered_map<mlir::Operation*, std::pair<mlir::relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distinct;
      distinct.insert({nullptr, {mlir::relalg::OrderedAttributes::fromVec({}), {}}});
      for (size_t i = 0; i < windowOp.getComputedCols().size(); i++) {
         auto destColumnAttr = windowOp.getComputedCols()[i].cast<mlir::tuples::ColumnDefAttr>();
         mlir::Value computedVal = terminator.getResults()[i];
         mlir::Value tupleStream;
         std::shared_ptr<DistAggrFunc> distAggrFunc;
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            tupleStream = aggrFn.getRel();
            auto sourceColumnAttr = aggrFn.getAttr();
            if (aggrFn.getFn() == mlir::relalg::AggrFunc::sum) {
               distAggrFunc = std::make_shared<SumAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::min) {
               distAggrFunc = std::make_shared<MinAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::max) {
               distAggrFunc = std::make_shared<MaxAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::any) {
               distAggrFunc = std::make_shared<AnyAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::count) {
               distAggrFunc = std::make_shared<CountAggrFunc>(destColumnAttr, sourceColumnAttr);
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            tupleStream = countOp.getRel();
            distAggrFunc = std::make_shared<CountStarAggrFunc>(destColumnAttr);
         }
         if (auto rankOp = mlir::dyn_cast_or_null<mlir::relalg::RankOp>(computedVal.getDefiningOp())) {
            analyzedWindow.orderedWindowFunctions.push_back(std::make_shared<RankWindowFunc>(destColumnAttr));
         }
         if (distAggrFunc) {
            if (!distinct.count(tupleStream.getDefiningOp())) {
               if (auto projectionOp = mlir::dyn_cast_or_null<mlir::relalg::ProjectionOp>(tupleStream.getDefiningOp())) {
                  distinct[tupleStream.getDefiningOp()] = {mlir::relalg::OrderedAttributes::fromRefArr(projectionOp.getCols()), {}};
               }
            }
            distinct.at(tupleStream.getDefiningOp()).second.push_back(distAggrFunc);
         }
      };
      for (auto d : distinct) {
         analyzedWindow.distAggrFuncs.push_back({d.second.first, d.second.second});
      }
   }
   void performWindowOp(mlir::relalg::WindowOp windowOp, mlir::Value inputStream, ConversionPatternRewriter& rewriter, std::function<mlir::Value(ConversionPatternRewriter&, mlir::Value, mlir::DictionaryAttr, mlir::Location)> evaluate) const {
      mlir::relalg::ColumnSet requiredColumns = getRequired(windowOp);
      auto& colManager = getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      requiredColumns.insert(windowOp.getUsedColumns());
      requiredColumns.remove(windowOp.getCreatedColumns());
      auto loc = windowOp->getLoc();
      if (windowOp.getPartitionBy().empty()) {
         MaterializationHelper helper(requiredColumns, rewriter.getContext());

         auto vectorType = mlir::subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());
         mlir::Value vector = rewriter.create<mlir::subop::GenericCreateOp>(loc, vectorType);
         rewriter.create<mlir::subop::MaterializeOp>(loc, inputStream, vector, helper.createColumnstateMapping());
         mlir::Value continuousView;
         if (windowOp.getOrderBy().empty()) {
            auto continuousViewType = mlir::subop::ContinuousViewType::get(rewriter.getContext(), vectorType);
            continuousView = rewriter.create<mlir::subop::CreateContinuousView>(loc, continuousViewType, vector);
         } else {
            auto sortedView = createSortedView(rewriter, vector, windowOp.getOrderBy(), loc, helper);
            auto continuousViewType = mlir::subop::ContinuousViewType::get(rewriter.getContext(), sortedView.getType().cast<mlir::subop::State>());
            continuousView = rewriter.create<mlir::subop::CreateContinuousView>(loc, continuousViewType, sortedView);
         }
         rewriter.replaceOp(windowOp, evaluate(rewriter, continuousView, helper.createStateColumnMapping(), loc));
      } else {
         auto keyAttributes = mlir::relalg::OrderedAttributes::fromRefArr(windowOp.getPartitionBy());
         auto valueColumns = requiredColumns;
         valueColumns.remove(mlir::relalg::ColumnSet::fromArrayAttr(windowOp.getPartitionBy()));
         MaterializationHelper helper(valueColumns, rewriter.getContext());
         auto bufferType = mlir::subop::BufferType::get(rewriter.getContext(), helper.createStateMembersAttr());

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

         auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {rewriter.getStringAttr(bufferMember)}), mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(bufferType)}));
         auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
         auto hashMapType = mlir::subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers);

         auto createOp = rewriter.create<mlir::subop::GenericCreateOp>(loc, hashMapType);
         mlir::Value hashMap = createOp.getRes();

         auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
         referenceDefAttr.getColumn().type = mlir::subop::LookupEntryRefType::get(context, hashMapType);

         auto lookupOp = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), inputStream, hashMap, keyAttributes.getArrayAttr(rewriter.getContext()), referenceDefAttr);
         mlir::Value afterLookup = lookupOp;
         Block* initialValueBlock = new Block;
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(initialValueBlock);
            std::vector<mlir::Value> defaultValues;
            mlir::Value buffer = rewriter.create<mlir::subop::GenericCreateOp>(loc, bufferType);
            buffer.getDefiningOp()->setAttr("initial_capacity", rewriter.getI64IntegerAttr(1));
            buffer.getDefiningOp()->setAttr("group", rewriter.getI64IntegerAttr(0));
            rewriter.create<mlir::tuples::ReturnOp>(loc, buffer);
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
            rewriter.create<mlir::tuples::ReturnOp>(loc, compared);
         }
         auto referenceRefAttr = colManager.createRef(&referenceDefAttr.getColumn());
         auto orderedValueColumns = mlir::relalg::OrderedAttributes::fromColumns(valueColumns);

         auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(loc, afterLookup, referenceRefAttr, orderedValueColumns.getArrayAttr(context), rewriter.getArrayAttr({rewriter.getStringAttr(bufferMember)}));
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
            mlir::Value stream = rewriter.create<mlir::subop::InFlightOp>(loc, columnValues, orderedValueColumns.getDefArrayAttr(context));
            rewriter.create<mlir::subop::MaterializeOp>(loc, stream, currentBuffer, helper.createColumnstateMapping());
            rewriter.create<mlir::tuples::ReturnOp>(loc, currentBuffer);
            reduceOp.getRegion().push_back(reduceBlock);
         }
         {
            mlir::Block* combineBlock = new Block;
            mlir::Value currentBuffer = combineBlock->addArgument(bufferType, loc);
            mlir::Value otherBuffer = combineBlock->addArgument(bufferType, loc);

            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(combineBlock);
            std::vector<mlir::Value> newStateValues;
            auto scan = rewriter.create<mlir::subop::ScanOp>(loc, otherBuffer, helper.createStateColumnMapping());
            rewriter.create<mlir::subop::MaterializeOp>(loc, scan.getRes(), currentBuffer, helper.createColumnstateMapping());
            rewriter.create<mlir::tuples::ReturnOp>(loc, currentBuffer);
            reduceOp.getCombine().push_back(combineBlock);
         }
         mlir::Value newStream = rewriter.create<mlir::subop::ScanOp>(loc, hashMap, rewriter.getDictionaryAttr(defMapping));
         auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), newStream, rewriter.getArrayAttr({bufferRef}));
         auto* b = new Block;
         b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
         mlir::Value buffer = b->addArgument(bufferType, loc);
         nestedMapOp.getRegion().push_back(b);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(b);
            mlir::Value continuousView;
            if (windowOp.getOrderBy().empty()) {
               auto continuousViewType = mlir::subop::ContinuousViewType::get(rewriter.getContext(), buffer.getType().cast<mlir::subop::State>());
               continuousView = rewriter.create<mlir::subop::CreateContinuousView>(loc, continuousViewType, buffer);
            } else {
               auto sortedView = createSortedView(rewriter, buffer, windowOp.getOrderBy(), loc, helper);
               auto continuousViewType = mlir::subop::ContinuousViewType::get(rewriter.getContext(), sortedView.getType().cast<mlir::subop::State>());
               continuousView = rewriter.create<mlir::subop::CreateContinuousView>(loc, continuousViewType, sortedView);
            }
            rewriter.create<mlir::tuples::ReturnOp>(loc, evaluate(rewriter, continuousView, helper.createStateColumnMapping(), loc));
         }

         rewriter.replaceOp(windowOp, nestedMapOp.getRes());
      }
   }

   std::tuple<Value, DictionaryAttr> buildSegmentTree(Location loc, ConversionPatternRewriter& rewriter, std::vector<std::shared_ptr<DistAggrFunc>> aggrFuncs, Value continuousView, mlir::DictionaryAttr continuousViewMapping) const {
      std::vector<mlir::Attribute> relevantMembers;
      std::unordered_map<mlir::tuples::Column*, mlir::Value> stateMap;
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
                  if (&x.getValue().cast<mlir::tuples::ColumnDefAttr>().getColumn() == &sourceColumn.getColumn()) {
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
         rewriter.create<mlir::tuples::ReturnOp>(loc, initialValues);
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
         rewriter.create<mlir::tuples::ReturnOp>(loc, combinedValues);
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
      auto continuousViewRefType = mlir::subop::ContinuousEntryRefType::get(rewriter.getContext(), continuousView.getType().cast<mlir::subop::ContinuousViewType>());
      auto cVRTAttr = mlir::TypeAttr::get(continuousViewRefType);
      auto keyStateMembers = mlir::subop::StateMembersAttr::get(rewriter.getContext(), mlir::ArrayAttr::get(rewriter.getContext(), {rewriter.getStringAttr(fromMemberName), rewriter.getStringAttr(toMemberName)}), mlir::ArrayAttr::get(rewriter.getContext(), {cVRTAttr, cVRTAttr}));
      auto valueStateMembers = mlir::subop::StateMembersAttr::get(rewriter.getContext(), mlir::ArrayAttr::get(rewriter.getContext(), names), mlir::ArrayAttr::get(rewriter.getContext(), types));
      auto segmentTreeViewType = mlir::subop::SegmentTreeViewType::get(rewriter.getContext(), keyStateMembers, valueStateMembers);
      auto createOp = rewriter.create<mlir::subop::CreateSegmentTreeView>(loc, segmentTreeViewType, continuousView, rewriter.getArrayAttr(relevantMembers));
      createOp.getInitialFn().push_back(initBlock);
      createOp.getCombineFn().push_back(combineBlock);
      return {createOp.getResult(), rewriter.getDictionaryAttr(defMapping)};
   }
   LogicalResult matchAndRewrite(mlir::relalg::WindowOp windowOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
         auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
         auto continuousViewRefType = mlir::subop::ContinuousEntryRefType::get(rewriter.getContext(), continuousView.getType().cast<mlir::subop::ContinuousViewType>());
         auto [beginReferenceDefAttr, beginReferenceRefAttr] = createColumn(continuousViewRefType, "view", "begin");
         auto [endReferenceDefAttr, endReferenceRefAttr] = createColumn(continuousViewRefType, "view", "end");
         auto [referenceDefAttr, referenceRefAttr] = createColumn(continuousViewRefType, "scan", "ref");

         std::tuple<mlir::Value, mlir::DictionaryAttr, mlir::DictionaryAttr> staticAggregateResults;
         std::tuple<mlir::Value, mlir::DictionaryAttr> segmentTreeViewResult;
         if (!distAggrFuncs.empty() && fromBegin && fromEnd) {
            mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, continuousView, columnMapping);
            staticAggregateResults = performAggregation(loc, rewriter, distAggrFuncs, mlir::relalg::OrderedAttributes::fromVec({}), scan, performAggrFuncReduce);
         } else if (!distAggrFuncs.empty()) {
            segmentTreeViewResult = buildSegmentTree(loc, rewriter, distAggrFuncs, continuousView, columnMapping);
         }
         mlir::Value scan = rewriter.create<mlir::subop::ScanRefsOp>(loc, continuousView, referenceDefAttr);
         mlir::Value afterGather = rewriter.create<mlir::subop::GatherOp>(loc, scan, referenceRefAttr, columnMapping);
         mlir::Value afterBegin = rewriter.create<mlir::subop::GetBeginReferenceOp>(loc, afterGather, continuousView, beginReferenceDefAttr);
         mlir::Value afterEnd = rewriter.create<mlir::subop::GetEndReferenceOp>(loc, afterBegin, continuousView, endReferenceDefAttr);
         mlir::Value current = afterEnd;
         mlir::tuples::ColumnRefAttr rangeBegin;
         mlir::tuples::ColumnRefAttr rangeEnd;
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
            current = rewriter.create<mlir::subop::OffsetReferenceBy>(loc, withConst, referenceRefAttr, colManager.createRef(constCol), fromDefAttr);
            rangeBegin = fromRefAttr;
         }
         if (to == 0) {
            rangeEnd = referenceRefAttr;
         } else {
            auto [toDefAttr, toRefAttr] = createColumn(continuousViewRefType, "frame", "to");
            auto [withConst, constCol] = mapIndex(current, rewriter, loc, to);
            current = rewriter.create<mlir::subop::OffsetReferenceBy>(loc, withConst, referenceRefAttr, colManager.createRef(constCol), toDefAttr);
            rangeEnd = toRefAttr;
         }
         assert(rangeBegin && rangeEnd);
         for (auto orderedWindowFn : analyzedWindow.orderedWindowFunctions) {
            current = orderedWindowFn->evaluate(rewriter, loc, current, rangeBegin, rangeEnd, colManager.createRef(&referenceDefAttr.getColumn()));
         }
         if (!distAggrFuncs.empty() && fromBegin && fromEnd) {
            mlir::Value state = std::get<0>(staticAggregateResults);
            mlir::DictionaryAttr stateColumnMapping = std::get<2>(staticAggregateResults);
            auto [referenceDef, referenceRef] = createColumn(mlir::subop::LookupEntryRefType::get(getContext(), state.getType().cast<mlir::subop::LookupAbleState>()), "lookup", "ref");
            mlir::Value afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), current, state, rewriter.getArrayAttr({}), referenceDef);
            current = rewriter.create<mlir::subop::GatherOp>(loc, afterLookup, referenceRef, stateColumnMapping);
         } else if (!distAggrFuncs.empty()) {
            mlir::Value segmentTreeView = std::get<0>(segmentTreeViewResult);
            mlir::DictionaryAttr stateColumnMapping = std::get<1>(segmentTreeViewResult);
            auto [referenceDef, referenceRef] = createColumn(mlir::subop::LookupEntryRefType::get(getContext(), segmentTreeView.getType().cast<mlir::subop::LookupAbleState>()), "lookup", "ref");
            mlir::Value afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(getContext()), current, segmentTreeView, rewriter.getArrayAttr({rangeBegin, rangeEnd}), referenceDef);
            current = rewriter.create<mlir::subop::GatherOp>(loc, afterLookup, referenceRef, stateColumnMapping);
         }
         return current;
      };
      performWindowOp(windowOp, adaptor.getRel(), rewriter, evaluate);
      return success();
   }
};
class AggregationLowering : public OpConversionPattern<mlir::relalg::AggregationOp> {
   public:
   using OpConversionPattern<mlir::relalg::AggregationOp>::OpConversionPattern;
   struct AnalyzedAggregation {
      std::vector<std::pair<mlir::relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distAggrFuncs;
   };

   void analyze(mlir::relalg::AggregationOp aggregationOp, AnalyzedAggregation& analyzedAggregation) const {
      mlir::tuples::ReturnOp terminator = mlir::cast<mlir::tuples::ReturnOp>(aggregationOp.getAggrFunc().front().getTerminator());
      std::unordered_map<mlir::Operation*, std::pair<mlir::relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distinct;
      distinct.insert({nullptr, {mlir::relalg::OrderedAttributes::fromVec({}), {}}});
      for (size_t i = 0; i < aggregationOp.getComputedCols().size(); i++) {
         auto destColumnAttr = aggregationOp.getComputedCols()[i].cast<mlir::tuples::ColumnDefAttr>();
         mlir::Value computedVal = terminator.getResults()[i];
         mlir::Value tupleStream;
         std::shared_ptr<DistAggrFunc> distAggrFunc;
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            tupleStream = aggrFn.getRel();
            auto sourceColumnAttr = aggrFn.getAttr();
            if (aggrFn.getFn() == mlir::relalg::AggrFunc::sum) {
               distAggrFunc = std::make_shared<SumAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::min) {
               distAggrFunc = std::make_shared<MinAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::max) {
               distAggrFunc = std::make_shared<MaxAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::any) {
               distAggrFunc = std::make_shared<AnyAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::count) {
               distAggrFunc = std::make_shared<CountAggrFunc>(destColumnAttr, sourceColumnAttr);
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            tupleStream = countOp.getRel();
            distAggrFunc = std::make_shared<CountStarAggrFunc>(destColumnAttr);
         }

         if (!distinct.count(tupleStream.getDefiningOp())) {
            if (auto projectionOp = mlir::dyn_cast_or_null<mlir::relalg::ProjectionOp>(tupleStream.getDefiningOp())) {
               distinct[tupleStream.getDefiningOp()] = {mlir::relalg::OrderedAttributes::fromRefArr(projectionOp.getCols()), {}};
            }
         }
         distinct.at(tupleStream.getDefiningOp()).second.push_back(distAggrFunc);
      };
      for (auto d : distinct) {
         analyzedAggregation.distAggrFuncs.push_back({d.second.first, d.second.second});
      }
   }

   LogicalResult matchAndRewrite(mlir::relalg::AggregationOp aggregationOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      AnalyzedAggregation analyzedAggregation;
      analyze(aggregationOp, analyzedAggregation);
      auto keyAttributes = mlir::relalg::OrderedAttributes::fromRefArr(aggregationOp.getGroupByColsAttr());
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
            tree = rewriter.create<mlir::relalg::ProjectionOp>(aggregationOp->getLoc(), mlir::relalg::SetSemantic::distinct, tree, mlir::relalg::OrderedAttributes::fromVec(projectionAttrs).getArrayAttr(rewriter.getContext()));
         }
         auto partialResult = performAggregation(aggregationOp->getLoc(), rewriter, x.second, keyAttributes, tree, performAggrFuncReduce);
         subResults.push_back(partialResult);
      }
      if (subResults.empty()) {
         //handle the case that aggregation is only used for distinct projection
         rewriter.replaceOpWithNewOp<mlir::relalg::ProjectionOp>(aggregationOp, mlir::relalg::SetSemantic::distinct, adaptor.getRel(), aggregationOp.getGroupByCols());
         return success();
      }

      mlir::Value newStream = rewriter.create<mlir::subop::ScanOp>(aggregationOp->getLoc(), std::get<0>(subResults.at(0)), std::get<1>(subResults.at(0)));
      ; //= scan %state of subresult 0
      for (size_t i = 1; i < subResults.size(); i++) {
         mlir::Value state = std::get<0>(subResults.at(i));
         mlir::DictionaryAttr stateColumnMapping = std::get<2>(subResults.at(i));
         auto [referenceDef, referenceRef] = createColumn(mlir::subop::LookupEntryRefType::get(getContext(), state.getType().cast<mlir::subop::LookupAbleState>()), "lookup", "ref");
         mlir::Value afterLookup = rewriter.create<mlir::subop::LookupOp>(aggregationOp->getLoc(), mlir::tuples::TupleStreamType::get(getContext()), newStream, state, aggregationOp.getGroupByCols(), referenceDef);
         newStream = rewriter.create<mlir::subop::GatherOp>(aggregationOp->getLoc(), afterLookup, referenceRef, stateColumnMapping);
      }

      rewriter.replaceOp(aggregationOp, newStream);

      return success();
   }
};
class GroupJoinLowering : public OpConversionPattern<mlir::relalg::GroupJoinOp> {
   public:
   using OpConversionPattern<mlir::relalg::GroupJoinOp>::OpConversionPattern;
   struct AnalyzedAggregation {
      std::vector<std::pair<mlir::relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distAggrFuncs;
   };

   void analyze(mlir::relalg::GroupJoinOp groupJoinOp, AnalyzedAggregation& analyzedAggregation) const {
      mlir::tuples::ReturnOp terminator = mlir::cast<mlir::tuples::ReturnOp>(groupJoinOp.getAggrFunc().front().getTerminator());
      std::unordered_map<mlir::Operation*, std::pair<mlir::relalg::OrderedAttributes, std::vector<std::shared_ptr<DistAggrFunc>>>> distinct;
      distinct.insert({nullptr, {mlir::relalg::OrderedAttributes::fromVec({}), {}}});
      for (size_t i = 0; i < groupJoinOp.getComputedCols().size(); i++) {
         auto destColumnAttr = groupJoinOp.getComputedCols()[i].cast<mlir::tuples::ColumnDefAttr>();
         mlir::Value computedVal = terminator.getResults()[i];
         mlir::Value tupleStream;
         std::shared_ptr<DistAggrFunc> distAggrFunc;
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            tupleStream = aggrFn.getRel();
            auto sourceColumnAttr = aggrFn.getAttr();
            if (aggrFn.getFn() == mlir::relalg::AggrFunc::sum) {
               distAggrFunc = std::make_shared<SumAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::min) {
               distAggrFunc = std::make_shared<MinAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::max) {
               distAggrFunc = std::make_shared<MaxAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::any) {
               distAggrFunc = std::make_shared<AnyAggrFunc>(destColumnAttr, sourceColumnAttr);
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::count) {
               distAggrFunc = std::make_shared<CountAggrFunc>(destColumnAttr, sourceColumnAttr);
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            tupleStream = countOp.getRel();
            distAggrFunc = std::make_shared<CountStarAggrFunc>(destColumnAttr);
         }

         if (!distinct.count(tupleStream.getDefiningOp())) {
            if (auto projectionOp = mlir::dyn_cast_or_null<mlir::relalg::ProjectionOp>(tupleStream.getDefiningOp())) {
               distinct[tupleStream.getDefiningOp()] = {mlir::relalg::OrderedAttributes::fromRefArr(projectionOp.getCols()), {}};
            }
         }
         distinct.at(tupleStream.getDefiningOp()).second.push_back(distAggrFunc);
      };
      for (auto d : distinct) {
         analyzedAggregation.distAggrFuncs.push_back({d.second.first, d.second.second});
      }
   }

   LogicalResult matchAndRewrite(mlir::relalg::GroupJoinOp groupJoinOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      AnalyzedAggregation analyzedAggregation;
      analyze(groupJoinOp, analyzedAggregation);
      if (analyzedAggregation.distAggrFuncs.size() != 1) return failure();
      if (!analyzedAggregation.distAggrFuncs[0].first.getAttrs().empty()) return failure();
      std::vector<std::shared_ptr<DistAggrFunc>> distAggrFuncs = analyzedAggregation.distAggrFuncs[0].second;

      auto* context = rewriter.getContext();
      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto loc = groupJoinOp->getLoc();
      auto storedColumns = groupJoinOp.getUsedColumns().intersect(groupJoinOp.getChildren()[0].getAvailableColumns());
      for (auto z : llvm::zip(groupJoinOp.getLeftCols(), groupJoinOp.getRightCols())) {
         auto leftType = std::get<0>(z).cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
         auto rightType = std::get<1>(z).cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
         if (leftType == rightType) {
            storedColumns.remove(mlir::relalg::ColumnSet::from(&std::get<0>(z).cast<mlir::tuples::ColumnRefAttr>().getColumn()));
         }
      }
      auto additionalColumns = mlir::relalg::OrderedAttributes::fromColumns(storedColumns);
      Block* initialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initialValueBlock);
         std::vector<mlir::Value> defaultValues;
         if (groupJoinOp.getBehavior() == mlir::relalg::GroupJoinBehavior::inner) {
            defaultValues.push_back(rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1));
         }
         for (auto* c : additionalColumns.getAttrs()) {
            defaultValues.push_back(rewriter.create<mlir::util::UndefOp>(loc, c->type));
         }
         for (auto aggrFn : distAggrFuncs) {
            defaultValues.push_back(aggrFn->createDefaultValue(rewriter, loc));
         }
         rewriter.create<mlir::tuples::ReturnOp>(loc, defaultValues);
      }
      std::vector<mlir::Attribute> names;
      std::vector<mlir::Attribute> reduceNames;
      std::vector<mlir::Attribute> types;
      std::vector<NamedAttribute> defMapping;
      std::vector<NamedAttribute> additionalColsDefMapping;
      std::vector<NamedAttribute> additionalColsRefMapping;
      mlir::tuples::ColumnRefAttr marker;
      std::string markerMember;
      if (groupJoinOp.getBehavior() == mlir::relalg::GroupJoinBehavior::inner) {
         markerMember = getUniqueMember(getContext(), "gjvalmarker");
         names.push_back(rewriter.getStringAttr(markerMember));
         types.push_back(mlir::TypeAttr::get(rewriter.getI1Type()));
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
         additionalColsRefMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createRef(c)));
      }

      for (auto aggrFn : distAggrFuncs) {
         auto memberName = getUniqueMember(getContext(), "aggrval");
         names.push_back(rewriter.getStringAttr(memberName));
         reduceNames.push_back(rewriter.getStringAttr(memberName));
         types.push_back(mlir::TypeAttr::get(aggrFn->getStateType()));
         auto def = aggrFn->getDestAttribute();
         defMapping.push_back(rewriter.getNamedAttr(memberName, def));
      }
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, names), mlir::ArrayAttr::get(context, types));
      auto leftKeys = mlir::relalg::OrderedAttributes::fromRefArr(groupJoinOp.getLeftCols());
      auto rightKeys = mlir::relalg::OrderedAttributes::fromRefArr(groupJoinOp.getRightCols());

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
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateType = mlir::subop::MapType::get(rewriter.getContext(), keyMembers, stateMembers);

      auto createOp = rewriter.create<mlir::subop::GenericCreateOp>(loc, stateType);
      auto state = createOp.getRes();
      auto [referenceDefAttr, referenceRefAttr] = createColumn(mlir::subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto lookupOp = rewriter.create<mlir::subop::LookupOrInsertOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getLeft(), state, groupJoinOp.getLeftCols(), referenceDefAttr);
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
            rewriter.create<mlir::tuples::ReturnOp>(loc, compared);
         }
      }
      rewriter.create<mlir::subop::ScatterOp>(loc, lookupOp, referenceRefAttr, rewriter.getDictionaryAttr(additionalColsRefMapping));

      auto [aggrDef, aggrRef] = createColumn(mlir::subop::OptionalType::get(getContext(), mlir::subop::LookupEntryRefType::get(context, stateType)), "lookup", "ref");
      auto [unwrappedAggrDef, unwrappedAggrRef] = createColumn(mlir::subop::LookupEntryRefType::get(context, stateType), "lookup", "ref");
      auto aggrLookupOp = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRight(), state, groupJoinOp.getRightCols(), aggrDef);
      {
         mlir::Block* equalBlock = new Block;
         aggrLookupOp.getEqFn().push_back(equalBlock);
         equalBlock->addArguments(keyTypes, locations);
         equalBlock->addArguments(otherKeyTypes, locations);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(equalBlock);
            mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
            rewriter.create<mlir::tuples::ReturnOp>(loc, compared);
         }
      }
      mlir::Value unwrap = rewriter.create<mlir::subop::UnwrapOptionalRefOp>(loc, aggrLookupOp.getRes(), aggrRef, unwrappedAggrDef);
      std::vector<mlir::Attribute> renameLeftDefs;
      std::vector<mlir::Attribute> renameRightDefs;
      for (auto z : llvm::zip(groupJoinOp.getLeftCols(), groupJoinOp.getRightCols())) {
         auto rightAttr = std::get<1>(z).cast<mlir::tuples::ColumnRefAttr>();
         auto leftAttr = std::get<0>(z).cast<mlir::tuples::ColumnRefAttr>();
         if (!storedColumns.contains(&leftAttr.getColumn())) {
            renameLeftDefs.push_back(mlir::tuples::ColumnDefAttr::get(getContext(), leftAttr.getName(), leftAttr.getColumnPtr(), rewriter.getArrayAttr(rightAttr)));
         }
         renameRightDefs.push_back(mlir::tuples::ColumnDefAttr::get(getContext(), rightAttr.getName(), rightAttr.getColumnPtr(), rewriter.getArrayAttr(leftAttr)));
      }
      if (!additionalColsDefMapping.empty()) {
         unwrap = rewriter.create<mlir::subop::GatherOp>(loc, unwrap, unwrappedAggrRef, rewriter.getDictionaryAttr(additionalColsDefMapping));
      }
      unwrap = rewriter.create<mlir::subop::RenamingOp>(loc, unwrap, rewriter.getArrayAttr(renameLeftDefs));
      auto filtered = translateSelection(unwrap, groupJoinOp.getPredicate(), rewriter, loc);
      if (groupJoinOp.getBehavior() == mlir::relalg::GroupJoinBehavior::inner) {
         auto [withTrueCol, trueCol] = mapBool(filtered, rewriter, loc, true);
         rewriter.create<mlir::subop::ScatterOp>(loc, withTrueCol, unwrappedAggrRef, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerMember, colManager.createRef(trueCol))));
      }
      if (!groupJoinOp.getMappedCols().empty()) {
         auto mapOp2 = rewriter.create<mlir::subop::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), filtered, groupJoinOp.getMappedCols());
         assert(safelyMoveRegion(rewriter, groupJoinOp.getMapFunc(), mapOp2.getFn()).succeeded());
         filtered = mapOp2;
      }
      performAggrFuncReduce(loc, rewriter, distAggrFuncs, unwrappedAggrRef, filtered, reduceNames, defMapping);

      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, state, rewriter.getDictionaryAttr(defMapping));
      if (groupJoinOp.getBehavior() == mlir::relalg::GroupJoinBehavior::inner) {
         scan = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(marker));
      }
      rewriter.replaceOpWithNewOp<mlir::subop::RenamingOp>(groupJoinOp, scan, rewriter.getArrayAttr(renameRightDefs));
      return success();
   }
};

class NestedLowering : public OpConversionPattern<mlir::relalg::NestedOp> {
   public:
   using OpConversionPattern<mlir::relalg::NestedOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::NestedOp nestedOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto* b = &nestedOp.getNestedFn().front();
      auto* terminator = b->getTerminator();
      auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(terminator);
      rewriter.inlineBlockBefore(b, &*rewriter.getInsertionPoint(), adaptor.getInputs());
      {
         auto* b2 = new mlir::Block;
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(b2);
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc());
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

class TrackTuplesLowering : public OpConversionPattern<mlir::relalg::TrackTuplesOP> {
   public:
   using OpConversionPattern<mlir::relalg::TrackTuplesOP>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::TrackTuplesOP trackTuplesOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = trackTuplesOp->getLoc();

      // Create counter as single i64 state initialized to 0
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto [counterState, counterName] = createCounterState(rewriter, loc);
      auto referenceDefAttr = colManager.createDef(colManager.getUniqueScope("lookup"), "ref");
      referenceDefAttr.getColumn().type = mlir::subop::LookupEntryRefType::get(rewriter.getContext(), counterState.getType().cast<mlir::subop::LookupAbleState>());
      auto lookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.getRel(), counterState, rewriter.getArrayAttr({}), referenceDefAttr);

      // Create reduce operation that increases counter for each seen tuple
      auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(loc, lookup, colManager.createRef(&referenceDefAttr.getColumn()), rewriter.getArrayAttr({}), rewriter.getArrayAttr({rewriter.getStringAttr(counterName)}));
      mlir::Block* reduceBlock = new Block;
      auto counter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(reduceBlock);
         auto one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
         mlir::Value updatedCounter = rewriter.create<mlir::arith::AddIOp>(loc, counter, one);
         rewriter.create<mlir::tuples::ReturnOp>(loc, updatedCounter);
      }
      reduceOp.getRegion().push_back(reduceBlock);

      // Saves counter state to execution context
      rewriter.create<mlir::subop::SetTrackedCountOp>(loc, counterState, adaptor.getResultId(), counterName);

      rewriter.eraseOp(trackTuplesOp);
      return mlir::success();
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
   target.addLegalDialect<arith::ArithDialect>();
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