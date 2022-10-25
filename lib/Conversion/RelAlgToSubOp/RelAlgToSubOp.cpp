#include "mlir-support/parsing.h"
#include "mlir/Conversion/RelAlgToSubOp/OrderedAttributes.h"
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
#include "mlir/IR/BlockAndValueMapping.h"
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
   return name + "n" + std::to_string(counts[name]++);
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
         required.insert(mlir::relalg::ColumnSet::fromArrayAttr(materializeOp.cols()));
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
      std::string scanDescription = R"({ "table": ")" + tableName + R"(", "columns": [ )";
      bool first = true;
      for (auto namedAttr : baseTableOp.columnsAttr().getValue()) {
         auto identifier = namedAttr.getName();
         auto attr = namedAttr.getValue();
         auto attrDef = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
         if (required.contains(&attrDef.getColumn())) {
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
static mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::ValueRange leftUnpacked, mlir::ValueRange rightUnpacked, mlir::Location loc) {
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
   mapOp.fn().push_back(mapBlock);
   return mapOp.result();
}
static mlir::Value translateSelection(mlir::Value stream, mlir::Region& predicate, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto terminator = mlir::cast<mlir::tuples::ReturnOp>(predicate.front().getTerminator());
   if (terminator.results().empty()) {
      auto [markAttrDef, markAttrRef] = createColumn(rewriter.getI1Type(), "map", "predicate");
      auto mapped = map(stream, rewriter, loc, rewriter.getArrayAttr(markAttrDef), [](mlir::ConversionPatternRewriter& rewriter, mlir::Value, mlir::Location loc) {
         return std::vector<mlir::Value>{rewriter.create<mlir::db::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1))};
      });
      return rewriter.create<mlir::subop::FilterOp>(loc, mapped, mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr(markAttrRef));

   } else {
      auto& predicateBlock = predicate.front();
      if (auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(predicateBlock.getTerminator())) {
         mlir::Value matched = returnOp.results()[0];
         std::vector<std::pair<int, mlir::Value>> conditions;
         if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(matched.getDefiningOp())) {
            for (auto c : andOp.vals()) {
               int p = 1000;
               if (auto* defOp = c.getDefiningOp()) {
                  if (auto betweenOp = mlir::dyn_cast_or_null<mlir::db::BetweenOp>(defOp)) {
                     auto t = betweenOp.val().getType();
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
               mlir::BlockAndValueMapping mapping;
               mapping.map(predicateBlock.getArgument(0), tuple);
               auto helperOp = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(0));
               mlir::relalg::detail::inlineOpIntoBlock(c.second.getDefiningOp(), c.second.getDefiningOp()->getParentOp(), b.getInsertionBlock(), mapping, helperOp);
               b.eraseOp(helperOp);
               return {mapping.lookupOrNull(c.second)};
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

static mlir::Block* createCompareBlock(std::vector<mlir::Type> keyTypes, ConversionPatternRewriter& rewriter, mlir::Location loc) {
   mlir::Block* equalBlock = new Block;
   std::vector<mlir::Location> locations(keyTypes.size(), loc);
   equalBlock->addArguments(keyTypes, locations);
   equalBlock->addArguments(keyTypes, locations);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(equalBlock);
      mlir::Value compared = compareKeys(rewriter, equalBlock->getArguments().drop_back(keyTypes.size()), equalBlock->getArguments().drop_front(keyTypes.size()), loc);
      rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc(), compared);
   }
   return equalBlock;
}

class ProjectionDistinctLowering : public OpConversionPattern<mlir::relalg::ProjectionOp> {
   public:
   using OpConversionPattern<mlir::relalg::ProjectionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::ProjectionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.set_semantic() != mlir::relalg::SetSemantic::distinct) return failure();
      auto* context = getContext();
      auto loc = projectionOp->getLoc();

      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      for (auto x : projectionOp.cols()) {
         auto ref = x.cast<mlir::tuples::ColumnRefAttr>();
         auto memberName = getUniqueMember("keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {}), mlir::ArrayAttr::get(context, {}));

      auto stateType = mlir::subop::HashMapType::get(rewriter.getContext(), keyMembers, stateMembers);
      mlir::Value state = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{}, 0);
      auto [referenceDef, referenceRef] = createColumn(mlir::subop::EntryRefType::get(context, stateType), "lookup", "ref");
      auto lookupOp = rewriter.create<mlir::subop::LookupOrInsertOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), adaptor.rel(), state, projectionOp.cols(), referenceDef);
      auto* initialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initialValueBlock);
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc());
      }
      lookupOp.initFn().push_back(initialValueBlock);
      lookupOp.eqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

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
   MaterializationHelper(mlir::ArrayAttr columnAttrs, mlir::MLIRContext* context) : context(context) {
      size_t i = 0;
      for (auto columnAttr : columnAttrs) {
         auto columnDef = columnAttr.cast<mlir::tuples::ColumnDefAttr>();
         auto* x = &columnDef.getColumn();
         types.push_back(mlir::TypeAttr::get(x->type));
         colToMemberPos[x] = i++;
         std::string name = getUniqueMember("member");
         auto nameAttr = mlir::StringAttr::get(context, name);
         names.push_back(nameAttr);
         defMapping.push_back(mlir::NamedAttribute(nameAttr, columnDef));
         refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(x)));
      }
   }
   mlir::Type getType(size_t i) {
      return types.at(i).cast<mlir::TypeAttr>().getValue();
   }

   std::string addFlag(mlir::tuples::ColumnDefAttr flagAttrDef) {
      auto i1Type = mlir::IntegerType::get(context, 1);
      types.push_back(mlir::TypeAttr::get(i1Type));
      colToMemberPos[&flagAttrDef.getColumn()] = names.size();
      std::string name = getUniqueMember("flag");
      auto nameAttr = mlir::StringAttr::get(context, name);
      names.push_back(nameAttr);
      defMapping.push_back(mlir::NamedAttribute(nameAttr, flagAttrDef));
      refMapping.push_back(mlir::NamedAttribute(nameAttr, context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(&flagAttrDef.getColumn())));
      return name;
   }
   mlir::subop::StateMembersAttr createStateMembersAttr() {
      //assert(!names.empty());
      return mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, names), mlir::ArrayAttr::get(context, types));
   }

   mlir::DictionaryAttr createStateColumnMapping() {
      return mlir::DictionaryAttr::get(context, defMapping);
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
      MaterializationHelper helper(constRelationOp.columns(), getContext());
      auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
         auto creationOp = rewriter.create<mlir::subop::CreateOp>(constRelationOp->getLoc(), vectorType, mlir::Attribute{}, constRelationOp.values().size());
         vector = creationOp.res();
         size_t rowId = 0;
         for (auto rowAttr : constRelationOp.valuesAttr()) {
            auto* currBlock = new Block;
            creationOp.initFn()[rowId++].push_back(currBlock);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(currBlock);
            auto row = rowAttr.cast<ArrayAttr>();
            std::vector<Value> values;
            size_t i = 0;
            for (auto entryAttr : row.getValue()) {
               if (helper.getType(i).isa<mlir::db::NullableType>() && entryAttr.isa<mlir::UnitAttr>()) {
                  auto entryVal = rewriter.create<mlir::db::NullOp>(constRelationOp->getLoc(), helper.getType(i));
                  values.push_back(entryVal);
                  i++;
               } else {
                  mlir::Value entryVal = rewriter.create<mlir::db::ConstantOp>(constRelationOp->getLoc(), getBaseType(helper.getType(i)), entryAttr);
                  if (helper.getType(i).isa<mlir::db::NullableType>()) {
                     entryVal = rewriter.create<mlir::db::AsNullableOp>(constRelationOp->getLoc(), helper.getType(i), entryVal);
                  }
                  values.push_back(entryVal);
                  i++;
               }
            }

            rewriter.create<mlir::tuples::ReturnOp>(constRelationOp->getLoc(), values);
         }
      }
      rewriter.replaceOpWithNewOp<mlir::subop::ScanOp>(constRelationOp, vector, helper.createStateColumnMapping());
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
static mlir::Value mapColsToNull(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping,mlir::relalg::ColumnSet excluded={}) {
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
         if(excluded.contains(&fromExisting.getColumn())) continue;
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
static mlir::Value mapColsToNullable(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, size_t exisingOffset = 0,mlir::relalg::ColumnSet excluded={}) {
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
         if(excluded.contains(&fromExisting.getColumn())) continue;
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
   mapOp.fn().push_back(mapBlock);
   return mapOp.result();
}
class UnionAllLowering : public OpConversionPattern<mlir::relalg::UnionOp> {
   public:
   using OpConversionPattern<mlir::relalg::UnionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::UnionOp unionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (unionOp.set_semantic() != mlir::relalg::SetSemantic::all) return failure();
      auto loc = unionOp->getLoc();
      mlir::Value left = mapColsToNullable(adaptor.left(), rewriter, loc, unionOp.mapping(), 0);
      mlir::Value right = mapColsToNullable(adaptor.right(), rewriter, loc, unionOp.mapping(), 1);
      rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(unionOp, mlir::ValueRange({left, right}));
      return success();
   }
};

class UnionDistinctLowering : public OpConversionPattern<mlir::relalg::UnionOp> {
   public:
   using OpConversionPattern<mlir::relalg::UnionOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::UnionOp projectionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (projectionOp.set_semantic() != mlir::relalg::SetSemantic::distinct) return failure();
      auto* context = getContext();
      auto loc = projectionOp->getLoc();

      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> keyNames;
      std::vector<mlir::Attribute> keyTypesAttr;
      std::vector<mlir::Type> keyTypes;
      std::vector<NamedAttribute> defMapping;
      std::vector<mlir::Attribute> refs;
      for (auto x : projectionOp.mapping()) {
         auto ref = x.cast<mlir::tuples::ColumnDefAttr>();
         refs.push_back(colManager.createRef(&ref.getColumn()));
         auto memberName = getUniqueMember("keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, {}), mlir::ArrayAttr::get(context, {}));

      auto stateType = mlir::subop::HashMapType::get(rewriter.getContext(), keyMembers, stateMembers);
      mlir::Value state = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{}, 0);
      auto [referenceDef, referenceRef] = createColumn(mlir::subop::EntryRefType::get(context, stateType), "lookup", "ref");
      mlir::Value left = mapColsToNullable(adaptor.left(), rewriter, loc, projectionOp.mapping(), 0);
      mlir::Value right = mapColsToNullable(adaptor.right(), rewriter, loc, projectionOp.mapping(), 1);
      auto lookupOpLeft = rewriter.create<mlir::subop::LookupOrInsertOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), left, state, rewriter.getArrayAttr(refs), referenceDef);
      auto lookupOpRight = rewriter.create<mlir::subop::LookupOrInsertOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), right, state, rewriter.getArrayAttr(refs), referenceDef);
      auto* leftInitialValueBlock = new Block;
      auto* rightInitialValueBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(leftInitialValueBlock);
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc());
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rightInitialValueBlock);
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc());
      }
      lookupOpLeft.initFn().push_back(leftInitialValueBlock);
      lookupOpRight.initFn().push_back(rightInitialValueBlock);
      lookupOpLeft.eqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));
      lookupOpRight.eqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

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
         auto memberName = getUniqueMember("keyval");
         keyNames.push_back(rewriter.getStringAttr(memberName));
         keyTypesAttr.push_back(mlir::TypeAttr::get(ref.getColumn().type));
         keyTypes.push_back((ref.getColumn().type));
         defMapping.push_back(rewriter.getNamedAttr(memberName, colManager.createDef(&ref.getColumn())));
      }
      std::string counterName1 = getUniqueMember("counter");
      std::string counterName2 = getUniqueMember("counter");
      auto [counter1Def, counter1Ref] = createColumn(rewriter.getI64Type(), "set", "counter");
      auto [counter2Def, counter2Ref] = createColumn(rewriter.getI64Type(), "set", "counter");
      defMapping.push_back(rewriter.getNamedAttr(counterName1, counter1Def));
      defMapping.push_back(rewriter.getNamedAttr(counterName2, counter2Def));
      std::vector<mlir::Attribute> counterNames = {rewriter.getStringAttr(counterName1), rewriter.getStringAttr(counterName2)};
      std::vector<mlir::Attribute> counterTypes = {mlir::TypeAttr::get(rewriter.getI64Type()), mlir::TypeAttr::get(rewriter.getI64Type())};
      auto keyMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, keyNames), mlir::ArrayAttr::get(context, keyTypesAttr));
      auto stateMembers = mlir::subop::StateMembersAttr::get(context, mlir::ArrayAttr::get(context, counterNames), mlir::ArrayAttr::get(context, counterTypes));

      auto stateType = mlir::subop::HashMapType::get(rewriter.getContext(), keyMembers, stateMembers);
      mlir::Value state = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{}, 0);
      auto [referenceDef, referenceRef] = createColumn(mlir::subop::EntryRefType::get(context, stateType), "lookup", "ref");
      mlir::Value left = mapColsToNullable(operands[0], rewriter, loc, mapping, 0);
      mlir::Value right = mapColsToNullable(operands[1], rewriter, loc, mapping, 1);
      auto lookupOpLeft = rewriter.create<mlir::subop::LookupOrInsertOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), left, state, rewriter.getArrayAttr(refs), referenceDef);
      auto* leftInitialValueBlock = new Block;
      auto* rightInitialValueBlock = new Block;

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(leftInitialValueBlock);
         mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc(), mlir::ValueRange({zeroI64, zeroI64}));
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(rightInitialValueBlock);
         mlir::Value zeroI64 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
         rewriter.create<mlir::tuples::ReturnOp>(rewriter.getUnknownLoc(), mlir::ValueRange({zeroI64, zeroI64}));
      }
      lookupOpLeft.initFn().push_back(leftInitialValueBlock);
      lookupOpLeft.eqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));
      {
         auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(rewriter.getUnknownLoc(), lookupOpLeft, referenceRef, rewriter.getArrayAttr({}), rewriter.getArrayAttr({counterNames[0]}));
         mlir::Block* reduceBlock = new Block;
         mlir::Value currCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceBlock);
            mlir::Value constOne = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
            rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({rewriter.create<mlir::arith::AddIOp>(loc, currCounter, constOne)}));
         }
         reduceOp.region().push_back(reduceBlock);
      }
      auto lookupOpRight = rewriter.create<mlir::subop::LookupOrInsertOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), right, state, rewriter.getArrayAttr(refs), referenceDef);
      lookupOpRight.initFn().push_back(rightInitialValueBlock);
      lookupOpRight.eqFn().push_back(createCompareBlock(keyTypes, rewriter, loc));

      {
         auto reduceOp = rewriter.create<mlir::subop::ReduceOp>(rewriter.getUnknownLoc(), lookupOpRight, referenceRef, rewriter.getArrayAttr({}), rewriter.getArrayAttr({counterNames[1]}));
         mlir::Block* reduceBlock = new Block;
         mlir::Value currCounter = reduceBlock->addArgument(rewriter.getI64Type(), loc);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceBlock);
            mlir::Value constOne = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
            rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange({rewriter.create<mlir::arith::AddIOp>(loc, currCounter, constOne)}));
         }
         reduceOp.region().push_back(reduceBlock);
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
         scan = rewriter.create<mlir::subop::RepeatOp>(loc, scan, repeatRef);
      }
      rewriter.replaceOp(op, scan);
   }
};
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
   auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute(), 1);
   createOp.initFn().front().push_back(initialValueBlock);

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
   auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute(), 1);
   createOp.initFn().front().push_back(initialValueBlock);

   return {createOp.res(), memberName};
}

static mlir::Value translateNLJ(mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   MaterializationHelper helper(columns, rewriter.getContext());
   auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector = rewriter.create<mlir::subop::CreateOp>(loc, vectorType, mlir::Attribute(), 0);
   rewriter.create<mlir::subop::MaterializeOp>(loc, right, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), left, rewriter.getArrayAttr({}));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   nestedMapOp.region().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      mlir::Value scan = rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping());
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, scan, tuple);
      rewriter.create<mlir::tuples::ReturnOp>(loc, fn(combined, rewriter));
   }
   return nestedMapOp.res();
}
static mlir::Value translateHJ(mlir::Value left, mlir::Value right, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   MaterializationHelper helper(columns, rewriter.getContext());
   auto hashMember = getUniqueMember("hash");
   auto [hashDef, hashRef] = createColumn(rewriter.getIndexType(), "hj", "hash");
   auto lmmType = mlir::subop::LazyMultiMapType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(hashMember)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getIndexType())})), helper.createStateMembersAttr());
   mlir::Value lmm = rewriter.create<mlir::subop::CreateOp>(loc, lmmType, mlir::Attribute(), 0);
   right = map(right, rewriter, loc, rewriter.getArrayAttr(hashDef), [&](mlir::OpBuilder& rewriter, mlir::Value tuple, mlir::Location loc) {
      std::vector<mlir::Value> values;
      for (auto hashAttr : hashRight) {
         auto hashAttrRef = hashAttr.cast<mlir::tuples::ColumnRefAttr>();
         values.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, hashAttrRef.getColumn().type, hashAttrRef, tuple));
      }
      mlir::Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.create<mlir::util::PackOp>(loc, values));
      return std::vector<mlir::Value>({hashed});
   });
   rewriter.create<mlir::subop::MaterializeOp>(loc, right, lmm, helper.createColumnstateMapping({rewriter.getNamedAttr(hashMember, hashRef)}));
   rewriter.create<mlir::subop::MaintainOp>(loc, lmm, "finalize");
   auto entryRefType = mlir::subop::EntryRefType::get(rewriter.getContext(), lmmType);
   auto entryRefListType = mlir::subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   left = map(left, rewriter, loc, rewriter.getArrayAttr(hashDef), [&](mlir::OpBuilder& rewriter, mlir::Value tuple, mlir::Location loc) {
      std::vector<mlir::Value> values;
      for (auto hashAttr : hashLeft) {
         auto hashAttrRef = hashAttr.cast<mlir::tuples::ColumnRefAttr>();
         values.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, hashAttrRef.getColumn().type, hashAttrRef, tuple));
      }
      mlir::Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.create<mlir::util::PackOp>(loc, values));
      return std::vector<mlir::Value>({hashed});
   });
   auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), left, lmm, rewriter.getArrayAttr({hashRef}), listDef);

   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);
   nestedMapOp.region().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      auto [markerState, markerName] = createMarkerState(rewriter, loc);
      mlir::Value scan = rewriter.create<mlir::subop::ScanListOp>(loc, list, entryDef);
      mlir::Value gathered = rewriter.create<mlir::subop::GatherOp>(loc, scan, entryRef, helper.createStateColumnMapping());
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, gathered, tuple);
      rewriter.create<mlir::tuples::ReturnOp>(loc, fn(combined, rewriter));
   }
   return nestedMapOp.res();
}
static mlir::Value translateNL(mlir::Value left, mlir::Value right, bool useHash, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, std::function<mlir::Value(mlir::Value, mlir::ConversionPatternRewriter& rewriter)> fn) {
   if (useHash) {
      return translateHJ(left, right, hashLeft, hashRight, columns, rewriter, loc, fn);
   } else {
      return translateNLJ(left, right, columns, rewriter, loc, fn);
   }
}

static std::pair<mlir::Value, mlir::Value> translateNLJWithMarker(mlir::Value left, mlir::Value right, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr, std::string markerName)> fn) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   MaterializationHelper helper(columns, rewriter.getContext());
   auto flagMember = helper.addFlag(markerDefAttr);
   auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
   mlir::Value vector = rewriter.create<mlir::subop::CreateOp>(loc, vectorType, mlir::Attribute(), 0);
   left = mapBool(left, rewriter, loc, false, &markerDefAttr.getColumn());
   rewriter.create<mlir::subop::MaterializeOp>(loc, left, vector, helper.createColumnstateMapping());
   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), right, rewriter.getArrayAttr({}));
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
      auto res = fn(combined,tuple, rewriter, colManager.createRef(&referenceDefAttr.getColumn()), flagMember);
      if (res) {
         rewriter.create<mlir::tuples::ReturnOp>(loc, res);
      } else {
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
   }
   return {nestedMapOp.res(), rewriter.create<mlir::subop::ScanOp>(loc, vector, helper.createStateColumnMapping())};
}
static std::pair<mlir::Value, mlir::Value> translateHJWithMarker(mlir::Value left, mlir::Value right, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr, std::string markerName)> fn) {
   MaterializationHelper helper(columns, rewriter.getContext());
   auto flagMember = helper.addFlag(markerDefAttr);
   auto hashMember = getUniqueMember("hash");
   auto [hashDef, hashRef] = createColumn(rewriter.getIndexType(), "hj", "hash");
   auto lmmType = mlir::subop::LazyMultiMapType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(hashMember)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getIndexType())})), helper.createStateMembersAttr());
   mlir::Value lmm = rewriter.create<mlir::subop::CreateOp>(loc, lmmType, mlir::Attribute(), 0);
   left = mapBool(left, rewriter, loc, false, &markerDefAttr.getColumn());
   left = map(left, rewriter, loc, rewriter.getArrayAttr(hashDef), [&](mlir::OpBuilder& rewriter, mlir::Value tuple, mlir::Location loc) {
      std::vector<mlir::Value> values;
      for (auto hashAttr : hashLeft) {
         auto hashAttrRef = hashAttr.cast<mlir::tuples::ColumnRefAttr>();
         values.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, hashAttrRef.getColumn().type, hashAttrRef, tuple));
      }
      mlir::Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.create<mlir::util::PackOp>(loc, values));
      return std::vector<mlir::Value>({hashed});
   });
   rewriter.create<mlir::subop::MaterializeOp>(loc, left, lmm, helper.createColumnstateMapping({rewriter.getNamedAttr(hashMember, hashRef)}));
   rewriter.create<mlir::subop::MaintainOp>(loc, lmm, "finalize");
   auto entryRefType = mlir::subop::EntryRefType::get(rewriter.getContext(), lmmType);
   auto entryRefListType = mlir::subop::ListType::get(rewriter.getContext(), entryRefType);
   auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
   auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
   right = map(right, rewriter, loc, rewriter.getArrayAttr(hashDef), [&](mlir::OpBuilder& rewriter, mlir::Value tuple, mlir::Location loc) {
      std::vector<mlir::Value> values;
      for (auto hashAttr : hashRight) {
         auto hashAttrRef = hashAttr.cast<mlir::tuples::ColumnRefAttr>();
         values.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, hashAttrRef.getColumn().type, hashAttrRef, tuple));
      }
      mlir::Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.create<mlir::util::PackOp>(loc, values));
      return std::vector<mlir::Value>({hashed});
   });
   auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), right, lmm, rewriter.getArrayAttr({hashRef}), listDef);

   auto nestedMapOp = rewriter.create<mlir::subop::NestedMapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), afterLookup, rewriter.getArrayAttr(listRef));
   auto* b = new Block;
   mlir::Value tuple = b->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
   mlir::Value list = b->addArgument(entryRefListType, loc);

   nestedMapOp.region().push_back(b);
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(b);
      mlir::Value scan = rewriter.create<mlir::subop::ScanListOp>(loc, list, entryDef);
      mlir::Value gathered = rewriter.create<mlir::subop::GatherOp>(loc, scan, entryRef, helper.createStateColumnMapping());
      mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, gathered, tuple);
      auto res = fn(combined, tuple,rewriter, entryRef, flagMember);
      if (res) {
         rewriter.create<mlir::tuples::ReturnOp>(loc, res);
      } else {
         rewriter.create<mlir::tuples::ReturnOp>(loc);
      }
   }
   return {nestedMapOp.res(), rewriter.create<mlir::subop::ScanOp>(loc, lmm, helper.createStateColumnMapping())};
}
static std::pair<mlir::Value, mlir::Value> translateNLWithMarker(mlir::Value left, mlir::Value right, bool useHash, mlir::ArrayAttr hashLeft, mlir::ArrayAttr hashRight, mlir::relalg::ColumnSet columns, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::tuples::ColumnDefAttr markerDefAttr, std::function<mlir::Value(mlir::Value,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr, std::string markerName)> fn) {
   if (useHash) {
      return translateHJWithMarker(left, right, hashLeft, hashRight, columns, rewriter, loc, markerDefAttr, fn);
   } else {
      return translateNLJWithMarker(left, right, columns, rewriter, loc, markerDefAttr, fn);
   }
}

static mlir::Value anyTuple(mlir::Value stream, mlir::tuples::ColumnDefAttr markerDefAttr, mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
   auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   auto [markerState, markerName] = createMarkerState(rewriter, loc);
   auto [mapped, boolColumn] = mapBool(stream, rewriter, loc, true);
   auto [referenceDefAttr, referenceRefAttr] = createColumn(mlir::subop::EntryRefType::get(rewriter.getContext(), markerState.getType()), "lookup", "ref");
   auto afterLookup = rewriter.create<mlir::subop::LookupOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(rewriter.getContext()), mapped, markerState, rewriter.getArrayAttr({}), referenceDefAttr);
   rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterLookup, referenceRefAttr, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, colManager.createRef(boolColumn))));
   return rewriter.create<mlir::subop::ScanOp>(loc, markerState, rewriter.getDictionaryAttr(rewriter.getNamedAttr(markerName, markerDefAttr)));
}

class CrossProductLowering : public OpConversionPattern<mlir::relalg::CrossProductOp> {
   public:
   using OpConversionPattern<mlir::relalg::CrossProductOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::relalg::CrossProductOp crossProductOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = crossProductOp->getLoc();
      rewriter.replaceOp(crossProductOp, translateNL(adaptor.right(), adaptor.left(), false, mlir::ArrayAttr(), mlir::ArrayAttr(), getRequired(mlir::cast<Operator>(crossProductOp.left().getDefiningOp())), rewriter, loc, [](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
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
      auto rightHash = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      rewriter.replaceOp(innerJoinOp, translateNL(adaptor.right(), adaptor.left(), useHash, rightHash, leftHash, getRequired(mlir::cast<Operator>(innerJoinOp.left().getDefiningOp())), rewriter, loc, [loc, &innerJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                            return translateSelection(v, innerJoinOp.predicate(), rewriter, loc);
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
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.right().getDefiningOp())), rewriter, loc, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.predicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               return rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::all_true, rewriter.getArrayAttr({markerRefAttr}));
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [_, scan] = translateNLWithMarker(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.left().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.predicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
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
      auto rightHash = markJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = markJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      if (!reverse) {
         rewriter.replaceOp(markJoinOp, translateNL(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(markJoinOp.right().getDefiningOp())), rewriter, loc, [loc, &markJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, markJoinOp.predicate(), rewriter, loc);
                               return anyTuple(filtered, markJoinOp.markattr(), rewriter, loc);
                            }));
      } else {
         auto [_, scan] = translateNLWithMarker(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(markJoinOp.left().getDefiningOp())), rewriter, loc, markJoinOp.markattr(), [loc, &markJoinOp](mlir::Value v,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, markJoinOp.predicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
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
      auto rightHash = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = antiSemiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      if (!reverse) {
         rewriter.replaceOp(antiSemiJoinOp, translateNL(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(antiSemiJoinOp.right().getDefiningOp())), rewriter, loc, [loc, &antiSemiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, antiSemiJoinOp.predicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               return rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [_, scan] = translateNLWithMarker(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(antiSemiJoinOp.left().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &antiSemiJoinOp](mlir::Value v,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, antiSemiJoinOp.predicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
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
      auto leftColumns=getRequired(mlir::cast<Operator>(semiJoinOp.left().getDefiningOp()));
      auto rightColumns=getRequired(mlir::cast<Operator>(semiJoinOp.right().getDefiningOp()));
      auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
      auto [stream, scan] = translateNLWithMarker(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.left().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp,leftColumns,rightColumns](mlir::Value v,mlir::Value tuple, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
         auto filtered = translateSelection(v, semiJoinOp.predicate(), rewriter, loc);
         auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
         auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
         rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
         auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.mapping());
         auto [markerDefAttr2, markerRefAttr2] = createColumn(rewriter.getI1Type(), "marker", "marker");
         Value filteredNoMatch = rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr2, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr2}));
         mlir::Value combined = rewriter.create<mlir::subop::CombineTupleOp>(loc, filteredNoMatch, tuple);

         auto mappedNullable2 = mapColsToNullable(combined,rewriter,loc,semiJoinOp.mapping(),0,leftColumns);

         auto mappedNull = mapColsToNull(mappedNullable2, rewriter, loc, semiJoinOp.mapping(),rightColumns);
         return rewriter.create<mlir::subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
      });
      auto noMatches = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
      auto mappedNullable = mapColsToNullable(noMatches,rewriter,loc,semiJoinOp.mapping(),0,rightColumns);
      auto mappedNull = mapColsToNull(mappedNullable, rewriter, loc, semiJoinOp.mapping(),leftColumns);
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
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.right().getDefiningOp())), rewriter, loc, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.predicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               Value filteredNoMatch = rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                               auto mappedNull = mapColsToNull(filteredNoMatch, rewriter, loc, semiJoinOp.mapping());
                               auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.mapping());
                               return rewriter.create<mlir::subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [stream, scan] = translateNLWithMarker(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.left().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.predicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.mapping());

            return mappedNullable;
         });
         auto noMatches = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
         auto mappedNull = mapColsToNull(noMatches, rewriter, loc, semiJoinOp.mapping());
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
      bool isConstantJoin = semiJoinOp->hasAttr("constantJoin");
      auto rightHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto leftHash = semiJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      if (isConstantJoin) {
         auto columnsToMaterialize = getRequired(mlir::cast<Operator>(semiJoinOp.right().getDefiningOp()));
         MaterializationHelper helper(columnsToMaterialize, rewriter.getContext());
         auto constantStateType = mlir::subop::SimpleStateType::get(rewriter.getContext(), helper.createStateMembersAttr());
         mlir::Value constantState = rewriter.create<mlir::subop::CreateOp>(loc, constantStateType, mlir::Attribute(), 0);
         auto entryRefType = mlir::subop::EntryRefType::get(rewriter.getContext(), constantStateType);
         auto [entryDef, entryRef] = createColumn(entryRefType, "lookup", "entryref");
         auto afterLookup = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.right(), constantState, rewriter.getArrayAttr({}), entryDef);
         rewriter.create<mlir::subop::ScatterOp>(loc, afterLookup, entryRef, helper.createColumnstateMapping());
         auto [entryDefLeft, entryRefLeft] = createColumn(entryRefType, "lookup", "entryref");

         auto afterLookupLeft = rewriter.create<mlir::subop::LookupOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), adaptor.left(), constantState, rewriter.getArrayAttr({}), entryDefLeft);
         auto gathered = rewriter.create<mlir::subop::GatherOp>(loc, afterLookupLeft, entryRefLeft, helper.createStateColumnMapping());
         auto mappedNullable = mapColsToNullable(gathered.res(), rewriter, loc, semiJoinOp.mapping());
         rewriter.replaceOp(semiJoinOp, mappedNullable);
      } else if (!reverse) {
         rewriter.replaceOp(semiJoinOp, translateNL(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.right().getDefiningOp())), rewriter, loc, [loc, &semiJoinOp](mlir::Value v, mlir::ConversionPatternRewriter& rewriter) -> mlir::Value {
                               auto filtered = translateSelection(v, semiJoinOp.predicate(), rewriter, loc);
                               auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
                               Value filteredNoMatch = rewriter.create<mlir::subop::FilterOp>(loc, anyTuple(filtered, markerDefAttr, rewriter, loc), mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({markerRefAttr}));
                               auto mappedNull = mapColsToNull(filteredNoMatch, rewriter, loc, semiJoinOp.mapping());
                               auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.mapping());
                               return rewriter.create<mlir::subop::UnionOp>(loc, mlir::ValueRange{mappedNullable, mappedNull});
                            }));
      } else {
         auto [flagAttrDef, flagAttrRef] = createColumn(rewriter.getI1Type(), "materialized", "marker");
         auto [stream, scan] = translateNLWithMarker(adaptor.left(), adaptor.right(), useHash, leftHash, rightHash, getRequired(mlir::cast<Operator>(semiJoinOp.left().getDefiningOp())), rewriter, loc, flagAttrDef, [loc, &semiJoinOp](mlir::Value v,mlir::Value, mlir::ConversionPatternRewriter& rewriter, mlir::tuples::ColumnRefAttr ref, std::string flagMember) -> mlir::Value {
            auto filtered = translateSelection(v, semiJoinOp.predicate(), rewriter, loc);
            auto [markerDefAttr, markerRefAttr] = createColumn(rewriter.getI1Type(), "marker", "marker");
            auto afterBool = mapBool(filtered, rewriter, loc, true, &markerDefAttr.getColumn());
            rewriter.create<mlir::subop::ScatterOp>(rewriter.getUnknownLoc(), afterBool, ref, rewriter.getDictionaryAttr(rewriter.getNamedAttr(flagMember, markerRefAttr)));
            auto mappedNullable = mapColsToNullable(filtered, rewriter, loc, semiJoinOp.mapping());

            return mappedNullable;
         });
         auto noMatches = rewriter.create<mlir::subop::FilterOp>(loc, scan, mlir::subop::FilterSemantic::none_true, rewriter.getArrayAttr({flagAttrRef}));
         auto mappedNull = mapColsToNull(noMatches, rewriter, loc, semiJoinOp.mapping());
         rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(semiJoinOp, mlir::ValueRange{stream, mappedNull});
      }
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
      mlir::relalg::ColumnSet requiredColumns = getRequired(sortOp);
      requiredColumns.insert(sortOp.getUsedColumns());
      MaterializationHelper helper(requiredColumns, rewriter.getContext());
      auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector = rewriter.create<mlir::subop::CreateOp>(sortOp->getLoc(), vectorType, mlir::Attribute{}, 0);
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
      MaterializationHelper helper(getRequired(tmpOp), rewriter.getContext());

      auto vectorType = mlir::subop::VectorType::get(rewriter.getContext(), helper.createStateMembersAttr());
      mlir::Value vector = rewriter.create<mlir::subop::CreateOp>(tmpOp->getLoc(), vectorType, mlir::Attribute{}, 0);
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
      mlir::Value table = rewriter.create<mlir::subop::CreateOp>(materializeOp->getLoc(), tableRefType, rewriter.getArrayAttr(colNames), 0);
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
               analyzedAggregation.finalizeFunctions.push_back([loc, currDestIdx = currDestIdx, destAttr = destAttr, resultingType = resultingType, counterType](mlir::ValueRange range, mlir::OpBuilder builder) {
                  mlir::Value casted=builder.create<mlir::db::CastOp>(loc, getBaseType(resultingType), range[currDestIdx+1]);
                  if(resultingType.isa<mlir::db::NullableType>()&&casted.getType()!=resultingType){
                     casted=builder.create<mlir::db::AsNullableOp>(loc, resultingType, casted);
                  }
                  //TypeRange resultTypes, Value cond, function_ref<void(OpBuilder &, Location)> thenBuilder = buildTerminatedBody, function_ref<void(OpBuilder &, Location)> elseBuilder = nullptr
                  mlir::Value initCounterVal = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(0));
                  mlir::Value isZero=builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::eq,initCounterVal,range[currDestIdx+1]);
                  mlir::Value average=builder.create<mlir::scf::IfOp>(loc,resultingType,isZero,[&](mlir::OpBuilder& builder,mlir::Location loc){
                        builder.create<mlir::scf::YieldOp>(loc,range[currDestIdx]);
                     },[&](mlir::OpBuilder& builder,mlir::Location loc){
                        mlir::Value average=builder.create<mlir::db::DivOp>(loc, resultingType, range[currDestIdx], casted);
                        builder.create<mlir::scf::YieldOp>(loc,average);
                     }).getResult(0);
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
         auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{}, 1);
         createOp.initFn().front().push_back(initialValueBlock);
         state = createOp.res();
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

         auto createOp = rewriter.create<mlir::subop::CreateOp>(loc, stateType, mlir::Attribute{}, 0);
         state = createOp.res();
         auto lookupOp = rewriter.create<mlir::subop::LookupOrInsertOp>(rewriter.getUnknownLoc(), mlir::tuples::TupleStreamType::get(getContext()), adaptor.rel(), state, aggregationOp.group_by_cols(), referenceDefAttr);
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
   patterns.insert<ConstRelationLowering>(typeConverter, ctxt);
   patterns.insert<MarkJoinLowering>(typeConverter, ctxt);
   patterns.insert<CrossProductLowering>(typeConverter, ctxt);
   patterns.insert<InnerJoinNLLowering>(typeConverter, ctxt);
   patterns.insert<AggregationLowering>(typeConverter, ctxt);
   patterns.insert<SemiJoinLowering>(typeConverter, ctxt);
   patterns.insert<AntiSemiJoinLowering>(typeConverter, ctxt);
   patterns.insert<OuterJoinLowering>(typeConverter, ctxt);
   patterns.insert<FullOuterJoinLowering>(typeConverter, ctxt);
   patterns.insert<SingleJoinLowering>(typeConverter, ctxt);
   patterns.insert<LimitLowering>(typeConverter, ctxt);
   patterns.insert<UnionAllLowering>(typeConverter, ctxt);
   patterns.insert<UnionDistinctLowering>(typeConverter, ctxt);
   patterns.insert<CountingSetOperationLowering>(ctxt);

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