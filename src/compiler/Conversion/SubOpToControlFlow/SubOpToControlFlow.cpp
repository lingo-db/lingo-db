#include "lingodb/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/runtime/Buffer.h"
#include "lingodb/compiler/runtime/DataSourceIteration.h"
#include "lingodb/compiler/runtime/ExecutionContext.h"
#include "lingodb/compiler/runtime/GrowingBuffer.h"
#include "lingodb/compiler/runtime/HashIndex.h"
#include "lingodb/compiler/runtime/HashMultiMap.h"
#include "lingodb/compiler/runtime/Hashtable.h"
#include "lingodb/compiler/runtime/Heap.h"
#include "lingodb/compiler/runtime/LazyJoinHashtable.h"
#include "lingodb/compiler/runtime/PreAggregationHashtable.h"
#include "lingodb/compiler/runtime/RelationHelper.h"
#include "lingodb/compiler/runtime/SegmentTreeView.h"
#include "lingodb/compiler/runtime/SimpleState.h"
#include "lingodb/compiler/runtime/ThreadLocal.h"
#include "lingodb/compiler/runtime/Tracing.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <stack>
using namespace mlir;

#ifdef NDEBUG
#define ASSERT_WITH_OP(cond,op, msg)
#else
#define ASSERT_WITH_OP(cond,op, msg)                                                                                   \
   if (!(cond)) {                                                                                                       \
      op->emitOpError(msg);                                                                                             \
   }                                                                                                                    \
   assert(cond)
#endif
namespace {
using namespace lingodb::compiler::dialect;
namespace rt = lingodb::compiler::runtime;
struct SubOpToControlFlowLoweringPass
   : public mlir::PassWrapper<SubOpToControlFlowLoweringPass, OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubOpToControlFlowLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "lower-subop-to-cf"; }

   SubOpToControlFlowLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect, dsa::DSADialect, subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};

static std::vector<mlir::Value> inlineBlock(mlir::Block* b, mlir::OpBuilder& rewriter, mlir::ValueRange arguments) {
   auto* terminator = b->getTerminator();
   auto returnOp = mlir::cast<tuples::ReturnOp>(terminator);
   mlir::IRMapping mapper;
   assert(b->getNumArguments() == arguments.size());
   for (auto i = 0ull; i < b->getNumArguments(); i++) {
      mapper.map(b->getArgument(i), arguments[i]);
   }
   for (auto& x : b->getOperations()) {
      if (&x != terminator) {
         rewriter.clone(x, mapper);
      }
   }
   std::vector<mlir::Value> res;
   for (auto val : returnOp.getResults()) {
      res.push_back(mapper.lookup(val));
   }
   return res;
}
static std::vector<Type> unpackTypes(mlir::ArrayAttr arr) {
   std::vector<Type> res;
   for (auto x : arr) { res.push_back(mlir::cast<mlir::TypeAttr>(x).getValue()); }
   return res;
};
class ColumnMapping {
   std::unordered_map<const tuples::Column*, mlir::Value> mapping;

   public:
   ColumnMapping() : mapping() {}
   ColumnMapping(subop::InFlightOp inFlightOp) {
      assert(!!inFlightOp);
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   ColumnMapping(subop::InFlightTupleOp inFlightOp) {
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   void merge(subop::InFlightOp inFlightOp) {
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   void merge(subop::InFlightTupleOp inFlightOp) {
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   mlir::Value resolve(mlir::Operation* op, tuples::ColumnRefAttr ref) {
      if (!mapping.contains(&ref.getColumn())) {
         std::string wrongReference;
         llvm::raw_string_ostream wrongReferenceStream(wrongReference);
         ((mlir::Attribute) ref).print(wrongReferenceStream);

         op->emitOpError("Could not resolve column reference," + wrongReference);
         assert(false);
      }
      mlir::Value r = mapping.at(&ref.getColumn());
      assert(r);
      return r;
   }
   std::vector<mlir::Value> resolve(mlir::Operation* op, mlir::ArrayAttr arr) {
      std::vector<mlir::Value> res;
      for (auto attr : arr) {
         res.push_back(resolve(op, mlir::cast<tuples::ColumnRefAttr>(attr)));
      }
      return res;
   }
   mlir::Value createInFlight(mlir::OpBuilder& builder) {
      std::vector<mlir::Value> values;
      std::vector<mlir::Attribute> columns;

      for (auto m : mapping) {
         columns.push_back(builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createDef(m.first));
         values.push_back(m.second);
      }
      return builder.create<subop::InFlightOp>(builder.getUnknownLoc(), values, builder.getArrayAttr(columns));
   }
   mlir::Value createInFlightTuple(mlir::OpBuilder& builder) {
      std::vector<mlir::Value> values;
      std::vector<mlir::Attribute> columns;

      for (auto m : mapping) {
         columns.push_back(builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createDef(m.first));
         values.push_back(m.second);
      }
      return builder.create<subop::InFlightTupleOp>(builder.getUnknownLoc(), values, builder.getArrayAttr(columns));
   }
   void define(tuples::ColumnDefAttr columnDefAttr, mlir::Value v) {
      mapping.insert(std::make_pair(&columnDefAttr.getColumn(), v));
   }
   void define(mlir::ArrayAttr columns, mlir::ValueRange values) {
      for (auto i = 0ul; i < columns.size(); i++) {
         define(mlir::cast<tuples::ColumnDefAttr>(columns[i]), values[i]);
      }
   }
   const auto& getMapping() {
      return mapping;
   }
};
class EntryStorageHelper {
   mlir::Operation* op;
   subop::StateMembersAttr members;
   mlir::TupleType storageType;
   size_t nullBitSetPos; //position of the nullBitSet in the entry
   mlir::Type nullBitsetType; //physical type of the nullBitSet (e.g. i8,i32, i64,...)

   struct MemberInfo {
      bool isNullable;
      size_t nullBitOffset;
      mlir::Type stored;
      size_t offset;
   };
   std::unordered_map<std::string, MemberInfo> memberInfos;

   public:
   static bool compressionEnabled;
   EntryStorageHelper(mlir::Operation* op, subop::StateMembersAttr members, mlir::TypeConverter* typeConverter) : op(op),members(members) {
      std::vector<mlir::Type> types;
      size_t nullBitOffset = 0;
      for (auto m : llvm::zip(members.getNames(), members.getTypes())) {
         auto memberName = mlir::cast<StringAttr>(std::get<0>(m)).str();
         auto type = mlir::cast<mlir::TypeAttr>(std::get<1>(m)).getValue();
         auto converted = typeConverter->convertType(type);
         type = converted ? converted : type;
         MemberInfo memberInfo;
         if (auto nullableType = mlir::dyn_cast_or_null<db::NullableType>(type)) {
            if (compressionEnabled) {
               memberInfo.isNullable = true;
               if (nullBitOffset == 0) {
                  nullBitSetPos = types.size();
                  types.push_back(mlir::Type());
               }
               memberInfo.nullBitOffset = nullBitOffset++;
               memberInfo.stored = nullableType.getType();
            } else {
               memberInfo.isNullable = false;
               memberInfo.stored = type;
            }
         } else {
            memberInfo.isNullable = false;
            memberInfo.stored = type;
         }
         memberInfo.offset = types.size();
         memberInfos.insert({memberName, memberInfo});
         types.push_back(memberInfo.stored);
      }
      if (nullBitOffset == 0) {
      } else if (nullBitOffset <= 8) {
         nullBitsetType = mlir::IntegerType::get(members.getContext(), 8);
      } else if (nullBitOffset <= 16) {
         nullBitsetType = mlir::IntegerType::get(members.getContext(), 16);
      } else if (nullBitOffset <= 32) {
         nullBitsetType = mlir::IntegerType::get(members.getContext(), 32);
      } else if (nullBitOffset <= 64) {
         nullBitsetType = mlir::IntegerType::get(members.getContext(), 64);
      } else {
         assert(false && "should not happen");
      }
      if (nullBitOffset > 0) {
         types[nullBitSetPos] = nullBitsetType;
      }
      storageType = mlir::TupleType::get(members.getContext(), types);
   }
   mlir::Value getPointer(mlir::Value ref, std::string member, mlir::OpBuilder& rewriter, mlir::Location loc) {
      const auto& memberInfo = memberInfos.at(member);
      assert(!memberInfo.isNullable);
      return rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), memberInfo.stored), ref, memberInfo.offset);
   }
   void storeValues(mlir::Value ref, std::unordered_map<std::string, mlir::Value> values, mlir::OpBuilder& rewriter, mlir::Location loc) {
      ref = ensureRefType(ref, rewriter, loc);

      mlir::Value nullBitSetRef;
      mlir::Value nullBitSet;
      if (nullBitsetType) {
         nullBitSetRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), nullBitsetType), ref, nullBitSetPos);
         nullBitSet = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, nullBitsetType);
      }
      for (auto i = 0ul; i < members.getTypes().size(); i++) {
         auto name = mlir::cast<mlir::StringAttr>(members.getNames()[i]).str();
         const MemberInfo& memberInfo = memberInfos.at(name);
         auto valueToStore = values[name];
         if (memberInfo.isNullable) {
            mlir::Value nullBit = rewriter.create<db::IsNullOp>(loc, valueToStore);
            mlir::Value shiftAmount = rewriter.create<mlir::arith::ConstantIntOp>(loc, memberInfo.nullBitOffset, nullBitsetType);
            mlir::Value shiftedNullBit = rewriter.create<mlir::arith::ShLIOp>(loc, rewriter.create<mlir::arith::ExtUIOp>(loc, nullBitsetType, nullBit), shiftAmount);
            valueToStore = rewriter.create<db::NullableGetVal>(loc, valueToStore);
            nullBitSet = rewriter.create<mlir::arith::OrIOp>(loc, nullBitSet, shiftedNullBit);
         }
         auto memberRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), memberInfo.stored), ref, memberInfo.offset);
         rewriter.create<util::StoreOp>(loc, valueToStore, memberRef, mlir::Value());
      }
      if (nullBitsetType) {
         rewriter.create<util::StoreOp>(loc, nullBitSet, nullBitSetRef, mlir::Value());
      }
   }
   std::unordered_map<std::string, mlir::Value> loadValues(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) {
      ref = ensureRefType(ref, rewriter, loc);
      std::unordered_map<std::string, mlir::Value> res;
      mlir::Value nullBitSet;
      if (nullBitsetType) {
         mlir::Value nullBitSetRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), nullBitsetType), ref, nullBitSetPos);
         nullBitSet = rewriter.create<util::LoadOp>(loc, nullBitSetRef);
      }
      for (auto i = 0ul; i < members.getTypes().size(); i++) {
         auto name = mlir::cast<mlir::StringAttr>(members.getNames()[i]).str();
         const MemberInfo& memberInfo = memberInfos.at(name);
         auto memberRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), memberInfo.stored), ref, memberInfo.offset);
         mlir::Value loadedValue = rewriter.create<util::LoadOp>(loc, memberRef);
         if (memberInfo.isNullable) {
            mlir::Value shiftedNullBit = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1ull << memberInfo.nullBitOffset, nullBitsetType);
            mlir::Value isNull = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, rewriter.create<mlir::arith::AndIOp>(loc, nullBitSet, shiftedNullBit), shiftedNullBit);
            loadedValue = rewriter.create<db::AsNullableOp>(loc, db::NullableType::get(rewriter.getContext(), memberInfo.stored), loadedValue, isNull);
         }
         res[name] = loadedValue;
      }
      return res;
   }
   mlir::TupleType getStorageType() {
      return storageType;
   }
   util::RefType getRefType() {
      return util::RefType::get(members.getContext(), getStorageType());
   }
   mlir::Value ensureRefType(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) {
      auto refType = mlir::cast<util::RefType>(ref.getType());
      auto expectedType = getRefType();
      if (refType != expectedType) {
         ref = rewriter.create<util::GenericMemrefCastOp>(loc, expectedType, ref);
      }
      return ref;
   }
   std::vector<mlir::Value> resolve(mlir::Operation* op,mlir::DictionaryAttr mapping, ColumnMapping columnMapping) {
      std::vector<mlir::Value> result;
      for (auto m : members.getNames()) {
         result.push_back(columnMapping.resolve(op,mlir::cast<tuples::ColumnRefAttr>(mapping.get(mlir::cast<mlir::StringAttr>(m).str()))));
      }
      return result;
   }
   std::vector<mlir::Value> loadValuesOrdered(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc, ArrayAttr relevantMembers = {}) {
      auto values = loadValues(ref, rewriter, loc);
      if (!relevantMembers) {
         relevantMembers = members.getNames();
      }
      std::vector<mlir::Value> res;
      for (auto m : relevantMembers) {
         res.push_back(values.at(mlir::cast<mlir::StringAttr>(m).str()));
      }
      return res;
   }
   template <class L>
   void storeOrderedValues(mlir::Value ref, L list, mlir::OpBuilder& rewriter, mlir::Location loc) {
      std::unordered_map<std::string, mlir::Value> toStore;
      for (auto x : llvm::zip(members.getNames(), list)) {
         toStore[mlir::cast<mlir::StringAttr>(std::get<0>(x)).str()] = std::get<1>(x);
      }
      storeValues(ref, toStore, rewriter, loc);
   }

   void storeFromColumns(mlir::DictionaryAttr mapping, ColumnMapping& columnMapping, mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) {
      std::unordered_map<std::string, mlir::Value> toStore;
      for (auto x : mapping) {
         toStore[x.getName().str()] = columnMapping.resolve(op,mlir::cast<tuples::ColumnRefAttr>(x.getValue()));
      }
      storeValues(ref, toStore, rewriter, loc);
   }
   void loadIntoColumns(mlir::DictionaryAttr mapping, ColumnMapping& columnMapping, mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) {
      auto values = loadValues(ref, rewriter, loc);
      for (auto x : mapping) {
         auto memberName = x.getName().str();
         if (values.contains(memberName)) {
            columnMapping.define(mlir::cast<tuples::ColumnDefAttr>(x.getValue()), values[memberName]);
         }
      }
   }
};
bool EntryStorageHelper::compressionEnabled = false;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// State management ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SubOpRewriter;
class AbstractSubOpConversionPattern {
   protected:
   mlir::TypeConverter* typeConverter;
   std::string operationName;
   PatternBenefit benefit;
   mlir::MLIRContext* context;

   public:
   AbstractSubOpConversionPattern(TypeConverter* typeConverter, const std::string& operationName, const PatternBenefit& benefit, MLIRContext* context) : typeConverter(typeConverter), operationName(operationName), benefit(benefit), context(context) {}
   virtual LogicalResult matchAndRewrite(mlir::Operation*, SubOpRewriter& rewriter) = 0;
   mlir::MLIRContext* getContext() const {
      return context;
   }
   const std::string& getOperationName() const {
      return operationName;
   }
   const PatternBenefit& getBenefit() const {
      return benefit;
   }
   virtual ~AbstractSubOpConversionPattern(){};
};
struct InFlightTupleStream {
   subop::InFlightOp inFlightOp;
};
class SubOpRewriter {
   mlir::OpBuilder builder;
   std::vector<mlir::IRMapping> valueMapping;
   std::unordered_map<std::string, std::vector<std::unique_ptr<AbstractSubOpConversionPattern>>> patterns;
   llvm::DenseMap<mlir::Value, InFlightTupleStream> inFlightTupleStreams;
   std::vector<mlir::Operation*> toErase;
   std::unordered_set<mlir::Operation*> isErased;
   std::vector<mlir::Operation*> toRewrite;
   mlir::Operation* currentStreamLoc = nullptr;
   struct ExecutionStepContext {
      subop::ExecutionStepOp executionStep;
      mlir::IRMapping& outerMapping;
   };
   std::stack<ExecutionStepContext> executionStepContexts;

   public:
   SubOpRewriter(subop::ExecutionStepOp executionStep, mlir::IRMapping& outerMapping) : builder(executionStep), executionStepContexts{} {
      valueMapping.push_back(mlir::IRMapping());
      executionStepContexts.push({executionStep, outerMapping});
   }
   auto getIndexType() { return builder.getIndexType(); }
   auto setInsertionPointAfter(mlir::Operation* op) { return builder.setInsertionPointAfter(op); }
   auto getIntegerAttr(mlir::Type t, int64_t v) { return builder.getIntegerAttr(t, v); }
   auto getNamedAttr(llvm::StringRef s, mlir::Attribute v) { return builder.getNamedAttr(s, v); }
   auto getArrayAttr(llvm::ArrayRef<mlir::Attribute> v) { return builder.getArrayAttr(v); }
   auto getDictionaryAttr(llvm::ArrayRef<mlir::NamedAttribute> v) { return builder.getDictionaryAttr(v); }
   auto getI8Type() { return builder.getI8Type(); }
   auto getI1Type() { return builder.getI1Type(); }
   auto getI64Type() { return builder.getI64Type(); }
   auto getStringAttr(const Twine& bytes) { return builder.getStringAttr(bytes); }
   mlir::Value storeStepRequirements() {
      auto outerMapping = executionStepContexts.top().outerMapping;
      auto executionStep = executionStepContexts.top().executionStep;
      std::vector<mlir::Type> types;
      for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
         mlir::Value input = outerMapping.lookup(param);
         types.push_back(input.getType());
      }
      auto tupleType = mlir::TupleType::get(getContext(), types);
      mlir::Value contextPtr = create<util::AllocaOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), tupleType), mlir::Value());
      size_t offset = 0;
      for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
         mlir::Value input = outerMapping.lookup(param);
         auto memberRef = create<util::TupleElementPtrOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), input.getType()), contextPtr, offset++);
         create<util::StoreOp>(builder.getUnknownLoc(), input, memberRef, mlir::Value());
      }
      contextPtr = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), getI8Type()), contextPtr);
      return contextPtr;
   }
   void cleanup() {
      for (auto* op : toErase) {
         op->dropAllReferences();
         op->dropAllUses();
         op->dropAllDefinedValueUses();
         op->remove();
         op->erase();
      }
   }
   class Guard {
      SubOpRewriter& rewriter;

      public:
      Guard(SubOpRewriter& rewriter) : rewriter(rewriter) {
         rewriter.valueMapping.push_back(mlir::IRMapping());
      }
      ~Guard() {
         rewriter.valueMapping.pop_back();
      }
   };
   Guard loadStepRequirements(mlir::Value contextPtr, mlir::TypeConverter* typeConverter) {
      auto outerMapping = executionStepContexts.top().outerMapping;
      auto executionStep = executionStepContexts.top().executionStep;
      std::vector<mlir::Type> types;
      for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
         mlir::Value input = outerMapping.lookup(param);
         types.push_back(input.getType());
      }
      auto tupleType = mlir::TupleType::get(getContext(), types);
      contextPtr = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), tupleType), contextPtr);
      Guard guard(*this);
      size_t offset = 0;
      for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
         mlir::Value input = outerMapping.lookup(param);
         auto memberRef = create<util::TupleElementPtrOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), input.getType()), contextPtr, offset++);
         mlir::Value value = create<util::LoadOp>(builder.getUnknownLoc(), memberRef);
         if (mlir::cast<mlir::BoolAttr>(isThreadLocal).getValue()) {
            value = rt::ThreadLocal::getLocal(builder, builder.getUnknownLoc())({value})[0];
            value = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), typeConverter->convertType(arg.getType()), value);
         }
         map(arg, value);
      }
      return guard;
   }
   class NestingGuard {
      SubOpRewriter& rewriter;

      public:
      NestingGuard(SubOpRewriter& rewriter, mlir::IRMapping& outerMapping, subop::ExecutionStepOp executionStepOp) : rewriter(rewriter) {
         rewriter.executionStepContexts.push({executionStepOp, outerMapping});
      }
      ~NestingGuard() {
         rewriter.executionStepContexts.pop();
      }
   };
   NestingGuard nest(mlir::IRMapping& outerMapping, subop::ExecutionStepOp executionStepOp) {
      return NestingGuard(*this, outerMapping, executionStepOp);
   }
   mlir::Value getMapped(mlir::Value v) {
      for (auto it = valueMapping.rbegin(); it != valueMapping.rend(); it++) {
         if (it->contains(v)) {
            return getMapped(it->lookup(v));
         }
      }
      return v;
   }
   void atStartOf(mlir::Block* block, const std::function<void(SubOpRewriter&)>& fn) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(block);
      return fn(*this);
   }
   template <typename OpTy, typename... Args>
   OpTy create(Location location, Args&&... args) {
      OpTy res = builder.create<OpTy>(location, std::forward<Args>(args)...);
      if (res->getDialect()->getNamespace() == "subop" || mlir::isa<mlir::UnrealizedConversionCastOp>(res.getOperation())) {
         toRewrite.push_back(res.getOperation());
         //   rewrite(res.getOperation());
      }
      return res;
   }
   void registerOpInserted(mlir::Operation* op) {
      if (op->getDialect()->getNamespace() == "subop") {
         toRewrite.push_back(op);
      } else {
         op->walk([&](mlir::Operation* nestedOp) {
            if (nestedOp->getDialect()->getNamespace() != "subop") {
               for (auto& operand : nestedOp->getOpOperands()) {
                  operand.set(getMapped(operand.get()));
               }
            }
         });
      }
   }
   template <typename OpTy, typename... Args>
   OpTy replaceOpWithNewOp(mlir::Operation* op, Args&&... args) {
      auto res = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
      replaceOp(op, res->getResults());
      return res;
   }
   void eraseOp(Operation* op) {
      assert(!isErased.contains(op));
      toErase.push_back(op);
      isErased.insert(op);
   }
   void replaceOp(Operation* op, ValueRange newValues) {
      assert(op->getNumResults() == newValues.size());
      for (auto z : llvm::zip(op->getResults(), newValues)) {
         valueMapping[0].map(std::get<0>(z), std::get<1>(z));
      }
      eraseOp(op);
   }
   mlir::Block* cloneBlock(mlir::Block* block, IRMapping& mapping) {
      mlir::Block* clonedBlock = new mlir::Block;
      for (auto arg : block->getArguments()) {
         auto clonedArg = clonedBlock->addArgument(arg.getType(), arg.getLoc());
         mapping.map(arg, clonedArg);
      }
      atStartOf(clonedBlock, [&](SubOpRewriter& rewriter) {
         for (auto& op : block->getOperations()) {
            rewriter.builder.insert(clone(&op, mapping));
         }
      });

      return clonedBlock;
   }
   void map(mlir::Value v, mlir::Value mapped) {
      valueMapping[0].map(v, mapped);
   }
   mlir::Operation* clone(mlir::Operation* op, IRMapping& mapping) {
      auto* cloned = op->cloneWithoutRegions(mapping);
      for (auto r : llvm::zip(op->getRegions(), cloned->getRegions())) {
         for (auto& b : std::get<0>(r).getBlocks()) {
            std::get<1>(r).push_back(cloneBlock(&b, mapping));
         }
      }

      return cloned;
   }
   InFlightTupleStream getTupleStream(mlir::Value v) {
      return inFlightTupleStreams[v];
   }
   subop::InFlightOp createInFlight(ColumnMapping mapping) {
      auto newInFlight = mapping.createInFlight(builder);
      inFlightTupleStreams[newInFlight] = InFlightTupleStream{mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp())};
      return mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp());
   }
   void replaceTupleStream(mlir::Value tupleStream, InFlightTupleStream previous) {
      inFlightTupleStreams[tupleStream] = previous;
      if (auto* definingOp = tupleStream.getDefiningOp()) {
         eraseOp(definingOp);
      }
   }
   void replaceTupleStream(mlir::Value tupleStream, ColumnMapping& mapping) {
      auto newInFlight = mapping.createInFlight(builder);
      eraseOp(newInFlight.getDefiningOp());
      inFlightTupleStreams[tupleStream] = InFlightTupleStream{mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp())};
      if (auto* definingOp = tupleStream.getDefiningOp()) {
         eraseOp(definingOp);
      }
   }

   template <class AdaptorType>
   void inlineBlock(mlir::Block* block, mlir::ValueRange values, const std::function<void(AdaptorType)> processTerminator) {
      for (auto z : llvm::zip(block->getArguments(), values)) {
         std::get<0>(z).replaceAllUsesWith(std::get<1>(z));
      }
      std::vector<mlir::Operation*> toInsert;
      mlir::Operation* terminator;
      for (auto& op : block->getOperations()) {
         if (&op != block->getTerminator()) {
            toInsert.push_back(&op);
         } else {
            terminator = &op;
            break;
         }
      }
      for (auto* op : toInsert) {
         op->remove();
         builder.insert(op);
         registerOpInserted(op);
      }
      std::vector<mlir::Value> adaptorVals;
      for (auto operand : terminator->getOperands()) {
         adaptorVals.push_back(getMapped(operand));
      }
      AdaptorType adaptor(adaptorVals);
      processTerminator(adaptor);
      terminator->remove();
      eraseOp(terminator);
   }

   mlir::MLIRContext* getContext() {
      return builder.getContext();
   }
   operator mlir::OpBuilder&() { return builder; }

   template <class PatternT, typename... Args>
   void insertPattern(Args&&... args) {
      auto uniquePtr = std::make_unique<PatternT>(std::forward<Args>(args)...);
      patterns[uniquePtr->getOperationName()].push_back(std::move(uniquePtr));
   }
   void rewrite(mlir::Operation* op, mlir::Operation* before = nullptr) {
      if (isErased.contains(op)) return;
      if (before) {
         builder.setInsertionPoint(before);
      } else {
         builder.setInsertionPoint(op);
      }
      for (auto& p : patterns[op->getName().getStringRef().str()]) { //todo: ordering
         if (p->matchAndRewrite(op, *this).succeeded()) {
            std::vector<mlir::Operation*> localRewrite = std::move(toRewrite);
            for (auto* r : localRewrite) {
               if (shouldRewrite(r)) {
                  rewrite(r);
               }
            }
            return;
         }
      }
      op->dump();
      llvm::dbgs() << "Could not rewrite" << op->getName() << "\n";
      assert(false);
   }

   bool shouldRewrite(mlir::Operation* op) {
      if (op->getDialect()->getNamespace() == "subop") {
         return true;
      }
      if (auto unrealizedCast = mlir::dyn_cast_or_null<mlir::UnrealizedConversionCastOp>(op)) {
         return llvm::any_of(unrealizedCast.getOutputs().getTypes(), [&](mlir::Type t) {
                   return t.getDialect().getNamespace() == "subop";
                }) ||
            llvm::any_of(unrealizedCast.getInputs().getTypes(), [&](mlir::Type t) {
                   return t.getDialect().getNamespace() == "subop";
                });
      }
      return false;
   }
   void rewrite(mlir::Block* block) {
      block->walk<WalkOrder::PreOrder>([this](mlir::Operation* op) {
         if (shouldRewrite(op)) {
            rewrite(op);
            return WalkResult::skip();
         }
         return WalkResult::advance();
      });
      for (auto* op : toErase) {
         op->dropAllReferences();
         op->dropAllUses();
         op->remove();
         op->erase();
      }
   }
   void insert(mlir::Operation* op) {
      builder.insert(op);
      registerOpInserted(op);
   }
   void insertAndRewrite(mlir::Operation* op) {
      builder.insert(op);
      if (op->getDialect()->getNamespace() == "subop") {
         rewrite(op);
      }
   }

   mlir::Operation* getCurrentStreamLoc() {
      return currentStreamLoc;
   }

   mlir::LogicalResult implementStreamConsumer(mlir::Value stream, const std::function<mlir::LogicalResult(SubOpRewriter&, ColumnMapping&)>& impl) {
      auto& streamInfo = inFlightTupleStreams[stream];
      ColumnMapping mapping(streamInfo.inFlightOp);
      mlir::OpBuilder::InsertionGuard guard(builder);
      currentStreamLoc = streamInfo.inFlightOp.getOperation();
      builder.setInsertionPoint(streamInfo.inFlightOp);
      mlir::LogicalResult res = impl(*this, mapping);
      currentStreamLoc = nullptr;
      return res;
   }
};

template <class OpT>
class SubOpConversionPattern : public AbstractSubOpConversionPattern {
   public:
   using OpAdaptor = typename OpT::Adaptor;
   SubOpConversionPattern(TypeConverter& typeConverter, MLIRContext* context,
                          PatternBenefit benefit = 1)
      : AbstractSubOpConversionPattern(&typeConverter, std::string(OpT::getOperationName()), benefit,
                                       context) {}
   LogicalResult matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) override {
      std::vector<mlir::Value> newOperands;
      for (auto operand : op->getOperands()) {
         newOperands.push_back(rewriter.getMapped(operand));
      }
      OpAdaptor adaptor(newOperands);
      return matchAndRewrite(mlir::cast<OpT>(op), adaptor, rewriter);
   }
   virtual LogicalResult matchAndRewrite(OpT op, OpAdaptor adaptor, SubOpRewriter& rewriter) const = 0;
   virtual ~SubOpConversionPattern(){};
};

template <class OpT, size_t B = 1>
class SubOpTupleStreamConsumerConversionPattern : public AbstractSubOpConversionPattern {
   public:
   using OpAdaptor = typename OpT::Adaptor;
   SubOpTupleStreamConsumerConversionPattern(TypeConverter& typeConverter, MLIRContext* context,
                                             PatternBenefit benefit = B)
      : AbstractSubOpConversionPattern(&typeConverter, std::string(OpT::getOperationName()), benefit,
                                       context) {}
   LogicalResult matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) override {
      auto castedOp = mlir::cast<OpT>(op);
      auto stream = castedOp.getStream();
      return rewriter.implementStreamConsumer(stream, [&](SubOpRewriter& rewriter, ColumnMapping& mapping) {
         std::vector<mlir::Value> newOperands;
         for (auto operand : op->getOperands()) {
            newOperands.push_back(rewriter.getMapped(operand));
         }
         OpAdaptor adaptor(newOperands);
         return matchAndRewrite(castedOp, adaptor, rewriter, mapping);
      });
   }
   virtual LogicalResult matchAndRewrite(OpT op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const = 0;
   virtual ~SubOpTupleStreamConsumerConversionPattern(){};
};

static mlir::TupleType getHtKVType(subop::HashMapType t, mlir::TypeConverter& converter) {
   auto keyTupleType = EntryStorageHelper(nullptr,t.getKeyMembers(), &converter).getStorageType();
   auto valTupleType = EntryStorageHelper(nullptr,t.getValueMembers(), &converter).getStorageType();
   return mlir::cast<mlir::TupleType>(converter.convertType(mlir::TupleType::get(t.getContext(), {keyTupleType, valTupleType})));
}
static mlir::TupleType getHtKVType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter) {
   auto keyTupleType = EntryStorageHelper(nullptr,t.getKeyMembers(), &converter).getStorageType();
   auto valTupleType = EntryStorageHelper(nullptr,t.getValueMembers(), &converter).getStorageType();
   return mlir::cast<mlir::TupleType>(converter.convertType(mlir::TupleType::get(t.getContext(), {keyTupleType, valTupleType})));
}
static mlir::TupleType getHtKVType(subop::PreAggrHtType t, mlir::TypeConverter& converter) {
   auto keyTupleType = EntryStorageHelper(nullptr,t.getKeyMembers(), &converter).getStorageType();
   auto valTupleType = EntryStorageHelper(nullptr,t.getValueMembers(), &converter).getStorageType();
   return mlir::cast<mlir::TupleType>(converter.convertType(mlir::TupleType::get(t.getContext(), {keyTupleType, valTupleType})));
}
static mlir::TupleType getHtEntryType(subop::HashMapType t, mlir::TypeConverter& converter) {
   auto i8PtrType = util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));
   return mlir::TupleType::get(t.getContext(), {i8PtrType, mlir::IndexType::get(t.getContext()), getHtKVType(t, converter)});
}
static mlir::TupleType getHtEntryType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter) {
   auto i8PtrType = util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));
   return mlir::TupleType::get(t.getContext(), {i8PtrType, mlir::IndexType::get(t.getContext()), getHtKVType(t, converter)});
}
static mlir::TupleType getHtEntryType(subop::PreAggrHtType t, mlir::TypeConverter& converter) {
   auto i8PtrType = util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));
   return mlir::TupleType::get(t.getContext(), {i8PtrType, mlir::IndexType::get(t.getContext()), getHtKVType(t, converter)});
}
static mlir::TupleType getHashMultiMapEntryType(subop::HashMultiMapType t, mlir::TypeConverter& converter) {
   auto keyTupleType = EntryStorageHelper(nullptr,t.getKeyMembers(), &converter).getStorageType();
   auto i8PtrType = util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));
   return mlir::TupleType::get(t.getContext(), {i8PtrType, mlir::IndexType::get(t.getContext()), i8PtrType, keyTupleType});
}
static mlir::TupleType getHashMultiMapValueType(subop::HashMultiMapType t, mlir::TypeConverter& converter) {
   auto valTupleType = EntryStorageHelper(nullptr,t.getValueMembers(), &converter).getStorageType();
   auto i8PtrType = util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));
   return mlir::TupleType::get(t.getContext(), {i8PtrType, valTupleType});
}

static TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      Type converted = typeConverter.convertType(t);
      converted = converted ? converted : t;
      types.push_back(converted);
   }
   return TupleType::get(tupleType.getContext(), TypeRange(types));
}

class TableRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::TableEntryRefType>(refType)) { return failure(); }
      auto columns = mlir::cast<subop::TableEntryRefType>(refType).getMembers();
      auto currRow = mapping.resolve(gatherOp,gatherOp.getRef());
      for (size_t i = 0; i < columns.getTypes().size(); i++) {
         auto memberName = mlir::cast<mlir::StringAttr>(columns.getNames()[i]).str();
         if (gatherOp.getMapping().contains(memberName)) {
            auto columnDefAttr = mlir::cast<tuples::ColumnDefAttr>(gatherOp.getMapping().get(memberName));
            auto type = columnDefAttr.getColumn().type;
            size_t accessOffset = i;
            std::vector<mlir::Type> types;
            types.push_back(getBaseType(type));
            if (mlir::isa<db::NullableType>(type)) {
               types.push_back(rewriter.getI1Type());
            }
            auto atOp = rewriter.create<dsa::At>(gatherOp->getLoc(), types, currRow, accessOffset);
            if (mlir::isa<db::NullableType>(type)) {
               mlir::Value isNull = rewriter.create<db::NotOp>(gatherOp->getLoc(), atOp.getValid());
               mlir::Value val = rewriter.create<db::AsNullableOp>(gatherOp->getLoc(), type, atOp.getVal(), isNull);
               mapping.define(columnDefAttr, val);
            } else {
               mapping.define(columnDefAttr, atOp.getVal());
            }
         }
      }
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};
class MaterializeTableLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::ResultTableType>(materializeOp.getState().getType())) return failure();
      auto stateType = mlir::cast<subop::ResultTableType>(materializeOp.getState().getType());
      mlir::Value loaded = rewriter.create<util::LoadOp>(materializeOp->getLoc(), adaptor.getState());
      auto columnBuilders = rewriter.create<util::UnPackOp>(materializeOp->getLoc(), loaded);
      for (size_t i = 0; i < stateType.getMembers().getTypes().size(); i++) {
         auto memberName = mlir::cast<mlir::StringAttr>(stateType.getMembers().getNames()[i]).str();
         auto attribute = mlir::cast<tuples::ColumnRefAttr>(materializeOp.getMapping().get(memberName));
         auto val = mapping.resolve(materializeOp,attribute);
         mlir::Value valid;
         if (mlir::isa<db::NullableType>(val.getType())) {
            valid = rewriter.create<db::IsNullOp>(materializeOp->getLoc(), val);
            valid = rewriter.create<db::NotOp>(materializeOp->getLoc(), valid);
            val = rewriter.create<db::NullableGetVal>(materializeOp->getLoc(), getBaseType(val.getType()), val);
         }
         rewriter.create<dsa::Append>(materializeOp->getLoc(), columnBuilders.getResult(i), val, valid);
      }
      rewriter.eraseOp(materializeOp);
      return mlir::success();
   }
};
class CreateThreadLocalLowering : public SubOpConversionPattern<subop::CreateThreadLocalOp> {
   public:
   using SubOpConversionPattern<subop::CreateThreadLocalOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateThreadLocalOp createThreadLocal, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      static size_t id = 0;
      ModuleOp parentModule = createThreadLocal->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;
      auto loc = createThreadLocal->getLoc();
      auto i8PtrType = util::RefType::get(rewriter.getI8Type());
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "thread_local_init" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({i8PtrType}), TypeRange(i8PtrType)));
      });
      auto* funcBody = new Block;
      mlir::Value funcArg = funcBody->addArgument(i8PtrType, loc);
      funcOp.getBody().push_back(funcBody);
      mlir::Block* newBlock = new Block;
      mlir::OpBuilder builder(rewriter.getContext());
      std::vector<mlir::Operation*> toInsert;
      builder.setInsertionPointToStart(newBlock);
      mlir::Value argRef = newBlock->addArgument(util::RefType::get(rewriter.getI8Type()), createThreadLocal.getLoc());
      argRef = builder.create<util::GenericMemrefCastOp>(createThreadLocal->getLoc(), util::RefType::get(rewriter.getContext(), util::RefType::get(rewriter.getContext(), rewriter.getI8Type())), argRef);
      for (auto& op : createThreadLocal.getInitFn().front().getOperations()) {
         toInsert.push_back(&op);
      }
      for (auto* op : toInsert) {
         op->remove();
         builder.insert(op);
      }
      std::vector<mlir::Value> toStore;
      std::vector<mlir::Operation*> toDelete;
      newBlock->walk([&](tuples::GetParamVal op) {
         builder.setInsertionPointAfter(op);
         auto idx = toStore.size();
         toStore.push_back(op.getParam());
         mlir::Value rawPtr = builder.create<util::LoadOp>(createThreadLocal->getLoc(), argRef, builder.create<mlir::arith::ConstantIndexOp>(loc, idx));
         mlir::Value ptr = builder.create<util::GenericMemrefCastOp>(loc, util::RefType::get(rewriter.getContext(), op.getParam().getType()), rawPtr);
         mlir::Value value = builder.create<util::LoadOp>(loc, ptr);
         op.replaceAllUsesWith(value);
         toDelete.push_back(op);
      });
      for (auto* op : toDelete) {
         op->erase();
      }
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(newBlock, mlir::ValueRange{funcArg}, [&](tuples::ReturnOpAdaptor adaptor) {
            mlir::Value unrealized = rewriter.create<mlir::UnrealizedConversionCastOp>(createThreadLocal->getLoc(), createThreadLocal.getType().getWrapped(), adaptor.getResults()[0]).getOutputs()[0];
            mlir::Value casted = rewriter.create<util::GenericMemrefCastOp>(createThreadLocal->getLoc(), util::RefType::get(rewriter.getI8Type()), unrealized);
            rewriter.create<mlir::func::ReturnOp>(loc, casted);
         });
      });
      Value arg = rewriter.create<util::AllocOp>(loc, util::RefType::get(i8PtrType), rewriter.create<mlir::arith::ConstantIndexOp>(loc, toStore.size()));
      for (size_t i = 0; i < toStore.size(); i++) {
         Value valPtr = rewriter.create<util::AllocOp>(loc, util::RefType::get(toStore[i].getType()), mlir::Value());

         rewriter.create<util::StoreOp>(loc, toStore[i], valPtr, mlir::Value());
         valPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(rewriter.getI8Type()), valPtr);
         rewriter.create<util::StoreOp>(loc, valPtr, arg, rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
      }
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      rewriter.replaceOp(createThreadLocal, rt::ThreadLocal::create(rewriter, loc)({functionPointer, arg}));
      return mlir::success();
   }
};
class UnrealizedConversionCastLowering : public SubOpConversionPattern<mlir::UnrealizedConversionCastOp> {
   using SubOpConversionPattern<mlir::UnrealizedConversionCastOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      SmallVector<Type> convertedTypes;
      if (succeeded(typeConverter->convertTypes(op.getOutputs().getTypes(),
                                                convertedTypes)) &&
          convertedTypes == adaptor.getInputs().getTypes()) {
         rewriter.replaceOp(op, adaptor.getInputs());
         for (auto z : llvm::zip(op.getOutputs(), adaptor.getInputs())) {
            auto [output, input] = z;
            output.replaceUsesWithIf(input, [&](auto& use) -> bool {
               bool res = !(use.getOwner()->getDialect()->getNamespace() == "subop" || mlir::isa<mlir::UnrealizedConversionCastOp>(use.getOwner()));
               return res;
            });
         }
         return success();
      }
      convertedTypes.clear();
      if (succeeded(typeConverter->convertTypes(adaptor.getInputs().getTypes(),
                                                convertedTypes)) &&
          convertedTypes == op.getOutputs().getType()) {
         rewriter.replaceOp(op, adaptor.getInputs());
         for (auto z : llvm::zip(op.getOutputs(), op.getInputs())) {
            auto [output, input] = z;
            output.replaceUsesWithIf(input, [&](auto& use) -> bool {
               bool res = !(use.getOwner()->getDialect()->getNamespace() == "subop" || mlir::isa<mlir::UnrealizedConversionCastOp>(use.getOwner()));
               return res;
            });
         }
         return success();
      }
      convertedTypes.clear();
      if (succeeded(typeConverter->convertTypes(op.getOutputs().getTypes(), convertedTypes))) {
         auto newOp = rewriter.create<mlir::UnrealizedConversionCastOp>(op->getLoc(), convertedTypes, adaptor.getInputs());
         for (auto z : llvm::zip(op.getOutputs(), newOp.getOutputs())) {
            auto [output, newOutput] = z;
            output.replaceUsesWithIf(newOutput, [&](auto& use) -> bool {
               bool res = !(use.getOwner()->getDialect()->getNamespace() == "subop" || mlir::isa<mlir::UnrealizedConversionCastOp>(use.getOwner()));
               return res;
            });
            rewriter.replaceOp(op, newOp.getOutputs());
         }
         return success();
      }
      return failure();
   }
};
class ScanRefsTableLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::TableType>(scanOp.getState().getType())) return failure();
      auto loc = scanOp->getLoc();
      auto refType = mlir::cast<subop::TableEntryRefType>(scanOp.getRef().getColumn().type);
      std::string memberMapping = "[";
      std::vector<mlir::Type> accessedColumnTypes;
      auto members = refType.getMembers();
      for (auto i = 0ul; i < members.getTypes().size(); i++) {
         auto type = mlir::cast<mlir::TypeAttr>(members.getTypes()[i]).getValue();
         auto name = mlir::cast<mlir::StringAttr>(members.getNames()[i]).str();
         accessedColumnTypes.push_back(type);
         if (memberMapping.length() > 1) {
            memberMapping += ",";
         }
         memberMapping += "\"" + name + "\"";
      }
      memberMapping += "]";
      mlir::Value memberMappingValue = rewriter.create<util::CreateConstVarLen>(scanOp->getLoc(), util::VarLen32Type::get(rewriter.getContext()), memberMapping);
      mlir::Value iterator = rt::DataSourceIteration::init(rewriter, scanOp->getLoc())({adaptor.getState(), memberMappingValue})[0];
      ColumnMapping mapping;

      auto baseTypes = [](mlir::TypeRange arr) {
         std::vector<Type> res;
         for (auto x : arr) { res.push_back(getBaseType(x)); }
         return res;
      };
      auto* ctxt = rewriter.getContext();
      auto tupleType = mlir::TupleType::get(ctxt, baseTypes(accessedColumnTypes));
      auto recordBatchType = dsa::RecordBatchType::get(ctxt, tupleType);
      ModuleOp parentModule = scanOp->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;
      static size_t funcIds;
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "scan_func" + std::to_string(funcIds++), mlir::FunctionType::get(getContext(), TypeRange{ptrType, ptrType}, TypeRange()));
      });
      auto* funcBody = new Block;
      mlir::Value recordBatchPointer = funcBody->addArgument(ptrType, loc);
      mlir::Value contextPtr = funcBody->addArgument(ptrType, loc);
      funcOp.getBody().push_back(funcBody);
      auto ptr = rewriter.storeStepRequirements();
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         rewriter.loadStepRequirements(contextPtr, typeConverter);
         recordBatchPointer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), recordBatchType), recordBatchPointer);

         mlir::Value recordBatch = rewriter.create<util::LoadOp>(loc, recordBatchPointer, mlir::Value());

         auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         auto end = rewriter.create<dsa::GetRecordBatchLen>(loc, recordBatch);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp2 = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
         rewriter.atStartOf(forOp2.getBody(), [&](SubOpRewriter& rewriter) {
            auto currentRecord = rewriter.create<dsa::GetRecord>(loc, recordBatchType.getElementType(), recordBatch, forOp2.getInductionVar());
            mapping.define(scanOp.getRef(), currentRecord);
            rewriter.replaceTupleStream(scanOp, mapping);
         });
         rewriter.create<mlir::func::ReturnOp>(loc);
      });
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      Value parallelConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, scanOp->hasAttr("parallel"), rewriter.getI1Type());
      rt::DataSourceIteration::iterate(rewriter, scanOp->getLoc())({iterator, parallelConst, functionPointer, ptr});
      return success();
   }
};
class MergeThreadLocalResultTable : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ResultTableType>(mergeOp.getType())) return mlir::failure();
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "result_table_merge" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftPtr = rewriter.create<util::GenericMemrefCastOp>(loc, typeConverter->convertType(mergeOp.getType()), left);
            auto rightPtr = rewriter.create<util::GenericMemrefCastOp>(loc, typeConverter->convertType(mergeOp.getType()), right);
            auto leftLoaded = rewriter.create<util::LoadOp>(loc, leftPtr);
            auto rightLoaded = rewriter.create<util::LoadOp>(loc, rightPtr);
            auto leftBuilders = rewriter.create<util::UnPackOp>(loc, leftLoaded);
            auto rightBuilders = rewriter.create<util::UnPackOp>(loc, rightLoaded);
            std::vector<mlir::Value> results;
            for (size_t i = 0; i < leftBuilders.getNumResults(); i++) {
               mlir::Value concatenated = rewriter.create<dsa::Concat>(loc, leftBuilders.getResults()[i].getType(), leftBuilders.getResults()[i], rightBuilders.getResults()[i]);
               results.push_back(concatenated);
            }
            auto packed = rewriter.create<util::PackOp>(loc, results);
            rewriter.create<util::StoreOp>(loc, packed, leftPtr, mlir::Value());
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }

      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));

      mlir::Value merged = rt::ThreadLocal::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};
class CreateFromResultTableLowering : public SubOpConversionPattern<subop::CreateFrom> {
   using SubOpConversionPattern<subop::CreateFrom>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateFrom createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto resultTableType = mlir::dyn_cast<subop::ResultTableType>(createOp.getState().getType());
      if (!resultTableType) return failure();
      mlir::Value loaded = rewriter.create<util::LoadOp>(createOp->getLoc(), adaptor.getState());
      auto columnBuilders = rewriter.create<util::UnPackOp>(createOp->getLoc(), loaded);
      std::vector<mlir::Value> columns;
      std::vector<mlir::Type> types;
      for (auto i = 0ul; i < columnBuilders.getNumResults(); i++) {
         auto columnBuilder = columnBuilders.getResult(i);
         auto eleType = mlir::cast<dsa::ColumnBuilderType>(columnBuilder.getType()).getType();
         types.push_back(eleType);
         auto column = rewriter.create<dsa::FinishColumn>(createOp->getLoc(), dsa::ColumnType::get(getContext(), eleType), columnBuilder);
         columns.push_back(column);
      }
      mlir::Value localTable = rewriter.create<dsa::CreateTable>(createOp->getLoc(), dsa::TableType::get(getContext(), mlir::TupleType::get(getContext(), types)), createOp.getColumns(), columns);
      localTable = rewriter.create<dsa::DownCast>(createOp->getLoc(), typeConverter->convertType(createOp.getType()), localTable);
      rewriter.replaceOp(createOp, localTable);
      return success();
   }
};
class CreateTableLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ResultTableType>(createOp.getType())) return failure();
      auto tableType = mlir::cast<subop::ResultTableType>(createOp.getType());
      std::string descr;
      std::vector<mlir::Value> columnBuilders;
      for (size_t i = 0; i < tableType.getMembers().getTypes().size(); i++) {
         auto type = mlir::cast<mlir::TypeAttr>(tableType.getMembers().getTypes()[i]).getValue();
         auto baseType = getBaseType(type);
         columnBuilders.push_back(rewriter.create<dsa::CreateDS>(createOp.getLoc(), dsa::ColumnBuilderType::get(getContext(), baseType)));
      }
      mlir::Value tpl = rewriter.create<util::PackOp>(createOp->getLoc(), columnBuilders);
      mlir::Value ref = rewriter.create<util::AllocOp>(createOp->getLoc(), util::RefType::get(tpl.getType()), mlir::Value());
      rewriter.create<util::StoreOp>(createOp->getLoc(), tpl, ref, mlir::Value());
      rewriter.replaceOp(createOp, ref);
      return mlir::success();
   }
};
mlir::Value getExecutionContext(SubOpRewriter& rewriter, mlir::Operation* op) {
   auto parentModule = op->getParentOfType<ModuleOp>();
   mlir::func::FuncOp funcOp = parentModule.lookupSymbol<mlir::func::FuncOp>("rt_get_execution_context");
   if (!funcOp) {
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(op->getLoc(), "rt_get_execution_context", mlir::FunctionType::get(op->getContext(), {}, {util::RefType::get(op->getContext(), rewriter.getI8Type())}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
      });
   }
   mlir::Value executionContext = rewriter.create<mlir::func::CallOp>(op->getLoc(), funcOp, mlir::ValueRange{}).getResult(0);
   return executionContext;
}
class GetExternalTableLowering : public SubOpConversionPattern<subop::GetExternalOp> {
   public:
   using SubOpConversionPattern<subop::GetExternalOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GetExternalOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::TableType>(op.getType())) return failure();
      mlir::Value description = rewriter.create<util::CreateConstVarLen>(op->getLoc(), util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());
      rewriter.replaceOp(op, rt::DataSource::get(rewriter, op->getLoc())({getExecutionContext(rewriter, op), description})[0]);
      return mlir::success();
   }
};
class GenerateLowering : public SubOpConversionPattern<subop::GenerateOp> {
   public:
   using SubOpConversionPattern<subop::GenerateOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GenerateOp generateOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      ColumnMapping mapping;
      std::vector<subop::GenerateEmitOp> emitOps;
      generateOp.getRegion().walk([&](subop::GenerateEmitOp emitOp) {
         emitOps.push_back(emitOp);
      });
      std::vector<mlir::Value> streams;
      for (auto emitOp : emitOps) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointAfter(emitOp);
         ColumnMapping mapping;
         mapping.define(generateOp.getGeneratedColumns(), emitOp.getValues());
         mlir::Value newInFlight = rewriter.createInFlight(mapping);
         streams.push_back(newInFlight);
         rewriter.eraseOp(emitOp);
      }

      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&generateOp.getRegion().front(), {}, [](auto x) {});
      for (auto [inflight, stream] : llvm::zip(streams, generateOp.getStreams())) {
         stream.replaceAllUsesWith(inflight);
      }
      rewriter.eraseOp(generateOp);

      return success();
   }
};
class MapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MapOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MapOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MapOp mapOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto args = mapping.resolve(mapOp,mapOp.getInputCols());
      std::vector<Value> res;

      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&mapOp.getFn().front(), args, [&](tuples::ReturnOpAdaptor adaptor) {
         res.insert(res.end(), adaptor.getResults().begin(), adaptor.getResults().end());
      });
      for (auto& r : res) {
         r = rewriter.getMapped(r);
      }
      mapping.define(mapOp.getComputedCols(), res);

      rewriter.replaceTupleStream(mapOp, mapping);

      return success();
   }
};
class InFlightLowering : public SubOpConversionPattern<subop::InFlightOp> {
   public:
   using SubOpConversionPattern<subop::InFlightOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::InFlightOp inFlightOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      ColumnMapping mapping(inFlightOp);
      rewriter.replaceTupleStream(inFlightOp, mapping);
      return success();
   }
};

class CreateBufferLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(createOp.getType());
      if (!bufferType) return failure();
      auto loc = createOp->getLoc();
      EntryStorageHelper storageHelper(createOp,bufferType.getMembers(), typeConverter);
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, createOp->hasAttr("initial_capacity") ? mlir::cast<mlir::IntegerAttr>(createOp->getAttr("initial_capacity")).getInt() : 1024);
      auto elementType = storageHelper.getStorageType();
      auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      mlir::Value executionContext;
      mlir::Value allocator;
      rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getBody().front(), [&](SubOpRewriter& rewriter) {
         executionContext = getExecutionContext(rewriter, createOp);
         if (createOp->hasAttrOfType<mlir::IntegerAttr>("group")) {
            Value groupId = rewriter.create<arith::ConstantIndexOp>(loc, mlir::cast<mlir::IntegerAttr>(createOp->getAttr("group")).getInt());
            allocator = rt::GrowingBufferAllocator::getGroupAllocator(rewriter, loc)({executionContext, groupId})[0];
         } else {
            allocator = rt::GrowingBufferAllocator::getDefaultAllocator(rewriter, loc)({})[0];
         }
      });
      mlir::Value vector = rt::GrowingBuffer::create(rewriter, loc)({allocator, executionContext, typeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, vector);
      return mlir::success();
   }
};

void implementBufferIterationRuntime(bool parallel, mlir::Value bufferIterator, mlir::Type entryType, mlir::Location loc, SubOpRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Operation* op, std::function<void(SubOpRewriter& rewriter, mlir::Value)> fn) {
   auto* ctxt = rewriter.getContext();
   ModuleOp parentModule = bufferIterator.getDefiningOp()->getParentOfType<ModuleOp>();
   mlir::func::FuncOp funcOp;
   static size_t funcIds;
   auto ptrType = util::RefType::get(ctxt, IntegerType::get(ctxt, 8));
   auto plainBufferType = util::BufferType::get(ctxt, mlir::IntegerType::get(ctxt, 8));
   rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
      funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "scan_buffer_func" + std::to_string(funcIds++), mlir::FunctionType::get(ctxt, TypeRange{plainBufferType, ptrType}, TypeRange()));
   });
   auto* funcBody = new Block;
   mlir::Value buffer = funcBody->addArgument(plainBufferType, loc);
   mlir::Value contextPtr = funcBody->addArgument(ptrType, loc);
   funcOp.getBody().push_back(funcBody);
   auto ptr = rewriter.storeStepRequirements();
   rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
      auto guard = rewriter.loadStepRequirements(contextPtr, &typeConverter);
      auto castedBuffer = rewriter.create<util::BufferCastOp>(loc, util::BufferType::get(rewriter.getContext(), entryType), buffer);
      auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end = rewriter.create<util::BufferGetLen>(loc, rewriter.getIndexType(), castedBuffer);
      auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
      rewriter.atStartOf(forOp.getBody(), [&](SubOpRewriter& rewriter) {
         auto currElementPtr = rewriter.create<util::BufferGetElementRef>(loc, util::RefType::get(entryType), castedBuffer, forOp.getInductionVar());
         fn(rewriter, currElementPtr);
      });
      rewriter.create<mlir::func::ReturnOp>(loc);
   });
   Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
   Value parallelConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, parallel, rewriter.getI1Type());
   rt::BufferIterator::iterate(rewriter, loc)({bufferIterator, parallelConst, functionPointer, ptr});
}
void implementBufferIteration(bool parallel, mlir::Value bufferIterator, mlir::Type entryType, mlir::Location loc, SubOpRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Operation* op, std::function<void(SubOpRewriter& rewriter, mlir::Value)> fn) {
   implementBufferIterationRuntime(parallel, bufferIterator, entryType, loc, rewriter, typeConverter, op, fn);
}
class ScanRefsVectorLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(scanOp.getState().getType());
      if (!bufferType) return failure();
      ColumnMapping mapping;
      auto elementType = EntryStorageHelper(scanOp,bufferType.getMembers(), typeConverter).getStorageType();

      auto iterator = rt::GrowingBuffer::createIterator(rewriter, scanOp->getLoc())(adaptor.getState())[0];
      implementBufferIteration(scanOp->hasAttr("parallel"), iterator, elementType, scanOp->getLoc(), rewriter, *typeConverter, scanOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         mapping.define(scanOp.getRef(), ptr);
         rewriter.replaceTupleStream(scanOp, mapping);
      });
      return success();
   }
};

class CreateHashMapLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMapType>(createOp.getType())) return failure();
      auto t = mlir::cast<subop::HashMapType>(createOp.getType());

      auto typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHtEntryType(t, *typeConverter));
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(createOp->getLoc(), 4);
      auto ptr = rt::Hashtable::create(rewriter, createOp->getLoc())({getExecutionContext(rewriter, createOp), typeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, ptr);
      return mlir::success();
   }
};
class CreateHashMultiMapLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMultiMapType>(createOp.getType())) return failure();
      auto t = mlir::cast<subop::HashMultiMapType>(createOp.getType());

      auto entryTypeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHashMultiMapEntryType(t, *typeConverter));
      auto valueTypeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHashMultiMapValueType(t, *typeConverter));
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(createOp->getLoc(), 4);
      auto ptr = rt::HashMultiMap::create(rewriter, createOp->getLoc())({getExecutionContext(rewriter, createOp), entryTypeSize, valueTypeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, ptr);
      return mlir::success();
   }
};
class CreateOpenHtFragmentLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::PreAggrHtFragmentType>(createOp.getType())) return failure();
      auto t = mlir::cast<subop::PreAggrHtFragmentType>(createOp.getType());

      auto typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHtEntryType(t, *typeConverter));
      auto ptr = rt::PreAggregationHashtableFragment::create(rewriter, createOp->getLoc())({getExecutionContext(rewriter, createOp), typeSize})[0];
      rewriter.replaceOpWithNewOp<util::GenericMemrefCastOp>(createOp, typeConverter->convertType(t), ptr);
      return mlir::success();
   }
};

class CreateArrayLowering : public SubOpConversionPattern<subop::CreateArrayOp> {
   public:
   using SubOpConversionPattern<subop::CreateArrayOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateArrayOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto arrayType = createOp.getType();
      auto loc = createOp->getLoc();
      EntryStorageHelper storageHelper(createOp,arrayType.getMembers(), typeConverter);

      Value tpl = rewriter.create<util::LoadOp>(loc, adaptor.getNumElements());
      Value numElements = rewriter.create<util::UnPackOp>(loc, tpl).getResults()[0];
      auto elementType = storageHelper.getStorageType();
      auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      auto numBytes = rewriter.create<mlir::arith::MulIOp>(loc, typeSize, numElements);
      mlir::Value vector = rt::Buffer::createZeroed(rewriter, loc)({getExecutionContext(rewriter, createOp), numBytes})[0];
      rewriter.replaceOpWithNewOp<util::BufferCastOp>(createOp, typeConverter->convertType(createOp.getType()), vector);
      return mlir::success();
   }
};

class SetResultOpLowering : public SubOpConversionPattern<subop::SetResultOp> {
   public:
   using SubOpConversionPattern<subop::SetResultOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::SetResultOp setResultOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<dsa::SetResultOp>(setResultOp, setResultOp.getResultId(), adaptor.getState());
      return mlir::success();
   }
};

class CreateSegmentTreeViewLowering : public SubOpConversionPattern<subop::CreateSegmentTreeView> {
   public:
   using SubOpConversionPattern<subop::CreateSegmentTreeView>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateSegmentTreeView createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      static size_t id = 0;
      auto loc = createOp->getLoc();
      auto continuousType = createOp.getSource().getType();

      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));

      ModuleOp parentModule = createOp->getParentOfType<ModuleOp>();
      EntryStorageHelper sourceStorageHelper(createOp,continuousType.getMembers(), typeConverter);
      EntryStorageHelper viewStorageHelper(createOp,createOp.getType().getValueMembers(), typeConverter);
      mlir::TupleType sourceElementType = sourceStorageHelper.getStorageType();
      mlir::TupleType viewElementType = viewStorageHelper.getStorageType();

      mlir::func::FuncOp initialFn;
      mlir::func::FuncOp combineFn;
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         initialFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "segment_tree_initial_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "segment_tree_combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
         initialFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            Value dest = funcBody->getArgument(0);
            Value src = funcBody->getArgument(1);
            auto sourceValues = sourceStorageHelper.loadValues(src, rewriter, loc);
            std::vector<mlir::Value> args;
            for (auto relevantMember : createOp.getRelevantMembers()) {
               args.push_back(sourceValues.at(mlir::cast<mlir::StringAttr>(relevantMember).str()));
            }
            Block* sortLambda = &createOp.getInitialFn().front();
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
               viewStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
            });
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }
      {
         auto* funcBody = new Block;
         Value dest = funcBody->addArgument(ptrType, loc);
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftValues = viewStorageHelper.loadValuesOrdered(left, rewriter, loc);
            auto rightValues = viewStorageHelper.loadValuesOrdered(right, rewriter, loc);
            std::vector<mlir::Value> args;
            args.insert(args.end(), leftValues.begin(), leftValues.end());
            args.insert(args.end(), rightValues.begin(), rightValues.end());
            Block* sortLambda = &createOp.getCombineFn().front();
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
               viewStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
            });
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }

      Value initialFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, initialFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(initialFn.getSymName())));
      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      //auto genericBuffer = rt::GrowingBuffer::sort(rewriter, loc)({adaptor.getToSort(), functionPointer})[0];
      Value sourceEntryTypeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), sourceElementType);
      Value stateTypeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), viewElementType);
      mlir::Value res = rt::SegmentTreeView::build(rewriter, loc)({getExecutionContext(rewriter, createOp), adaptor.getSource(), sourceEntryTypeSize, initialFnPtr, combineFnPtr, stateTypeSize})[0];
      rewriter.replaceOp(createOp, res);
      return mlir::success();
   }
};

class CreateHeapLowering : public SubOpConversionPattern<subop::CreateHeapOp> {
   public:
   using SubOpConversionPattern<subop::CreateHeapOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateHeapOp heapOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      static size_t id = 0;
      auto heapType = heapOp.getType();
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      EntryStorageHelper storageHelper(heapOp,heapType.getMembers(), typeConverter);
      ModuleOp parentModule = heapOp->getParentOfType<ModuleOp>();
      mlir::TupleType elementType = storageHelper.getStorageType();
      auto loc = heapOp.getLoc();
      mlir::func::FuncOp funcOp;
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "dsa_heap_compare" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
      });
      auto* funcBody = new Block;
      Value left = funcBody->addArgument(ptrType, loc);
      Value right = funcBody->addArgument(ptrType, loc);
      funcOp.getBody().push_back(funcBody);
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto leftVals = storageHelper.loadValuesOrdered(left, rewriter, loc, heapOp.getSortBy());
         auto rightVals = storageHelper.loadValuesOrdered(right, rewriter, loc, heapOp.getSortBy());
         std::vector<mlir::Value> args;
         args.insert(args.end(), leftVals.begin(), leftVals.end());
         args.insert(args.end(), rightVals.begin(), rightVals.end());
         Block* sortLambda = &heapOp.getRegion().front();
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
            rewriter.create<mlir::func::ReturnOp>(loc, adaptor.getResults());
         });
      });
      Value typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      Value maxElements = rewriter.create<mlir::arith::ConstantIndexOp>(loc, heapType.getMaxElements());
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      auto heap = rt::Heap::create(rewriter, loc)({getExecutionContext(rewriter, heapOp), maxElements, typeSize, functionPointer})[0];
      rewriter.replaceOp(heapOp, heap);
      return mlir::success();
   }
};

class MergeThreadLocalBuffer : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::BufferType>(mergeOp.getType())) return mlir::failure();
      mlir::Value merged = rt::GrowingBuffer::merge(rewriter, mergeOp->getLoc())(adaptor.getThreadLocal())[0];
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};
class MergeThreadLocalHeap : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HeapType>(mergeOp.getType())) return mlir::failure();
      mlir::Value merged = rt::Heap::merge(rewriter, mergeOp->getLoc())(adaptor.getThreadLocal())[0];
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};
class MergeThreadLocalSimpleState : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::SimpleStateType>(mergeOp.getType())) return mlir::failure();
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();
      EntryStorageHelper storageHelper(mergeOp,mlir::cast<subop::SimpleStateType>(mergeOp.getType()).getMembers(), typeConverter);

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "simple_state__combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value dest = left;
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftValues = storageHelper.loadValuesOrdered(left, rewriter, loc);
            auto rightValues = storageHelper.loadValuesOrdered(right, rewriter, loc);
            std::vector<mlir::Value> args;
            args.insert(args.end(), leftValues.begin(), leftValues.end());
            args.insert(args.end(), rightValues.begin(), rightValues.end());
            for (size_t i = 0; i < args.size(); i++) {
               auto expectedType = mergeOp.getCombineFn().front().getArgument(i).getType();
               if (args[i].getType() != expectedType) {
                  args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
               }
            }
            Block* sortLambda = &mergeOp.getCombineFn().front();
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
               storageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
            });
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }

      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      mlir::Value merged = rt::SimpleState::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};
class MergeThreadLocalHashMap : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMapType>(mergeOp.getType())) return mlir::failure();
      auto hashMapType = mlir::cast<subop::HashMapType>(mergeOp.getType());
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      mlir::func::FuncOp eqFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();
      EntryStorageHelper keyStorageHelper(mergeOp,hashMapType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(mergeOp,hashMapType.getValueMembers(), typeConverter);

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         eqFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "hashmap_eq_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "hashmap_combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            if (!mergeOp.getCombineFn().empty()) {
               auto kvType = getHtKVType(hashMapType, *typeConverter);
               auto kvPtrType = util::RefType::get(context, kvType);
               left = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, left);
               right = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, right);

               left = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), left, 1);
               right = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), right, 1);
               Value dest = left;
               auto leftValues = valStorageHelper.loadValuesOrdered(left, rewriter, loc);
               auto rightValues = valStorageHelper.loadValuesOrdered(right, rewriter, loc);
               std::vector<mlir::Value> args;
               args.insert(args.end(), leftValues.begin(), leftValues.end());
               args.insert(args.end(), rightValues.begin(), rightValues.end());
               for (size_t i = 0; i < args.size(); i++) {
                  auto expectedType = mergeOp.getCombineFn().front().getArgument(i).getType();
                  if (args[i].getType() != expectedType) {
                     args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
                  }
               }
               Block* sortLambda = &mergeOp.getCombineFn().front();
               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
                  valStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
               });
            }
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         eqFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftKeys = keyStorageHelper.loadValuesOrdered(left, rewriter, loc);
            auto rightKeys = keyStorageHelper.loadValuesOrdered(right, rewriter, loc);
            std::vector<mlir::Value> args;
            args.insert(args.end(), leftKeys.begin(), leftKeys.end());
            args.insert(args.end(), rightKeys.begin(), rightKeys.end());
            auto res = inlineBlock(&mergeOp.getEqFn().front(), rewriter, args)[0];
            rewriter.create<mlir::func::ReturnOp>(loc, res);
         });
      }

      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      Value eqFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, eqFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(eqFn.getSymName())));
      mlir::Value merged = rt::Hashtable::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), eqFnPtr, combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};
class MergePreAggrHashMap : public SubOpConversionPattern<subop::MergeOp> {
   public:
   using SubOpConversionPattern<subop::MergeOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::MergeOp mergeOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::PreAggrHtType>(mergeOp.getType())) return mlir::failure();
      auto hashMapType = mlir::cast<subop::PreAggrHtType>(mergeOp.getType());
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      auto loc = mergeOp->getLoc();
      mlir::func::FuncOp combineFn;
      mlir::func::FuncOp eqFn;
      static size_t id = 0;
      ModuleOp parentModule = mergeOp->getParentOfType<ModuleOp>();
      EntryStorageHelper keyStorageHelper(mergeOp,hashMapType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(mergeOp,hashMapType.getValueMembers(), typeConverter);

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         eqFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "optimistic_ht_eq_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
         combineFn = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "optimistic_ht_combine_fn" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange()));
      });
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         combineFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            if (!mergeOp.getCombineFn().empty()) {
               auto kvType = getHtKVType(hashMapType, *typeConverter);
               auto kvPtrType = util::RefType::get(context, kvType);
               left = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, left);
               right = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), kvPtrType, right);

               left = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), left, 1);
               right = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), right, 1);
               Value dest = left;
               auto leftValues = valStorageHelper.loadValuesOrdered(left, rewriter, loc);
               auto rightValues = valStorageHelper.loadValuesOrdered(right, rewriter, loc);
               std::vector<mlir::Value> args;
               args.insert(args.end(), leftValues.begin(), leftValues.end());
               args.insert(args.end(), rightValues.begin(), rightValues.end());
               for (size_t i = 0; i < args.size(); i++) {
                  auto expectedType = mergeOp.getCombineFn().front().getArgument(i).getType();
                  if (args[i].getType() != expectedType) {
                     args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
                  }
               }
               Block* sortLambda = &mergeOp.getCombineFn().front();
               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
                  valStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
               });
            }
            rewriter.create<mlir::func::ReturnOp>(loc);
         });
      }
      {
         auto* funcBody = new Block;
         Value left = funcBody->addArgument(ptrType, loc);
         Value right = funcBody->addArgument(ptrType, loc);
         eqFn.getBody().push_back(funcBody);
         rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
            auto leftKeys = keyStorageHelper.loadValuesOrdered(left, rewriter, loc);
            auto rightKeys = keyStorageHelper.loadValuesOrdered(right, rewriter, loc);
            std::vector<mlir::Value> args;
            args.insert(args.end(), leftKeys.begin(), leftKeys.end());
            args.insert(args.end(), rightKeys.begin(), rightKeys.end());
            auto res = inlineBlock(&mergeOp.getEqFn().front(), rewriter, args)[0];
            rewriter.create<mlir::func::ReturnOp>(loc, res);
         });
      }
      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      Value eqFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, eqFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(eqFn.getSymName())));
      mlir::Value merged = rt::PreAggregationHashtable::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{getExecutionContext(rewriter, mergeOp), adaptor.getThreadLocal(), eqFnPtr, combineFnPtr})[0];
      merged = rewriter.create<util::GenericMemrefCastOp>(mergeOp->getLoc(), typeConverter->convertType(mergeOp.getType()), merged);
      rewriter.replaceOp(mergeOp, merged);
      return mlir::success();
   }
};
class SortLowering : public SubOpConversionPattern<subop::CreateSortedViewOp> {
   public:
   using SubOpConversionPattern<subop::CreateSortedViewOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::CreateSortedViewOp sortOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      static size_t id = 0;
      auto bufferType = mlir::cast<subop::BufferType>(sortOp.getToSort().getType());
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      EntryStorageHelper storageHelper(sortOp,bufferType.getMembers(), typeConverter);
      ModuleOp parentModule = sortOp->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "dsa_sort_compare" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
      });
      auto* funcBody = new Block;
      funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
      funcOp.getBody().push_back(funcBody);
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto leftVals = storageHelper.loadValuesOrdered(funcBody->getArgument(0), rewriter, sortOp->getLoc(), sortOp.getSortBy());
         auto rightVals = storageHelper.loadValuesOrdered(funcBody->getArgument(1), rewriter, sortOp->getLoc(), sortOp.getSortBy());
         std::vector<mlir::Value> args;
         args.insert(args.end(), leftVals.begin(), leftVals.end());
         args.insert(args.end(), rightVals.begin(), rightVals.end());
         Block* sortLambda = &sortOp.getRegion().front();
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
            rewriter.create<mlir::func::ReturnOp>(sortOp->getLoc(), adaptor.getResults());
         });
      });

      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(sortOp->getLoc(), funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      auto genericBuffer = rt::GrowingBuffer::sort(rewriter, sortOp->getLoc())({adaptor.getToSort(), getExecutionContext(rewriter, sortOp), functionPointer})[0];
      rewriter.replaceOpWithNewOp<util::BufferCastOp>(sortOp, typeConverter->convertType(sortOp.getType()), genericBuffer);
      return mlir::success();
   }
};

class ScanRefsSimpleStateLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::SimpleStateType>(scanOp.getState().getType())) return failure();
      ColumnMapping mapping;
      mapping.define(scanOp.getRef(), adaptor.getState());
      rewriter.replaceTupleStream(scanOp, mapping);
      return success();
   }
};

class ScanRefsSortedViewLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto sortedViewType = mlir::dyn_cast_or_null<subop::SortedViewType>(scanOp.getState().getType());
      if (!sortedViewType) return failure();
      ColumnMapping mapping;
      auto elementType = util::RefType::get(getContext(), EntryStorageHelper(scanOp,sortedViewType.getMembers(), typeConverter).getStorageType());
      auto loc = scanOp->getLoc();
      auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end = rewriter.create<util::BufferGetLen>(loc, rewriter.getIndexType(), adaptor.getState());
      auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
      rewriter.atStartOf(forOp.getBody(), [&](SubOpRewriter& rewriter) {
         auto currElementPtr = rewriter.create<util::BufferGetElementRef>(loc, util::RefType::get(elementType), adaptor.getState(), forOp.getInductionVar());
         mapping.define(scanOp.getRef(), currElementPtr);
         rewriter.replaceTupleStream(scanOp, mapping);
      });

      return success();
   }
};

class ScanRefsHeapLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto heapType = mlir::dyn_cast_or_null<subop::HeapType>(scanOp.getState().getType());
      if (!heapType) return failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      EntryStorageHelper storageHelper(scanOp,heapType.getMembers(), typeConverter);
      mlir::TupleType elementType = storageHelper.getStorageType();
      auto buffer = rt::Heap::getBuffer(rewriter, scanOp->getLoc())({adaptor.getState()})[0];
      auto castedBuffer = rewriter.create<util::BufferCastOp>(loc, util::BufferType::get(rewriter.getContext(), elementType), buffer);
      auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end = rewriter.create<util::BufferGetLen>(loc, rewriter.getIndexType(), castedBuffer);
      auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
      rewriter.atStartOf(forOp.getBody(), [&](SubOpRewriter& rewriter) {
         auto currElementPtr = rewriter.create<util::BufferGetElementRef>(loc, util::RefType::get(elementType), castedBuffer, forOp.getInductionVar());
         mapping.define(scanOp.getRef(), currElementPtr);
         rewriter.replaceTupleStream(scanOp, mapping);
      });

      return success();
   }
};
class ScanRefsContinuousViewLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ContinuousViewType, subop::ArrayType>(scanOp.getState().getType())) return failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      auto bufferType = mlir::cast<util::BufferType>(adaptor.getState().getType());
      mlir::Value typeSize = rewriter.create<util::SizeOfOp>(scanOp->getLoc(), rewriter.getIndexType(), typeConverter->convertType(bufferType.getT()));

      auto* ctxt = rewriter.getContext();
      ModuleOp parentModule = typeSize.getDefiningOp()->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;
      static size_t funcIds;
      auto ptrType = util::RefType::get(ctxt, IntegerType::get(ctxt, 8));
      auto plainBufferType = util::BufferType::get(ctxt, mlir::IntegerType::get(ctxt, 8));
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "scan_cv_func" + std::to_string(funcIds++), mlir::FunctionType::get(ctxt, TypeRange{plainBufferType, rewriter.getI64Type(), rewriter.getI64Type(), ptrType}, TypeRange()));
      });
      auto* funcBody = new Block;
      mlir::Value buffer = funcBody->addArgument(plainBufferType, loc);
      mlir::Value startPos = funcBody->addArgument(rewriter.getI64Type(), loc);
      mlir::Value endPos = funcBody->addArgument(rewriter.getI64Type(), loc);

      mlir::Value contextPtr = funcBody->addArgument(ptrType, loc);
      funcOp.getBody().push_back(funcBody);
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto guard = rewriter.loadStepRequirements(contextPtr, typeConverter);
         auto castedBuffer = rewriter.create<util::BufferCastOp>(loc, bufferType, buffer);
         startPos = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), startPos);
         endPos = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), endPos);
         auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp = rewriter.create<mlir::scf::ForOp>(scanOp->getLoc(), startPos, endPos, one);
         mlir::Block* block = forOp.getBody();
         rewriter.atStartOf(block, [&](SubOpRewriter& rewriter) {
            auto pair = rewriter.create<util::PackOp>(loc, mlir::ValueRange{forOp.getInductionVar(), castedBuffer});
            mapping.define(scanOp.getRef(), pair);
            rewriter.replaceTupleStream(scanOp, mapping);
         });
         rewriter.create<mlir::func::ReturnOp>(loc);
      });
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      Value parallelConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, scanOp->hasAttr("parallel"), rewriter.getI1Type());
      rt::Buffer::iterate(rewriter, loc)({parallelConst, adaptor.getState(), typeSize, functionPointer, rewriter.storeStepRequirements()});
      return mlir::success();

      return success();
   }
};

class ScanHashMapLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanRefsOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::HashMapType>(scanRefsOp.getState().getType())) return failure();
      auto hashMapType = mlir::cast<subop::HashMapType>(scanRefsOp.getState().getType());
      ColumnMapping mapping;
      auto loc = scanRefsOp->getLoc();
      auto it = rt::Hashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashMapType, *typeConverter));
      implementBufferIteration(scanRefsOp->hasAttr("parallel"), it, getHtEntryType(hashMapType, *typeConverter), loc, rewriter, *typeConverter, scanRefsOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         mapping.define(scanRefsOp.getRef(), kvPtr);
         rewriter.replaceTupleStream(scanRefsOp, mapping);
      });
      return success();
   }
};
class ScanPreAggrHtLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanRefsOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::PreAggrHtType>(scanRefsOp.getState().getType())) return failure();
      auto hashMapType = mlir::cast<subop::PreAggrHtType>(scanRefsOp.getState().getType());
      ColumnMapping mapping;
      auto loc = scanRefsOp->getLoc();
      auto it = rt::PreAggregationHashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashMapType, *typeConverter));
      implementBufferIteration(scanRefsOp->hasAttr("parallel"), it, util::RefType::get(getContext(), getHtEntryType(hashMapType, *typeConverter)), loc, rewriter, *typeConverter, scanRefsOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         ptr = rewriter.create<util::LoadOp>(loc, ptr, mlir::Value());
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         mapping.define(scanRefsOp.getRef(), kvPtr);
         rewriter.replaceTupleStream(scanRefsOp, mapping);
      });
      return success();
   }
};
class ScanHashMultiMap : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanRefsOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(scanRefsOp.getState().getType());
      if (!hashMultiMapType) return failure();
      ColumnMapping mapping;
      auto loc = scanRefsOp->getLoc();
      auto it = rt::Hashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      EntryStorageHelper valStorageHelper(scanRefsOp,hashMultiMapType.getValueMembers(), typeConverter);
      EntryStorageHelper keyStorageHelper(scanRefsOp,hashMultiMapType.getKeyMembers(), typeConverter);
      auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
      auto i8PtrPtrType = util::RefType::get(getContext(), i8PtrType);

      implementBufferIteration(scanRefsOp->hasAttr("parallel"), it, getHashMultiMapEntryType(hashMultiMapType, *typeConverter), loc, rewriter, *typeConverter, scanRefsOp.getOperation(), [&](SubOpRewriter& rewriter, mlir::Value ptr) {
         auto keyPtr = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ptr, 3);
         auto valueListPtr = rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, ptr, 2);
         mlir::Value valuePtr = rewriter.create<util::LoadOp>(loc, valueListPtr);
         auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, i8PtrType, valuePtr);
         Block* before = new Block;
         Block* after = new Block;
         whileOp.getBefore().push_back(before);
         whileOp.getAfter().push_back(after);
         mlir::Value beforePtr = before->addArgument(i8PtrType, loc);
         mlir::Value afterPtr = after->addArgument(i8PtrType, loc);
         rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
            mlir::Value cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), beforePtr);
            rewriter.create<mlir::scf::ConditionOp>(loc, cond, beforePtr);
         });
         rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
            Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, valStorageHelper.getStorageType()})), afterPtr);
            Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), castedPtr, 1);
            Value packed = rewriter.create<util::PackOp>(loc, mlir::ValueRange{keyPtr, valuePtr});
            mapping.define(scanRefsOp.getRef(), packed);
            rewriter.replaceTupleStream(scanRefsOp, mapping);
            Value nextPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), i8PtrType), castedPtr, 0);
            mlir::Value next = rewriter.create<util::LoadOp>(loc, nextPtr, mlir::Value());
            rewriter.create<mlir::scf::YieldOp>(loc, next);
         });
      });
      return success();
   }
};
class ScanHashMapListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      bool onlyValues = false;
      subop::HashMapType hashmapType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         hashmapType = mlir::dyn_cast_or_null<subop::HashMapType>(lookupRefType.getState());
         onlyValues = true;
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::HashMapEntryRefType>(listType.getT())) {
         hashmapType = entryRefType.getHashMap();
      }

      if (!hashmapType) return mlir::failure();
      auto loc = scanOp.getLoc();
      ColumnMapping mapping;
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), adaptor.getList());
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      auto valPtrType = util::RefType::get(getContext(), mlir::TupleType::get(getContext(), unpackTypes(hashmapType.getValueMembers().getTypes())));
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, scanOp->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, adaptor.getList());
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         if (onlyValues) {
            auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
            mapping.define(scanOp.getElem(), valuePtr);
         } else {
            mapping.define(scanOp.getElem(), kvPtr);
         }
         rewriter.replaceTupleStream(scanOp, mapping);
      });
      return success();
   }
};
class ScanPreAggregationHtListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      bool onlyValues = false;
      subop::PreAggrHtType hashmapType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         hashmapType = mlir::dyn_cast_or_null<subop::PreAggrHtType>(lookupRefType.getState());
         onlyValues = true;
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::PreAggrHTEntryRefType>(listType.getT())) {
         hashmapType = entryRefType.getHashMap();
      }

      if (!hashmapType) return mlir::failure();
      auto loc = scanOp.getLoc();
      ColumnMapping mapping;
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), adaptor.getList());
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      auto valPtrType = util::RefType::get(getContext(), mlir::TupleType::get(getContext(), unpackTypes(hashmapType.getValueMembers().getTypes())));
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, scanOp->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, adaptor.getList());
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         if (onlyValues) {
            auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
            mapping.define(scanOp.getElem(), valuePtr);
         } else {
            mapping.define(scanOp.getElem(), kvPtr);
         }
         rewriter.replaceTupleStream(scanOp, mapping);
      });
      return success();
   }
};

class ScanListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT());
      if (!lookupRefType) return mlir::failure();
      auto hashIndexedViewType = mlir::dyn_cast_or_null<subop::HashIndexedViewType>(lookupRefType.getState());
      if (!hashIndexedViewType) return mlir::failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      auto unpacked = rewriter.create<util::UnPackOp>(loc, adaptor.getList());
      auto ptr = unpacked.getResult(0);
      auto hash = unpacked.getResult(1);
      auto iteratorType = ptr.getType();
      auto referenceType = mlir::cast<subop::ListType>(scanOp.getList().getType()).getT();

      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, iteratorType, ptr);
      Block* before = new Block;
      Block* after = new Block;
      whileOp.getBefore().push_back(before);
      whileOp.getAfter().push_back(after);

      mlir::Value beforePtr = before->addArgument(iteratorType, loc);
      mlir::Value afterPtr = after->addArgument(iteratorType, loc);
      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         mlir::Value cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), beforePtr);
         rewriter.create<mlir::scf::ConditionOp>(loc, cond, beforePtr);
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         auto tupleType = mlir::TupleType::get(getContext(), unpackTypes(referenceType.getMembers().getTypes()));
         auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
         Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, rewriter.getIndexType(), tupleType})), afterPtr);
         Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), tupleType), castedPtr, 2);
         Value hashPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), rewriter.getIndexType()), castedPtr, 1);
         mlir::Value currHash = rewriter.create<util::LoadOp>(loc, hashPtr, mlir::Value());
         mlir::Value hashEq = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, currHash, hash);
         rewriter.create<mlir::scf::IfOp>(
            loc, hashEq, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               mapping.define(scanOp.getElem(), valuePtr);
               rewriter.replaceTupleStream(scanOp, mapping);
               builder1.create<mlir::scf::YieldOp>(loc);
            });

         Value nextPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), i8PtrType), castedPtr, 0);
         mlir::Value next = rewriter.create<util::LoadOp>(loc, nextPtr, mlir::Value());
         next = rewriter.create<util::FilterTaggedPtr>(loc, next.getType(), next, hash);
         rewriter.create<mlir::scf::YieldOp>(loc, next);
      });

      return success();
   }
};
class ScanExternalHashIndexListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();

      subop::ExternalHashIndexType externalHashIndexType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         if (!(externalHashIndexType = mlir::dyn_cast_or_null<subop::ExternalHashIndexType>(lookupRefType.getState()))) {
            return mlir::failure();
         };
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::ExternalHashIndexEntryRefType>(listType.getT())) {
         externalHashIndexType = entryRefType.getExternalHashIndex();
      } else {
         return mlir::failure();
      }

      auto loc = scanOp->getLoc();
      auto* ctxt = rewriter.getContext();

      // Get correct types
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(externalHashIndexType.getMembers().getTypes()));
      auto baseTypes = [](mlir::TypeRange arr) {
         std::vector<Type> res;
         for (auto x : arr) { res.push_back(getBaseType(x)); }
         return res;
      };
      mlir::TypeRange typeRange{tupleType.getTypes()};
      auto convertedTupleType = mlir::TupleType::get(ctxt, baseTypes(typeRange));
      auto recordBatchType = dsa::RecordBatchType::get(ctxt, convertedTupleType);
      auto convertedListType = typeConverter->convertType(listType);

      // Create while loop to extract all chained values from hash table
      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, convertedListType, adaptor.getList());
      Block* conditionBlock = new Block;
      Block* bodyBlock = new Block;
      whileOp.getBefore().push_back(conditionBlock);
      whileOp.getAfter().push_back(bodyBlock);

      conditionBlock->addArgument(convertedListType, loc);
      bodyBlock->addArgument(convertedListType, loc);
      ColumnMapping mapping;

      // Check if iterator contains another value
      rewriter.atStartOf(conditionBlock, [&](SubOpRewriter& rewriter) {
         mlir::Value list = conditionBlock->getArgument(0);
         mlir::Value cont = rt::HashIndexIteration::hasNext(rewriter, loc)({list})[0];
         rewriter.create<scf::ConditionOp>(loc, cont, ValueRange({list}));
      });

      // Load record batch from iterator
      rewriter.atStartOf(bodyBlock, [&](SubOpRewriter& rewriter) {
         mlir::Value list = bodyBlock->getArgument(0);
         mlir::Value recordBatchPointer;
         rewriter.atStartOf(&scanOp->getParentOfType<mlir::func::FuncOp>().getBody().front(), [&](SubOpRewriter& rewriter) {
            recordBatchPointer = rewriter.create<util::AllocaOp>(loc, util::RefType::get(rewriter.getContext(), recordBatchType), mlir::Value());
         });
         rt::HashIndexIteration::consumeRecordBatch(rewriter, loc)({list, recordBatchPointer});
         mlir::Value recordBatch = rewriter.create<util::LoadOp>(loc, recordBatchPointer, mlir::Value());

         auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         auto end = rewriter.create<dsa::GetRecordBatchLen>(loc, recordBatch);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp2 = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
         rewriter.atStartOf(forOp2.getBody(), [&](SubOpRewriter& rewriter) {
            auto currentRecord = rewriter.create<dsa::GetRecord>(loc, recordBatchType.getElementType(), recordBatch, forOp2.getInductionVar());
            mapping.define(scanOp.getElem(), currentRecord);
            rewriter.replaceTupleStream(scanOp, mapping);
         });
         rewriter.create<mlir::scf::YieldOp>(loc, list);
      });

      // Close iterator
      rt::HashIndexIteration::close(rewriter, loc)({adaptor.getList()});
      return success();
   }
};
class ScanMultiMapListLowering : public SubOpConversionPattern<subop::ScanListOp> {
   public:
   using SubOpConversionPattern<subop::ScanListOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanListOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto listType = mlir::dyn_cast_or_null<subop::ListType>(scanOp.getList().getType());
      if (!listType) return mlir::failure();
      bool onlyValues = false;
      subop::HashMultiMapType hashMultiMapType;
      if (auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(listType.getT())) {
         hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(lookupRefType.getState());
         onlyValues = true;
      } else if (auto entryRefType = mlir::dyn_cast_or_null<subop::HashMultiMapEntryRefType>(listType.getT())) {
         hashMultiMapType = entryRefType.getHashMultimap();
      }
      if (!hashMultiMapType) return mlir::failure();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      auto ptr = adaptor.getList();
      EntryStorageHelper keyStorageHelper(scanOp,hashMultiMapType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(scanOp,hashMultiMapType.getValueMembers(), typeConverter);
      auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
      auto i8PtrPtrType = util::RefType::get(getContext(), i8PtrType);
      Value ptrValid = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), ptr);
      mlir::Value valuePtr = rewriter.create<scf::IfOp>(
                                        loc, ptrValid, [&](OpBuilder& b, Location loc) {
                                           Value valuePtrPtr = rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, ptr, 2);
                                           Value valuePtr = rewriter.create<util::LoadOp>(loc, valuePtrPtr);
                                           b.create<scf::YieldOp>(loc,valuePtr); }, [&](OpBuilder& b, Location loc) {
                                           Value invalidPtr=rewriter.create<util::InvalidRefOp>(loc,i8PtrType);
                                           b.create<scf::YieldOp>(loc, invalidPtr); })
                                .getResult(0);
      Value keyPtr = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ptr, 3);

      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, i8PtrType, valuePtr);
      Block* before = new Block;
      Block* after = new Block;
      whileOp.getBefore().push_back(before);
      whileOp.getAfter().push_back(after);

      mlir::Value beforePtr = before->addArgument(i8PtrType, loc);
      mlir::Value afterPtr = after->addArgument(i8PtrType, loc);
      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         mlir::Value cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), beforePtr);
         rewriter.create<mlir::scf::ConditionOp>(loc, cond, beforePtr);
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         auto i8PtrType = util::RefType::get(getContext(), rewriter.getI8Type());
         Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, valStorageHelper.getStorageType()})), afterPtr);
         Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), castedPtr, 1);
         if (onlyValues) {
            mapping.define(scanOp.getElem(), valuePtr);
            rewriter.replaceTupleStream(scanOp, mapping);
         } else {
            Value packed = rewriter.create<util::PackOp>(loc, mlir::ValueRange{keyPtr, valuePtr});
            mapping.define(scanOp.getElem(), packed);
            rewriter.replaceTupleStream(scanOp, mapping);
         }
         Value nextPtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), i8PtrType), castedPtr, 0);
         mlir::Value next = rewriter.create<util::LoadOp>(loc, nextPtr, mlir::Value());
         rewriter.create<mlir::scf::YieldOp>(loc, next);
      });
      return success();
   }
};

class FilterLowering : public SubOpTupleStreamConsumerConversionPattern<subop::FilterOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::FilterOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::FilterOp filterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      mlir::Value cond = rewriter.create<db::AndOp>(filterOp.getLoc(), mapping.resolve(filterOp,filterOp.getConditions()));
      cond = rewriter.create<db::DeriveTruth>(filterOp.getLoc(), cond);
      if (filterOp.getFilterSemantic() == subop::FilterSemantic::none_true) {
         cond = rewriter.create<db::NotOp>(filterOp->getLoc(), cond);
      }
      auto ifOp = rewriter.create<mlir::scf::IfOp>(filterOp->getLoc(), mlir::TypeRange{}, cond);
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, filterOp->getLoc());
      rewriter.atStartOf(ifOp.thenBlock(), [&](SubOpRewriter& rewriter) {
         rewriter.replaceTupleStream(filterOp, mapping);
      });

      return success();
   }
};

class RenameLowering : public SubOpTupleStreamConsumerConversionPattern<subop::RenamingOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::RenamingOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::RenamingOp renamingOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      for (mlir::Attribute attr : renamingOp.getColumns()) {
         auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         mlir::Attribute from = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[0];
         auto relationRefAttr = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(from);
         mapping.define(relationDefAttr, mapping.resolve(renamingOp,relationRefAttr));
      }
      rewriter.replaceTupleStream(renamingOp, mapping);
      return success();
   }
};

class MaterializeHeapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto heapType = mlir::dyn_cast_or_null<subop::HeapType>(materializeOp.getState().getType());
      if (!heapType) return failure();
      EntryStorageHelper storageHelper(materializeOp,heapType.getMembers(), typeConverter);
      mlir::Value ref;
      rewriter.atStartOf(&rewriter.getCurrentStreamLoc()->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
         ref = rewriter.create<util::AllocaOp>(materializeOp->getLoc(), util::RefType::get(storageHelper.getStorageType()), mlir::Value());
      });
      storageHelper.storeFromColumns(materializeOp.getMapping(), mapping, ref, rewriter, materializeOp->getLoc());
      rt::Heap::insert(rewriter, materializeOp->getLoc())({adaptor.getState(), ref});
      rewriter.eraseOp(materializeOp);
      return mlir::success();
   }
};

class MaterializeVectorLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(materializeOp.getState().getType());
      if (!bufferType) return failure();
      EntryStorageHelper storageHelper(materializeOp,bufferType.getMembers(), typeConverter);
      mlir::Value ref = rt::GrowingBuffer::insert(rewriter, materializeOp->getLoc())({adaptor.getState()})[0];
      storageHelper.storeFromColumns(materializeOp.getMapping(), mapping, ref, rewriter, materializeOp->getLoc());
      rewriter.eraseOp(materializeOp);
      return mlir::success();
   }
};

class LookupSimpleStateLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::SimpleStateType>(lookupOp.getState().getType())) return failure();
      mapping.define(lookupOp.getRef(), adaptor.getState());
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

class LookupHashIndexedViewLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::HashIndexedViewType>(lookupOp.getState().getType())) return failure();
      auto loc = lookupOp->getLoc();
      mlir::Value hash = mapping.resolve(lookupOp,lookupOp.getKeys())[0];
      auto* context = getContext();
      auto indexType = rewriter.getIndexType();
      auto htType = util::RefType::get(context, util::RefType::get(context, rewriter.getI8Type()));

      Value castedPointer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, TupleType::get(context, {htType, indexType})), adaptor.getState());

      auto loaded = rewriter.create<util::LoadOp>(loc, mlir::cast<util::RefType>(castedPointer.getType()).getElementType(), castedPointer, Value());
      auto unpacked = rewriter.create<util::UnPackOp>(loc, loaded);
      Value ht = unpacked.getResult(0);
      Value htMask = unpacked.getResult(1);
      Value buckedPos = rewriter.create<arith::AndIOp>(loc, htMask, hash);
      Value ptr = rewriter.create<util::LoadOp>(loc, util::RefType::get(getContext(), rewriter.getI8Type()), ht, buckedPos);
      //optimization
      ptr = rewriter.create<util::FilterTaggedPtr>(loc, ptr.getType(), ptr, hash);
      Value matches = rewriter.create<util::PackOp>(loc, ValueRange{ptr, hash});

      mapping.define(lookupOp.getRef(), matches);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};
class LookupSegmentTreeViewLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::SegmentTreeViewType>(lookupOp.getState().getType())) return failure();

      auto valueMembers = mlir::cast<subop::SegmentTreeViewType>(lookupOp.getState().getType()).getValueMembers();
      mlir::TupleType stateType = mlir::TupleType::get(getContext(), unpackTypes(valueMembers.getTypes()));

      auto loc = lookupOp->getLoc();
      auto unpackedLeft = rewriter.create<util::UnPackOp>(loc, mapping.resolve(lookupOp,lookupOp.getKeys())[0]).getResults();
      auto unpackedRight = rewriter.create<util::UnPackOp>(loc, mapping.resolve(lookupOp,lookupOp.getKeys())[1]).getResults();
      auto idxLeft = unpackedLeft[0];
      auto idxRight = unpackedRight[0];
      mlir::Value ref;
      rewriter.atStartOf(&rewriter.getCurrentStreamLoc()->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
         ref = rewriter.create<util::AllocaOp>(lookupOp->getLoc(), util::RefType::get(typeConverter->convertType(stateType)), mlir::Value());
      });
      rt::SegmentTreeView::lookup(rewriter, loc)({adaptor.getState(), ref, idxLeft, idxRight});
      mapping.define(lookupOp.getRef(), ref);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

class PureLookupHashMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::HashMapType>(lookupOp.getState().getType())) return failure();
      subop::HashMapType htStateType = mlir::cast<subop::HashMapType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp,htStateType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp,htStateType.getValueMembers(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp,lookupOp.getKeys());
      auto packed = rewriter.create<util::PackOp>(lookupOp->getLoc(), lookupKey);

      mlir::Value hash = rewriter.create<db::Hash>(lookupOp->getLoc(), packed); //todo: external hash
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, std::vector<Value> left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), left.begin(), left.end());
         arguments.insert(arguments.end(), right.begin(), right.end());
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;
      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(htStateType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      ptr = rewriter.create<util::FilterTaggedPtr>(loc, ptr.getType(), ptr, hashed);

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
      Block* before = new Block;
      whileOp.getBefore().push_back(before);
      before->addArgument(bucketPtrType, loc);
      Block* after = new Block;
      whileOp.getAfter().push_back(after);
      after->addArgument(bucketPtrType, loc);
      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value currEntryPtr = rewriter.create<util::UnTagPtr>(loc, before->getArgument(0).getType(), before->getArgument(0));
         Value ptr = currEntryPtr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                     Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                     auto keyValues=keyStorageHelper.loadValuesOrdered(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc, keyMatches, [&](OpBuilder& b, Location loc) {


                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });

      Value currEntryPtr = whileOp.getResult(0);
      currEntryPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), rewriter.getI8Type()), currEntryPtr);
      mapping.define(lookupOp.getRef(), currEntryPtr);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

class PureLookupPreAggregationHtLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::PreAggrHtType>(lookupOp.getState().getType())) return failure();
      subop::PreAggrHtType htStateType = mlir::cast<subop::PreAggrHtType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp,htStateType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp,htStateType.getValueMembers(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp,lookupOp.getKeys());
      auto packed = rewriter.create<util::PackOp>(lookupOp->getLoc(), lookupKey);

      mlir::Value hash = rewriter.create<db::Hash>(lookupOp->getLoc(), packed); //todo: external hash
      ASSERT_WITH_OP(!lookupOp.getEqFn().empty(), lookupOp, "LookupOp must have an equality function");
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, std::vector<Value> left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), left.begin(), left.end());
         arguments.insert(arguments.end(), right.begin(), right.end());
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;
      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(htStateType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      Type bucketPtrType = util::RefType::get(context, entryType);

      mlir::Value partition = rewriter.create<mlir::arith::AndIOp>(loc, hashed, rewriter.create<mlir::arith::ConstantIndexOp>(loc, 64 - 1));
      Type partitionHtType = mlir::TupleType::get(rewriter.getContext(), {util::RefType::get(context, bucketPtrType), rewriter.getIndexType()});
      Value preaggregationHt = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, partitionHtType), adaptor.getState());
      Value partitionHt = rewriter.create<util::LoadOp>(loc, preaggregationHt, partition);
      auto unpacked = rewriter.create<util::UnPackOp>(loc, partitionHt).getResults();
      Value ht = unpacked[0];
      Value htMask = unpacked[1];
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, rewriter.create<arith::ShRUIOp>(loc, hashed, rewriter.create<mlir::arith::ConstantIndexOp>(loc, 6)));
      Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      ptr = rewriter.create<util::FilterTaggedPtr>(loc, ptr.getType(), ptr, hashed);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
      Block* before = new Block;
      whileOp.getBefore().push_back(before);
      before->addArgument(bucketPtrType, loc);
      Block* after = new Block;
      whileOp.getAfter().push_back(after);
      after->addArgument(bucketPtrType, loc);
      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value currEntryPtr = rewriter.create<util::UnTagPtr>(loc, before->getArgument(0).getType(), before->getArgument(0));
         Value ptr = currEntryPtr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                     Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                     auto keyValues=keyStorageHelper.loadValuesOrdered(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc, keyMatches, [&](OpBuilder& b, Location loc) {


                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });

      Value currEntryPtr = whileOp.getResult(0);
      currEntryPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), rewriter.getI8Type()), currEntryPtr);
      mapping.define(lookupOp.getRef(), currEntryPtr);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};
class LookupHashMultiMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      subop::HashMultiMapType hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(lookupOp.getState().getType());
      if (!hashMultiMapType) return failure();
      EntryStorageHelper keyStorageHelper(lookupOp,hashMultiMapType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp,hashMultiMapType.getValueMembers(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp,lookupOp.getKeys());
      auto packed = rewriter.create<util::PackOp>(lookupOp->getLoc(), lookupKey);

      mlir::Value hash = rewriter.create<db::Hash>(lookupOp->getLoc(), packed); //todo: external hash
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, std::vector<Value> left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), left.begin(), left.end());
         arguments.insert(arguments.end(), right.begin(), right.end());
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;
      auto* context = rewriter.getContext();
      auto entryType = getHashMultiMapEntryType(hashMultiMapType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      ptr = rewriter.create<util::FilterTaggedPtr>(loc, ptr.getType(), ptr, hashed);

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
      Block* before = new Block;
      whileOp.getBefore().push_back(before);
      before->addArgument(bucketPtrType, loc);
      Block* after = new Block;
      whileOp.getAfter().push_back(after);
      after->addArgument(bucketPtrType, loc);
      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value ptr = before->getArgument(0);

         Value currEntryPtr = ptr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, currEntryPtr, 3);
                     auto keyValues=keyStorageHelper.loadValuesOrdered(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc,  keyMatches, [&](OpBuilder& b, Location loc) {
                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });
      Value currEntryPtr = whileOp.getResult(0);
      mapping.define(lookupOp.getRef(), currEntryPtr);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};
class InsertMultiMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::InsertOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::InsertOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::InsertOp insertOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      subop::HashMultiMapType htStateType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(insertOp.getState().getType());
      if (!htStateType) return failure();
      EntryStorageHelper keyStorageHelper(insertOp,htStateType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(insertOp,htStateType.getValueMembers(), typeConverter);
      auto loc = insertOp->getLoc();
      std::vector<mlir::Value> lookupKey = keyStorageHelper.resolve(insertOp,insertOp.getMapping(), mapping);
      auto packed = rewriter.create<util::PackOp>(loc, lookupKey);

      mlir::Value hash = rewriter.create<db::Hash>(loc, packed); //todo: external hash
      mlir::Value hashTable = adaptor.getState();
      auto equalFnBuilder = [&insertOp](mlir::OpBuilder& rewriter, std::vector<Value> left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), left.begin(), left.end());
         arguments.insert(arguments.end(), right.begin(), right.end());
         auto res = inlineBlock(&insertOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      Value hashed = hash;

      auto* context = rewriter.getContext();
      auto entryType = getHashMultiMapEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value firstPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      firstPtr = rewriter.create<util::FilterTaggedPtr>(loc, firstPtr.getType(), firstPtr, hashed);

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({firstPtr}));
      Block* before = new Block;
      before->addArgument(bucketPtrType, loc);
      whileOp.getBefore().push_back(before);
      Block* after = new Block;
      after->addArgument(bucketPtrType, loc);
      whileOp.getAfter().push_back(after);

      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value ptr = before->getArgument(0);

         Value currEntryPtr = ptr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value keyRef=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, currEntryPtr, 3);
                     auto keyValues=keyStorageHelper.loadValuesOrdered(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc, keyMatches, [&](OpBuilder& b, Location loc) {
                           Value valRef=rt::HashMultiMap::insertValue(rewriter,loc)({hashTable,currEntryPtr})[0];
                           valRef=rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(),mlir::TupleType::get(getContext(),{i8PtrType,valStorageHelper.getStorageType()})),valRef);
                           valRef=rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), valRef, 1);
                           valStorageHelper.storeFromColumns(insertOp.getMapping(),mapping,valRef,rewriter,loc);
                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {

                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) {
               Value entryRef=rt::HashMultiMap::insertEntry(b,loc)({hashTable,hashed})[0];
               Value entryRefCasted= rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, entryRef);
               Value keyRef=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, entryRefCasted, 3);

               keyStorageHelper.storeFromColumns(insertOp.getMapping(),mapping,keyRef,rewriter,loc);
               Value valRef=rt::HashMultiMap::insertValue(rewriter,loc)({hashTable,entryRef})[0];
               valRef=rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(),mlir::TupleType::get(getContext(),{i8PtrType,valStorageHelper.getStorageType()})),valRef);
               valRef=rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), valRef, 1);
               valStorageHelper.storeFromColumns(insertOp.getMapping(),mapping,valRef,rewriter,loc);
               b.create<scf::YieldOp>(loc, ValueRange{falseValue, entryRefCasted}); });
         //       if(compare(entry.key,key)){

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });
      rewriter.eraseOp(insertOp);
      return mlir::success();
   }
};
class LookupPreAggrHtFragment : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::PreAggrHtFragmentType>(lookupOp.getState().getType())) return failure();
      subop::PreAggrHtFragmentType fragmentType = mlir::cast<subop::PreAggrHtFragmentType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp,fragmentType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp,fragmentType.getValueMembers(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp,lookupOp.getKeys());
      auto packed = rewriter.create<util::PackOp>(lookupOp->getLoc(), lookupKey);

      mlir::Value hash = rewriter.create<db::Hash>(lookupOp->getLoc(), packed); //todo: external hash
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, std::vector<Value> left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), left.begin(), left.end());
         arguments.insert(arguments.end(), right.begin(), right.end());
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto initValBuilder = [&lookupOp, this](SubOpRewriter& rewriter) -> std::vector<mlir::Value> {
         std::vector<mlir::Value> res;
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lookupOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor adaptor) {
            res = std::vector<mlir::Value>{adaptor.getResults().begin(), adaptor.getResults().end()};
         });
         for (size_t i = 0; i < res.size(); i++) {
            auto convertedType = typeConverter->convertType(res[i].getType());
            if (res[i].getType() != convertedType) {
               res[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(lookupOp->getLoc(), convertedType, res[i]).getResult(0);
            }
         }
         return res;
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;

      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(fragmentType, *typeConverter);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(fragmentType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto valPtrType = valStorageHelper.getRefType();

      Value ht = adaptor.getState();
      Value htMask = rewriter.create<arith::ConstantIndexOp>(loc, 1024 - 1);
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, rewriter.create<arith::ShRUIOp>(loc, hashed, rewriter.create<mlir::arith::ConstantIndexOp>(loc, 6)));
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value currEntryPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      ;
      //    if (*ptr != nullptr){
      Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
      auto ifOp = rewriter.create<scf::IfOp>(
         loc, cmp, [&](OpBuilder& b, Location loc) {
            Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
            Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
            Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
            auto ifOpH = b.create<scf::IfOp>(
               loc, hashMatches, [&](OpBuilder& b, Location loc) {
                  Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                  Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                  auto keyValues=keyStorageHelper.loadValuesOrdered(keyRef,rewriter,loc);
                  Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                  b.create<scf::YieldOp>(loc, mlir::ValueRange{keyMatches});
               }, [&](OpBuilder& b, Location loc) {  b.create<scf::YieldOp>(loc, falseValue);});
            b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue}); });
      auto ifOp2 = rewriter.create<scf::IfOp>(
         loc, ifOp.getResults()[0], [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, currEntryPtr); },
         [&](OpBuilder& b, Location loc) {
            auto initialVals = initValBuilder(rewriter);
            Value entryRef = rt::PreAggregationHashtableFragment::insert(b, loc)({adaptor.getState(), hashed})[0];
            Value entryRefCasted = rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, entryRef);
            Value kvRef = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, entryRefCasted, 2);
            Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvRef, 0);
            Value valRef = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvRef, 1);
            keyStorageHelper.storeOrderedValues(keyRef, lookupKey, rewriter, loc);
            valStorageHelper.storeOrderedValues(valRef, initialVals, rewriter, loc);
            b.create<scf::YieldOp>(loc, ValueRange{entryRefCasted});
         });
      currEntryPtr = ifOp2.getResult(0);
      Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
      mapping.define(lookupOp.getRef(), rewriter.create<util::TupleElementPtrOp>(lookupOp->getLoc(), util::RefType::get(getContext(), kvType.getType(1)), kvAddress, 1));
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};
class LookupHashMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::HashMapType>(lookupOp.getState().getType())) return failure();
      subop::HashMapType htStateType = mlir::cast<subop::HashMapType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp,htStateType.getKeyMembers(), typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp,htStateType.getValueMembers(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp,lookupOp.getKeys());
      auto packed = rewriter.create<util::PackOp>(lookupOp->getLoc(), lookupKey);

      mlir::Value hash = rewriter.create<db::Hash>(lookupOp->getLoc(), packed); //todo: external hash
      mlir::Value hashTable = adaptor.getState();
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, std::vector<Value> left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), left.begin(), left.end());
         arguments.insert(arguments.end(), right.begin(), right.end());
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto initValBuilder = [&lookupOp, this](SubOpRewriter& rewriter) -> std::vector<mlir::Value> {
         std::vector<mlir::Value> res;
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lookupOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor adaptor) {
            res = std::vector<mlir::Value>{adaptor.getResults().begin(), adaptor.getResults().end()};
         });
         for (size_t i = 0; i < res.size(); i++) {
            auto convertedType = typeConverter->convertType(res[i].getType());
            if (res[i].getType() != convertedType) {
               res[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(lookupOp->getLoc(), convertedType, res[i]).getResult(0);
            }
         }
         return res;
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;

      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(htStateType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto valPtrType = valStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value firstPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      firstPtr = rewriter.create<util::FilterTaggedPtr>(loc, firstPtr.getType(), firstPtr, hashed);

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({firstPtr}));
      Block* before = new Block;
      before->addArgument(bucketPtrType, loc);
      whileOp.getBefore().push_back(before);
      Block* after = new Block;
      after->addArgument(bucketPtrType, loc);
      whileOp.getAfter().push_back(after);

      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value ptr = rewriter.create<util::UnTagPtr>(loc, before->getArgument(0).getType(), before->getArgument(0));

         Value currEntryPtr = ptr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                     Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                     auto keyValues=keyStorageHelper.loadValuesOrdered(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc,  keyMatches, [&](OpBuilder& b, Location loc) {


                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) {
               auto initialVals = initValBuilder(rewriter);
               //       %newEntry = ...
               Value entryRef=rt::Hashtable::insert(b,loc)({hashTable,hashed})[0];
               Value entryRefCasted= rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, entryRef);
               Value kvRef = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, entryRefCasted, 2);
               Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvRef, 0);
               Value valRef = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvRef, 1);
               keyStorageHelper.storeOrderedValues(keyRef,lookupKey,rewriter,loc);
               valStorageHelper.storeOrderedValues(valRef,initialVals,rewriter,loc);

               b.create<scf::YieldOp>(loc, ValueRange{falseValue, entryRefCasted}); });
         //       if(compare(entry.key,key)){

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });
      Value currEntryPtr = whileOp.getResult(0);
      Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
      mapping.define(lookupOp.getRef(), rewriter.create<util::TupleElementPtrOp>(lookupOp->getLoc(), util::RefType::get(getContext(), kvType.getType(1)), kvAddress, 1));
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};
class LookupExternalHashIndexLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::ExternalHashIndexType>(lookupOp.getState().getType())) return failure();

      auto loc = lookupOp->getLoc();

      // Calculate hash value and perform lookup in external index hashmap
      auto hashValue = rewriter.create<db::Hash>(loc, rewriter.create<util::PackOp>(loc, mapping.resolve(lookupOp,lookupOp.getKeys())));
      mlir::Value list = rt::HashIndexAccess::lookup(rewriter, loc)({adaptor.getState(), hashValue})[0];

      mapping.define(lookupOp.getRef(), list);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

class DefaultGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      EntryStorageHelper storageHelper(gatherOp,mlir::cast<subop::StateEntryReference>(gatherOp.getRef().getColumn().type).getMembers(), typeConverter);
      storageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, mapping.resolve(gatherOp,gatherOp.getRef()), rewriter, gatherOp->getLoc());
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};
class ContinuousRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(gatherOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { return failure(); }
      auto unpackedReference = rewriter.create<util::UnPackOp>(gatherOp->getLoc(), mapping.resolve(gatherOp,gatherOp.getRef())).getResults();
      EntryStorageHelper storageHelper(gatherOp,continuousRefEntryType.getMembers(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      auto baseRef = rewriter.create<util::BufferGetRef>(gatherOp->getLoc(), ptrType, unpackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(gatherOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
      storageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, elementRef, rewriter, gatherOp->getLoc());
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};
class HashMapRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::HashMapEntryRefType>(refType)) { return failure(); }
      auto keyMembers = mlir::cast<subop::HashMapEntryRefType>(refType).getHashMap().getKeyMembers();
      auto valMembers = mlir::cast<subop::HashMapEntryRefType>(refType).getHashMap().getValueMembers();
      auto loc = gatherOp->getLoc();
      EntryStorageHelper keyStorageHelper(gatherOp,keyMembers, typeConverter);
      EntryStorageHelper valStorageHelper(gatherOp,valMembers, typeConverter);
      auto ref = mapping.resolve(gatherOp,gatherOp.getRef());
      auto keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ref, 0);
      auto valRef = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), ref, 1);
      keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, keyRef, rewriter, loc);
      valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, valRef, rewriter, loc);
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};
class PreAggrHtRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::PreAggrHTEntryRefType>(refType)) { return failure(); }
      auto keyMembers = mlir::cast<subop::PreAggrHTEntryRefType>(refType).getHashMap().getKeyMembers();
      auto valMembers = mlir::cast<subop::PreAggrHTEntryRefType>(refType).getHashMap().getValueMembers();
      auto loc = gatherOp->getLoc();
      EntryStorageHelper keyStorageHelper(gatherOp,keyMembers, typeConverter);
      EntryStorageHelper valStorageHelper(gatherOp,valMembers, typeConverter);
      auto ref = mapping.resolve(gatherOp,gatherOp.getRef());
      auto keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ref, 0);
      auto valRef = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), ref, 1);
      keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, keyRef, rewriter, loc);
      valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, valRef, rewriter, loc);
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};
class HashMultiMapRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::HashMultiMapEntryRefType>(refType)) { return failure(); }
      auto hashMultiMap = mlir::cast<subop::HashMultiMapEntryRefType>(refType).getHashMultimap();
      auto keyMembers = hashMultiMap.getKeyMembers();
      auto valMembers = hashMultiMap.getValueMembers();
      auto loc = gatherOp->getLoc();
      EntryStorageHelper keyStorageHelper(gatherOp,keyMembers, typeConverter);
      EntryStorageHelper valStorageHelper(gatherOp,valMembers, typeConverter);
      auto packed = mapping.resolve(gatherOp,gatherOp.getRef());
      auto unpacked = rewriter.create<util::UnPackOp>(loc, packed).getResults();
      keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, unpacked[0], rewriter, loc);
      valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, unpacked[1], rewriter, loc);
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};
class ExternalHashIndexRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::ExternalHashIndexEntryRefType>(refType)) { return failure(); }
      auto columns = mlir::cast<subop::ExternalHashIndexEntryRefType>(refType).getMembers();
      auto currRow = mapping.resolve(gatherOp,gatherOp.getRef());

      // Define mapping for values of gathered tuple
      for (size_t i = 0; i < columns.getTypes().size(); i++) {
         auto memberName = mlir::cast<mlir::StringAttr>(columns.getNames()[i]).str();
         if (gatherOp.getMapping().contains(memberName)) {
            auto columnDefAttr = mlir::cast<tuples::ColumnDefAttr>(gatherOp.getMapping().get(memberName));
            auto type = columnDefAttr.getColumn().type;
            size_t accessOffset = i;
            std::vector<mlir::Type> types;
            types.push_back(getBaseType(type));
            if (mlir::isa<db::NullableType>(type)) {
               types.push_back(rewriter.getI1Type());
            }
            auto atOp = rewriter.create<dsa::At>(gatherOp->getLoc(), types, currRow, accessOffset);
            if (mlir::isa<db::NullableType>(type)) {
               mlir::Value isNull = rewriter.create<db::NotOp>(gatherOp->getLoc(), atOp.getValid());
               mlir::Value val = rewriter.create<db::AsNullableOp>(gatherOp->getLoc(), type, atOp.getVal(), isNull);
               mapping.define(columnDefAttr, val);
            } else {
               mapping.define(columnDefAttr, atOp.getVal());
            }
         }
      }
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};

static bool checkAtomicStore(mlir::Operation* op) {
   //on x86, stores are always atomic (if aligned)
#ifdef __x86_64__
   return true;
#else
   return !op->hasAttr("atomic");
#endif
}
class ContinuousRefScatterOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!checkAtomicStore(scatterOp)) return failure();
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(scatterOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { return failure(); }
      auto unpackedReference = rewriter.create<util::UnPackOp>(scatterOp->getLoc(), mapping.resolve(scatterOp,scatterOp.getRef())).getResults();
      EntryStorageHelper storageHelper(scatterOp,continuousRefEntryType.getMembers(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      auto baseRef = rewriter.create<util::BufferGetRef>(scatterOp->getLoc(), ptrType, unpackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(scatterOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
      auto values = storageHelper.loadValues(elementRef, rewriter, scatterOp->getLoc());
      for (auto x : scatterOp.getMapping()) {
         values[x.getName().str()] = mapping.resolve(scatterOp,mlir::cast<tuples::ColumnRefAttr>(x.getValue()));
      }
      storageHelper.storeValues(elementRef, values, rewriter, scatterOp->getLoc());
      rewriter.eraseOp(scatterOp);
      return success();
   }
};

class ScatterOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 1> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!checkAtomicStore(scatterOp)) return failure();
      auto columns = mlir::cast<subop::StateEntryReference>(scatterOp.getRef().getColumn().type).getMembers();
      EntryStorageHelper storageHelper(scatterOp,columns, typeConverter);
      auto ref = mapping.resolve(scatterOp,scatterOp.getRef());
      auto values = storageHelper.loadValues(ref, rewriter, scatterOp->getLoc());
      for (auto x : scatterOp.getMapping()) {
         values[x.getName().str()] = mapping.resolve(scatterOp,mlir::cast<tuples::ColumnRefAttr>(x.getValue()));
      }
      storageHelper.storeValues(ref, values, rewriter, scatterOp->getLoc());
      rewriter.eraseOp(scatterOp);
      return success();
   }
};
class HashMultiMapScatterOp : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 1> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto hashMultiMapEntryRef = mlir::dyn_cast_or_null<subop::HashMultiMapEntryRefType>(scatterOp.getRef().getColumn().type);
      if (!hashMultiMapEntryRef) return failure();
      auto columns = hashMultiMapEntryRef.getHashMultimap().getValueMembers();
      EntryStorageHelper storageHelper(scatterOp,columns, typeConverter);
      auto ref = rewriter.create<util::UnPackOp>(scatterOp.getLoc(), mapping.resolve(scatterOp,scatterOp.getRef())).getResult(1);
      auto values = storageHelper.loadValues(ref, rewriter, scatterOp->getLoc());
      for (auto x : scatterOp.getMapping()) {
         values[x.getName().str()] = mapping.resolve(scatterOp,mlir::cast<tuples::ColumnRefAttr>(x.getValue()));
      }
      storageHelper.storeValues(ref, values, rewriter, scatterOp->getLoc());
      rewriter.eraseOp(scatterOp);
      return success();
   }
};

class ReduceContinuousRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { return failure(); }
      if (reduceOp->hasAttr("atomic")) {
         return mlir::failure();
      }
      auto unpackedReference = rewriter.create<util::UnPackOp>(reduceOp->getLoc(), mapping.resolve(reduceOp,reduceOp.getRef())).getResults();
      EntryStorageHelper storageHelper(reduceOp,continuousRefEntryType.getMembers(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      auto baseRef = rewriter.create<util::BufferGetRef>(reduceOp->getLoc(), ptrType, unpackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(reduceOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
      auto values = storageHelper.loadValues(elementRef, rewriter, reduceOp->getLoc());
      std::vector<mlir::Value> arguments;
      for (auto attr : reduceOp.getColumns()) {
         mlir::Value arg = mapping.resolve(reduceOp,mlir::cast<tuples::ColumnRefAttr>(attr));
         if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
         }
         arguments.push_back(arg);
      }
      for (auto member : reduceOp.getMembers()) {
         mlir::Value arg = values.at(mlir::cast<mlir::StringAttr>(member).str());
         if (arg.getType() != reduceOp.getRegion().getArgument(arguments.size()).getType()) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), reduceOp.getRegion().getArgument(arguments.size()).getType(), arg).getResult(0);
         }
         arguments.push_back(arg);
      }

      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
         for (size_t i = 0; i < reduceOp.getMembers().size(); i++) {
            auto& memberVal = values[mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[i]).str()];
            auto updatedVal = adaptor.getResults()[i];
            if (updatedVal.getType() != memberVal.getType()) {
               updatedVal = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), memberVal.getType(), updatedVal).getResult(0);
            }
            memberVal = updatedVal;
         }
         storageHelper.storeValues(elementRef, values, rewriter, reduceOp->getLoc());
         rewriter.eraseOp(reduceOp);
      });

      return success();
   }
};
static void implementAtomicReduce(subop::ReduceOp reduceOp, SubOpRewriter& rewriter, mlir::Value valueRef, ColumnMapping& mapping) {
   auto loc = reduceOp->getLoc();
   auto elementType = mlir::cast<util::RefType>(valueRef.getType()).getElementType();
   auto origElementType = mlir::cast<util::RefType>(valueRef.getType()).getElementType();
   if (elementType.isInteger(1)) {
      elementType = rewriter.getI8Type();
      valueRef = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(rewriter.getContext(), elementType), valueRef);
   }
   auto memRefType = mlir::MemRefType::get({}, elementType);
   auto memRef = rewriter.create<util::ToMemrefOp>(reduceOp->getLoc(), memRefType, valueRef);
   auto returnOp = mlir::cast<tuples::ReturnOp>(reduceOp.getRegion().front().getTerminator());
   ::mlir::arith::AtomicRMWKind atomicKind = mlir::arith::AtomicRMWKind::maximumf; //maxf is invalid value;
   mlir::Value memberValue = reduceOp.getRegion().front().getArguments().back();
   mlir::Value atomicOperand;

   if (auto* defOp = returnOp.getResults()[0].getDefiningOp()) {
      if (auto addFOp = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(defOp)) {
         if (addFOp.getLhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addf;
            atomicOperand = addFOp.getRhs();
         } else if (addFOp.getRhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addf;
            atomicOperand = addFOp.getLhs();
         }
      }

      if (auto addIOp = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(defOp)) {
         if (addIOp.getLhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addi;
            atomicOperand = addIOp.getRhs();
         } else if (addIOp.getRhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addi;
            atomicOperand = addIOp.getLhs();
         }
      }
      if (auto orIOp = mlir::dyn_cast_or_null<mlir::arith::OrIOp>(defOp)) {
         if (orIOp.getLhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::ori;
            atomicOperand = orIOp.getRhs();
         } else if (orIOp.getRhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::ori;
            atomicOperand = orIOp.getLhs();
         }
      }
      //todo: continue
   }

   if (atomicOperand) {
      std::vector<mlir::Value> arguments;
      for (auto attr : reduceOp.getColumns()) {
         mlir::Value arg = mapping.resolve(reduceOp,mlir::cast<tuples::ColumnRefAttr>(attr));
         if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
         }
         arguments.push_back(arg);
      }

      arguments.push_back(rewriter.create<util::UndefOp>(loc, origElementType));
      mlir::IRMapping mapper;
      mlir::Block* b = &reduceOp.getRegion().front();
      assert(b->getNumArguments() == arguments.size());
      for (auto i = 0ull; i < b->getNumArguments(); i++) {
         mapper.map(b->getArgument(i), arguments[i]);
      }
      for (auto& x : b->getOperations()) {
         if (&x != returnOp.getOperation()) {
            rewriter.insert(rewriter.clone(&x, mapper));
         }
      }
      atomicOperand = mapper.lookup(atomicOperand);
      if (atomicOperand.getType().isInteger(1)) {
         atomicOperand = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI8Type(), atomicOperand);
      }
      rewriter.create<memref::AtomicRMWOp>(loc, atomicOperand.getType(), atomicKind, atomicOperand, memRef, ValueRange{});
      rewriter.eraseOp(reduceOp);
   } else {
      auto genericOp = rewriter.create<memref::GenericAtomicRMWOp>(loc, memRef, mlir::ValueRange{});
      std::vector<mlir::Value> arguments;
      for (auto attr : reduceOp.getColumns()) {
         mlir::Value arg = mapping.resolve(reduceOp,mlir::cast<tuples::ColumnRefAttr>(attr));
         if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
         }
         arguments.push_back(arg);
      }

      arguments.push_back(genericOp.getCurrentValue());

      rewriter.atStartOf(genericOp.getBody(), [&](SubOpRewriter& rewriter) {
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
            mlir::Value atomicResult = adaptor.getResults()[0];
            if (atomicResult.getType().isInteger(1)) {
               atomicResult = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI8Type(), atomicResult);
            }
            rewriter.create<memref::AtomicYieldOp>(loc, atomicResult);
            rewriter.eraseOp(reduceOp);
         });
      });
   }
}
class ReduceContinuousRefAtomicLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { return failure(); }
      if (!reduceOp->hasAttr("atomic")) {
         return mlir::failure();
      }
      auto loc = reduceOp->getLoc();
      auto unpackedReference = rewriter.create<util::UnPackOp>(reduceOp->getLoc(), mapping.resolve(reduceOp,reduceOp.getRef())).getResults();
      EntryStorageHelper storageHelper(reduceOp,continuousRefEntryType.getMembers(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      auto baseRef = rewriter.create<util::BufferGetRef>(reduceOp->getLoc(), ptrType, unpackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(reduceOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
      auto valueRef = storageHelper.getPointer(elementRef, mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[0]).str(), rewriter, loc);
      implementAtomicReduce(reduceOp, rewriter, valueRef, mapping);

      return success();
   }
};
class ReduceOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto members = mlir::cast<subop::StateEntryReference>(reduceOp.getRef().getColumn().type).getMembers();
      auto ref = mapping.resolve(reduceOp,reduceOp.getRef());
      EntryStorageHelper storageHelper(reduceOp,members, typeConverter);
      if (reduceOp->hasAttr("atomic")) {
         auto valueRef = storageHelper.getPointer(ref, mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[0]).str(), rewriter, reduceOp->getLoc());
         implementAtomicReduce(reduceOp, rewriter, valueRef, mapping);
      } else {
         auto values = storageHelper.loadValues(ref, rewriter, reduceOp->getLoc());
         std::vector<mlir::Value> arguments;
         for (auto attr : reduceOp.getColumns()) {
            mlir::Value arg = mapping.resolve(reduceOp,mlir::cast<tuples::ColumnRefAttr>(attr));
            if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
               arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
            }
            arguments.push_back(arg);
         }
         for (auto member : reduceOp.getMembers()) {
            mlir::Value arg = values.at(mlir::cast<mlir::StringAttr>(member).str());
            if (arg.getType() != reduceOp.getRegion().getArgument(arguments.size()).getType()) {
               arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), reduceOp.getRegion().getArgument(arguments.size()).getType(), arg).getResult(0);
            }
            arguments.push_back(arg);
         }

         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
            for (size_t i = 0; i < reduceOp.getMembers().size(); i++) {
               auto& memberVal = values[mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[i]).str()];
               auto updatedVal = adaptor.getResults()[i];
               if (updatedVal.getType() != memberVal.getType()) {
                  updatedVal = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), memberVal.getType(), updatedVal).getResult(0);
               }
               memberVal = updatedVal;
            }
            storageHelper.storeValues(ref, values, rewriter, reduceOp->getLoc());
            rewriter.eraseOp(reduceOp);
         });
      }

      return success();
   }
};

class CreateHashIndexedViewLowering : public SubOpConversionPattern<subop::CreateHashIndexedView> {
   using SubOpConversionPattern<subop::CreateHashIndexedView>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateHashIndexedView createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto bufferType = mlir::dyn_cast<subop::BufferType>(createOp.getSource().getType());
      if (!bufferType) return failure();
      auto linkIsFirst = mlir::cast<mlir::StringAttr>(bufferType.getMembers().getNames()[0]).str() == createOp.getLinkMember();
      auto hashIsSecond = mlir::cast<mlir::StringAttr>(bufferType.getMembers().getNames()[1]).str() == createOp.getHashMember();
      if (!linkIsFirst || !hashIsSecond) return failure();
      auto htView = rt::HashIndexedView::build(rewriter, createOp->getLoc())({getExecutionContext(rewriter, createOp), adaptor.getSource()})[0];
      rewriter.replaceOp(createOp, htView);
      return success();
   }
};
class CreateContinuousViewLowering : public SubOpConversionPattern<subop::CreateContinuousView> {
   using SubOpConversionPattern<subop::CreateContinuousView>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateContinuousView createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (mlir::isa<subop::ArrayType>(createOp.getSource().getType())) {
         //todo: for now: every sorted view is equivalent to continuous view
         rewriter.replaceOp(createOp, adaptor.getSource());
         return success();
      }
      if (mlir::isa<subop::SortedViewType>(createOp.getSource().getType())) {
         //todo: for now: every sorted view is equivalent to continuous view
         rewriter.replaceOp(createOp, adaptor.getSource());
         return success();
      }
      auto bufferType = mlir::dyn_cast<subop::BufferType>(createOp.getSource().getType());
      if (!bufferType) return failure();
      auto genericBuffer = rt::GrowingBuffer::asContinuous(rewriter, createOp->getLoc())({adaptor.getSource(), getExecutionContext(rewriter, createOp)})[0];
      rewriter.replaceOpWithNewOp<util::BufferCastOp>(createOp, typeConverter->convertType(createOp.getType()), genericBuffer);
      return success();
   }
};

class GetBeginLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GetBeginReferenceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GetBeginReferenceOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::GetBeginReferenceOp getBeginReferenceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(getBeginReferenceOp->getLoc(), 0);
      auto packed = rewriter.create<util::PackOp>(getBeginReferenceOp->getLoc(), mlir::ValueRange{zero, adaptor.getState()});
      mapping.define(getBeginReferenceOp.getRef(), packed);
      rewriter.replaceTupleStream(getBeginReferenceOp, mapping);
      return success();
   }
};
class GetEndLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GetEndReferenceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GetEndReferenceOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::GetEndReferenceOp getEndReferenceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto len = rewriter.create<util::BufferGetLen>(getEndReferenceOp->getLoc(), rewriter.getIndexType(), adaptor.getState());
      auto one = rewriter.create<mlir::arith::ConstantIndexOp>(getEndReferenceOp->getLoc(), 1);
      auto lastOffset = rewriter.create<mlir::arith::SubIOp>(getEndReferenceOp->getLoc(), len, one);
      auto packed = rewriter.create<util::PackOp>(getEndReferenceOp->getLoc(), mlir::ValueRange{lastOffset, adaptor.getState()});
      mapping.define(getEndReferenceOp.getRef(), packed);
      rewriter.replaceTupleStream(getEndReferenceOp, mapping);
      return success();
   }
};
class EntriesBetweenLowering : public SubOpTupleStreamConsumerConversionPattern<subop::EntriesBetweenOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::EntriesBetweenOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::EntriesBetweenOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto unpackedLeft = rewriter.create<util::UnPackOp>(op->getLoc(), mapping.resolve(op,op.getLeftRef())).getResults();
      auto unpackedRight = rewriter.create<util::UnPackOp>(op->getLoc(), mapping.resolve(op,op.getRightRef())).getResults();
      mlir::Value difference = rewriter.create<mlir::arith::SubIOp>(op->getLoc(), unpackedRight[0], unpackedLeft[0]);
      if (!op.getBetween().getColumn().type.isIndex()) {
         difference = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), op.getBetween().getColumn().type, difference);
      }
      mapping.define(op.getBetween(), difference);
      rewriter.replaceTupleStream(op, mapping);
      return success();
   }
};
class OffsetReferenceByLowering : public SubOpTupleStreamConsumerConversionPattern<subop::OffsetReferenceBy> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::OffsetReferenceBy>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::OffsetReferenceBy op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto unpackedRef = rewriter.create<util::UnPackOp>(op->getLoc(), mapping.resolve(op,op.getRef())).getResults();
      auto offset = mapping.resolve(op,op.getIdx());
      if (!offset.getType().isIndex()) {
         offset = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getIndexType(), offset);
      }
      offset = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getI64Type(), offset);
      auto currIdx = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getI64Type(), unpackedRef[0]);
      mlir::Value newIdx = rewriter.create<mlir::arith::AddIOp>(op->getLoc(), currIdx, offset);
      newIdx = rewriter.create<mlir::arith::MaxSIOp>(op->getLoc(), rewriter.create<mlir::arith::ConstantIntOp>(op->getLoc(), 0, rewriter.getI64Type()), newIdx);
      newIdx = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), rewriter.getIndexType(), newIdx);
      auto length = rewriter.create<util::BufferGetLen>(op->getLoc(), rewriter.getIndexType(), unpackedRef[1]);
      auto maxIdx = rewriter.create<mlir::arith::SubIOp>(op->getLoc(), length, rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 1));
      newIdx = rewriter.create<mlir::arith::MinUIOp>(op->getLoc(), maxIdx, newIdx);

      auto newRef = rewriter.create<util::PackOp>(op->getLoc(), mlir::ValueRange{newIdx, unpackedRef[1]});
      mapping.define(op.getNewRef(), newRef);
      rewriter.replaceTupleStream(op, mapping);
      return success();
   }
};
class UnwrapOptionalHashmapRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::UnwrapOptionalRefOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto optionalType = mlir::dyn_cast_or_null<subop::OptionalType>(op.getOptionalRef().getColumn().type);
      if (!optionalType) return mlir::failure();
      auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(optionalType.getT());
      if (!lookupRefType) return mlir::failure();
      auto hashmapType = mlir::dyn_cast_or_null<subop::HashMapType>(lookupRefType.getState());
      if (!hashmapType) return mlir::failure();
      auto loc = op.getLoc();
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), mapping.resolve(op,op.getOptionalRef()));
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      EntryStorageHelper valStorageHelper(op,hashmapType.getValueMembers(), typeConverter);
      auto valPtrType = valStorageHelper.getRefType();
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, op->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, mapping.resolve(op,op.getOptionalRef()));
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
         mapping.define(op.getRef(), valuePtr);
         rewriter.replaceTupleStream(op, mapping);
      });
      return mlir::success();
   }
};
class UnwrapOptionalPreAggregationHtRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::UnwrapOptionalRefOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto optionalType = mlir::dyn_cast_or_null<subop::OptionalType>(op.getOptionalRef().getColumn().type);
      if (!optionalType) return mlir::failure();
      auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(optionalType.getT());
      if (!lookupRefType) return mlir::failure();
      auto hashmapType = mlir::dyn_cast_or_null<subop::PreAggrHtType>(lookupRefType.getState());
      if (!hashmapType) return mlir::failure();
      auto loc = op.getLoc();
      auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), mapping.resolve(op,op.getOptionalRef()));
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
      auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
      auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
      auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
      EntryStorageHelper valStorageHelper(op,hashmapType.getValueMembers(), typeConverter);
      auto valPtrType = valStorageHelper.getRefType();
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, op->getLoc());
      rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
         Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, mapping.resolve(op,op.getOptionalRef()));
         auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
         mapping.define(op.getRef(), valuePtr);
         rewriter.replaceTupleStream(op, mapping);
      });
      return mlir::success();
   }
};

class GetExternalHashIndexLowering : public SubOpConversionPattern<subop::GetExternalOp> {
   public:
   using SubOpConversionPattern<subop::GetExternalOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GetExternalOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ExternalHashIndexType>(op.getType())) return failure();
      mlir::Value description = rewriter.create<util::CreateConstVarLen>(op->getLoc(), util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());

      rewriter.replaceOp(op, rt::RelationHelper::getIndex(rewriter, op->getLoc())({getExecutionContext(rewriter, op), description})[0]);
      return mlir::success();
   }
};

class CreateSimpleStateLowering : public SubOpConversionPattern<subop::CreateSimpleStateOp> {
   public:
   using SubOpConversionPattern<subop::CreateSimpleStateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateSimpleStateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto simpleStateType = mlir::dyn_cast_or_null<subop::SimpleStateType>(createOp.getType());
      if (!simpleStateType) return failure();

      mlir::Value ref;
      if (createOp->hasAttr("allocateOnHeap")) {
         auto loweredType = mlir::cast<util::RefType>(typeConverter->convertType(createOp.getType()));
         mlir::Value typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), loweredType.getElementType());
         ref = rt::SimpleState::create(rewriter, createOp->getLoc())(mlir::ValueRange{getExecutionContext(rewriter, createOp), typeSize})[0];
         ref = rewriter.create<util::GenericMemrefCastOp>(createOp->getLoc(), loweredType, ref);

      } else {
         rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
            ref = rewriter.create<util::AllocaOp>(createOp->getLoc(), typeConverter->convertType(createOp.getType()), mlir::Value());
         });
      }
      if (!createOp.getInitFn().empty()) {
         EntryStorageHelper storageHelper(createOp,simpleStateType.getMembers(), typeConverter);
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&createOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor returnOpAdaptor) {
            storageHelper.storeOrderedValues(ref, returnOpAdaptor.getResults(), rewriter, createOp->getLoc());
         });
      }
      rewriter.replaceOp(createOp, ref);
      return mlir::success();
   }
};

class NestedMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::NestedMapOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::NestedMapOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::NestedMapOp nestedMapOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      for (auto [p, a] : llvm::zip(nestedMapOp.getParameters(), nestedMapOp.getRegion().front().getArguments().drop_front())) {
         rewriter.map(a, mapping.resolve(nestedMapOp,mlir::cast<tuples::ColumnRefAttr>(p)));
      }
      auto nestedExecutionGroup = mlir::dyn_cast_or_null<subop::NestedExecutionGroupOp>(&nestedMapOp.getRegion().front().front());
      if (!nestedExecutionGroup) {
         nestedMapOp.emitError("NestedMapOp should have a NestedExecutionGroupOp as the first operation in the region");
         return failure();
      }
      auto returnOp = mlir::cast<tuples::ReturnOp>(nestedMapOp.getRegion().front().getTerminator());
      if (!returnOp.getResults().empty()) {
         nestedMapOp.emitError("NestedMapOp should not return any values for the lowering");
         return failure();
      }
      mlir::IRMapping outerMapping;
      for (auto [i, b] : llvm::zip(nestedExecutionGroup.getInputs(), nestedExecutionGroup.getRegion().front().getArguments())) {
         outerMapping.map(b, rewriter.getMapped(i));
      }
      for (auto& op : nestedExecutionGroup.getRegion().front().getOperations()) {
         if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
            auto guard = rewriter.nest(outerMapping, step);
            for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
               mlir::Value input = outerMapping.lookup(param);
               rewriter.map(arg, input);
            }
            mlir::IRMapping cloneMapping;
            std::vector<mlir::Operation*> ops;
            for (auto& op : step.getSubOps().front()) {
               if (&op == step.getSubOps().front().getTerminator())
                  break;
               ops.push_back(&op);
            }
            for (auto* op : ops) {
               op->remove();
               rewriter.insertAndRewrite(op);
            }
            auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
            for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
               auto mapped = rewriter.getMapped(i);
               outerMapping.map(o, mapped);
            }
         }
      }
      rewriter.eraseOp(nestedMapOp);

      return success();
   }
};
template <class T>
static std::vector<T> repeat(T val, size_t times) {
   std::vector<T> res{};
   for (auto i = 0ul; i < times; i++) res.push_back(val);
   return res;
}

class LoopLowering : public SubOpConversionPattern<subop::LoopOp> {
   public:
   using SubOpConversionPattern<subop::LoopOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::LoopOp loopOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto loc = loopOp->getLoc();
      auto* b = loopOp.getBody();
      auto* terminator = b->getTerminator();
      auto continueOp = mlir::cast<subop::LoopContinueOp>(terminator);
      mlir::Value trueValue = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, rewriter.getI1Type());
      std::vector<mlir::Type> iterTypes;
      std::vector<mlir::Value> iterArgs;

      iterTypes.push_back(rewriter.getI1Type());
      for (auto argumentType : loopOp.getBody()->getArgumentTypes()) {
         iterTypes.push_back(typeConverter->convertType(argumentType));
      }
      iterArgs.push_back(trueValue);
      iterArgs.insert(iterArgs.end(), adaptor.getArgs().begin(), adaptor.getArgs().end());
      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, iterTypes, iterArgs);
      auto* before = new Block;
      before->addArguments(iterTypes, repeat(loc, iterTypes.size()));
      whileOp.getBefore().push_back(before);

      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         rewriter.create<mlir::scf::ConditionOp>(loc, before->getArgument(0), before->getArguments());
      });

      auto nestedExecutionGroup = mlir::dyn_cast_or_null<subop::NestedExecutionGroupOp>(&loopOp.getBody()->front());
      if (!nestedExecutionGroup) {
         loopOp.emitError("LoopOp should have a NestedExecutionGroupOp as the first operation in the region");
         return failure();
      }
      auto* after = new Block;
      after->addArguments(iterTypes, repeat(loc, iterTypes.size()));
      whileOp.getAfter().push_back(after);
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         std::vector<mlir::Value> args;
         for (size_t i = 0; i < loopOp.getBody()->getNumArguments(); i++) {
            mlir::Value whileArg = after->getArguments()[i + 1];
            rewriter.map(loopOp.getBody()->getArgument(i), whileArg);
         }
         mlir::IRMapping nestedGroupResultMapping;

         mlir::IRMapping outerMapping;
         for (auto [i, b] : llvm::zip(nestedExecutionGroup.getInputs(), nestedExecutionGroup.getRegion().front().getArguments())) {
            outerMapping.map(b, rewriter.getMapped(i));
         }
         for (auto& op : nestedExecutionGroup.getRegion().front().getOperations()) {
            if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
               auto guard = rewriter.nest(outerMapping, step);
               for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
                  mlir::Value input = outerMapping.lookup(param);
                  rewriter.map(arg, input);
               }
               mlir::IRMapping cloneMapping;
               std::vector<mlir::Operation*> ops;
               for (auto& op : step.getSubOps().front()) {
                  if (&op == step.getSubOps().front().getTerminator())
                     break;
                  ops.push_back(&op);
               }
               for (auto* op : ops) {
                  rewriter.operator mlir::OpBuilder&().setInsertionPointToEnd(after);
                  op->remove();
                  rewriter.insertAndRewrite(op);
               }
               auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
               for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
                  auto mapped = rewriter.getMapped(i);
                  outerMapping.map(o, mapped);
               }
            } else if (auto returnOp = mlir::dyn_cast_or_null<subop::NestedExecutionGroupReturnOp>(&op)) {
               for (auto [i, o] : llvm::zip(returnOp.getInputs(), nestedExecutionGroup.getResults())) {
                  nestedGroupResultMapping.map(o, outerMapping.lookup(i));
               }
            }
         }
         rewriter.operator mlir::OpBuilder&().setInsertionPointToEnd(after);
         std::vector<mlir::Value> res;
         EntryStorageHelper storageHelper(loopOp,mlir::cast<subop::SimpleStateType>(continueOp.getOperandTypes()[0]).getMembers(), typeConverter);
         auto shouldContinueBool = storageHelper.loadValues(nestedGroupResultMapping.lookup(continueOp.getOperand(0)), rewriter, loc)[continueOp.getCondMember().str()];
         res.push_back(shouldContinueBool);
         for (auto operand : continueOp->getOperands().drop_front()) {
            res.push_back(nestedGroupResultMapping.lookup(operand));
         }
         rewriter.create<mlir::scf::YieldOp>(loc, res);
      });
      rewriter.replaceOp(loopOp, whileOp.getResults().drop_front());
      return success();
   }
};

class NestedExecutionGroupLowering : public SubOpConversionPattern<subop::NestedExecutionGroupOp> {
   public:
   using SubOpConversionPattern<subop::NestedExecutionGroupOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::NestedExecutionGroupOp nestedExecutionGroup, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto loc = nestedExecutionGroup->getLoc();
      auto dummyOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, rewriter.getI1Type());
      rewriter.operator mlir::OpBuilder&().setInsertionPoint(dummyOp);
      mlir::IRMapping nestedGroupResultMapping;

      mlir::IRMapping outerMapping;
      for (auto [i, b] : llvm::zip(adaptor.getInputs(), nestedExecutionGroup.getRegion().front().getArguments())) {
         outerMapping.map(b, i);
      }
      std::vector<mlir::Value> toReplaceWith;
      for (auto& op : nestedExecutionGroup.getRegion().front().getOperations()) {
         if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
            auto guard = rewriter.nest(outerMapping, step);
            for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
               mlir::Value input = outerMapping.lookup(param);
               rewriter.map(arg, input);
            }
            mlir::IRMapping cloneMapping;
            std::vector<mlir::Operation*> ops;
            for (auto& op : step.getSubOps().front()) {
               if (&op == step.getSubOps().front().getTerminator())
                  break;
               ops.push_back(&op);
            }
            for (auto* op : ops) {
               rewriter.operator mlir::OpBuilder&().setInsertionPoint(dummyOp);
               op->remove();
               rewriter.insertAndRewrite(op);
            }
            auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
            for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
               auto mapped = rewriter.getMapped(i);
               outerMapping.map(o, mapped);
            }
         } else if (auto returnOp = mlir::dyn_cast_or_null<subop::NestedExecutionGroupReturnOp>(&op)) {
            for (auto [i, o] : llvm::zip(returnOp.getInputs(), nestedExecutionGroup.getResults())) {
               toReplaceWith.push_back(outerMapping.lookup(i));
            }
         }
      }
      rewriter.replaceOp(nestedExecutionGroup, toReplaceWith);
      return success();
   }
};

class SetTrackedCountLowering : public SubOpConversionPattern<subop::SetTrackedCountOp> {
   public:
   using SubOpConversionPattern<subop::SetTrackedCountOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::SetTrackedCountOp setTrackedCountOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto loc = setTrackedCountOp->getLoc();
      mlir::Value executionContext = getExecutionContext(rewriter, setTrackedCountOp);

      // Get resultId
      mlir::Value resultId = rewriter.create<mlir::arith::ConstantIntOp>(loc, setTrackedCountOp.getResultId(), mlir::IntegerType::get(rewriter.getContext(), 32));

      // Get tupleCount
      Value loadedTuple = rewriter.create<util::LoadOp>(loc, adaptor.getTupleCount());
      Value tupleCount = rewriter.create<util::UnPackOp>(loc, loadedTuple).getResults()[0];

      rt::ExecutionContext::setTupleCount(rewriter, loc)({executionContext, resultId, tupleCount});
      rewriter.eraseOp(setTrackedCountOp);
      return mlir::success();
   }
};

}; // namespace
namespace {

void handleExecutionStepCPU(subop::ExecutionStepOp step, subop::ExecutionGroupOp executionGroup, mlir::IRMapping& mapping, mlir::TypeConverter& typeConverter) {
   // llvm::dbgs() << "[CPU] HANDLING STEP " << step << "\n";
   auto* ctxt = step->getContext();
   SubOpRewriter rewriter(step, mapping);
   rewriter.insertPattern<MapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<FilterLowering>(typeConverter, ctxt);
   rewriter.insertPattern<RenameLowering>(typeConverter, ctxt);
   //external
   rewriter.insertPattern<GetExternalTableLowering>(typeConverter, ctxt);
   rewriter.insertPattern<GetExternalHashIndexLowering>(typeConverter, ctxt);
   //ResultTable
   rewriter.insertPattern<CreateTableLowering>(typeConverter, ctxt);
   rewriter.insertPattern<MaterializeTableLowering>(typeConverter, ctxt);
   rewriter.insertPattern<CreateFromResultTableLowering>(typeConverter, ctxt);
   //SimpleState
   rewriter.insertPattern<CreateSimpleStateLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanRefsSimpleStateLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LookupSimpleStateLowering>(typeConverter, ctxt);
   //Table
   rewriter.insertPattern<ScanRefsTableLowering>(typeConverter, ctxt);
   //rewriter.insertPattern<ScanRefsLocalTableLowering>(typeConverter, ctxt);
   rewriter.insertPattern<TableRefGatherOpLowering>(typeConverter, ctxt);
   //Buffer
   rewriter.insertPattern<CreateBufferLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanRefsVectorLowering>(typeConverter, ctxt);
   rewriter.insertPattern<MaterializeVectorLowering>(typeConverter, ctxt);

   //Hashmap
   rewriter.insertPattern<CreateHashMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanHashMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LookupHashMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<PureLookupHashMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<HashMapRefGatherOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanHashMapListLowering>(typeConverter, ctxt);
   //rewriter.insertPattern<LockLowering>(typeConverter, ctxt);

   //HashMultiMap
   rewriter.insertPattern<CreateHashMultiMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<InsertMultiMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LookupHashMultiMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanMultiMapListLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanHashMultiMap>(typeConverter, ctxt);
   rewriter.insertPattern<HashMultiMapRefGatherOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<HashMultiMapScatterOp>(typeConverter, ctxt);

   // ExternalHashIndex
   rewriter.insertPattern<ScanExternalHashIndexListLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LookupExternalHashIndexLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ExternalHashIndexRefGatherOpLowering>(typeConverter, ctxt);

   //SortedView
   rewriter.insertPattern<SortLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanRefsSortedViewLowering>(typeConverter, ctxt);
   //HashIndexedView
   rewriter.insertPattern<CreateHashIndexedViewLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LookupHashIndexedViewLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanListLowering>(typeConverter, ctxt);
   //ContinuousView
   rewriter.insertPattern<CreateContinuousViewLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanRefsContinuousViewLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ContinuousRefGatherOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ContinuousRefScatterOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ReduceContinuousRefLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ReduceContinuousRefAtomicLowering>(typeConverter, ctxt);

   //Heap
   rewriter.insertPattern<CreateHeapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanRefsHeapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<MaterializeHeapLowering>(typeConverter, ctxt);
   //SegmentTreeView
   rewriter.insertPattern<CreateSegmentTreeViewLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LookupSegmentTreeViewLowering>(typeConverter, ctxt);
   //Array
   rewriter.insertPattern<CreateArrayLowering>(typeConverter, ctxt);
   //ThreadLocal
   rewriter.insertPattern<CreateThreadLocalLowering>(typeConverter, ctxt);
   rewriter.insertPattern<MergeThreadLocalResultTable>(typeConverter, ctxt);
   rewriter.insertPattern<MergeThreadLocalBuffer>(typeConverter, ctxt);
   rewriter.insertPattern<MergeThreadLocalHeap>(typeConverter, ctxt);
   rewriter.insertPattern<MergeThreadLocalSimpleState>(typeConverter, ctxt);
   rewriter.insertPattern<MergeThreadLocalHashMap>(typeConverter, ctxt);
   rewriter.insertPattern<CreateOpenHtFragmentLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LookupPreAggrHtFragment>(typeConverter, ctxt);
   rewriter.insertPattern<MergePreAggrHashMap>(typeConverter, ctxt);
   rewriter.insertPattern<ScanPreAggrHtLowering>(typeConverter, ctxt);
   rewriter.insertPattern<PreAggrHtRefGatherOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<PureLookupPreAggregationHtLowering>(typeConverter, ctxt);
   rewriter.insertPattern<UnwrapOptionalPreAggregationHtRefLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScanPreAggregationHtListLowering>(typeConverter, ctxt);

   rewriter.insertPattern<DefaultGatherOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ScatterOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<ReduceOpLowering>(typeConverter, ctxt);
   rewriter.insertPattern<NestedMapLowering>(typeConverter, ctxt);
   rewriter.insertPattern<UnrealizedConversionCastLowering>(typeConverter, ctxt);
   rewriter.insertPattern<UnwrapOptionalHashmapRefLowering>(typeConverter, ctxt);
   rewriter.insertPattern<OffsetReferenceByLowering>(typeConverter, ctxt);
   rewriter.insertPattern<GetBeginLowering>(typeConverter, ctxt);
   rewriter.insertPattern<GetEndLowering>(typeConverter, ctxt);
   rewriter.insertPattern<EntriesBetweenLowering>(typeConverter, ctxt);
   rewriter.insertPattern<InFlightLowering>(typeConverter, ctxt);
   rewriter.insertPattern<GenerateLowering>(typeConverter, ctxt);
   rewriter.insertPattern<LoopLowering>(typeConverter, ctxt);
   rewriter.insertPattern<NestedExecutionGroupLowering>(typeConverter, ctxt);
   //rewriter.insertPattern<GetSingleValLowering>(typeConverter, ctxt);
   rewriter.insertPattern<SetTrackedCountLowering>(typeConverter, ctxt);
   //rewriter.insertPattern<SimpleStateGetScalarLowering>(typeConverter, ctxt);

   for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
      mlir::Value input = mapping.lookup(param);
      if (!mlir::cast<mlir::BoolAttr>(isThreadLocal).getValue()) {
         rewriter.map(arg, input);
      } else {
         mlir::OpBuilder b(executionGroup);
         mlir::Value threadLocal = rt::ThreadLocal::getLocal(b, b.getUnknownLoc())({mapping.lookup(param)})[0];
         threadLocal = b.create<util::GenericMemrefCastOp>(threadLocal.getLoc(), typeConverter.convertType(arg.getType()), threadLocal);
         rewriter.map(arg, threadLocal);
      }
   }
   std::vector<mlir::Operation*> ops;
   for (auto& op : step.getSubOps().front()) {
      if (&op == step.getSubOps().front().getTerminator()) {
         break;
      }
      ops.push_back(&op);
   }
   for (auto* op : ops) {
      // llvm::dbgs() << "====OP: " << *op <<"\n";
      rewriter.rewrite(op, executionGroup);
   }
   auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
   for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
      mapping.map(o, rewriter.getMapped(i));
   }
   rewriter.cleanup();
}
} // namespace

void SubOpToControlFlowLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<util::UtilDialect>()->getFunctionHelper().setParentModule(module);
   auto* ctxt = &getContext();

   TypeConverter typeConverter;
   typeConverter.addConversion([](mlir::Type t) { return t; });
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](subop::TableType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::LocalTableType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ResultTableType t) -> Type {
      std::vector<mlir::Type> types;
      for (auto typeAttr : t.getMembers().getTypes()) {
         types.push_back(dsa::ColumnBuilderType::get(ctxt, getBaseType(mlir::cast<mlir::TypeAttr>(typeAttr).getValue())));
      }
      return util::RefType::get(mlir::TupleType::get(ctxt, types));
   });
   typeConverter.addConversion([&](subop::BufferType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::SortedViewType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr,t.getBasedOn().getMembers(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::ArrayType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr,t.getMembers(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::ContinuousViewType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr,t.getBasedOn().getMembers(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::SimpleStateType t) -> Type {
      return util::RefType::get(t.getContext(), EntryStorageHelper(nullptr,t.getMembers(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::HashMapType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::PreAggrHtFragmentType t) -> Type {
      return util::RefType::get(t.getContext(), util::RefType::get(t.getContext(), getHtEntryType(t, typeConverter)));
   });
   typeConverter.addConversion([&](subop::PreAggrHtType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::HashMultiMapType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ThreadLocalType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::SegmentTreeViewType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::HashIndexedViewType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::HeapType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ExternalHashIndexType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::ListType t) -> Type {
      if (auto lookupEntryRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(t.getT())) {
         if (mlir::isa<subop::HashMapType>(lookupEntryRefType.getState())) {
            return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
         }
         if (auto hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(lookupEntryRefType.getState())) {
            return util::RefType::get(t.getContext(), getHashMultiMapEntryType(hashMultiMapType, typeConverter));
         }
         if (auto externalHashIndexRefType = mlir::dyn_cast_or_null<subop::ExternalHashIndexType>(t.getT())) {
            return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
         }
         return mlir::TupleType::get(t.getContext(), {util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8)), mlir::IndexType::get(t.getContext())});
      }
      if (auto hashMapEntryRefType = mlir::dyn_cast_or_null<subop::HashMapEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      if (auto hashMapEntryRefType = mlir::dyn_cast_or_null<subop::PreAggrHTEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      if (auto hashMultiMapEntryRefType = mlir::dyn_cast_or_null<subop::HashMultiMapEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), getHashMultiMapEntryType(hashMultiMapEntryRefType.getHashMultimap(), typeConverter));
      }
      if (auto externalHashIndexRefType = mlir::dyn_cast_or_null<subop::ExternalHashIndexEntryRefType>(t.getT())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      return mlir::Type();
   });
   typeConverter.addConversion([&](subop::HashMapEntryRefType t) -> Type {
      auto hashMapType = t.getHashMap();
      return util::RefType::get(t.getContext(), getHtKVType(hashMapType, typeConverter));
   });
   typeConverter.addConversion([&](subop::PreAggrHTEntryRefType t) -> Type {
      auto hashMapType = t.getHashMap();
      return util::RefType::get(t.getContext(), getHtKVType(hashMapType, typeConverter));
   });
   typeConverter.addConversion([&](subop::LookupEntryRefType t) -> Type {
      if (mlir::isa<subop::HashMapType, subop::PreAggrHtFragmentType>(t.getState())) {
         return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
      }
      if (auto hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(t.getState())) {
         return util::RefType::get(t.getContext(), getHashMultiMapEntryType(hashMultiMapType, typeConverter));
      }
      return mlir::TupleType::get(t.getContext(), {util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8)), mlir::IndexType::get(t.getContext())});
   });

   //basic tuple stream manipulation

   //rewriter.rewrite(module.getBody());
   std::vector<mlir::Operation*> toRemove;
   module->walk([&](subop::ExecutionGroupOp executionGroup) { // walk over "queries"
      mlir::IRMapping mapping;
      //todo: handle arguments of executionGroup
      for (auto& op : executionGroup.getRegion().front().getOperations()) {
         if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
#ifdef TRACER
            mlir::Value tracingStep;
            {
               std::string stepLocStr = "";

               if (auto lineLoc = op.getLoc()->dyn_cast_or_null<mlir::FileLineColLoc>()) {
                  auto fileName = lineLoc.getFilename().str();
                  auto baseNameStarts = fileName.find_last_of("/");
                  if (baseNameStarts != std::string::npos) {
                     fileName = fileName.substr(baseNameStarts + 1);
                  }
                  auto endingStarts = fileName.find(".");
                  if (endingStarts != std::string::npos) {
                     fileName = fileName.substr(0, endingStarts);
                  };
                  stepLocStr = fileName + std::string(":") + std::to_string(lineLoc.getLine());
               }
               mlir::OpBuilder builder(executionGroup);

               mlir::Value stepLoc = builder.create<util::CreateConstVarLen>(op.getLoc(), util::VarLen32Type::get(ctxt), stepLocStr);
               tracingStep = rt::ExecutionStepTracing::start(builder, op.getLoc())({stepLoc})[0];
            }
#endif
            handleExecutionStepCPU(step, executionGroup, mapping, typeConverter);
#if TRACER
            {
               mlir::OpBuilder builder(executionGroup);
               rt::ExecutionStepTracing::end(builder, op.getLoc())({tracingStep});
            }
#endif
         }
      }
      auto returnOp = mlir::cast<subop::ExecutionGroupReturnOp>(executionGroup.getRegion().front().getTerminator());
      std::vector<mlir::Value> results;
      for (auto i : returnOp.getInputs()) {
         results.push_back(mapping.lookup(i));
      }
      executionGroup.replaceAllUsesWith(results);
      toRemove.push_back(executionGroup);
   });
   getOperation()->walk([&](subop::SetResultOp setResultOp) {
      mlir::OpBuilder builder(setResultOp);
      builder.create<dsa::SetResultOp>(setResultOp->getLoc(), setResultOp.getResultId(), setResultOp.getState());
      toRemove.push_back(setResultOp);
   });
   for (auto* op : toRemove) {
      op->dropAllReferences();
      op->dropAllDefinedValueUses();
      op->erase();
   }
   std::vector<mlir::Operation*> defs;
   for (auto& op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(&op)) {
         if (funcOp.getBody().empty()) {
            defs.push_back(&op);
         }
      }
   }
   for (auto* op : defs) {
      op->moveBefore(&module.getBody()->getOperations().front());
   }
   module->walk([&](tuples::GetParamVal getParamVal) {
      getParamVal.replaceAllUsesWith(getParamVal.getParam());
   });
}
//} //namespace
std::unique_ptr<mlir::Pass> subop::createLowerSubOpPass() {
   return std::make_unique<SubOpToControlFlowLoweringPass>();
}
void subop::setCompressionEnabled(bool compressionEnabled) {
   EntryStorageHelper::compressionEnabled = compressionEnabled;
}
void subop::createLowerSubOpPipeline(mlir::OpPassManager& pm) {
   pm.addPass(subop::createGlobalOptPass());
   pm.addPass(subop::createFoldColumnsPass());
   pm.addPass(subop::createReuseLocalPass());
   pm.addPass(subop::createSpecializeSubOpPass(true));
   pm.addPass(subop::createNormalizeSubOpPass());
   pm.addPass(subop::createPullGatherUpPass());
   pm.addPass(subop::createParallelizePass());
   pm.addPass(subop::createEnforceOrderPass());
   pm.addPass(subop::createInlineNestedMapPass());
   pm.addPass(subop::createFinalizePass());
   pm.addPass(subop::createSplitIntoExecutionStepsPass());
   pm.addNestedPass<mlir::func::FuncOp>(subop::createParallelizePass());
   pm.addPass(subop::createSpecializeParallelPass());
   pm.addPass(subop::createPrepareLoweringPass());
   pm.addPass(subop::createLowerSubOpPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addPass(mlir::createCSEPass());
}
void subop::registerSubOpToControlFlowConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createLowerSubOpPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-subop",
      "",
      subop::createLowerSubOpPipeline);
}