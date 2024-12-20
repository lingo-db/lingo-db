#include "lingodb/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"

#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/runtime/ArrowColumn.h"
#include "lingodb/compiler/runtime/ArrowTable.h"
#include "lingodb/compiler/runtime/Buffer.h"
#include "lingodb/compiler/runtime/DataSourceIteration.h"
#include "lingodb/compiler/runtime/EntryLock.h"
#include "lingodb/compiler/runtime/ExecutionContext.h"
#include "lingodb/compiler/runtime/GPU/DeviceRTWrapper.h"
#include "lingodb/compiler/runtime/GrowingBuffer.h"
#include "lingodb/compiler/runtime/HashMultiMap.h"
#include "lingodb/compiler/runtime/Hashtable.h"
#include "lingodb/compiler/runtime/Heap.h"
#include "lingodb/compiler/runtime/LazyJoinHashtable.h"
#include "lingodb/compiler/runtime/LingoDBHashIndex.h"
#include "lingodb/compiler/runtime/PreAggregationHashtable.h"
#include "lingodb/compiler/runtime/RelationHelper.h"
#include "lingodb/compiler/runtime/SegmentTreeView.h"
#include "lingodb/compiler/runtime/SimpleState.h"
#include "lingodb/compiler/runtime/ThreadLocal.h"
#include "lingodb/compiler/runtime/Tracing.h"
#include "lingodb/runtime/GPU/CUDA/GrowingBuffer.cuh"
#include "lingodb/runtime/GPU/CUDA/PreAggregationHashTable.cuh"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

#include <stack>
using namespace mlir;

#ifdef NDEBUG
#define ASSERT_WITH_OP(cond, op, msg)
#else
#define ASSERT_WITH_OP(cond, op, msg) \
   if (!(cond)) {                     \
      op->emitOpError(msg);           \
   }                                  \
   assert(cond)
#endif
namespace {
constexpr uint8_t powerOfTwo(int64_t n, int power = 0) { return (n == 1) ? power : powerOfTwo(n / 2, power + 1); }

using namespace lingodb::compiler::dialect;
namespace rt = lingodb::compiler::runtime;
using Member = subop::Member;
struct SubOpToControlFlowLoweringPass
   : public mlir::PassWrapper<SubOpToControlFlowLoweringPass, OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubOpToControlFlowLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "lower-subop-to-cf"; }

   SubOpToControlFlowLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect, arrow::ArrowDialect, subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};

static llvm::SmallVector<mlir::Value> inlineBlock(mlir::Block* b, mlir::OpBuilder& rewriter, mlir::ValueRange arguments) {
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
   llvm::SmallVector<mlir::Value> res;
   for (auto val : returnOp.getResults()) {
      res.push_back(mapper.lookup(val));
   }
   return res;
}
static llvm::SmallVector<Type> unpackTypes(subop::StateMembersAttr membersAttr) {
   auto& memberManager = membersAttr.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::SmallVector<Type> res;
   for (auto x : membersAttr.getMembers()) {
      res.push_back(memberManager.getType(x));
   }
   return res;
};
class ColumnMapping {
   llvm::SmallDenseMap<const tuples::Column*, mlir::Value> mapping;

   public:
   ColumnMapping() : mapping() {}

   // Move constructor
   ColumnMapping(ColumnMapping&& other) noexcept
      : mapping(std::move(other.mapping)) {}

   // Move assignment operator
   ColumnMapping& operator=(ColumnMapping&& other) noexcept {
      if (this != &other) {
         mapping = std::move(other.mapping);
      }
      return *this;
   }

   // Copy constructor and assignment can be defaulted or deleted as needed
   ColumnMapping(const ColumnMapping&) = default;
   ColumnMapping& operator=(const ColumnMapping&) = default;

   ColumnMapping(subop::InFlightOp inFlightOp) {
      assert(!!inFlightOp);
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
   llvm::SmallVector<mlir::Value> resolve(mlir::Operation* op, mlir::ArrayAttr arr) {
      llvm::SmallVector<mlir::Value> res;
      for (auto attr : arr) {
         res.push_back(resolve(op, mlir::cast<tuples::ColumnRefAttr>(attr)));
      }
      return res;
   }
   mlir::Value createInFlight(mlir::OpBuilder& builder) {
      llvm::SmallVector<mlir::Value> values;
      llvm::SmallVector<mlir::Attribute> columns;

      for (auto m : mapping) {
         columns.push_back(builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createDef(m.first));
         values.push_back(m.second);
      }
      return builder.create<subop::InFlightOp>(builder.getUnknownLoc(), values, builder.getArrayAttr(columns));
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
   bool withLock;

   struct MemberInfo {
      bool isNullable;
      size_t nullBitOffset;
      mlir::Type stored;
      size_t offset;
   };
   llvm::SmallDenseMap<Member, MemberInfo, 16> memberInfos;

   public:
   static bool compressionEnabled;
   EntryStorageHelper(mlir::Operation* op, subop::StateMembersAttr members, bool withLock, mlir::TypeConverter* typeConverter) : op(op), members(members), withLock(withLock) {
      auto& memberManager = members.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      llvm::SmallVector<mlir::Type> types;
      size_t nullBitOffset = 0;
      for (auto m : members.getMembers()) {
         auto type = memberManager.getType(m);
         auto converted = typeConverter->convertType(type);
         type = converted ? converted : type;
         MemberInfo memberInfo;
         if (auto nullableType = mlir::dyn_cast_or_null<db::NullableType>(type)) {
            auto charType = mlir::dyn_cast_or_null<db::CharType>(nullableType.getType());
            if (mlir::isa<db::StringType>(nullableType.getType()) || (charType && charType.getLen() > 1)) {
               memberInfo.isNullable = false;
               memberInfo.stored = type;
            } else {
               // Compression is bounded
               if (compressionEnabled && nullBitOffset <= 63) {
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
            }
         } else {
            memberInfo.isNullable = false;
            memberInfo.stored = type;
         }
         memberInfo.offset = types.size();
         memberInfos.insert({m, memberInfo});
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
      if (withLock) {
         auto lockType = mlir::IntegerType::get(members.getContext(), 8);
         types.push_back(lockType);
      }
      storageType = mlir::TupleType::get(members.getContext(), types);
   }
   mlir::Value getPointer(mlir::Value ref, Member member, mlir::OpBuilder& rewriter, mlir::Location loc) {
      const auto& memberInfo = memberInfos.at(member);
      assert(!memberInfo.isNullable);
      return rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), memberInfo.stored), ref, memberInfo.offset);
   }
   mlir::Value getLockPointer(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) {
      assert(withLock);
      assert(mlir::isa<mlir::IntegerType>(storageType.getTypes().back()));
      return rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), storageType.getTypes().back()), ref, storageType.getTypes().size() - 1);
   }

   class LazyValueMap {
      llvm::SmallDenseMap<Member, mlir::Value> values;
      // name -> (isNull: i1) mapping
      llvm::SmallDenseMap<Member, mlir::Value> nullBitCache;
      std::optional<mlir::Value> nullBitSet;
      mlir::Value ref;
      bool refIsRefType = false;

      mlir::OpBuilder& rewriter;
      mlir::Location loc;
      const EntryStorageHelper& esh;
      llvm::SmallVector<Member> relevantMembers;

      public:
      LazyValueMap(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc, const EntryStorageHelper& esh, std::optional<llvm::SmallVector<Member>> relevantMembers) : ref(ref), rewriter(rewriter), loc(loc), esh(esh) {
         if (!relevantMembers) {
            this->relevantMembers = esh.members.getMembers();
         } else {
            this->relevantMembers = *relevantMembers;
         }
      }
      void set(const Member& member, mlir::Value value) {
         values[member] = std::move(value);
      }
      mlir::Value& get(const Member& member) {
         assert(esh.memberInfos.contains(member) && "Member not found");
         return values[member] = loadValue(member);
      }

      void store() {
         ensureRefIsRefType();
         if (esh.nullBitsetType) {
            if (values.size() < esh.memberInfos.size()) {
               // load null bit set if not already loaded (not loaded if only set() was called on the map)
               // null bit set not required if all members are loaded
               populateNullBitSet();
            } else if (!nullBitSet) {
               nullBitSet = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, esh.nullBitsetType);
            }
         }
         for (auto& [name, value] : values) {
            const MemberInfo& memberInfo = esh.memberInfos.at(name);
            if (memberInfo.isNullable) {
               llvm::SmallVector<mlir::Value> isNullVals;
               rewriter.createOrFold<db::IsNullOp>(isNullVals, loc, value);
               const mlir::Value nullBit = isNullVals[0];
               nullBitSet = rewriter.create<util::SetBitConstOp>(loc, esh.nullBitsetType, *nullBitSet, nullBit, memberInfo.nullBitOffset);
               llvm::SmallVector<mlir::Value> rawValues;
               rewriter.createOrFold<db::NullableGetVal>(rawValues, loc, value);
               value = rawValues[0];
            }
            // now, actually store the value (not just whether it's null)
            rewriter.create<util::StoreElementOp>(loc, value, ref, memberInfo.offset);
         }
         if (esh.nullBitsetType) {
            rewriter.create<util::StoreElementOp>(loc, *nullBitSet, ref, esh.nullBitSetPos);
         }
      }

      struct Iterator {
         LazyValueMap* lvm;
         size_t index;

         using iterator_category = std::input_iterator_tag;
         using value_type = mlir::Value;
         using difference_type = std::ptrdiff_t;
         using pointer = mlir::Value*;
         using reference = mlir::Value&;

         Iterator& operator++() {
            ++index;
            return *this;
         }

         Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
         }

         bool operator==(const Iterator& other) const {
            return lvm == other.lvm && index == other.index;
         }

         value_type operator*() const {
            return lvm->loadValue(lvm->relevantMembers[index]);
         }
      };
      static_assert(std::input_iterator<Iterator>);
      friend Iterator;

      Iterator begin() {
         return Iterator(this, 0);
      }

      Iterator end() {
         return Iterator(this, std::distance(relevantMembers.begin(), relevantMembers.end()));
      }

      private:
      mlir::Value loadValue(const Member& member) {
         ensureRefIsRefType();
         const MemberInfo& memberInfo = esh.memberInfos.at(member);
         mlir::Value value = rewriter.create<util::LoadElementOp>(loc, memberInfo.stored, ref, memberInfo.offset);
         if (memberInfo.isNullable) {
            populateNullBitSet();
            assert(nullBitSet);
            mlir::Value isNull = rewriter.create<util::IsBitSetConstOp>(loc, rewriter.getI1Type(), *nullBitSet, memberInfo.nullBitOffset);
            value = rewriter.create<db::AsNullableOp>(loc, db::NullableType::get(rewriter.getContext(), memberInfo.stored), value, isNull);
            nullBitCache.insert({member, isNull});
         }
         return value;
      }

      void populateNullBitSet() {
         if (nullBitSet) {
            return;
         }
         assert(esh.nullBitsetType && "NullBitSetType must be set if one of the fields is nullable.");
         nullBitSet = rewriter.create<util::LoadElementOp>(loc, esh.nullBitsetType, ref, esh.nullBitSetPos);
      }

      void ensureRefIsRefType() {
         if (!refIsRefType) {
            ref = esh.ensureRefType(ref, rewriter, loc);
            refIsRefType = true;
         }
      }
   };

   mlir::TupleType getStorageType() const {
      return storageType;
   }
   util::RefType getRefType() const {
      return util::RefType::get(members.getContext(), getStorageType());
   }
   mlir::Value ensureRefType(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) const {
      auto refType = mlir::cast<util::RefType>(ref.getType());
      auto expectedType = getRefType();
      if (refType != expectedType) {
         ref = rewriter.create<util::GenericMemrefCastOp>(loc, expectedType, ref);
      }
      return ref;
   }
   llvm::SmallVector<mlir::Value> resolve(mlir::Operation* op, subop::ColumnRefMemberMappingAttr mapping, ColumnMapping columnMapping) {
      llvm::SmallVector<mlir::Value> result;
      for (auto m : members.getMembers()) {
         result.push_back(columnMapping.resolve(op, mapping.getColumnRef(m)));
      }
      return result;
   }
   LazyValueMap getValueMap(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc, ArrayAttr relevantMembers = {}) {
      llvm::SmallVector<Member> relevantMembersVec;
      if (relevantMembers) {
         for (auto mAttr : relevantMembers) {
            relevantMembersVec.push_back(mlir::cast<subop::MemberAttr>(mAttr).getMember());
         }
      }
      return LazyValueMap(ref, rewriter, loc, *this, relevantMembers ? relevantMembersVec : std::optional<llvm::SmallVector<Member>>());
   }
   template <class L>
   void storeOrderedValues(mlir::Value ref, L list, mlir::OpBuilder& rewriter, mlir::Location loc) {
      auto values = getValueMap(ref, rewriter, loc);
      auto membersList = members.getMembers();
      for (size_t i = 0; i < membersList.size() && i < list.size(); ++i) {
         values.set(membersList[i], list[i]);
      }
      values.store();
   }

   void storeFromColumns(subop::ColumnRefMemberMappingAttr mapping, ColumnMapping& columnMapping, mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) {
      auto values = getValueMap(ref, rewriter, loc);
      for (auto x : mapping.getMapping()) {
         values.set(x.first, columnMapping.resolve(op, x.second));
      }
      values.store();
   }
   void loadIntoColumns(subop::ColumnDefMemberMappingAttr mapping, ColumnMapping& columnMapping, mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) {
      auto values = getValueMap(ref, rewriter, loc);
      for (auto x : mapping.getMapping()) {
         if (memberInfos.contains(x.first)) {
            columnMapping.define(x.second, values.get(x.first));
         }
      }
   }
};
bool EntryStorageHelper::compressionEnabled = true;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// State management ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SubOpRewriter;
class AbstractSubOpConversionPattern {
   protected:
   mlir::TypeConverter* typeConverter;
   mlir::OperationName operationName;
   PatternBenefit benefit;
   mlir::MLIRContext* context;

   public:
   AbstractSubOpConversionPattern(TypeConverter* typeConverter, const mlir::OperationName& operationName, const PatternBenefit& benefit, MLIRContext* context) : typeConverter(typeConverter), operationName(operationName), benefit(benefit), context(context) {}
   virtual LogicalResult matchAndRewrite(mlir::Operation*, SubOpRewriter& rewriter) = 0;
   mlir::MLIRContext* getContext() const {
      return context;
   }
   const mlir::OperationName& getOperationName() const {
      return operationName;
   }
   const PatternBenefit& getBenefit() const {
      return benefit;
   }
   virtual ~AbstractSubOpConversionPattern() {};
};
struct InFlightTupleStream {
   subop::InFlightOp inFlightOp;
   ColumnMapping columnMapping;
};

struct PatternList {
   llvm::DenseMap<OperationName, llvm::SmallVector<std::unique_ptr<AbstractSubOpConversionPattern>>> patterns;
   template <class PatternT, typename... Args>
   void insertPattern(Args&&... args) {
      auto uniquePtr = std::make_unique<PatternT>(std::forward<Args>(args)...);
      patterns[uniquePtr->getOperationName()].push_back(std::move(uniquePtr));
   }
};
class SubOpRewriter {
   PatternList& patternList;
   mlir::OpBuilder builder;
   llvm::SmallVector<mlir::IRMapping> valueMapping;
   llvm::DenseMap<mlir::Value, InFlightTupleStream> inFlightTupleStreams;
   llvm::SmallVector<mlir::Operation*> toErase;
   llvm::DenseSet<mlir::Operation*> isErased;
   llvm::SmallVector<mlir::Operation*> toRewrite;
   mlir::Operation* currentStreamLoc = nullptr;

   mlir::gpu::GPUModuleOp gpuModule; // container for device funcs
   mlir::gpu::GPUFuncOp gpuStepKernel; // contains execution step code
   mlir::Block* kernelBlock{nullptr}; // aggregate arguments to later create kernel from it.
   mlir::func::FuncOp gpuStepWrapper; // callback passed to runtime

   struct ExecutionStepContext {
      subop::ExecutionStepOp executionStep;
      mlir::IRMapping& outerMapping;
   };
   std::stack<ExecutionStepContext> executionStepContexts;

   util::RefType barePtrType;

   public:
   SubOpRewriter(PatternList& patternList, subop::ExecutionStepOp executionStep, mlir::IRMapping& outerMapping) : patternList(patternList), builder(executionStep), executionStepContexts{} {
      valueMapping.push_back(mlir::IRMapping());
      executionStepContexts.push({executionStep, outerMapping});
      barePtrType = util::RefType::get(getContext(), builder.getI8Type());
   }
   mlir::Value entryWithNextPtr; // Used for locking in PreAggrHTFragment building
   std::vector<mlir::Value> wrapperKernelLocals;
   mlir::Operation* firstKernelLocalFreeInWrapper{nullptr}; // we insert frees to the end of wrapper, before creating a kernel launch, use this ptr as insertion ptr for launch

   mlir::gpu::GPUFuncOp finalizeGPUKernel() {
      // We have aggregated all input types, now we need to create a function
      static uint32_t kernelId{0};
      llvm::SmallVector<mlir::Type, 4> kernelInputTypes;
      for (auto arg : kernelBlock->getArguments()) {
         kernelInputTypes.push_back(arg.getType());
      }
      atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
         gpuStepKernel = rewriter.create<mlir::gpu::GPUFuncOp>(gpuModule.getLoc(), "kernel_" + std::to_string(kernelId++), mlir::FunctionType::get(getContext(), kernelInputTypes, {}));
      });
      gpuStepKernel.getBlocks().pop_front(); // We already have a block, no need for an implicit one.
      assert(gpuStepKernel.getBlocks().empty());
      gpuStepKernel.getBlocks().push_back(kernelBlock);
      gpuStepKernel->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());
      return gpuStepKernel;
   }
   void setGPUModule(mlir::gpu::GPUModuleOp newGPUModule) {
      assert(!gpuModule && "We expect only one gpu module for a rewriter!");
      gpuModule = newGPUModule;
   }
   void setGPUStepWrapper(mlir::func::FuncOp wrapper) {
      assert(!gpuStepWrapper && "We expect only one wrapper per step for a rewriter!");
      gpuStepWrapper = wrapper;
   }
   void setGPUKernelBody(mlir::Block* block) {
      assert(!kernelBlock && "We expect only one block per step for a rewriter!");
      kernelBlock = block;
   }
   auto getGPUKernelFunc() {
      assert(gpuStepKernel && "GPU kernel func must be valid!");
      return gpuStepKernel;
   }
   auto getGPUKernelBody() {
      assert(kernelBlock && "GPU module must be valid!");
      return kernelBlock;
   }
   auto getGPUModule() { return gpuModule; }
   auto getGPUStepWrapper() {
      assert(gpuStepWrapper && "GPU wrapper must be valid!");
      return gpuStepWrapper;
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
   auto getI32Type() { return builder.getI32Type(); }
   auto getLLVMPtrType() { return mlir::LLVM::LLVMPointerType::get(getContext()); }
   auto getI8PtrType() { return util::RefType::get(getI8Type()); }
   auto getStringAttr(const Twine& bytes) { return builder.getStringAttr(bytes); }
   auto getPtrType() { return barePtrType; }
   std::vector<mlir::Type> getExecutionStepParamTypes(subop::ExecutionStepOp& executionStep, mlir::IRMapping& outerMapping, mlir::Value exclude = {}) {
      std::vector<mlir::Type> types;
      for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
         if (exclude && arg == exclude && arg.hasOneUse()) continue;
         mlir::Value input = outerMapping.lookup(param);
         types.push_back(input.getType());
      }
      return types;
   }
   mlir::Value storeStepRequirements(mlir::Value exclude = {}) {
      auto outerMapping = executionStepContexts.top().outerMapping;
      auto executionStep = executionStepContexts.top().executionStep;
      std::vector<mlir::Type> types = getExecutionStepParamTypes(executionStep, outerMapping, exclude);
      auto tupleType = mlir::TupleType::get(getContext(), types);
      mlir::Value contextPtr = create<util::AllocaOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), tupleType), mlir::Value());
      size_t offset = 0;
      for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
         if (exclude && arg == exclude && arg.hasOneUse()) continue;
         mlir::Value input = outerMapping.lookup(param);
         create<util::StoreElementOp>(builder.getUnknownLoc(), input, contextPtr, offset++);
      }
      contextPtr = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), barePtrType, contextPtr);
      return contextPtr;
   }
   mlir::Value storeStepRequirementsGPU() {
      auto outerMapping = executionStepContexts.top().outerMapping;
      auto executionStep = executionStepContexts.top().executionStep;
      auto loc = executionStep->getLoc();
      auto* ctxt = executionStep->getContext();
      // Collect step inputs that will represent a context: a tuple of input states, ptrs, etc.
      std::vector<mlir::Type> types = getExecutionStepParamTypes(executionStep, outerMapping);
      // Allocate the context tuple on CPU and GPU
      mlir::TupleType tupleType = mlir::TupleType::get(ctxt, types);
      util::RefType tupleRefType = util::RefType::get(ctxt, tupleType);
      mlir::Value contextPtrCPU = create<util::AllocaOp>(loc, tupleRefType, mlir::Value()); // allocation size is deduced from type
      mlir::Value tupleTySize = create<util::SizeOfOp>(loc, getIndexType(), tupleType); // for gpu the size is more explicit
      mlir::Value contextPtrGPU = rt::DeviceMemoryFuncs::getPtrForArray(*this, loc)({tupleTySize})[0];
      contextPtrGPU = create<util::GenericMemrefCastOp>(loc, tupleRefType, contextPtrGPU);

      // Store each step input to the corresponding offset in the allocated CPU tuple
      size_t offset{0};
      for (auto param : executionStep.getInputs()) {
         mlir::Value input = outerMapping.lookup(param);
         auto memberRef = create<util::TupleElementPtrOp>(loc, util::RefType::get(ctxt, input.getType()), contextPtrCPU, offset++);
         create<util::StoreOp>(loc, input, memberRef, mlir::Value());
      }
      // Send the CPU tuple to GPU (we assume they have the same size).
      rt::DeviceMemoryFuncs::threadSendToGPUSync(*this, loc)({contextPtrCPU, contextPtrGPU, tupleTySize});

      return contextPtrGPU;
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
   Guard loadStepRequirements(mlir::Value contextPtr, mlir::TypeConverter* typeConverter, mlir::Value exclude = {}) {
      auto loc = builder.getUnknownLoc();
      auto* ctxt = getContext();
      // Maps arguments to the parameters loaded from contextPtr. Returns rewriter with a new IRMapping.
      auto outerMapping = executionStepContexts.top().outerMapping;
      auto executionStep = executionStepContexts.top().executionStep;
      std::vector<mlir::Type> types = getExecutionStepParamTypes(executionStep, outerMapping, exclude);
      auto tupleType = mlir::TupleType::get(getContext(), types);
      contextPtr = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), tupleType), contextPtr);
      Guard guard(*this);
      size_t offset = 0;
      for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
         if (exclude && arg == exclude && arg.hasOneUse()) continue;
         mlir::Value input = outerMapping.lookup(param);
         mlir::Value value = create<util::LoadElementOp>(builder.getUnknownLoc(), input.getType(), contextPtr, offset++);
         if (mlir::cast<mlir::BoolAttr>(isThreadLocal).getValue()) {
            value = rt::ThreadLocal::getLocal(builder, loc)({value})[0];
            value = create<util::GenericMemrefCastOp>(loc, typeConverter->convertType(arg.getType()), value);
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
   template <class Fn>
   void atStartOf(mlir::Block* block, const Fn& fn) {
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
   template <typename OpTy, typename... Args>
   void createOrFold(llvm::SmallVector<Value>& results, Location location, Args&&... args) {
      OpTy res = builder.create<OpTy>(location, std::forward<Args>(args)...);
      if (succeeded(builder.tryFold(res, results))) {
         // If folding was successful, we don't need to rewrite the operation.
         res->erase();
         return;
      }
      ResultRange opResults = res->getResults();
      results.assign(opResults.begin(), opResults.end());
      if (res->getDialect()->getNamespace() == "subop" || mlir::isa<mlir::UnrealizedConversionCastOp>(res.getOperation())) {
         toRewrite.push_back(res.getOperation());
         //   rewrite(res.getOperation());
      }
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
      inFlightTupleStreams[newInFlight] = InFlightTupleStream{mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp()), mapping};
      return mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp());
   }
   void replaceTupleStream(mlir::Value tupleStream, ColumnMapping& mapping) {
      mlir::Value newInFlight = builder.create<subop::InFlightOp>(builder.getUnknownLoc(), mlir::ValueRange{}, mlir::ArrayAttr::get(getContext(), {}));
      eraseOp(newInFlight.getDefiningOp());
      inFlightTupleStreams[tupleStream] = InFlightTupleStream{mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp()), std::move(mapping)};
      if (auto* definingOp = tupleStream.getDefiningOp()) {
         eraseOp(definingOp);
      }
   }

   template <class AdaptorType>
   void inlineBlock(mlir::Block* block, mlir::ValueRange values, const std::function<void(AdaptorType)> processTerminator) {
      for (auto z : llvm::zip(block->getArguments(), values)) {
         std::get<0>(z).replaceAllUsesWith(std::get<1>(z));
      }
      llvm::SmallVector<mlir::Operation*> toInsert;
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
         llvm::SmallVector<mlir::Value> res;
         if (builder.tryFold(op, res).succeeded()) {
            op->replaceAllUsesWith(res);
            eraseOp(op);
         } else {
            builder.insert(op);
            registerOpInserted(op);
         }
      }
      llvm::SmallVector<mlir::Value> adaptorVals;
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

   void rewrite(mlir::Operation* op, mlir::Operation* before = nullptr) {
      if (isErased.contains(op)) return;
      if (before) {
         builder.setInsertionPoint(before);
      } else {
         builder.setInsertionPoint(op);
      }
      for (auto& p : patternList.patterns[op->getName()]) { //todo: ordering
         if (p->matchAndRewrite(op, *this).succeeded()) {
            llvm::SmallVector<mlir::Operation*> localRewrite = std::move(toRewrite);
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
   template <class Fn>
   mlir::LogicalResult implementStreamConsumer(mlir::Value stream, const Fn& impl) {
      auto& streamInfo = inFlightTupleStreams[stream];
      ColumnMapping mapping;
      if (stream.hasOneUse()) {
         mapping = std::move(streamInfo.columnMapping);
      } else {
         mapping = streamInfo.columnMapping;
      }

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
      : AbstractSubOpConversionPattern(&typeConverter, OperationName(OpT::getOperationName(), context), benefit,
                                       context) {}
   LogicalResult matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) override {
      llvm::SmallVector<mlir::Value> newOperands;
      for (auto operand : op->getOperands()) {
         newOperands.push_back(rewriter.getMapped(operand));
      }
      OpAdaptor adaptor(newOperands);
      return matchAndRewrite(mlir::cast<OpT>(op), adaptor, rewriter);
   }
   virtual LogicalResult matchAndRewrite(OpT op, OpAdaptor adaptor, SubOpRewriter& rewriter) const = 0;
   virtual ~SubOpConversionPattern() {};
};

template <class OpT, size_t B = 1>
class SubOpTupleStreamConsumerConversionPattern : public AbstractSubOpConversionPattern {
   public:
   using OpAdaptor = typename OpT::Adaptor;
   SubOpTupleStreamConsumerConversionPattern(TypeConverter& typeConverter, MLIRContext* context,
                                             PatternBenefit benefit = B)
      : AbstractSubOpConversionPattern(&typeConverter, OperationName(OpT::getOperationName(), context), benefit,
                                       context) {}
   LogicalResult matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) override {
      auto castedOp = mlir::cast<OpT>(op);

      // First check if the pattern matches before doing expensive work
      if (failed(match(castedOp))) {
         return failure();
      }

      auto stream = castedOp.getStream();
      return rewriter.implementStreamConsumer(stream, [&](SubOpRewriter& rewriter, ColumnMapping& mapping) {
         llvm::SmallVector<mlir::Value> newOperands;
         for (auto operand : op->getOperands()) {
            newOperands.push_back(rewriter.getMapped(operand));
         }
         OpAdaptor adaptor(newOperands);
         rewrite(castedOp, adaptor, rewriter, mapping);
         return success();
      });
   }
   virtual LogicalResult match(OpT op) const = 0;
   virtual void rewrite(OpT op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const = 0;
   virtual ~SubOpTupleStreamConsumerConversionPattern() {};
};

static mlir::TupleType getHtKVType(subop::HashMapType t, mlir::TypeConverter& converter) {
   auto keyTupleType = EntryStorageHelper(nullptr, t.getKeyMembers(), false, &converter).getStorageType();
   auto valTupleType = EntryStorageHelper(nullptr, t.getValueMembers(), t.getWithLock(), &converter).getStorageType();
   return mlir::cast<mlir::TupleType>(converter.convertType(mlir::TupleType::get(t.getContext(), {keyTupleType, valTupleType})));
}
static mlir::TupleType getHtKVType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter) {
   auto keyTupleType = EntryStorageHelper(nullptr, t.getKeyMembers(), false, &converter).getStorageType();
   auto valTupleType = EntryStorageHelper(nullptr, t.getValueMembers(), t.getWithLock(), &converter).getStorageType();
   return mlir::cast<mlir::TupleType>(converter.convertType(mlir::TupleType::get(t.getContext(), {keyTupleType, valTupleType})));
}
static mlir::TupleType getHtKVType(subop::PreAggrHtType t, mlir::TypeConverter& converter) {
   auto keyTupleType = EntryStorageHelper(nullptr, t.getKeyMembers(), false, &converter).getStorageType();
   auto valTupleType = EntryStorageHelper(nullptr, t.getValueMembers(), t.getWithLock(), &converter).getStorageType();
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
   auto keyTupleType = EntryStorageHelper(nullptr, t.getKeyMembers(), false, &converter).getStorageType();
   auto i8PtrType = util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));
   return mlir::TupleType::get(t.getContext(), {i8PtrType, mlir::IndexType::get(t.getContext()), i8PtrType, keyTupleType});
}
static mlir::TupleType getHashMultiMapValueType(subop::HashMultiMapType t, mlir::TypeConverter& converter) {
   auto valTupleType = EntryStorageHelper(nullptr, t.getValueMembers(), false, &converter).getStorageType();
   auto i8PtrType = util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));
   return mlir::TupleType::get(t.getContext(), {i8PtrType, valTupleType});
}

static TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   llvm::SmallVector<Type> types;
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

   LogicalResult match(subop::GatherOp gatherOp) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::TableEntryRefType>(refType)) { return failure(); }
      return success();
   }

   void rewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      auto columns = mlir::cast<subop::TableEntryRefType>(refType).getMembers();
      auto tableRefVal = mapping.resolve(gatherOp, gatherOp.getRef());
      llvm::SmallVector<mlir::Value> unpacked;
      rewriter.createOrFold<util::UnPackOp>(unpacked, gatherOp->getLoc(), tableRefVal);
      auto currRow = unpacked[0];
      llvm::SmallVector<mlir::Value> unPackedColumns;
      rewriter.createOrFold<util::UnPackOp>(unPackedColumns, gatherOp->getLoc(), unpacked[1]);
      for (size_t i = 0; i < columns.getMembers().size(); i++) {
         auto c = columns.getMembers()[i];
         if (gatherOp.getMapping().hasMember(c)) {
            auto columnDefAttr = gatherOp.getMapping().getColumnDef(c);
            auto colArray = unPackedColumns[i];
            auto type = columnDefAttr.getColumn().type;
            //todo: use MLIR interfaces to get the "right" operation for loading a certain type from an arrow array?
            mlir::Value loaded = rewriter.create<db::LoadArrowOp>(gatherOp->getLoc(), type, colArray, currRow);
            mapping.define(columnDefAttr, loaded);
         }
      }
      rewriter.replaceTupleStream(gatherOp, mapping);
   }
};
class MaterializeTableLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult match(subop::MaterializeOp materializeOp) const override {
      if (!mlir::isa<subop::ResultTableType>(materializeOp.getState().getType())) return failure();
      return success();
   }

   void rewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto stateType = mlir::cast<subop::ResultTableType>(materializeOp.getState().getType());
      mlir::Value loaded = rewriter.create<util::LoadOp>(materializeOp->getLoc(), adaptor.getState());
      auto columnBuilders = rewriter.create<util::UnPackOp>(materializeOp->getLoc(), loaded);
      for (size_t i = 0; i < stateType.getMembers().getMembers().size(); i++) {
         auto attribute = materializeOp.getMapping().getColumnRef(stateType.getMembers().getMembers()[i]);
         auto val = mapping.resolve(materializeOp, attribute);
         auto asArrayBuilder = rewriter.create<arrow::BuilderFromPtr>(materializeOp->getLoc(), columnBuilders.getResult(i));
         rewriter.create<db::AppendArrowOp>(materializeOp->getLoc(), asArrayBuilder, val);
      }
      rewriter.eraseOp(materializeOp);
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
      auto i8PtrType = rewriter.getPtrType();
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "thread_local_init" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({i8PtrType}), TypeRange(i8PtrType)));
      });
      auto* funcBody = new Block;
      mlir::Value funcArg = funcBody->addArgument(i8PtrType, loc);
      funcOp.getBody().push_back(funcBody);
      mlir::Block* newBlock = new Block;
      mlir::OpBuilder builder(rewriter.getContext());
      llvm::SmallVector<mlir::Operation*> toInsert;
      builder.setInsertionPointToStart(newBlock);
      mlir::Value argRef = newBlock->addArgument(util::RefType::get(rewriter.getI8Type()), createThreadLocal.getLoc());
      argRef = builder.create<util::GenericMemrefCastOp>(createThreadLocal->getLoc(), util::RefType::get(rewriter.getContext(), i8PtrType), argRef);
      for (auto& op : createThreadLocal.getInitFn().front().getOperations()) {
         toInsert.push_back(&op);
      }
      for (auto* op : toInsert) {
         op->remove();
         builder.insert(op);
      }
      llvm::SmallVector<mlir::Value> toStore;
      llvm::SmallVector<mlir::Operation*> toDelete;
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
            mlir::Value casted = rewriter.create<util::GenericMemrefCastOp>(createThreadLocal->getLoc(), i8PtrType, unrealized);
            rewriter.create<mlir::func::ReturnOp>(loc, casted);
         });
         delete newBlock;
      });
      auto ptrSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), i8PtrType);
      auto numPtrs = rewriter.create<mlir::arith::ConstantIndexOp>(loc, toStore.size());
      auto bytes = rewriter.create<mlir::arith::MulIOp>(loc, ptrSize, numPtrs);

      Value arg = rt::ExecutionContext::allocStateRaw(rewriter, loc)({bytes})[0];
      arg = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(rewriter.getContext(), i8PtrType), arg);
      for (size_t i = 0; i < toStore.size(); i++) {
         Value storeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), toStore[i].getType());
         Value valPtrOrig = rt::ExecutionContext::allocStateRaw(rewriter, loc)({storeSize})[0];
         Value valPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(toStore[i].getType()), valPtr);
         rewriter.create<util::StoreOp>(loc, toStore[i], valPtr, mlir::Value());
         rewriter.create<util::StoreOp>(loc, valPtrOrig, arg, rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
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
      auto tableType = mlir::dyn_cast_or_null<subop::TableType>(scanOp.getState().getType());
      if (!tableType) return failure();
      auto& memberManager = getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
      auto loc = scanOp->getLoc();
      auto refType = mlir::cast<subop::TableEntryRefType>(scanOp.getRef().getColumn().type);
      std::string memberMapping = "[";
      llvm::SmallVector<mlir::Type> accessedColumnTypes;
      auto members = refType.getMembers();
      for (auto m : members.getMembers()) {
         auto type = memberManager.getType(m);
         auto name = memberManager.getName(m);
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

      auto* ctxt = rewriter.getContext();
      auto i16T = mlir::IntegerType::get(rewriter.getContext(), 16);
      auto recordBatchInfoRepr = mlir::TupleType::get(ctxt, {rewriter.getIndexType(), rewriter.getIndexType(), util::RefType::get(i16T), util::RefType::get(arrow::ArrayType::get(ctxt))});
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
      auto ptr = rewriter.storeStepRequirements(scanOp.getState());
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         rewriter.loadStepRequirements(contextPtr, typeConverter, scanOp.getState());
         recordBatchPointer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), recordBatchInfoRepr), recordBatchPointer);
         mlir::Value end = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), recordBatchPointer, 0);
         mlir::Value globalOffset = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), recordBatchPointer, 1);
         mlir::Value selVecPtr;
         if (tableType.getFiltered()) {
            selVecPtr = rewriter.create<util::LoadElementOp>(loc, util::RefType::get(i16T), recordBatchPointer, 2);
         }
         mlir::Value ptrToColumns = rewriter.create<util::LoadElementOp>(loc, util::RefType::get(arrow::ArrayType::get(ctxt)), recordBatchPointer, 3);
         llvm::SmallVector<mlir::Value> arrays;
         for (size_t i = 0; i < accessedColumnTypes.size(); i++) {
            auto ci = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
            auto array = rewriter.create<util::LoadOp>(loc, ptrToColumns, ci);
            arrays.push_back(array);
         }
         auto arraysVal = rewriter.create<util::PackOp>(loc, arrays);
         auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp2 = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
         rewriter.atStartOf(forOp2.getBody(), [&](SubOpRewriter& rewriter) {
            mlir::Value index = forOp2.getInductionVar();
            if (tableType.getFiltered()) {
               auto idx = rewriter.create<util::LoadOp>(loc, selVecPtr, index);
               index = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), idx);
            }
            auto withOffset = rewriter.create<mlir::arith::AddIOp>(loc, index, globalOffset);
            auto currentRecord = rewriter.create<util::PackOp>(loc, mlir::ValueRange{withOffset, arraysVal});
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
            llvm::SmallVector<mlir::Value> results;
            for (size_t i = 0; i < leftBuilders.getNumResults(); i++) {
               rt::ArrowColumnBuilder::merge(rewriter, loc)({leftBuilders.getResults()[i], rightBuilders.getResults()[i]});
               results.push_back(leftBuilders.getResults()[i]);
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
      auto loc = createOp->getLoc();
      mlir::Value table = rt::ArrowTable::createEmpty(rewriter, loc)({})[0];
      for (auto i = 0ul; i < columnBuilders.getNumResults(); i++) {
         auto columnBuilder = columnBuilders.getResult(i);
         auto column = rt::ArrowColumnBuilder::finish(rewriter, loc)({columnBuilder})[0];
         mlir::Value columnName = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(getContext()), mlir::cast<mlir::StringAttr>(createOp.getColumns()[i]));
         table = rt::ArrowTable::addColumn(rewriter, loc)({table, columnName, column})[0];
      }
      rewriter.replaceOp(createOp, table);
      return success();
   }
};
class CreateTableLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   std::string arrowDescrFromType(mlir::Type type) const {
      if (type.isIndex()) {
         return "int[64]";
      } else if (isIntegerType(type, 1)) {
         return "bool";
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         return "int[" + std::to_string(intWidth) + "]";
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         return "uint[" + std::to_string(uIntWidth) + "]";
      } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
         return "float[" + std::to_string(floatType.getWidth()) + "]";
      } else if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
         auto prec = std::min(decimalType.getP(), 38);
         return "decimal[" + std::to_string(prec) + "," + std::to_string(decimalType.getS()) + "]";
      } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
         return dateType.getUnit() == db::DateUnitAttr::day ? "date[32]" : "date[64]";
      } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(type)) {
         return "timestamp[" + std::to_string(static_cast<uint32_t>(timestampType.getUnit())) + "]";
      } else if (mlir::isa<db::StringType>(type)) {
         return "string";
      } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(type)) {
         if (charType.getLen() <= 1) {
            return "fixed_sized[4]";
         } else {
            return "string";
         }
      }
      assert(false);
      return "";
   }

   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ResultTableType>(createOp.getType())) return failure();
      auto& memberManager = getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
      auto tableType = mlir::cast<subop::ResultTableType>(createOp.getType());
      std::string descr;
      llvm::SmallVector<mlir::Value> columnBuilders;
      auto loc = createOp->getLoc();
      auto members = tableType.getMembers().getMembers();
      for (auto m : members) {
         auto type = memberManager.getType(m);
         auto baseType = getBaseType(type);
         mlir::Value typeDescr = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(getContext()), arrowDescrFromType(baseType));
         Value columnBuilder = rt::ArrowColumnBuilder::create(rewriter, loc)({typeDescr})[0];
         columnBuilders.push_back(columnBuilder);
      }
      mlir::Value tpl = rewriter.create<util::PackOp>(createOp->getLoc(), columnBuilders);
      auto tplSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), tpl.getType());
      mlir::Value ref = rt::ExecutionContext::allocStateRaw(rewriter, loc)({tplSize})[0];
      ref = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(tpl.getType()), ref);
      rewriter.create<util::StoreOp>(createOp->getLoc(), tpl, ref, mlir::Value());
      rewriter.replaceOp(createOp, ref);
      return mlir::success();
   }
};

class GetExternalTableLowering : public SubOpConversionPattern<subop::GetExternalOp> {
   public:
   using SubOpConversionPattern<subop::GetExternalOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GetExternalOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::TableType>(op.getType())) return failure();
      mlir::Value description = rewriter.create<util::CreateConstVarLen>(op->getLoc(), util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());
      rewriter.replaceOp(op, rt::DataSource::get(rewriter, op->getLoc())({description})[0]);
      return mlir::success();
   }
};
class GenerateLowering : public SubOpConversionPattern<subop::GenerateOp> {
   public:
   using SubOpConversionPattern<subop::GenerateOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GenerateOp generateOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      ColumnMapping mapping;
      llvm::SmallVector<subop::GenerateEmitOp> emitOps;
      generateOp.getRegion().walk([&](subop::GenerateEmitOp emitOp) {
         emitOps.push_back(emitOp);
      });
      llvm::SmallVector<mlir::Value> streams;
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

   LogicalResult match(subop::MapOp mapOp) const override {
      return success();
   }

   void rewrite(subop::MapOp mapOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto args = mapping.resolve(mapOp, mapOp.getInputCols());
      llvm::SmallVector<Value> res;

      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&mapOp.getFn().front(), args, [&](tuples::ReturnOpAdaptor adaptor) {
         res.insert(res.end(), adaptor.getResults().begin(), adaptor.getResults().end());
      });
      for (auto& r : res) {
         r = rewriter.getMapped(r);
      }
      mapping.define(mapOp.getComputedCols(), res);

      rewriter.replaceTupleStream(mapOp, mapping);
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
class CreateThreadLocalBufferLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto threadLocalType = mlir::dyn_cast_or_null<subop::ThreadLocalType>(createOp.getType());
      if (!threadLocalType) return failure();
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(threadLocalType.getWrapped());
      if (!bufferType) return failure();
      auto loc = createOp->getLoc();
      EntryStorageHelper storageHelper(createOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter);
      auto elementType = storageHelper.getStorageType();
      auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      mlir::Value vector = rt::GrowingBuffer::createThreadLocal(rewriter, loc)({typeSize})[0];
      rewriter.replaceOp(createOp, vector);
      return mlir::success();
   }
};

class CreateBufferLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(createOp.getType());
      if (!bufferType) return failure();
      bool usedForGPU{false};
      for (mlir::OpResult result : createOp->getResults()) {
         for (mlir::OpOperand& use : result.getUses()) {
            mlir::Operation* userOp = use.getOwner();
            if (mlir::dyn_cast_or_null<subop::StateContextSwitchOp>(userOp))
               usedForGPU = true;
         }
      }
      auto loc = createOp->getLoc();
      EntryStorageHelper storageHelper(createOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter);
      auto elementType = storageHelper.getStorageType();
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, createOp->hasAttr("initial_capacity") ? mlir::cast<mlir::IntegerAttr>(createOp->getAttr("initial_capacity")).getInt() : 1024);
      auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      mlir::Value vector;
      if (usedForGPU) {
         // If we are on CPU -> properly construct the class by setting *plain class members*, no heap allocations within class.
         mlir::Value executionContext;
         rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getBody().front(), [&](SubOpRewriter& rewriter) {
            auto classSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::GrowingBuffer));
            vector = rewriter.create<util::AllocOp>(loc, rewriter.getI8PtrType(), classSize);
         });
         rt::DeviceMemoryFuncs::initializeGrowingBufferOnCPU(rewriter, loc)({vector, initialCapacity, typeSize});
      } else {
         mlir::Value executionContext;
         mlir::Value allocator;
         rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getBody().front(), [&](SubOpRewriter& rewriter) {
            if (createOp->hasAttrOfType<mlir::IntegerAttr>("group")) {
               Value groupId = rewriter.create<arith::ConstantIndexOp>(loc, mlir::cast<mlir::IntegerAttr>(createOp->getAttr("group")).getInt());
               allocator = rt::GrowingBufferAllocator::getGroupAllocator(rewriter, loc)({groupId})[0];
            } else {
               allocator = rt::GrowingBufferAllocator::getDefaultAllocator(rewriter, loc)({})[0];
            }
         });
         vector = rt::GrowingBuffer::create(rewriter, loc)({allocator, typeSize, initialCapacity})[0];
      }
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
      auto elementType = EntryStorageHelper(scanOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter).getStorageType();

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
      auto ptr = rt::Hashtable::create(rewriter, createOp->getLoc())({typeSize, initialCapacity})[0];
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
      auto ptr = rt::HashMultiMap::create(rewriter, createOp->getLoc())({entryTypeSize, valueTypeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, ptr);
      return mlir::success();
   }
};
class CreateOpenHtFragmentLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::PreAggrHtFragmentType>(createOp.getType())) return failure();
      bool usedForGPU{false};
      for (mlir::OpResult result : createOp->getResults()) {
         for (mlir::OpOperand& use : result.getUses()) {
            mlir::Operation* userOp = use.getOwner();
            if (mlir::dyn_cast_or_null<subop::StateContextSwitchOp>(userOp))
               usedForGPU = true;
         }
      }
      auto t = mlir::cast<subop::PreAggrHtFragmentType>(createOp.getType());
      auto typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHtEntryType(t, *typeConverter));
      mlir::Value ptr;
      assert(!usedForGPU || !t.getWithLock());
      auto withLocks = rewriter.create<mlir::arith::ConstantIntOp>(createOp->getLoc(), t.getWithLock(), rewriter.getI1Type());
      if (usedForGPU) {
         // If we are on CPU -> properly construct the class by setting *plain class members*, no heap allocations within class.
         mlir::Value classSize = rewriter.create<mlir::arith::ConstantIndexOp>(createOp->getLoc(), sizeof(cudaRT::PreAggregationHashtableFragment));
         ptr = rewriter.create<util::AllocOp>(createOp->getLoc(), rewriter.getI8PtrType(), classSize);
         rt::DeviceMemoryFuncs::initializePreAggrFragmentOnCPU(rewriter, createOp->getLoc())({ptr, typeSize});
      } else {
         ptr = rt::PreAggregationHashtableFragment::create(rewriter, createOp->getLoc())({typeSize, withLocks})[0];
      }
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
      EntryStorageHelper storageHelper(createOp, arrayType.getMembers(), arrayType.hasLock(), typeConverter);

      Value tpl = rewriter.create<util::LoadOp>(loc, adaptor.getNumElements());
      Value numElements = rewriter.create<util::UnPackOp>(loc, tpl).getResults()[0];
      auto elementType = storageHelper.getStorageType();
      auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      auto numBytes = rewriter.create<mlir::arith::MulIOp>(loc, typeSize, numElements);
      mlir::Value vector = rt::Buffer::createZeroed(rewriter, loc)({numBytes})[0];
      rewriter.replaceOpWithNewOp<util::BufferCastOp>(createOp, typeConverter->convertType(createOp.getType()), vector);
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
      EntryStorageHelper sourceStorageHelper(createOp, continuousType.getMembers(), continuousType.hasLock(), typeConverter);
      EntryStorageHelper viewStorageHelper(createOp, createOp.getType().getValueMembers(), continuousType.hasLock(), typeConverter);
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
            auto sourceValues = sourceStorageHelper.getValueMap(src, rewriter, loc);
            llvm::SmallVector<mlir::Value> args;
            for (auto relevantMember : createOp.getRelevantMembers()) {
               args.push_back(sourceValues.get(mlir::cast<subop::MemberAttr>(relevantMember).getMember()));
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
            auto leftValues = viewStorageHelper.getValueMap(left, rewriter, loc);
            auto rightValues = viewStorageHelper.getValueMap(right, rewriter, loc);
            llvm::SmallVector<mlir::Value> args;
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
      mlir::Value res = rt::SegmentTreeView::build(rewriter, loc)({adaptor.getSource(), sourceEntryTypeSize, initialFnPtr, combineFnPtr, stateTypeSize})[0];
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
      EntryStorageHelper storageHelper(heapOp, heapType.getMembers(), heapType.hasLock(), typeConverter);
      ModuleOp parentModule = heapOp->getParentOfType<ModuleOp>();
      mlir::TupleType elementType = storageHelper.getStorageType();
      auto loc = heapOp.getLoc();
      mlir::func::FuncOp funcOp;
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "heap_compare" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
      });
      auto* funcBody = new Block;
      Value left = funcBody->addArgument(ptrType, loc);
      Value right = funcBody->addArgument(ptrType, loc);
      funcOp.getBody().push_back(funcBody);
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto leftVals = storageHelper.getValueMap(left, rewriter, loc, heapOp.getSortBy());
         auto rightVals = storageHelper.getValueMap(right, rewriter, loc, heapOp.getSortBy());
         llvm::SmallVector<mlir::Value> args;
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
      auto heap = rt::Heap::create(rewriter, loc)({maxElements, typeSize, functionPointer})[0];
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
      auto simpleStateType = mlir::cast<subop::SimpleStateType>(mergeOp.getType());
      EntryStorageHelper storageHelper(mergeOp, simpleStateType.getMembers(), simpleStateType.hasLock(), typeConverter);

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
            auto leftValues = storageHelper.getValueMap(left, rewriter, loc);
            auto rightValues = storageHelper.getValueMap(right, rewriter, loc);
            llvm::SmallVector<mlir::Value> args;
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
      EntryStorageHelper keyStorageHelper(mergeOp, hashMapType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(mergeOp, hashMapType.getValueMembers(), hashMapType.hasLock(), typeConverter);

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
               auto leftValues = valStorageHelper.getValueMap(left, rewriter, loc);
               auto rightValues = valStorageHelper.getValueMap(right, rewriter, loc);
               llvm::SmallVector<mlir::Value> args;
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
            auto leftKeys = keyStorageHelper.getValueMap(left, rewriter, loc);
            auto rightKeys = keyStorageHelper.getValueMap(right, rewriter, loc);
            llvm::SmallVector<mlir::Value> args;
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
      EntryStorageHelper keyStorageHelper(mergeOp, hashMapType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(mergeOp, hashMapType.getValueMembers(), hashMapType.hasLock(), typeConverter);

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
               auto leftValues = valStorageHelper.getValueMap(left, rewriter, loc);
               auto rightValues = valStorageHelper.getValueMap(right, rewriter, loc);
               llvm::SmallVector<mlir::Value> args;
               args.insert(args.end(), leftValues.begin(), leftValues.end());
               args.insert(args.end(), rightValues.begin(), rightValues.end());
               assert(!mergeOp.getCombineFn().empty());
               assert(mergeOp.getCombineFn().front().getNumArguments() == args.size());
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
            auto leftKeys = keyStorageHelper.getValueMap(left, rewriter, loc);
            auto rightKeys = keyStorageHelper.getValueMap(right, rewriter, loc);
            llvm::SmallVector<mlir::Value> args;
            args.insert(args.end(), leftKeys.begin(), leftKeys.end());
            args.insert(args.end(), rightKeys.begin(), rightKeys.end());
            auto res = inlineBlock(&mergeOp.getEqFn().front(), rewriter, args)[0];
            rewriter.create<mlir::func::ReturnOp>(loc, res);
         });
      }
      Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
      Value eqFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, eqFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(eqFn.getSymName())));
      mlir::Value merged = rt::PreAggregationHashtable::merge(rewriter, mergeOp->getLoc())(mlir::ValueRange{adaptor.getThreadLocal(), eqFnPtr, combineFnPtr})[0];
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
      auto ptrType = rewriter.getPtrType();
      EntryStorageHelper storageHelper(sortOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter);
      ModuleOp parentModule = sortOp->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;

      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "sort_compare" + std::to_string(id++), mlir::FunctionType::get(getContext(), TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
      });
      auto* funcBody = new Block;
      funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
      funcOp.getBody().push_back(funcBody);
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         auto leftVals = storageHelper.getValueMap(funcBody->getArgument(0), rewriter, sortOp->getLoc(), sortOp.getSortBy());
         auto rightVals = storageHelper.getValueMap(funcBody->getArgument(1), rewriter, sortOp->getLoc(), sortOp.getSortBy());
         llvm::SmallVector<mlir::Value> args;
         args.insert(args.end(), leftVals.begin(), leftVals.end());
         args.insert(args.end(), rightVals.begin(), rightVals.end());
         Block* sortLambda = &sortOp.getRegion().front();
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
            rewriter.create<mlir::func::ReturnOp>(sortOp->getLoc(), adaptor.getResults());
         });
      });

      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(sortOp->getLoc(), funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      auto genericBuffer = rt::GrowingBuffer::sort(rewriter, sortOp->getLoc())({adaptor.getToSort(), functionPointer})[0];
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
      auto elementType = util::RefType::get(getContext(), EntryStorageHelper(scanOp, sortedViewType.getMembers(), sortedViewType.hasLock(), typeConverter).getStorageType());
      auto loc = scanOp->getLoc();
      auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end = rewriter.create<util::BufferGetLen>(loc, rewriter.getIndexType(), adaptor.getState());
      auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
      rewriter.atStartOf(forOp.getBody(), [&](SubOpRewriter& rewriter) {
         auto currElementPtr = rewriter.create<util::BufferGetElementRef>(loc, elementType, adaptor.getState(), forOp.getInductionVar());
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
      EntryStorageHelper storageHelper(scanOp, heapType.getMembers(), heapType.hasLock(), typeConverter);
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
      Value it;
      if (mlir::isa_and_nonnull<subop::StateContextSwitchOp>(scanRefsOp.getState().getDefiningOp())) {
         Value size = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), getHtEntryType(hashMapType, *typeConverter));
         size = rewriter.create<mlir::arith::IndexCastUIOp>(loc, rewriter.getI32Type(), size);
         it = rt::DeviceMemoryFuncs::createHTiteratorOnCPU(rewriter, loc)({adaptor.getState(), size})[0];
      } else {
         it = rt::PreAggregationHashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      }
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
      EntryStorageHelper valStorageHelper(scanRefsOp, hashMultiMapType.getValueMembers(), false, typeConverter);
      EntryStorageHelper keyStorageHelper(scanRefsOp, hashMultiMapType.getKeyMembers(), hashMultiMapType.hasLock(), typeConverter);
      auto i8PtrType = rewriter.getPtrType();
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
            mlir::Value next = rewriter.create<util::LoadElementOp>(loc, i8PtrType, castedPtr, 0);
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
      auto valPtrType = util::RefType::get(getContext(), mlir::TupleType::get(getContext(), unpackTypes(hashmapType.getValueMembers())));
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
      auto valPtrType = util::RefType::get(getContext(), mlir::TupleType::get(getContext(), unpackTypes(hashmapType.getValueMembers())));
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
      llvm::SmallVector<mlir::Value> unpacked;
      rewriter.createOrFold<util::UnPackOp>(unpacked, loc, adaptor.getList());
      auto ptr = unpacked[0];
      auto hash = unpacked[1];
      auto initialValid = unpacked[2];
      auto iteratorType = ptr.getType();
      auto referenceType = mlir::cast<subop::ListType>(scanOp.getList().getType()).getT();
      rewriter.create<mlir::scf::IfOp>(
         loc, initialValid, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, iteratorType, ptr);
            Block* before = new Block;
            Block* after = new Block;
            whileOp.getBefore().push_back(before);
            whileOp.getAfter().push_back(after);
            rewriter.create<scf::YieldOp>(loc);
            mlir::Value beforePtr = before->addArgument(iteratorType, loc);
            mlir::Value afterPtr = after->addArgument(iteratorType, loc);
            rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
               auto tupleType = mlir::TupleType::get(getContext(), unpackTypes(referenceType.getMembers()));
               auto i8PtrType = rewriter.getPtrType();
               Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, rewriter.getIndexType(), tupleType})), beforePtr);
               Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), tupleType), castedPtr, 2);
               if (hashIndexedViewType.getCompareHashForLookup()) {
                  mlir::Value currHash = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), castedPtr, 1);
                  mlir::Value hashEq = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, currHash, hash);
                  rewriter.create<mlir::scf::IfOp>(
                     loc, hashEq, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
                        mapping.define(scanOp.getElem(), valuePtr);
                        rewriter.replaceTupleStream(scanOp, mapping);
                        builder1.create<mlir::scf::YieldOp>(loc);
                     });
               } else {
                  mapping.define(scanOp.getElem(), valuePtr);
                  rewriter.replaceTupleStream(scanOp, mapping);
               }
               mlir::Value next = rewriter.create<util::LoadElementOp>(loc, i8PtrType, castedPtr, 0);
               mlir::Value cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), next);
               rewriter.create<mlir::scf::ConditionOp>(loc, cond, next);
            });
            rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
               rewriter.create<mlir::scf::YieldOp>(loc, afterPtr);
            });
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
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(externalHashIndexType.getMembers()));
      mlir::TypeRange typeRange{tupleType.getTypes()};
      auto i16T = mlir::IntegerType::get(rewriter.getContext(), 16);
      auto recordBatchInfoRepr = mlir::TupleType::get(ctxt, {rewriter.getIndexType(), rewriter.getIndexType(), util::RefType::get(i16T), util::RefType::get(arrow::ArrayType::get(ctxt))});

      // Create while loop to extract all chained values from hash table
      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{}, mlir::ValueRange{});
      Block* conditionBlock = new Block;
      Block* bodyBlock = new Block;
      whileOp.getBefore().push_back(conditionBlock);
      whileOp.getAfter().push_back(bodyBlock);

      ColumnMapping mapping;

      // Check if iterator contains another value
      rewriter.atStartOf(conditionBlock, [&](SubOpRewriter& rewriter) {
         mlir::Value cont = rt::HashIndexIteration::hasNext(rewriter, loc)({adaptor.getList()})[0];
         rewriter.create<scf::ConditionOp>(loc, cont, mlir::ValueRange{});
      });

      // Load record batch from iterator
      rewriter.atStartOf(bodyBlock, [&](SubOpRewriter& rewriter) {
         mlir::Value list = adaptor.getList();
         mlir::Value recordBatchPointer;
         rewriter.atStartOf(&scanOp->getParentOfType<mlir::func::FuncOp>().getBody().front(), [&](SubOpRewriter& rewriter) {
            recordBatchPointer = rewriter.create<util::AllocaOp>(loc, util::RefType::get(rewriter.getContext(), recordBatchInfoRepr), mlir::Value());
         });
         rt::HashIndexIteration::consumeRecordBatch(rewriter, loc)({list, recordBatchPointer});
         mlir::Value ptrToColumns = rewriter.create<util::LoadElementOp>(loc, util::RefType::get(arrow::ArrayType::get(ctxt)), recordBatchPointer, 3);
         llvm::SmallVector<mlir::Value> arrays;
         for (size_t i = 0; i < externalHashIndexType.getMembers().getMembers().size(); i++) {
            auto ci = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
            auto array = rewriter.create<util::LoadOp>(loc, ptrToColumns, ci);
            arrays.push_back(array);
         }
         auto arraysVal = rewriter.create<util::PackOp>(loc, arrays);
         auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         auto globalOffset = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), recordBatchPointer, 1);
         auto end = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), recordBatchPointer, 0);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp2 = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
         rewriter.atStartOf(forOp2.getBody(), [&](SubOpRewriter& rewriter) {
            auto withOffset = rewriter.create<mlir::arith::AddIOp>(loc, forOp2.getInductionVar(), globalOffset);
            auto currentRecord = rewriter.create<util::PackOp>(loc, mlir::ValueRange{withOffset, arraysVal});
            mapping.define(scanOp.getElem(), currentRecord);
            rewriter.replaceTupleStream(scanOp, mapping);
         });
         rewriter.create<mlir::scf::YieldOp>(loc);
      });

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
      EntryStorageHelper keyStorageHelper(scanOp, hashMultiMapType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(scanOp, hashMultiMapType.getValueMembers(), hashMultiMapType.hasLock(), typeConverter);
      auto i8PtrType = rewriter.getPtrType();
      Value ptrValid = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), ptr);
      mlir::Value valuePtr = rewriter.create<scf::IfOp>(
                                        loc, ptrValid, [&](OpBuilder& b, Location loc) {
                                           Value valuePtr = rewriter.create<util::LoadElementOp>(loc, i8PtrType, ptr, 2);
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
         mlir::Value next = rewriter.create<util::LoadElementOp>(loc, i8PtrType, castedPtr, 0);
         rewriter.create<mlir::scf::YieldOp>(loc, next);
      });
      return success();
   }
};

class FilterLowering : public SubOpTupleStreamConsumerConversionPattern<subop::FilterOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::FilterOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult match(subop::FilterOp filterOp) const override {
      return success();
   }

   void rewrite(subop::FilterOp filterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto providedVals = mapping.resolve(filterOp, filterOp.getConditions());
      mlir::Value cond;
      if (providedVals.size() == 1) {
         cond = providedVals[0];
      } else {
         cond = rewriter.create<db::AndOp>(filterOp.getLoc(), providedVals);
      }
      if (!cond.getType().isInteger(1)) {
         cond = rewriter.create<db::DeriveTruth>(filterOp.getLoc(), cond);
      }
      if (filterOp.getFilterSemantic() == subop::FilterSemantic::none_true) {
         cond = rewriter.create<db::NotOp>(filterOp->getLoc(), cond);
      }
      auto ifOp = rewriter.create<mlir::scf::IfOp>(filterOp->getLoc(), mlir::TypeRange{}, cond);
      ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, filterOp->getLoc());
      rewriter.atStartOf(ifOp.thenBlock(), [&](SubOpRewriter& rewriter) {
         rewriter.replaceTupleStream(filterOp, mapping);
      });
   }
};

class RenameLowering : public SubOpTupleStreamConsumerConversionPattern<subop::RenamingOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::RenamingOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult match(subop::RenamingOp renamingOp) const override {
      return success();
   }

   void rewrite(subop::RenamingOp renamingOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      for (mlir::Attribute attr : renamingOp.getColumns()) {
         auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         mlir::Attribute from = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[0];
         auto relationRefAttr = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(from);
         mapping.define(relationDefAttr, mapping.resolve(renamingOp, relationRefAttr));
      }
      rewriter.replaceTupleStream(renamingOp, mapping);
   }
};

class MaterializeHeapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult match(subop::MaterializeOp materializeOp) const override {
      auto heapType = mlir::dyn_cast_or_null<subop::HeapType>(materializeOp.getState().getType());
      if (!heapType) return failure();
      return success();
   }

   void rewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto heapType = mlir::cast<subop::HeapType>(materializeOp.getState().getType());
      EntryStorageHelper storageHelper(materializeOp, heapType.getMembers(), heapType.hasLock(), typeConverter);
      mlir::Value ref;
      rewriter.atStartOf(&rewriter.getCurrentStreamLoc()->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
         ref = rewriter.create<util::AllocaOp>(materializeOp->getLoc(), util::RefType::get(storageHelper.getStorageType()), mlir::Value());
      });
      storageHelper.storeFromColumns(materializeOp.getMapping(), mapping, ref, rewriter, materializeOp->getLoc());
      rt::Heap::insert(rewriter, materializeOp->getLoc())({adaptor.getState(), ref});
      rewriter.eraseOp(materializeOp);
   }
};
class MaterializeVectorLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult match(subop::MaterializeOp materializeOp) const override {
      auto kernelLocalStateType = mlir::dyn_cast_or_null<subop::KernelLocalType>(materializeOp.getState().getType());
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(materializeOp.getState().getType());
      if (kernelLocalStateType)
         bufferType = mlir::dyn_cast_or_null<subop::BufferType>(kernelLocalStateType.getWrapped());
      if (!bufferType) return failure();
      return success();
   }

   void rewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(materializeOp.getState().getType());
      if (auto kernelLocalStateType = mlir::dyn_cast_or_null<subop::KernelLocalType>(materializeOp.getState().getType()))
         bufferType = mlir::dyn_cast_or_null<subop::BufferType>(kernelLocalStateType.getWrapped());
      auto loc = materializeOp->getLoc();
      auto* ctxt = rewriter.getContext();
      mlir::Value ref;
      EntryStorageHelper storageHelper(materializeOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter);
      if (rewriter.getGPUModule()) {
         mlir::func::FuncOp insertDeviceFunc;
         rewriter.atStartOf(rewriter.getGPUModule().getBody(), [&](SubOpRewriter& rewriter) {
            insertDeviceFunc = rewriter.create<mlir::func::FuncOp>(loc, "GrowingBufferInsertWarp", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType()}, {rewriter.getI8PtrType()}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
         });
         ref = rewriter.create<mlir::func::CallOp>(loc, insertDeviceFunc, mlir::ValueRange{adaptor.getState()}).getResult(0);
      } else {
         ref = rt::GrowingBuffer::insert(rewriter, materializeOp->getLoc())({adaptor.getState()})[0];
      }
      storageHelper.storeFromColumns(materializeOp.getMapping(), mapping, ref, rewriter, loc);
      rewriter.eraseOp(materializeOp);
   }
};

class LookupSimpleStateLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult match(subop::LookupOp lookupOp) const override {
      auto stateType = lookupOp.getState().getType();
      if (mlir::isa<subop::KernelLocalType>(stateType)) {
         stateType = mlir::cast<subop::KernelLocalType>(stateType).getWrapped();
      }
      if (!mlir::isa<subop::SimpleStateType>(stateType)) return failure();
      return success();
   }

   void rewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
         mapping.define(lookupOp.getRef(), adaptor.getState());
         rewriter.replaceTupleStream(lookupOp, mapping);
      }
   };

   class LookupHashIndexedViewLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
      public:
      using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
      LogicalResult match(subop::LookupOp lookupOp) const override {
         if (!mlir::isa<subop::HashIndexedViewType>(lookupOp.getState().getType())) return failure();
         return success();
      }

      void rewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
         auto loc = lookupOp->getLoc();
         mlir::Value hash = mapping.resolve(lookupOp, lookupOp.getKeys())[0];
         auto* context = getContext();
         auto indexType = rewriter.getIndexType();
         auto htType = util::RefType::get(context, rewriter.getPtrType());

         Value castedPointer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, TupleType::get(context, {htType, indexType})), adaptor.getState());
         Value ht = rewriter.create<util::LoadElementOp>(loc, htType, castedPointer, 0);
         Value htMask = rewriter.create<util::LoadElementOp>(loc, indexType, castedPointer, 1);
         Value buckedPos = rewriter.create<arith::AndIOp>(loc, htMask, hash);
         Value ptr = rewriter.create<util::LoadOp>(loc, rewriter.getPtrType(), ht, buckedPos);
         //optimization
         Value refValid = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hash);
         ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
         Value matches = rewriter.create<util::PackOp>(loc, ValueRange{ptr, hash, refValid});

         mapping.define(lookupOp.getRef(), matches);
         rewriter.replaceTupleStream(lookupOp, mapping);
      }
   };
   class LookupSegmentTreeViewLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
      public:
      using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
      LogicalResult match(subop::LookupOp lookupOp) const override {
         if (!mlir::isa<subop::SegmentTreeViewType>(lookupOp.getState().getType())) return failure();
         return success();
      }

      void rewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
         auto valueMembers = mlir::cast<subop::SegmentTreeViewType>(lookupOp.getState().getType()).getValueMembers();
         mlir::TupleType stateType = mlir::TupleType::get(getContext(), unpackTypes(valueMembers));

         auto loc = lookupOp->getLoc();
         llvm::SmallVector<mlir::Value> unpackedLeft;
         llvm::SmallVector<mlir::Value> unpackedRight;

         rewriter.createOrFold<util::UnPackOp>(unpackedLeft, loc, mapping.resolve(lookupOp, lookupOp.getKeys())[0]);
         rewriter.createOrFold<util::UnPackOp>(unpackedRight, loc, mapping.resolve(lookupOp, lookupOp.getKeys())[1]);
         auto idxLeft = unpackedLeft[0];
         auto idxRight = unpackedRight[0];
         mlir::Value ref;
         rewriter.atStartOf(&rewriter.getCurrentStreamLoc()->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
            ref = rewriter.create<util::AllocaOp>(lookupOp->getLoc(), util::RefType::get(typeConverter->convertType(stateType)), mlir::Value());
         });
         rt::SegmentTreeView::lookup(rewriter, loc)({adaptor.getState(), ref, idxLeft, idxRight});
         mapping.define(lookupOp.getRef(), ref);
         rewriter.replaceTupleStream(lookupOp, mapping);
      }
   };
   mlir::Value hashKeys(llvm::SmallVector<mlir::Value> keys, OpBuilder& rewriter, Location loc) {
      return rewriter.create<db::Hash>(loc, keys);
   }
   class PureLookupHashMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
      public:
      using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
      LogicalResult match(subop::LookupOp lookupOp) const override {
         if (!mlir::isa<subop::HashMapType>(lookupOp.getState().getType())) return failure();
         return success();
      }

      void rewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
         subop::HashMapType htStateType = mlir::cast<subop::HashMapType>(lookupOp.getState().getType());
         EntryStorageHelper keyStorageHelper(lookupOp, htStateType.getKeyMembers(), false, typeConverter);
         EntryStorageHelper valStorageHelper(lookupOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
         auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());
         mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
         auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, llvm::SmallVector<Value> right) -> Value {
            llvm::SmallVector<mlir::Value> arguments;
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

         auto idxType = rewriter.getIndexType();
         auto kvType = getHtKVType(htStateType, *typeConverter);
         auto kvPtrType = util::RefType::get(context, kvType);
         auto keyPtrType = keyStorageHelper.getRefType();
         auto entryPtrType = util::RefType::get(context, entryType);
         auto htType = util::RefType::get(context, entryPtrType);

         Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
         Value ht = rewriter.create<util::LoadElementOp>(loc, htType, castedState, 0);
         Value htMask = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), castedState, 1);

         Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         //position = hash & hashTableMask
         Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
         // ptr = &hashtable[position]
         Type bucketPtrType = util::RefType::get(context, entryType);
         Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
         Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hashed);
         ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
         auto ifOpOuter = rewriter.create<mlir::scf::IfOp>(
            loc, tagMatches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
               Block* before = new Block;
               whileOp.getBefore().push_back(before);
               before->addArgument(bucketPtrType, loc);
               Block* after = new Block;
               whileOp.getAfter().push_back(after);
               after->addArgument(bucketPtrType, loc);
               rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
                  Value currEntryPtr = before->getArgument(0);
                  Value ptr = currEntryPtr;
                  //    if (*ptr != nullptr){
                  Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
                  auto ifOp = rewriter.create<scf::IfOp>(
                     loc, cmp, [&](OpBuilder& b, Location loc) {
                  Value entryHash=rewriter.create<util::LoadElementOp>(loc, idxType, currEntryPtr, 1);
                  Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                  auto ifOpH = b.create<scf::IfOp>(
                     loc, hashMatches, [&](OpBuilder& b, Location loc) {
                        Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                        Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                        auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                        Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                        auto ifOp2 = b.create<scf::IfOp>(
                           loc, keyMatches, [&](OpBuilder& b, Location loc) {


                              b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                              //          ptr = &entry.next
                              Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
                              Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                              //          yield ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                        b.create<scf::YieldOp>(loc, ifOp2.getResults());
                     }, [&](OpBuilder& b, Location loc) {
                        //          ptr = &entry.next
                        Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
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
               rewriter.create<scf::YieldOp>(loc, ValueRange{whileOp.getResult(0)}); },
            [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               Value invalidPtr = rewriter.create<util::InvalidRefOp>(loc, bucketPtrType);
               rewriter.create<scf::YieldOp>(loc, ValueRange{invalidPtr});
            });

         Value currEntryPtr = ifOpOuter.getResult(0);
         currEntryPtr = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getPtrType(), currEntryPtr);
         mapping.define(lookupOp.getRef(), currEntryPtr);
         rewriter.replaceTupleStream(lookupOp, mapping);
      }
   };

   class PureLookupPreAggregationHtLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
      public:
      using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
      LogicalResult match(subop::LookupOp lookupOp) const override {
         if (!mlir::isa<subop::PreAggrHtType>(lookupOp.getState().getType())) return failure();
         return success();
      }

      void rewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
         subop::PreAggrHtType htStateType = mlir::cast<subop::PreAggrHtType>(lookupOp.getState().getType());
         EntryStorageHelper keyStorageHelper(lookupOp, htStateType.getKeyMembers(), false, typeConverter);
         EntryStorageHelper valStorageHelper(lookupOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
         auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());

         mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
         ASSERT_WITH_OP(!lookupOp.getEqFn().empty(), lookupOp, "LookupOp must have an equality function");
         auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, llvm::SmallVector<Value> right) -> Value {
            llvm::SmallVector<mlir::Value> arguments;
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

         auto idxType = rewriter.getIndexType();
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
         Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hashed);
         ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
         Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         auto ifOpOuter = rewriter.create<mlir::scf::IfOp>(
            loc, tagMatches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
               Block* before = new Block;
               whileOp.getBefore().push_back(before);
               before->addArgument(bucketPtrType, loc);
               Block* after = new Block;
               whileOp.getAfter().push_back(after);
               after->addArgument(bucketPtrType, loc);
               rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
                  Value currEntryPtr = before->getArgument(0);
                  Value ptr = currEntryPtr;
                  //    if (*ptr != nullptr){
                  Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
                  auto ifOp = rewriter.create<scf::IfOp>(
                     loc, cmp, [&](OpBuilder& b, Location loc) {
                        Value entryHash = rewriter.create<util::LoadElementOp>(loc, idxType, currEntryPtr, 1);
                        Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                        auto ifOpH = b.create<scf::IfOp>(
                           loc, hashMatches, [&](OpBuilder& b, Location loc) {
                              Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                              Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                              auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                              Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                              auto ifOp2 = b.create<scf::IfOp>(
                                 loc, keyMatches, [&](OpBuilder& b, Location loc) {


                                    b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                                    //          ptr = &entry.next
                                    Value newEntryPtr=b.create<util::LoadElementOp> (loc, i8PtrType, currEntryPtr, 0);
                                    Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                                    //          yield ptr,done=false
                                    b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                              b.create<scf::YieldOp>(loc, ifOp2.getResults());
                           }, [&](OpBuilder& b, Location loc) {
                              //          ptr = &entry.next
                              Value newEntryPtr=b.create<util::LoadElementOp> (loc, i8PtrType, currEntryPtr, 0);
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
               rewriter.create<scf::YieldOp>(loc, ValueRange{whileOp.getResult(0)}); },
            [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               Value invalidPtr = rewriter.create<util::InvalidRefOp>(loc, bucketPtrType);
               rewriter.create<scf::YieldOp>(loc, ValueRange{invalidPtr});
            });

         Value currEntryPtr = ifOpOuter.getResult(0);
         currEntryPtr = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getPtrType(), currEntryPtr);
         mapping.define(lookupOp.getRef(), currEntryPtr);
         rewriter.replaceTupleStream(lookupOp, mapping);
      }
   };

   class LookupHashMultiMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
      public:
      using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
      LogicalResult match(subop::LookupOp lookupOp) const override {
         subop::HashMultiMapType hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(lookupOp.getState().getType());
         if (!hashMultiMapType) return failure();
         return success();
      }

      void rewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
         subop::HashMultiMapType hashMultiMapType = mlir::cast<subop::HashMultiMapType>(lookupOp.getState().getType());
         EntryStorageHelper keyStorageHelper(lookupOp, hashMultiMapType.getKeyMembers(), false, typeConverter);
         EntryStorageHelper valStorageHelper(lookupOp, hashMultiMapType.getValueMembers(), hashMultiMapType.hasLock(), typeConverter);
         auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());

         mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
         auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, llvm::SmallVector<Value> right) -> Value {
            llvm::SmallVector<mlir::Value> arguments;
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

         auto idxType = rewriter.getIndexType();
         auto keyPtrType = keyStorageHelper.getRefType();
         auto entryPtrType = util::RefType::get(context, entryType);
         auto htType = util::RefType::get(context, entryPtrType);

         Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
         Value ht = rewriter.create<util::LoadElementOp>(loc, htType, castedState, 0);
         Value htMask = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), castedState, 1);

         Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         //position = hash & hashTableMask
         Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
         // ptr = &hashtable[position]
         Type bucketPtrType = util::RefType::get(context, entryType);
         Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
         Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hashed);
         ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
         auto ifOpOuter = rewriter.create<mlir::scf::IfOp>(
            loc, tagMatches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {

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
                  Value entryHash = rewriter.create<util::LoadElementOp>(loc, idxType, currEntryPtr, 1);
                  Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                  auto ifOpH = b.create<scf::IfOp>(
                     loc, hashMatches, [&](OpBuilder& b, Location loc) {
                        Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, currEntryPtr, 3);
                        auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                        Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                        auto ifOp2 = b.create<scf::IfOp>(
                           loc,  keyMatches, [&](OpBuilder& b, Location loc) {
                              b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                              //          ptr = &entry.next
                              Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
                              Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                              //          yield ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                        b.create<scf::YieldOp>(loc, ifOp2.getResults());
                     }, [&](OpBuilder& b, Location loc) { //          ptr = &entry.next
                        Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
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
               rewriter.create<scf::YieldOp>(loc, ValueRange{whileOp.getResult(0)}); },
            [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               Value invalidPtr = rewriter.create<util::InvalidRefOp>(loc, bucketPtrType);
               rewriter.create<scf::YieldOp>(loc, ValueRange{invalidPtr});
            });

         Value currEntryPtr = ifOpOuter.getResult(0);
         mapping.define(lookupOp.getRef(), currEntryPtr);
         rewriter.replaceTupleStream(lookupOp, mapping);
      }
   };
   class InsertMultiMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::InsertOp> {
      public:
      using SubOpTupleStreamConsumerConversionPattern<subop::InsertOp>::SubOpTupleStreamConsumerConversionPattern;
      LogicalResult match(subop::InsertOp insertOp) const override {
         subop::HashMultiMapType htStateType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(insertOp.getState().getType());
         if (!htStateType) return failure();
         return success();
      }

      void rewrite(subop::InsertOp insertOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
         subop::HashMultiMapType htStateType = mlir::cast<subop::HashMultiMapType>(insertOp.getState().getType());
         EntryStorageHelper keyStorageHelper(insertOp, htStateType.getKeyMembers(), false, typeConverter);
         EntryStorageHelper valStorageHelper(insertOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
         auto loc = insertOp->getLoc();
         llvm::SmallVector<mlir::Value> lookupKey = keyStorageHelper.resolve(insertOp, insertOp.getMapping(), mapping);

         mlir::Value hash = hashKeys(lookupKey, rewriter, loc);
         mlir::Value hashTable = adaptor.getState();
         auto equalFnBuilder = [&insertOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, llvm::SmallVector<Value> right) -> Value {
            llvm::SmallVector<mlir::Value> arguments;
            arguments.insert(arguments.end(), left.begin(), left.end());
            arguments.insert(arguments.end(), right.begin(), right.end());
            auto res = inlineBlock(&insertOp.getEqFn().front(), rewriter, arguments);
            return res[0];
         };
         Value hashed = hash;

         auto* context = rewriter.getContext();
         auto entryType = getHashMultiMapEntryType(htStateType, *typeConverter);
         auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));

         auto idxType = rewriter.getIndexType();
         auto keyPtrType = keyStorageHelper.getRefType();
         auto entryPtrType = util::RefType::get(context, entryType);
         auto htType = util::RefType::get(context, entryPtrType);

         Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
         Value ht = rewriter.create<util::LoadElementOp>(loc, htType, castedState, 0);
         Value htMask = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), castedState, 1);

         Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         //position = hash & hashTableMask
         Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
         // ptr = &hashtable[position]
         Type bucketPtrType = util::RefType::get(context, entryType);
         Value firstPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
         Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), firstPtr, hashed);
         firstPtr = rewriter.create<arith::SelectOp>(loc, tagMatches, rewriter.create<util::UnTagPtr>(loc, firstPtr.getType(), firstPtr), rewriter.create<util::InvalidRefOp>(loc, bucketPtrType));

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
                  Value entryHash = rewriter.create<util::LoadElementOp>(loc, idxType, currEntryPtr, 1);
                  Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                  auto ifOpH = b.create<scf::IfOp>(
                     loc, hashMatches, [&](OpBuilder& b, Location loc) {
                        Value keyRef=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, currEntryPtr, 3);
                        auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                        Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                        auto ifOp2 = b.create<scf::IfOp>(
                           loc, keyMatches, [&](OpBuilder& b, Location loc) {
                              Value valRef=rt::HashMultiMap::insertValue(rewriter,loc)({hashTable,currEntryPtr})[0];
                              valRef=rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(),mlir::TupleType::get(getContext(),{i8PtrType,valStorageHelper.getStorageType()})),valRef);
                              valRef=rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), valRef, 1);
                              valStorageHelper.storeFromColumns(insertOp.getMapping(),mapping,valRef,rewriter,loc);
                              b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                              //          ptr = &entry.next
                              Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
                              Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                              //          yield ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                        b.create<scf::YieldOp>(loc, ifOp2.getResults());
                     }, [&](OpBuilder& b, Location loc) {
                  //          ptr = &entry.next
                        Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
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
      }
   };

   class LookupPreAggrHtFragment : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp> {
      public:
      using SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp>::SubOpTupleStreamConsumerConversionPattern;
      LogicalResult match(subop::LookupOrInsertOp lookupOp) const override {
         auto stateType = mlir::dyn_cast_or_null<subop::PreAggrHtFragmentType>(lookupOp.getState().getType());
         auto kernelLocalStateType = mlir::dyn_cast_or_null<subop::KernelLocalType>(lookupOp.getState().getType());
         if (kernelLocalStateType) {
            stateType = mlir::dyn_cast_or_null<subop::PreAggrHtFragmentType>(kernelLocalStateType.getWrapped());
         }
         if (!stateType) return failure();
         return success();
      }

      void rewrite(subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {

         subop::PreAggrHtFragmentType fragmentType = mlir::dyn_cast_or_null<subop::PreAggrHtFragmentType>(lookupOp.getState().getType());
         auto kernelLocalStateType = mlir::dyn_cast_or_null<subop::KernelLocalType>(lookupOp.getState().getType());
         if (kernelLocalStateType) {
            fragmentType = mlir::dyn_cast_or_null<subop::PreAggrHtFragmentType>(kernelLocalStateType.getWrapped());
         }
            auto loc = lookupOp->getLoc();
            auto* ctxt = rewriter.getContext();
            mlir::func::FuncOp insertDeviceFunc;
            mlir::func::FuncOp atomicExchBlock8B;
            mlir::func::FuncOp getFragCacheMask;
            if (rewriter.getGPUModule()) {
               // If we are on the GPU side, then ops are created in the GPU module
               rewriter.atStartOf(rewriter.getGPUModule().getBody(), [&](SubOpRewriter& rewriter) {
                  insertDeviceFunc = rewriter.create<mlir::func::FuncOp>(loc, "PreAggregationHTFragInsertWarp", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI64Type()}, {rewriter.getI8PtrType()}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
                  atomicExchBlock8B = rewriter.create<mlir::func::FuncOp>(loc, "AtomicExchBlock8B", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType()}, {rewriter.getI8PtrType()}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
                  getFragCacheMask = rewriter.create<mlir::func::FuncOp>(loc, "PreAggregationHTFragGetCacheMask", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType()}, {rewriter.getI64Type()}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
               });
            }
            EntryStorageHelper keyStorageHelper(lookupOp, fragmentType.getKeyMembers(), false, typeConverter);
            EntryStorageHelper valStorageHelper(lookupOp, fragmentType.getValueMembers(), fragmentType.hasLock(), typeConverter);
            auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());
            mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
            auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, llvm::SmallVector<Value> right) -> Value {
               llvm::SmallVector<mlir::Value> arguments;
               arguments.insert(arguments.end(), left.begin(), left.end());
               arguments.insert(arguments.end(), right.begin(), right.end());
               auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
               return res[0];
            };
            auto initValBuilder = [&lookupOp, this](SubOpRewriter& rewriter) -> llvm::SmallVector<mlir::Value> {
               llvm::SmallVector<mlir::Value> res;
               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lookupOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor adaptor) {
                  res = llvm::SmallVector<mlir::Value>{adaptor.getResults().begin(), adaptor.getResults().end()};
               });
               for (size_t i = 0; i < res.size(); i++) {
                  auto convertedType = typeConverter->convertType(res[i].getType());
                  if (res[i].getType() != convertedType) {
                     res[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(lookupOp->getLoc(), convertedType, res[i]).getResult(0);
                  }
               }
               return res;
            };
            Value hashed = hash;

            auto* context = rewriter.getContext();
            auto entryType = getHtEntryType(fragmentType, *typeConverter);

            auto idxType = rewriter.getIndexType();
            auto kvType = getHtKVType(fragmentType, *typeConverter);
            auto kvPtrType = util::RefType::get(context, kvType);
            auto keyPtrType = keyStorageHelper.getRefType();
            auto valPtrType = valStorageHelper.getRefType();

            Type bucketPtrType = util::RefType::get(context, entryType);
            Value stateAsI8Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getI8PtrType(), adaptor.getState());

            Value ht = adaptor.getState();
            Value htMask;
            // if(rewriter.getGPUModule()){
            //    ht = rewriter.create<util::LoadOp>(loc, bucketPtrPtrType, ht); // first member of PreAggrFrag is Entry**, not Entry*[1024], load base address first
            //    htMask = rewriter.create<mlir::func::CallOp>(loc, getFragCacheMask, ValueRange{stateAsI8Ptr}).getResult(0); // get i64 mask (should depend on SMEM size)
            //    htMask = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), htMask); // convert to index to match CPU path
            // } else {
            htMask = rewriter.create<arith::ConstantIndexOp>(loc, 1024 - 1);
            // }
            Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

            //position = hash & hashTableMask
            Value position = rewriter.create<arith::AndIOp>(loc, htMask, rewriter.create<arith::ShRUIOp>(loc, hashed, rewriter.create<mlir::arith::ConstantIndexOp>(loc, 6)));
            // ptr = hashtable[position]
            Value currEntryPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position); // exchnage 0, result is the cached entry
            //    if (ptr != nullptr){
            Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
            auto ifOp = rewriter.create<scf::IfOp>(
               loc, cmp, [&](OpBuilder& b, Location loc) {
                  Value entryHash = rewriter.create<util::LoadElementOp>(loc, idxType, currEntryPtr, 1);
                  Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                  auto ifOpH = b.create<scf::IfOp>(
                     loc, hashMatches, [&](OpBuilder& b, Location loc) {
                        Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                        Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                        auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                        Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                        b.create<scf::YieldOp>(loc, mlir::ValueRange{keyMatches});
                     }, [&](OpBuilder& b, Location loc) {  b.create<scf::YieldOp>(loc, falseValue);});
                  b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue}); });
            auto ifOp2 = rewriter.create<scf::IfOp>(
               loc, ifOp.getResults()[0], [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, currEntryPtr); },
               [&](OpBuilder& b, Location loc) {
                  auto initialVals = initValBuilder(rewriter);
                  Value entryRef;
                  if (rewriter.getGPUModule()) {
                     Value hashedAsI64 = rewriter.create<mlir::arith::IndexCastUIOp>(loc, rewriter.getI64Type(), hashed);
                     entryRef = rewriter.create<mlir::func::CallOp>(loc, insertDeviceFunc, mlir::ValueRange{stateAsI8Ptr, hashedAsI64}).getResult(0);
                  } else {
                     entryRef = rt::PreAggregationHashtableFragment::insert(b, loc)({adaptor.getState(), hashed})[0];
                  }
                  Value entryRefCasted = rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, entryRef);
                  Value kvRef = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, entryRefCasted, 2);
                  Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvRef, 0);
                  Value valRef = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvRef, 1);
                  keyStorageHelper.storeOrderedValues(keyRef, lookupKey, rewriter, loc);
                  valStorageHelper.storeOrderedValues(valRef, initialVals, rewriter, loc);
                  if (fragmentType.hasLock()) {
                     rt::EntryLock::initialize(rewriter, loc)({valStorageHelper.getLockPointer(valRef, rewriter, loc)});
                  }
                  if (rewriter.getGPUModule()) {
                     // GPU rt call does not store entries. Unlike with CPU, GPU's cache is shared across thread block, hence an update
                     // must only become visible when the future pointee has been fully initialized
                     auto htEntryRef = rewriter.create<util::ArrayElementPtrOp>(loc, bucketPtrType, ht, position); // &ht[pos]
                     Value htEntryRefAsI8 = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getI8PtrType(), htEntryRef);
                     Value entryRefAsI8 = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getI8PtrType(), entryRef);
                     rewriter.create<mlir::func::CallOp>(loc, atomicExchBlock8B, mlir::ValueRange{htEntryRefAsI8, entryRefAsI8}).getResult(0); // atomicExch(&ht[pos], entryRef)
                  }
                  b.create<scf::YieldOp>(loc, ValueRange{entryRefCasted});
               });
            currEntryPtr = ifOp2.getResult(0);
            if (rewriter.getGPUModule()) {
               rewriter.entryWithNextPtr = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getI8PtrType(), currEntryPtr);
            }

            Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
            mapping.define(lookupOp.getRef(), rewriter.create<util::TupleElementPtrOp>(lookupOp->getLoc(), util::RefType::get(getContext(), kvType.getType(1)), kvAddress, 1));
            rewriter.replaceTupleStream(lookupOp, mapping);
         }
      };

      class LookupHashMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::LookupOrInsertOp lookupOp) const override {
            if (!mlir::isa<subop::HashMapType>(lookupOp.getState().getType())) return failure();
            return success();
         }

         void rewrite(subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            subop::HashMapType htStateType = mlir::cast<subop::HashMapType>(lookupOp.getState().getType());
            EntryStorageHelper keyStorageHelper(lookupOp, htStateType.getKeyMembers(), false, typeConverter);
            EntryStorageHelper valStorageHelper(lookupOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
            auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());

            mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
            mlir::Value hashTable = adaptor.getState();
            auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, llvm::SmallVector<Value> right) -> Value {
               llvm::SmallVector<mlir::Value> arguments;
               arguments.insert(arguments.end(), left.begin(), left.end());
               arguments.insert(arguments.end(), right.begin(), right.end());
               auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
               return res[0];
            };
            auto initValBuilder = [&lookupOp, this](SubOpRewriter& rewriter) -> llvm::SmallVector<mlir::Value> {
               llvm::SmallVector<mlir::Value> res;
               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lookupOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor adaptor) {
                  res = llvm::SmallVector<mlir::Value>{adaptor.getResults().begin(), adaptor.getResults().end()};
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

            auto idxType = rewriter.getIndexType();
            auto kvType = getHtKVType(htStateType, *typeConverter);
            auto kvPtrType = util::RefType::get(context, kvType);
            auto keyPtrType = keyStorageHelper.getRefType();
            auto valPtrType = valStorageHelper.getRefType();
            auto entryPtrType = util::RefType::get(context, entryType);
            auto htType = util::RefType::get(context, entryPtrType);

            Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
            Value ht = rewriter.create<util::LoadElementOp>(loc, htType, castedState, 0);
            Value htMask = rewriter.create<util::LoadElementOp>(loc, rewriter.getIndexType(), castedState, 1);

            Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
            Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

            //position = hash & hashTableMask
            Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
            // ptr = &hashtable[position]
            Type bucketPtrType = util::RefType::get(context, entryType);
            Value firstPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
            Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), firstPtr, hashed);
            firstPtr = rewriter.create<arith::SelectOp>(loc, tagMatches, rewriter.create<util::UnTagPtr>(loc, firstPtr.getType(), firstPtr), rewriter.create<util::InvalidRefOp>(loc, firstPtr.getType()));

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
                     Value entryHash = rewriter.create<util::LoadElementOp>(loc, idxType, currEntryPtr, 1);
                     Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                     auto ifOpH = b.create<scf::IfOp>(
                        loc, hashMatches, [&](OpBuilder& b, Location loc) {
                           Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                           Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                           auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                           Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                           auto ifOp2 = b.create<scf::IfOp>(
                              loc,  keyMatches, [&](OpBuilder& b, Location loc) {


                                 b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                                 //          ptr = &entry.next
                                 Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
                                 Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                                 //          yield ptr,done=false
                                 b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                           b.create<scf::YieldOp>(loc, ifOp2.getResults());
                        }, [&](OpBuilder& b, Location loc) {
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadElementOp>(loc, i8PtrType, currEntryPtr, 0);
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
         }
      };

      class LookupExternalHashIndexLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::LookupOp lookupOp) const override {
            if (!mlir::isa<subop::ExternalHashIndexType>(lookupOp.getState().getType())) return failure();
            return success();
         }

         void rewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto loc = lookupOp->getLoc();

            // Calculate hash value and perform lookup in external index hashmap
            auto hashValue = rewriter.create<db::Hash>(loc, mapping.resolve(lookupOp, lookupOp.getKeys()));
            mlir::Value list = rt::HashIndexAccess::lookup(rewriter, loc)({adaptor.getState(), hashValue})[0];

            mapping.define(lookupOp.getRef(), list);
            rewriter.replaceTupleStream(lookupOp, mapping);
         }
      };
      class DefaultGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::GatherOp gatherOp) const override {
            return success();
         }

         void rewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto referenceType = mlir::cast<subop::StateEntryReference>(gatherOp.getRef().getColumn().type);
            EntryStorageHelper storageHelper(gatherOp, referenceType.getMembers(), referenceType.hasLock(), typeConverter);
            storageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, mapping.resolve(gatherOp, gatherOp.getRef()), rewriter, gatherOp->getLoc());
            rewriter.replaceTupleStream(gatherOp, mapping);
         }
      };
      class ContinuousRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::GatherOp gatherOp) const override {
            auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(gatherOp.getRef().getColumn().type);
            if (!continuousRefEntryType) { return failure(); }
            return success();
         }

         void rewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto continuousRefEntryType = mlir::cast<subop::ContinuousEntryRefType>(gatherOp.getRef().getColumn().type);
            llvm::SmallVector<mlir::Value> unPackedReference;
            rewriter.createOrFold<util::UnPackOp>(unPackedReference, gatherOp->getLoc(), mapping.resolve(gatherOp, gatherOp.getRef()));
            EntryStorageHelper storageHelper(gatherOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
            auto ptrType = storageHelper.getRefType();
            auto baseRef = rewriter.create<util::BufferGetRef>(gatherOp->getLoc(), ptrType, unPackedReference[1]);
            auto elementRef = rewriter.create<util::ArrayElementPtrOp>(gatherOp->getLoc(), ptrType, baseRef, unPackedReference[0]);
            storageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, elementRef, rewriter, gatherOp->getLoc());
            rewriter.replaceTupleStream(gatherOp, mapping);
         }
      };
      class HashMapRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::GatherOp gatherOp) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            auto referenceType = mlir::dyn_cast_or_null<subop::HashMapEntryRefType>(refType);
            if (!referenceType) { return failure(); }
            return success();
         }

         void rewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            auto referenceType = mlir::cast<subop::HashMapEntryRefType>(refType);
            auto keyMembers = referenceType.getHashMap().getKeyMembers();
            auto valMembers = referenceType.getHashMap().getValueMembers();
            auto loc = gatherOp->getLoc();
            EntryStorageHelper keyStorageHelper(gatherOp, keyMembers, false, typeConverter);
            EntryStorageHelper valStorageHelper(gatherOp, valMembers, referenceType.hasLock(), typeConverter);
            auto ref = mapping.resolve(gatherOp, gatherOp.getRef());
            auto keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ref, 0);
            auto valRef = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), ref, 1);
            keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, keyRef, rewriter, loc);
            valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, valRef, rewriter, loc);
            rewriter.replaceTupleStream(gatherOp, mapping);
         }
      };
      class PreAggrHtRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::GatherOp gatherOp) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            auto referenceType = mlir::dyn_cast_or_null<subop::PreAggrHTEntryRefType>(refType);
            if (!referenceType) { return failure(); }
            return success();
         }

         void rewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            auto referenceType = mlir::cast<subop::PreAggrHTEntryRefType>(refType);
            auto keyMembers = referenceType.getHashMap().getKeyMembers();
            auto valMembers = referenceType.getHashMap().getValueMembers();
            auto loc = gatherOp->getLoc();
            EntryStorageHelper keyStorageHelper(gatherOp, keyMembers, false, typeConverter);
            EntryStorageHelper valStorageHelper(gatherOp, valMembers, referenceType.hasLock(), typeConverter);
            auto ref = mapping.resolve(gatherOp, gatherOp.getRef());
            auto keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ref, 0);
            auto valRef = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), ref, 1);
            keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, keyRef, rewriter, loc);
            valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, valRef, rewriter, loc);
            rewriter.replaceTupleStream(gatherOp, mapping);
         }
      };
      class HashMultiMapRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::GatherOp gatherOp) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            if (!mlir::isa<subop::HashMultiMapEntryRefType>(refType)) { return failure(); }
            return success();
         }

         void rewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            auto hashMultiMap = mlir::cast<subop::HashMultiMapEntryRefType>(refType).getHashMultimap();
            auto keyMembers = hashMultiMap.getKeyMembers();
            auto valMembers = hashMultiMap.getValueMembers();
            auto loc = gatherOp->getLoc();
            EntryStorageHelper keyStorageHelper(gatherOp, keyMembers, false, typeConverter);
            EntryStorageHelper valStorageHelper(gatherOp, valMembers, hashMultiMap.hasLock(), typeConverter);
            auto packed = mapping.resolve(gatherOp, gatherOp.getRef());
            llvm::SmallVector<mlir::Value> unpacked;
            rewriter.createOrFold<util::UnPackOp>(unpacked, loc, packed);
            keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, unpacked[0], rewriter, loc);
            valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, unpacked[1], rewriter, loc);
            rewriter.replaceTupleStream(gatherOp, mapping);
         }
      };

      class ExternalHashIndexRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::GatherOp gatherOp) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            if (!mlir::isa<subop::ExternalHashIndexEntryRefType>(refType)) { return failure(); }
            return success();
         }

         void rewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto refType = gatherOp.getRef().getColumn().type;
            auto columns = mlir::cast<subop::ExternalHashIndexEntryRefType>(refType).getMembers();
            auto tableRefVal = mapping.resolve(gatherOp, gatherOp.getRef());
            llvm::SmallVector<mlir::Value> unPacked;
            rewriter.createOrFold<util::UnPackOp>(unPacked, gatherOp->getLoc(), tableRefVal);
            auto currRow = unPacked[0];
            llvm::SmallVector<mlir::Value> unPackedColumns;
            rewriter.createOrFold<util::UnPackOp>(unPackedColumns, gatherOp->getLoc(), unPacked[1]);
            auto members = columns.getMembers();
            for (size_t i = 0; i < members.size(); i++) {
               auto member = members[i];
               if (gatherOp.getMapping().hasMember(member)) {
                  auto columnDefAttr = gatherOp.getMapping().getColumnDef(member);
                  auto colArray = unPackedColumns[i];
                  auto type = columnDefAttr.getColumn().type;
                  //todo: use MLIR interfaces to get the "right" operation for loading a certain type from an arrow array?
                  mlir::Value loaded = rewriter.create<db::LoadArrowOp>(gatherOp->getLoc(), type, colArray, currRow);
                  mapping.define(columnDefAttr, loaded);
               }
            }
            rewriter.replaceTupleStream(gatherOp, mapping);
         }
      };

      class ContinuousRefScatterOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 2> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 2>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::ScatterOp scatterOp) const override {
            auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(scatterOp.getRef().getColumn().type);
            if (!continuousRefEntryType) { return failure(); }
            return success();
         }

         void rewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto continuousRefEntryType = mlir::cast<subop::ContinuousEntryRefType>(scatterOp.getRef().getColumn().type);
            llvm::SmallVector<mlir::Value> unpackedReference;
            rewriter.createOrFold<util::UnPackOp>(unpackedReference, scatterOp->getLoc(), mapping.resolve(scatterOp, scatterOp.getRef()));
            EntryStorageHelper storageHelper(scatterOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
            auto ptrType = storageHelper.getRefType();
            auto baseRef = rewriter.create<util::BufferGetRef>(scatterOp->getLoc(), ptrType, unpackedReference[1]);
            auto elementRef = rewriter.create<util::ArrayElementPtrOp>(scatterOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
            auto values = storageHelper.getValueMap(elementRef, rewriter, scatterOp->getLoc());
            for (auto x : scatterOp.getMapping().getMapping()) {
               values.set(x.first, mapping.resolve(scatterOp, x.second));
            }
            values.store();
            rewriter.eraseOp(scatterOp);
         }
      };

      class ScatterOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 1> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::ScatterOp scatterOp) const override {
            return success();
         }

         void rewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto referenceType = mlir::cast<subop::StateEntryReference>(scatterOp.getRef().getColumn().type);
            auto columns = referenceType.getMembers();
            EntryStorageHelper storageHelper(scatterOp, columns, referenceType.hasLock(), typeConverter);
            auto ref = mapping.resolve(scatterOp, scatterOp.getRef());
            auto values = storageHelper.getValueMap(ref, rewriter, scatterOp->getLoc());
            for (auto x : scatterOp.getMapping().getMapping()) {
               values.set(x.first, mapping.resolve(scatterOp, x.second));
            }
            values.store();
            rewriter.eraseOp(scatterOp);
         }
      };
      class HashMultiMapScatterOp : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 1> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::ScatterOp scatterOp) const override {
            auto hashMultiMapEntryRef = mlir::dyn_cast_or_null<subop::HashMultiMapEntryRefType>(scatterOp.getRef().getColumn().type);
            if (!hashMultiMapEntryRef) return failure();
            return success();
         }

         void rewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto hashMultiMapEntryRef = mlir::cast<subop::HashMultiMapEntryRefType>(scatterOp.getRef().getColumn().type);
            auto columns = hashMultiMapEntryRef.getHashMultimap().getValueMembers();
            EntryStorageHelper storageHelper(scatterOp, columns, hashMultiMapEntryRef.hasLock(), typeConverter);
            llvm::SmallVector<mlir::Value> unPacked;
            rewriter.createOrFold<util::UnPackOp>(unPacked, scatterOp.getLoc(), mapping.resolve(scatterOp, scatterOp.getRef()));
            auto ref = unPacked[1];
            auto values = storageHelper.getValueMap(ref, rewriter, scatterOp->getLoc());
            for (auto x : scatterOp.getMapping().getMapping()) {
               values.set(x.first, mapping.resolve(scatterOp, x.second));
            }
            values.store();
            rewriter.eraseOp(scatterOp);
         }
      };

      class ReduceContinuousRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::ReduceOp reduceOp) const override {
            auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
            if (!continuousRefEntryType) { return failure(); }
            if (reduceOp->hasAttr("atomic")) {
               return failure();
            }
            return success();
         }

         void rewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto continuousRefEntryType = mlir::cast<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
            llvm::SmallVector<mlir::Value> unpackedReference;
            rewriter.createOrFold<util::UnPackOp>(unpackedReference, reduceOp->getLoc(), mapping.resolve(reduceOp, reduceOp.getRef()));
            EntryStorageHelper storageHelper(reduceOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
            auto ptrType = storageHelper.getRefType();
            auto baseRef = rewriter.create<util::BufferGetRef>(reduceOp->getLoc(), ptrType, unpackedReference[1]);
            auto elementRef = rewriter.create<util::ArrayElementPtrOp>(reduceOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
            auto values = storageHelper.getValueMap(elementRef, rewriter, reduceOp->getLoc());
            llvm::SmallVector<mlir::Value> arguments;
            for (auto attr : reduceOp.getColumns()) {
               mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
               if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
                  arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
               }
               arguments.push_back(arg);
            }
            for (auto member : reduceOp.getMembers()) {
               mlir::Value arg = values.get(mlir::cast<subop::MemberAttr>(member).getMember());
               if (arg.getType() != reduceOp.getRegion().getArgument(arguments.size()).getType()) {
                  arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), reduceOp.getRegion().getArgument(arguments.size()).getType(), arg).getResult(0);
               }
               arguments.push_back(arg);
            }

            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
               for (size_t i = 0; i < reduceOp.getMembers().size(); i++) {
                  auto member = mlir::cast<subop::MemberAttr>(reduceOp.getMembers()[i]).getMember();
                  auto& memberVal = values.get(member);
                  auto updatedVal = adaptor.getResults()[i];
                  if (updatedVal.getType() != memberVal.getType()) {
                     updatedVal = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), memberVal.getType(), updatedVal).getResult(0);
                  }
                  memberVal = updatedVal;
               }
               values.store();
               rewriter.eraseOp(reduceOp);
            });
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
            llvm::SmallVector<mlir::Value> arguments;
            for (auto attr : reduceOp.getColumns()) {
               mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
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
            llvm::SmallVector<mlir::Value> arguments;
            for (auto attr : reduceOp.getColumns()) {
               mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
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

         LogicalResult match(subop::ReduceOp reduceOp) const override {
            auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
            if (!continuousRefEntryType) { return failure(); }
            if (!reduceOp->hasAttr("atomic")) {
               return failure();
            }
            return success();
         }

         void rewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto continuousRefEntryType = mlir::cast<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
            auto loc = reduceOp->getLoc();
            llvm::SmallVector<mlir::Value> unpackedReference;
            rewriter.createOrFold<util::UnPackOp>(unpackedReference, reduceOp->getLoc(), mapping.resolve(reduceOp, reduceOp.getRef()));
            EntryStorageHelper storageHelper(reduceOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
            auto ptrType = storageHelper.getRefType();
            auto baseRef = rewriter.create<util::BufferGetRef>(reduceOp->getLoc(), ptrType, unpackedReference[1]);
            auto elementRef = rewriter.create<util::ArrayElementPtrOp>(reduceOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
            auto valueRef = storageHelper.getPointer(elementRef, mlir::cast<subop::MemberAttr>(reduceOp.getMembers()[0]).getMember(), rewriter, loc);
            implementAtomicReduce(reduceOp, rewriter, valueRef, mapping);
         }
      };
      class ReduceOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::ReduceOp reduceOp) const override {
            return success();
         }

         void rewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto referenceType = mlir::cast<subop::StateEntryReference>(reduceOp.getRef().getColumn().type);
            auto members = referenceType.getMembers();
            auto ref = mapping.resolve(reduceOp, reduceOp.getRef());
            EntryStorageHelper storageHelper(reduceOp, members, referenceType.hasLock(), typeConverter);
            if (reduceOp->hasAttr("atomic")) {
               auto valueRef = storageHelper.getPointer(ref, mlir::cast<subop::MemberAttr>(reduceOp.getMembers()[0]).getMember(), rewriter, reduceOp->getLoc());
               implementAtomicReduce(reduceOp, rewriter, valueRef, mapping);
            } else {
               auto loc = reduceOp->getLoc();
               auto* ctxt = rewriter.getContext();

               Value entryNext;
               mlir::func::FuncOp lockFn;
               mlir::func::FuncOp unlockFn;
               if (rewriter.getGPUModule() && rewriter.entryWithNextPtr) {
                  // If we are on the GPU side, then ops are created in the GPU module
                  rewriter.atStartOf(rewriter.getGPUModule().getBody(), [&](SubOpRewriter& rewriter) {
                     lockFn = rewriter.create<mlir::func::FuncOp>(loc, "lockPreAggrHTEntry", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType()}, {rewriter.getI8PtrType()}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
                     unlockFn = rewriter.create<mlir::func::FuncOp>(loc, "unlockPreAggrHTEntry", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType()}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
                  });
                  entryNext = rewriter.create<mlir::func::CallOp>(loc, lockFn, mlir::ValueRange{rewriter.entryWithNextPtr}).getResult(0); // atomicExch(&ht[pos], entryRef)
               }

               auto values = storageHelper.getValueMap(ref, rewriter, reduceOp->getLoc());
               llvm::SmallVector<mlir::Value> arguments;
               for (auto attr : reduceOp.getColumns()) {
                  mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
                  if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
                     arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
                  }
                  arguments.push_back(arg);
               }
               for (auto member : reduceOp.getMembers()) {
                  mlir::Value arg = values.get(mlir::cast<subop::MemberAttr>(member).getMember());
                  if (arg.getType() != reduceOp.getRegion().getArgument(arguments.size()).getType()) {
                     arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), reduceOp.getRegion().getArgument(arguments.size()).getType(), arg).getResult(0);
                  }
                  arguments.push_back(arg);
               }

               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
                  for (size_t i = 0; i < reduceOp.getMembers().size(); i++) {
                     auto member = mlir::cast<subop::MemberAttr>(reduceOp.getMembers()[i]).getMember();
                     auto& memberVal = values.get(member);
                     auto updatedVal = adaptor.getResults()[i];
                     if (updatedVal.getType() != memberVal.getType()) {
                        updatedVal = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), memberVal.getType(), updatedVal).getResult(0);
                     }
                     memberVal = updatedVal;
                  }
                  values.store();
                  rewriter.eraseOp(reduceOp);
               });
               if (rewriter.getGPUModule() && rewriter.entryWithNextPtr) {
                  rewriter.create<mlir::func::CallOp>(loc, unlockFn, mlir::ValueRange{rewriter.entryWithNextPtr, entryNext});
               }
            }
         }
      };

      class CreateHashIndexedViewLowering : public SubOpConversionPattern<subop::CreateHashIndexedView> {
         using SubOpConversionPattern<subop::CreateHashIndexedView>::SubOpConversionPattern;
         LogicalResult matchAndRewrite(subop::CreateHashIndexedView createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
            auto bufferType = mlir::dyn_cast<subop::BufferType>(createOp.getSource().getType());
            if (!bufferType) return failure();
            auto linkIsFirst = bufferType.getMembers().getMembers()[0] == createOp.getLinkMember().getMember();
            auto hashIsSecond = bufferType.getMembers().getMembers()[1] == createOp.getHashMember().getMember();
            if (!linkIsFirst || !hashIsSecond) return failure();

            mlir::Value htView;
            if (createOp->getParentOfType<subop::ExecutionStepOp>().isOnGPU()) {
               // Special case where one op is the entire pipeline implemented in rt, we should be able to launch it as one rt call
               htView = rt::DeviceMemoryFuncs::createHashIndexedView(rewriter, createOp->getLoc())({adaptor.getSource()})[0];
            } else {
               htView = rt::HashIndexedView::build(rewriter, createOp->getLoc())({adaptor.getSource()})[0];
            }
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
            auto genericBuffer = rt::GrowingBuffer::asContinuous(rewriter, createOp->getLoc())({adaptor.getSource()})[0];
            rewriter.replaceOpWithNewOp<util::BufferCastOp>(createOp, typeConverter->convertType(createOp.getType()), genericBuffer);
            return success();
         }
      };

      class GetBeginLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GetBeginReferenceOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GetBeginReferenceOp>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::GetBeginReferenceOp getBeginReferenceOp) const override {
            return success();
         }

         void rewrite(subop::GetBeginReferenceOp getBeginReferenceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(getBeginReferenceOp->getLoc(), 0);
            auto packed = rewriter.create<util::PackOp>(getBeginReferenceOp->getLoc(), mlir::ValueRange{zero, adaptor.getState()});
            mapping.define(getBeginReferenceOp.getRef(), packed);
            rewriter.replaceTupleStream(getBeginReferenceOp, mapping);
         }
      };
      class GetEndLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GetEndReferenceOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::GetEndReferenceOp>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::GetEndReferenceOp getEndReferenceOp) const override {
            return success();
         }

         void rewrite(subop::GetEndReferenceOp getEndReferenceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto len = rewriter.create<util::BufferGetLen>(getEndReferenceOp->getLoc(), rewriter.getIndexType(), adaptor.getState());
            auto one = rewriter.create<mlir::arith::ConstantIndexOp>(getEndReferenceOp->getLoc(), 1);
            auto lastOffset = rewriter.create<mlir::arith::SubIOp>(getEndReferenceOp->getLoc(), len, one);
            auto packed = rewriter.create<util::PackOp>(getEndReferenceOp->getLoc(), mlir::ValueRange{lastOffset, adaptor.getState()});
            mapping.define(getEndReferenceOp.getRef(), packed);
            rewriter.replaceTupleStream(getEndReferenceOp, mapping);
         }
      };
      class EntriesBetweenLowering : public SubOpTupleStreamConsumerConversionPattern<subop::EntriesBetweenOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::EntriesBetweenOp>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::EntriesBetweenOp op) const override {
            return success();
         }

         void rewrite(subop::EntriesBetweenOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            llvm::SmallVector<mlir::Value> unpackedLeft;
            llvm::SmallVector<mlir::Value> unpackedRight;
            rewriter.createOrFold<util::UnPackOp>(unpackedLeft, op->getLoc(), mapping.resolve(op, op.getLeftRef()));
            rewriter.createOrFold<util::UnPackOp>(unpackedRight, op->getLoc(), mapping.resolve(op, op.getRightRef()));
            mlir::Value difference = rewriter.create<mlir::arith::SubIOp>(op->getLoc(), unpackedRight[0], unpackedLeft[0]);
            if (!op.getBetween().getColumn().type.isIndex()) {
               difference = rewriter.create<mlir::arith::IndexCastOp>(op->getLoc(), op.getBetween().getColumn().type, difference);
            }
            mapping.define(op.getBetween(), difference);
            rewriter.replaceTupleStream(op, mapping);
         }
      };
      class OffsetReferenceByLowering : public SubOpTupleStreamConsumerConversionPattern<subop::OffsetReferenceBy> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::OffsetReferenceBy>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::OffsetReferenceBy op) const override {
            return success();
         }

         void rewrite(subop::OffsetReferenceBy op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            llvm::SmallVector<mlir::Value> unpackedRef;
            rewriter.createOrFold<util::UnPackOp>(unpackedRef, op->getLoc(), mapping.resolve(op, op.getRef()));
            auto offset = mapping.resolve(op, op.getIdx());
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
         }
      };
      class UnwrapOptionalHashmapRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::UnwrapOptionalRefOp op) const override {
            auto optionalType = mlir::dyn_cast_or_null<subop::OptionalType>(op.getOptionalRef().getColumn().type);
            if (!optionalType) return failure();
            auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(optionalType.getT());
            if (!lookupRefType) return failure();
            auto hashmapType = mlir::dyn_cast_or_null<subop::HashMapType>(lookupRefType.getState());
            if (!hashmapType) return failure();
            return success();
         }

         void rewrite(subop::UnwrapOptionalRefOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto optionalType = mlir::cast<subop::OptionalType>(op.getOptionalRef().getColumn().type);
            auto lookupRefType = mlir::cast<subop::LookupEntryRefType>(optionalType.getT());
            auto hashmapType = mlir::cast<subop::HashMapType>(lookupRefType.getState());
            auto loc = op.getLoc();
            auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), mapping.resolve(op, op.getOptionalRef()));
            auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
            auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
            auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
            auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
            EntryStorageHelper valStorageHelper(op, hashmapType.getValueMembers(), hashmapType.hasLock(), typeConverter);
            auto valPtrType = valStorageHelper.getRefType();
            ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, op->getLoc());
            rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
               Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, mapping.resolve(op, op.getOptionalRef()));
               auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
               auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
               mapping.define(op.getRef(), valuePtr);
               rewriter.replaceTupleStream(op, mapping);
            });
         }
      };
      class UnwrapOptionalPreAggregationHtRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::UnwrapOptionalRefOp>::SubOpTupleStreamConsumerConversionPattern;
         LogicalResult match(subop::UnwrapOptionalRefOp op) const override {
            auto optionalType = mlir::dyn_cast_or_null<subop::OptionalType>(op.getOptionalRef().getColumn().type);
            if (!optionalType) return failure();
            auto lookupRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(optionalType.getT());
            if (!lookupRefType) return failure();
            auto hashmapType = mlir::dyn_cast_or_null<subop::PreAggrHtType>(lookupRefType.getState());
            if (!hashmapType) return failure();
            return success();
         }

         void rewrite(subop::UnwrapOptionalRefOp op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto optionalType = mlir::cast<subop::OptionalType>(op.getOptionalRef().getColumn().type);
            auto lookupRefType = mlir::cast<subop::LookupEntryRefType>(optionalType.getT());
            auto hashmapType = mlir::cast<subop::PreAggrHtType>(lookupRefType.getState());
            auto loc = op.getLoc();
            auto cond = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), mapping.resolve(op, op.getOptionalRef()));
            auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond);
            auto htEntryType = getHtEntryType(hashmapType, *typeConverter);
            auto htEntryPtrType = util::RefType::get(getContext(), htEntryType);
            auto kvPtrType = util::RefType::get(getContext(), getHtKVType(hashmapType, *typeConverter));
            EntryStorageHelper valStorageHelper(op, hashmapType.getValueMembers(), hashmapType.hasLock(), typeConverter);
            auto valPtrType = valStorageHelper.getRefType();
            ifOp.ensureTerminator(ifOp.getThenRegion(), rewriter, op->getLoc());
            rewriter.atStartOf(&ifOp.getThenRegion().front(), [&](SubOpRewriter& rewriter) {
               Value ptr = rewriter.create<util::GenericMemrefCastOp>(loc, htEntryPtrType, mapping.resolve(op, op.getOptionalRef()));
               auto kvPtr = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
               auto valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvPtr, 1);
               mapping.define(op.getRef(), valuePtr);
               rewriter.replaceTupleStream(op, mapping);
            });
         }
      };

      class GetExternalHashIndexLowering : public SubOpConversionPattern<subop::GetExternalOp> {
         public:
         using SubOpConversionPattern<subop::GetExternalOp>::SubOpConversionPattern;

         LogicalResult matchAndRewrite(subop::GetExternalOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
            if (!mlir::isa<subop::ExternalHashIndexType>(op.getType())) return failure();
            mlir::Value description = rewriter.create<util::CreateConstVarLen>(op->getLoc(), util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());

            rewriter.replaceOp(op, rt::RelationHelper::accessHashIndex(rewriter, op->getLoc())({description})[0]);
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
               ref = rt::SimpleState::create(rewriter, createOp->getLoc())(mlir::ValueRange{typeSize})[0];
               ref = rewriter.create<util::GenericMemrefCastOp>(createOp->getLoc(), loweredType, ref);

            } else {
               rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
                  ref = rewriter.create<util::AllocaOp>(createOp->getLoc(), typeConverter->convertType(createOp.getType()), mlir::Value());
               });
            }
            if (!createOp.getInitFn().empty()) {
               EntryStorageHelper storageHelper(createOp, simpleStateType.getMembers(), simpleStateType.hasLock(), typeConverter);
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

         LogicalResult match(subop::NestedMapOp nestedMapOp) const override {
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
            return success();
         }

         void rewrite(subop::NestedMapOp nestedMapOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            for (auto [p, a] : llvm::zip(nestedMapOp.getParameters(), nestedMapOp.getRegion().front().getArguments().drop_front())) {
               rewriter.map(a, mapping.resolve(nestedMapOp, mlir::cast<tuples::ColumnRefAttr>(p)));
            }
            auto nestedExecutionGroup = mlir::cast<subop::NestedExecutionGroupOp>(&nestedMapOp.getRegion().front().front());

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
                  llvm::SmallVector<mlir::Operation*> ops;
                  auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
                  for (auto& op : step.getSubOps().front()) {
                     if (&op == returnOp)
                        break;
                     ops.push_back(&op);
                  }
                  for (auto* op : ops) {
                     op->remove();
                     rewriter.insertAndRewrite(op);
                  }
                  for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
                     auto mapped = rewriter.getMapped(i);
                     outerMapping.map(o, mapped);
                  }
               }
            }
            rewriter.eraseOp(nestedMapOp);
         }
      };
      template <class T>
      static llvm::SmallVector<T> repeat(T val, size_t times) {
         llvm::SmallVector<T> res{};
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
            llvm::SmallVector<mlir::Type> iterTypes;
            llvm::SmallVector<mlir::Value> iterArgs;

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
               llvm::SmallVector<mlir::Value> args;
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
                     llvm::SmallVector<mlir::Operation*> ops;
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
               llvm::SmallVector<mlir::Value> res;
               auto simpleStateType = mlir::cast<subop::SimpleStateType>(continueOp.getOperandTypes()[0]);
               EntryStorageHelper storageHelper(loopOp, simpleStateType.getMembers(), simpleStateType.hasLock(), typeConverter);
               auto shouldContinueBool = storageHelper.getValueMap(nestedGroupResultMapping.lookup(continueOp.getOperand(0)), rewriter, loc).get(continueOp.getCondMember().getMember());
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
            llvm::SmallVector<mlir::Value> toReplaceWith;
            for (auto& op : nestedExecutionGroup.getRegion().front().getOperations()) {
               if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
                  auto guard = rewriter.nest(outerMapping, step);
                  for (auto [param, arg, isThreadLocal] : llvm::zip(step.getInputs(), step.getSubOps().front().getArguments(), step.getIsThreadLocal())) {
                     mlir::Value input = outerMapping.lookup(param);
                     rewriter.map(arg, input);
                  }
                  llvm::SmallVector<mlir::Operation*> ops;
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
            // Get resultId
            mlir::Value resultId = rewriter.create<mlir::arith::ConstantIntOp>(loc, setTrackedCountOp.getResultId(), mlir::IntegerType::get(rewriter.getContext(), 32));

            // Get tupleCount
            Value loadedTuple = rewriter.create<util::LoadOp>(loc, adaptor.getTupleCount());
            Value tupleCount = rewriter.create<util::UnPackOp>(loc, loadedTuple).getResults()[0];

            rt::ExecutionContext::setTupleCount(rewriter, loc)({resultId, tupleCount});
            rewriter.eraseOp(setTrackedCountOp);
            return mlir::success();
         }
      };

      class LockLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LockOp> {
         public:
         using SubOpTupleStreamConsumerConversionPattern<subop::LockOp>::SubOpTupleStreamConsumerConversionPattern;

         LogicalResult match(subop::LockOp lockOp) const override {
            auto refType = lockOp.getRef().getColumn().type;
            auto entryRefType = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(refType);
            if (!entryRefType) return failure();
            auto hashMapType = mlir::dyn_cast_or_null<subop::PreAggrHtType>(entryRefType.getState());
            if (!hashMapType || !hashMapType.getWithLock()) return failure();
            return success();
         }

         void rewrite(subop::LockOp lockOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
            auto refType = lockOp.getRef().getColumn().type;
            auto entryRefType = mlir::cast<subop::LookupEntryRefType>(refType);
            auto hashMapType = mlir::cast<subop::PreAggrHtType>(entryRefType.getState());
            assert(hashMapType.hasLock());
            auto storageHelper = EntryStorageHelper(lockOp, hashMapType.getValueMembers(), hashMapType.hasLock(), typeConverter);
            auto ref = mapping.resolve(lockOp, lockOp.getRef());
            auto lockPtr = storageHelper.getLockPointer(ref, rewriter, lockOp->getLoc());
            rt::EntryLock::lock(rewriter, lockOp->getLoc())({lockPtr});
            auto inflight = rewriter.createInFlight(mapping);
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lockOp.getNested().front(), inflight.getRes(), [&](tuples::ReturnOpAdaptor adaptor) {
               if (!adaptor.getResults().empty()) {
                  lockOp.getRes().replaceAllUsesWith(adaptor.getResults()[0]);
                  rewriter.eraseOp(lockOp);
               } else {
                  rewriter.eraseOp(lockOp);
               }
            });
            rt::EntryLock::unlock(rewriter, lockOp->getLoc())({lockPtr});
         }
      };
   }; // namespace

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////       GPU Patterns                                                                                                      //
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   class CreateSimpleStateLoweringGPU : public SubOpConversionPattern<subop::CreateSimpleStateOp> {
      public:
      using SubOpConversionPattern<subop::CreateSimpleStateOp>::SubOpConversionPattern;
      LogicalResult matchAndRewrite(subop::CreateSimpleStateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
         auto simpleStateType = mlir::dyn_cast_or_null<subop::SimpleStateType>(createOp.getType());
         if (!simpleStateType) return failure();

         mlir::Value ref;
         if (createOp->hasAttr("allocateOnHeap")) {
            auto loweredType = mlir::cast<util::RefType>(typeConverter->convertType(createOp.getType()));
            mlir::Value typeSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), loweredType.getElementType());
            ref = rt::SimpleState::create(rewriter, createOp->getLoc())(mlir::ValueRange{typeSize})[0];
            ref = rewriter.create<util::GenericMemrefCastOp>(createOp->getLoc(), loweredType, ref);

         } else {
            rewriter.atStartOf(&createOp->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
               ref = rewriter.create<util::AllocaOp>(createOp->getLoc(), typeConverter->convertType(createOp.getType()), mlir::Value());
            });
         }
         if (!createOp.getInitFn().empty()) {
            EntryStorageHelper storageHelper(createOp, simpleStateType.getMembers(), false, typeConverter);
            rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&createOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor returnOpAdaptor) {
               storageHelper.storeOrderedValues(ref, returnOpAdaptor.getResults(), rewriter, createOp->getLoc());
            });
         }
         rewriter.replaceOp(createOp, ref);
         return mlir::success();
      }
   };

   class MoveSimpleStateToFromCPU : public SubOpConversionPattern<subop::StateContextSwitchOp> {
      public:
      using SubOpConversionPattern<subop::StateContextSwitchOp>::SubOpConversionPattern;

      LogicalResult matchAndRewrite(subop::StateContextSwitchOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
         mlir::Value resultRef;
         mlir::Value typeSize;
         if (mlir::isa<subop::SimpleStateType>(op.getType())) {
            typeSize = rewriter.create<util::SizeOfOp>(op->getLoc(), rewriter.getIndexType(), mlir::cast<util::RefType>(typeConverter->convertType(op.getType())).getElementType());
         } else if (mlir::isa<subop::PreAggrHtFragmentType>(op.getType())) {
            typeSize = rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), sizeof(cudaRT::PreAggregationHashtableFragment));
         } else if (mlir::isa<subop::BufferType>(op.getType())) {
            typeSize = rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), sizeof(cudaRT::GrowingBuffer));
         } else if (mlir::isa<subop::PreAggrHtType>(op.getType())) {
            typeSize = rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), sizeof(cudaRT::PreAggregationHashtable));
         } else {
            assert(0 && "Unknown/unsupported state type");
            // return failure();
         }
         if (op.getDirectionAttr().getValue() == subop::DataMovementDirection::fromDevice) {
            resultRef = rewriter.create<util::AllocaOp>(op->getLoc(), rewriter.getI8PtrType(), typeSize);
            rt::DeviceMemoryFuncs::threadCopyFromGPUSync(rewriter, op->getLoc())({resultRef, adaptor.getInput(), typeSize});
         } else { // toDevice
            // A lock will come directly after the state: all states that go to GPU are lockable, represented as {bytes_for_state,bytes_for_lock}.
            mlir::Value lockSize = rewriter.create<util::SizeOfOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getI64Type());
            // Prepare a lock (set to 0)
            mlir::Value lockCPURef = rewriter.create<util::AllocaOp>(op->getLoc(), util::RefType::get(rewriter.getI64Type()), mlir::Value());
            mlir::Value lockInitVal = rewriter.create<mlir::arith::ConstantIntOp>(op->getLoc(), 0, rewriter.getI64Type());
            rewriter.create<util::StoreOp>(op->getLoc(), lockInitVal, lockCPURef, mlir::Value());
            mlir::Value lockableTypeSize = rewriter.create<mlir::arith::AddIOp>(op->getLoc(), typeSize, lockSize);
            // Allocate memory for state + lock
            resultRef = rt::DeviceMemoryFuncs::getPtrForArray(rewriter, op->getLoc())({lockableTypeSize})[0];
            // send type
            rt::DeviceMemoryFuncs::threadSendToGPUSync(rewriter, op->getLoc())({adaptor.getInput(), resultRef, typeSize});
            // send it to a location right after the state
            mlir::Value lockOffsetDevicePtr = rewriter.create<util::ArrayElementPtrOp>(op->getLoc(), rewriter.getI8PtrType(), resultRef, typeSize);
            rt::DeviceMemoryFuncs::threadSendToGPUSync(rewriter, op->getLoc())({lockCPURef, lockOffsetDevicePtr, lockSize});
         }
         rewriter.replaceOp(op, resultRef);
         return mlir::success();
      }
   };

   class MergeIntoOpLoweringGPU : public SubOpConversionPattern<subop::MergeIntoOp> {
      public:
      using SubOpConversionPattern<subop::MergeIntoOp>::SubOpConversionPattern;

      LogicalResult matchAndRewrite(subop::MergeIntoOp mergeIntoOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
         static uint32_t fnIdCounter{0};
         auto loc = mergeIntoOp->getLoc();
         auto* ctxt = rewriter.getContext();

         mlir::Type wrappedReturnType = mergeIntoOp.getKernelLocal().getType().getWrapped();
         util::RefType wrappedReturnLoweredRefType = mlir::cast<util::RefType>(typeConverter->convertType(wrappedReturnType));

         // [GPU module]
         //    Need to create a device function to pass its pointer to the runtime merge impl.
         mlir::gpu::GPUModuleOp gpuModule = rewriter.getGPUModule();
         mlir::func::FuncOp mergeDeviceFunc;
         mlir::func::FuncOp mergeThreadToGlobal;
         mlir::func::FuncOp mergeWarpToGlobal;
         mlir::func::FuncOp mergeThreadBlockToGlobal;

         mlir::FunctionType mergeDeviceFuncType = mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType()}, {});
         rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
            mergeDeviceFunc = rewriter.create<mlir::func::FuncOp>(loc, "merge_fn_" + std::to_string(fnIdCounter++), mergeDeviceFuncType);
            mergeThreadToGlobal = rewriter.create<mlir::func::FuncOp>(loc, "mergeThreadToGlobal",
                                                                      mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType(), rewriter.getI8PtrType(), mergeDeviceFuncType, rewriter.getI32Type()}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
            mergeWarpToGlobal = rewriter.create<mlir::func::FuncOp>(loc, "mergeWarpToGlobal",
                                                                    mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType(), rewriter.getI8PtrType(), mergeDeviceFuncType, rewriter.getI32Type()}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
            mergeThreadBlockToGlobal = rewriter.create<mlir::func::FuncOp>(loc, "mergeThreadBlockToGlobal",
                                                                           mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType(), rewriter.getI8PtrType(), mergeDeviceFuncType}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
         });

         // [MergeFn] (int8*, int8*) -> ()
         //    Need to inline the combine region
         mlir::Block* mergeDeviceFuncBody = new Block;
         mergeDeviceFuncBody->addArgument(rewriter.getI8PtrType(), loc);
         mergeDeviceFuncBody->addArgument(rewriter.getI8PtrType(), loc);
         if (auto simpleState = mlir::dyn_cast_or_null<subop::SimpleStateType>(wrappedReturnType)) {
            EntryStorageHelper storageHelper(mergeIntoOp, simpleState.getMembers(), false, typeConverter);
            rewriter.atStartOf(mergeDeviceFuncBody, [&](SubOpRewriter& rewriter) {
               mlir::Value dst = mergeDeviceFuncBody->getArgument(0);
               auto leftValues = storageHelper.getValueMap(dst, rewriter, loc);
               auto rightValues = storageHelper.getValueMap(mergeDeviceFuncBody->getArgument(1), rewriter, loc);
               std::vector<mlir::Value> args;
               args.insert(args.end(), leftValues.begin(), leftValues.end());
               args.insert(args.end(), rightValues.begin(), rightValues.end());
               for (size_t i = 0; i < args.size(); i++) {
                  auto expectedType = mergeIntoOp.getCombineFn().front().getArgument(i).getType();
                  if (args[i].getType() != expectedType) {
                     args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
                  }
               }
               mlir::Block* combineFn = &mergeIntoOp.getCombineFn().front();
               rewriter.inlineBlock<tuples::ReturnOpAdaptor>(combineFn, args, [&](tuples::ReturnOpAdaptor adaptor) {
                  storageHelper.storeOrderedValues(dst, adaptor.getResults(), rewriter, loc);
               });
               rewriter.create<mlir::func::ReturnOp>(loc);
            });
         } else if (auto growingBufState = mlir::dyn_cast_or_null<subop::BufferType>(wrappedReturnType)) {
            mlir::func::FuncOp mergeIntoDeviceFunc;
            rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
               mergeIntoDeviceFunc = rewriter.create<mlir::func::FuncOp>(loc, "GrowingBufferMergeIntoLeft", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType()}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
            });
            rewriter.atStartOf(mergeDeviceFuncBody, [&](SubOpRewriter& rewriter) {
               mlir::Value dst = mergeDeviceFuncBody->getArgument(0);
               mlir::Value src = mergeDeviceFuncBody->getArgument(1);
               rewriter.create<mlir::func::CallOp>(loc, mergeIntoDeviceFunc, mlir::ValueRange{dst, src});
               rewriter.create<mlir::func::ReturnOp>(loc);
            });
         } else if (auto preAggrHTFrag = mlir::dyn_cast_or_null<subop::PreAggrHtFragmentType>(wrappedReturnType)) {
            mlir::func::FuncOp mergeIntoDeviceFunc;
            rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
               mergeIntoDeviceFunc = rewriter.create<mlir::func::FuncOp>(loc, "PreAggregationHTFragMergeIntoLeft", mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI8PtrType()}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
            });
            rewriter.atStartOf(mergeDeviceFuncBody, [&](SubOpRewriter& rewriter) {
               mlir::Value dst = mergeDeviceFuncBody->getArgument(0);
               mlir::Value src = mergeDeviceFuncBody->getArgument(1);
               rewriter.create<mlir::func::CallOp>(loc, mergeIntoDeviceFunc, mlir::ValueRange{dst, src});
               rewriter.create<mlir::func::ReturnOp>(loc);
            });
         } else {
            llvm::dbgs() << "UNKNOWN  TYPE " << wrappedReturnType << "\n";
            assert(0);
         }
         assert(mergeDeviceFunc.getBlocks().empty());
         mergeDeviceFunc.getBlocks().push_back(mergeDeviceFuncBody);

         // [Kernel]
         //    Merge comes after scan, scan has finalized the kernel func signature and body, so we insert at the end of kernel funcOp body.
         mlir::gpu::GPUFuncOp kernelFuncOp = rewriter.getGPUKernelFunc();
         mlir::Block* kernelFuncOpBody = &kernelFuncOp.getBody().front();
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter.operator mlir::OpBuilder&());
            rewriter.operator mlir::OpBuilder&().setInsertionPoint(&kernelFuncOpBody->back());
            mlir::Value stateSize;
            if (mlir::isa<subop::SimpleStateType>(wrappedReturnType)) {
               stateSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), wrappedReturnLoweredRefType.getElementType());
            } else if (mlir::isa<subop::PreAggrHtFragmentType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::PreAggregationHashtableFragment));
            } else if (mlir::isa<subop::BufferType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::GrowingBuffer));
            } else if (mlir::isa<subop::PreAggrHtType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::PreAggregationHashtable));
            } else {
               assert(0 && "Unknown/unsupported state type");
               // return failure();
            }
            mlir::Value refGlobal = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getI8PtrType(), adaptor.getGlobalState());
            mlir::Value refLocal = rewriter.create<util::GenericMemrefCastOp>(loc, rewriter.getI8PtrType(), adaptor.getKernelLocal());
            // Index to the end of a state (we allocated a few more bytes when transferring it to GPU for locking purposes)
            mlir::Value lockPtr = rewriter.create<util::ArrayElementPtrOp>(loc, rewriter.getI8PtrType(), refGlobal, stateSize);
            auto kernelLocalDefOp = mlir::cast<subop::CreateKernelLocalOp>(mergeIntoOp.getKernelLocal().getDefiningOp());
            auto localityLvl = kernelLocalDefOp.getLocality();
            mlir::Value mergeDeviceFuncPtr = rewriter.create<mlir::func::ConstantOp>(loc, mergeDeviceFunc.getFunctionType(), FlatSymbolRefAttr::get(ctxt, mergeDeviceFunc.getSymName()));
            mlir::Value stateSizeAsI32 = rewriter.create<mlir::arith::IndexCastUIOp>(loc, rewriter.getI32Type(), stateSize);
            if (localityLvl == subop::KernelLocalityLvl::thread) {
               rewriter.create<mlir::func::CallOp>(loc, mergeThreadToGlobal, mlir::ValueRange{refGlobal, refLocal, lockPtr, mergeDeviceFuncPtr, stateSizeAsI32});
            } else if (localityLvl == subop::KernelLocalityLvl::warp) {
               rewriter.create<mlir::func::CallOp>(loc, mergeWarpToGlobal, mlir::ValueRange{refGlobal, refLocal, lockPtr, mergeDeviceFuncPtr, stateSizeAsI32});
            } else {
               rewriter.create<mlir::func::CallOp>(loc, mergeThreadBlockToGlobal, mlir::ValueRange{refGlobal, refLocal, lockPtr, mergeDeviceFuncPtr});
            }
         }
         return mlir::success();
      }
   };

   class MergeOneToOneOpLoweringGPU : public SubOpConversionPattern<subop::MergeOneToOneOp> {
      public:
      using SubOpConversionPattern<subop::MergeOneToOneOp>::SubOpConversionPattern;

      LogicalResult matchAndRewrite(subop::MergeOneToOneOp mergeOneToOneOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
         static uint32_t fnIdCounter{0};
         auto loc = mergeOneToOneOp->getLoc();
         auto* ctxt = rewriter.getContext();
         // A single runtime call is not enough as we have custom device functions eqFn, combineFn and host does not know their addresses.
         // Hence we need a GPU module with device functions eqFn, combineFn and deviceRT call to merge (merge takes pointers to eqFn and combineFn).
         mlir::gpu::GPUModuleOp gpuModule = rewriter.getGPUModule();
         // Step 1. Create eqFn and combineFn as device functions in GPUModuleOp
         mlir::func::FuncOp eqFn, combineFn, mergeFn;
         rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
            eqFn = rewriter.create<mlir::func::FuncOp>(loc, "ht_frag_merge_eqFn_" + std::to_string(fnIdCounter), mlir::FunctionType::get(ctxt, TypeRange({rewriter.getI8PtrType(), rewriter.getI8PtrType()}), TypeRange({rewriter.getI1Type()})));
            combineFn = rewriter.create<mlir::func::FuncOp>(loc, "ht_frag_merge_combineFn_" + std::to_string(fnIdCounter), mlir::FunctionType::get(ctxt, TypeRange({rewriter.getI8PtrType(), rewriter.getI8PtrType()}), {}));
            mergeFn = rewriter.create<mlir::func::FuncOp>(loc, "fragToHT", mlir::FunctionType::get(ctxt, {rewriter.getLLVMPtrType(), rewriter.getLLVMPtrType(), eqFn.getFunctionType(), combineFn.getFunctionType()}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
            fnIdCounter++;
         });
         auto hashMapType = mlir::cast<subop::PreAggrHtType>(mergeOneToOneOp.getType());
         EntryStorageHelper keyStorageHelper(mergeOneToOneOp, hashMapType.getKeyMembers(), false, typeConverter);
         EntryStorageHelper valStorageHelper(mergeOneToOneOp, hashMapType.getValueMembers(), false, typeConverter);
         {
            auto* funcBody = new Block;
            Value left = funcBody->addArgument(rewriter.getI8PtrType(), loc);
            Value right = funcBody->addArgument(rewriter.getI8PtrType(), loc);
            combineFn.getBody().push_back(funcBody);
            rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
               if (!mergeOneToOneOp.getCombineFn().empty()) {
                  auto kvType = getHtKVType(hashMapType, *typeConverter);
                  auto kvPtrType = util::RefType::get(context, kvType);
                  left = rewriter.create<util::GenericMemrefCastOp>(loc, kvPtrType, left);
                  right = rewriter.create<util::GenericMemrefCastOp>(loc, kvPtrType, right);

                  left = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), left, 1);
                  right = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(context, kvType.getType(1)), right, 1);
                  Value dest = left;
                  auto leftValues = valStorageHelper.getValueMap(left, rewriter, loc);
                  auto rightValues = valStorageHelper.getValueMap(right, rewriter, loc);
                  std::vector<mlir::Value> args;
                  args.insert(args.end(), leftValues.begin(), leftValues.end());
                  args.insert(args.end(), rightValues.begin(), rightValues.end());
                  for (size_t i = 0; i < args.size(); i++) {
                     auto expectedType = mergeOneToOneOp.getCombineFn().front().getArgument(i).getType();
                     if (args[i].getType() != expectedType) {
                        args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, expectedType, args[i]).getResult(0);
                     }
                  }
                  Block* sortLambda = &mergeOneToOneOp.getCombineFn().front();
                  rewriter.inlineBlock<tuples::ReturnOpAdaptor>(sortLambda, args, [&](tuples::ReturnOpAdaptor adaptor) {
                     valStorageHelper.storeOrderedValues(dest, adaptor.getResults(), rewriter, loc);
                  });
               }
               rewriter.create<mlir::func::ReturnOp>(loc);
            });
         }
         {
            auto* funcBody = new Block;
            Value left = funcBody->addArgument(rewriter.getI8PtrType(), loc);
            Value right = funcBody->addArgument(rewriter.getI8PtrType(), loc);
            eqFn.getBody().push_back(funcBody);
            rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
               auto leftKeys = keyStorageHelper.getValueMap(left, rewriter, loc);
               auto rightKeys = keyStorageHelper.getValueMap(right, rewriter, loc);
               std::vector<mlir::Value> args;
               args.insert(args.end(), leftKeys.begin(), leftKeys.end());
               args.insert(args.end(), rightKeys.begin(), rightKeys.end());
               auto res = inlineBlock(&mergeOneToOneOp.getEqFn().front(), rewriter, args)[0];
               rewriter.create<mlir::func::ReturnOp>(loc, res);
            });
         }

         // Step 2: Construct kernel body - add arguments and call the runtime merge (don't really care about ptr types here, the runtime reinterprets as it seems fit)
         mlir::Block* kernelBody = rewriter.getGPUKernelBody();
         Value preAggrHt = kernelBody->addArgument(rewriter.getLLVMPtrType(), loc);
         Value preAggrHtFrag = kernelBody->addArgument(rewriter.getLLVMPtrType(), loc);
         rewriter.atStartOf(kernelBody, [&](SubOpRewriter& rewriter) {
            Value combineFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, combineFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(combineFn.getSymName())));
            Value eqFnPtr = rewriter.create<mlir::func::ConstantOp>(loc, eqFn.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(eqFn.getSymName())));
            rewriter.create<mlir::func::CallOp>(loc, mergeFn, mlir::ValueRange{preAggrHt, preAggrHtFrag, eqFnPtr, combineFnPtr});
         });
         mlir::gpu::GPUFuncOp kernel = rewriter.finalizeGPUKernel();

         // Step 3: Construct an inlined kernel call - construct/retrieve kernel parameters on the host side
         mlir::Value launchGridDimX = rewriter.create<arith::ConstantIndexOp>(loc, cudaRT::PreAggregationHashtableFragment::numPartitions);
         mlir::Value launchBlockDimX = rewriter.create<arith::ConstantIndexOp>(loc, 256);
         mlir::gpu::AsyncTokenType tokenType = mlir::gpu::AsyncTokenType::get(ctxt);
         mlir::Value stream = rt::DeviceMemoryFuncs::getThreadStream(rewriter, loc)({})[0];
         mlir::Value launchStream = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, tokenType, stream).getResult(0);
         mlir::Value forUnusedLaunchDims = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
         mlir::Value dynamicSharedMemSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
         mlir::Value loweredTypeToSatisfyRTArg = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI8PtrType(), mergeOneToOneOp.getSourceState()).getResult(0);
         mlir::Value htPtrDevice = rt::DeviceMemoryFuncs::initHtFromFragMetadata(rewriter, loc)({loweredTypeToSatisfyRTArg})[0];
         mlir::Value hashTableLLVMPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getLLVMPtrType(), htPtrDevice).getResult(0);
         mlir::Value fragmentLLVMPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getLLVMPtrType(), mergeOneToOneOp.getSourceState()).getResult(0);
         std::vector<mlir::Value> kernelLaunchArgumentsInOrder{hashTableLLVMPtr, fragmentLLVMPtr};
         const auto blk = mlir::gpu::KernelDim3{.x = launchBlockDimX, .y = forUnusedLaunchDims, .z = forUnusedLaunchDims};
         const auto grd = mlir::gpu::KernelDim3{.x = launchGridDimX, .y = forUnusedLaunchDims, .z = forUnusedLaunchDims};
         mlir::Value streamPtr = rewriter.create<mlir::gpu::LaunchFuncOp>(loc, kernel, grd, blk, dynamicSharedMemSize, kernelLaunchArgumentsInOrder, tokenType, ValueRange{launchStream}).getAsyncToken();
         assert(streamPtr && "Expected a valid stream ptr!");
         streamPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI8PtrType(), streamPtr).getResult(0);
         rt::DeviceMemoryFuncs::syncStream(rewriter, loc)({streamPtr});
         rewriter.replaceOp(mergeOneToOneOp, htPtrDevice);
         return mlir::success();
      }
   };

   class CreateKernelLocalOpLoweringGPU : public SubOpConversionPattern<subop::CreateKernelLocalOp> {
      public:
      using SubOpConversionPattern<subop::CreateKernelLocalOp>::SubOpConversionPattern;
      /* Kernel local means:
            Wrapper: allocate arg2(blocks) * arg3(threads) * sizeof(state) bytes on GPU, pass this pointer to kernel.
            Kernel: calculate state index for a thread (depends on locality level) and use it to get state from the passed pointer.
                    initialize this state (depending on locality level).
      */
      LogicalResult matchAndRewrite(subop::CreateKernelLocalOp createKernelLocalOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
         static uint32_t idCounter{0};
         auto loc = createKernelLocalOp->getLoc();
         auto* ctxt = rewriter.getContext();

         mlir::func::FuncOp gpuStepWrapperFuncOp = rewriter.getGPUStepWrapper();
         mlir::Value gridDimX = gpuStepWrapperFuncOp.front().getArgument(3); // num blocks
         mlir::Value blockDimX = gpuStepWrapperFuncOp.front().getArgument(4); // num threads in a block
         mlir::Value devicePtr;

         mlir::Type wrappedReturnType = createKernelLocalOp.getRes().getType().getWrapped();
         util::RefType wrappedReturnLoweredRefType = mlir::cast<util::RefType>(typeConverter->convertType(wrappedReturnType));

         auto gpuModule = rewriter.getGPUModule();

         // [InitFn]
         mlir::func::FuncOp initStateFuncOp;
         rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
            initStateFuncOp = rewriter.create<mlir::func::FuncOp>(loc, "kernel_local_init_" + std::to_string(idCounter++), mlir::FunctionType::get(ctxt, TypeRange({rewriter.getI8PtrType()}), {}));
         });
         auto* initStateFuncBody = new Block;
         mlir::Value initStateFuncBodyArg = initStateFuncBody->addArgument(rewriter.getI8PtrType(), loc);
         initStateFuncOp.getBody().push_back(initStateFuncBody);
         // initFn wraps state creation ops, its first block's first op is the state creation with an init region, first block of the region is the initialization itself
         rewriter.atStartOf(initStateFuncBody, [&](SubOpRewriter& rewriter) {
            if (!createKernelLocalOp.getInitFn().empty()) {
               if (auto simpleStateType = mlir::dyn_cast<subop::SimpleStateType>(wrappedReturnType)) {
                  mlir::Block* initBlock = &createKernelLocalOp.getInitFn().front().getOperations().front().getRegion(0).front();
                  auto ref = rewriter.create<util::GenericMemrefCastOp>(loc, wrappedReturnLoweredRefType, initStateFuncBodyArg); // i8* -> ref<tuple<values>>
                  EntryStorageHelper storageHelper(createKernelLocalOp, simpleStateType.getMembers(), false, typeConverter);
                  rewriter.inlineBlock<tuples::ReturnOpAdaptor>(initBlock, {}, [&](tuples::ReturnOpAdaptor returnOpAdaptor) {
                     storageHelper.storeOrderedValues(ref, returnOpAdaptor.getResults(), rewriter, loc);
                  });
               } else if (auto growingBufferType = mlir::dyn_cast<subop::BufferType>(wrappedReturnType)) {
                  mlir::Block* initBlock = &createKernelLocalOp.getInitFn().front();
                  mlir::func::FuncOp initByPtrDeviceFunc;
                  mlir::FunctionType initByPtrDeviceFuncType = mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI32Type(), rewriter.getI32Type()}, {});
                  rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
                     initByPtrDeviceFunc = rewriter.create<mlir::func::FuncOp>(loc, "GrowingBufferConstruct", initByPtrDeviceFuncType, rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
                  });
                  auto createOp = mlir::cast<subop::GenericCreateOp>(initBlock->getOperations().front());
                  auto bufferType = mlir::cast<subop::BufferType>(createOp.getType());
                  EntryStorageHelper storageHelper(createKernelLocalOp, bufferType.getMembers(), false, typeConverter);
                  auto elementType = storageHelper.getStorageType();
                  Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, 1024);
                  Value typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
                  Value typeSizeAsI32 = rewriter.create<mlir::arith::IndexCastUIOp>(loc, rewriter.getI32Type(), typeSize);
                  Value initialCapacityAsI32 = rewriter.create<mlir::arith::IndexCastUIOp>(loc, rewriter.getI32Type(), initialCapacity);
                  rewriter.create<mlir::func::CallOp>(loc, initByPtrDeviceFunc, mlir::ValueRange{initStateFuncBodyArg, initialCapacityAsI32, typeSizeAsI32});
               } else if (auto htFragmentType = mlir::dyn_cast<subop::PreAggrHtFragmentType>(wrappedReturnType)) {
                  mlir::Block* initBlock = &createKernelLocalOp.getInitFn().front();
                  mlir::func::FuncOp initByPtrDeviceFunc;
                  // auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext(), static_cast<unsigned>(mlir::gpu::AddressSpace::Workgroup));

                  rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
                     mlir::FunctionType initByPtrDeviceFuncType = mlir::FunctionType::get(ctxt, {rewriter.getI8PtrType(), rewriter.getI32Type(), rewriter.getI8PtrType(), rewriter.getI32Type()}, {});
                     initByPtrDeviceFunc = rewriter.create<mlir::func::FuncOp>(loc, "PreAggregationHTFragConstruct", initByPtrDeviceFuncType, rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
                  });
                  auto createOp = mlir::cast<subop::GenericCreateOp>(initBlock->getOperations().front());
                  auto fragType = mlir::cast<subop::PreAggrHtFragmentType>(createOp.getType());
                  auto typeSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), getHtEntryType(fragType, *typeConverter));
                  mlir::Value typeSizeAsI32 = rewriter.create<mlir::arith::IndexCastUIOp>(loc, rewriter.getI32Type(), typeSize);
                  // mlir::Value smemPtr = rewriter.create<mlir::gpu::DynamicSharedMemoryOp>(loc, mlir::MemRefType::get(ShapedType::kDynamic, rewriter.getI8Type(), MemRefLayoutAttrInterface{}, mlir::gpu::AddressSpaceAttr::get(ctxt, mlir::gpu::AddressSpace::Workgroup))); // lowering only uses addr. space and element type
                  // smemPtr = rewriter.create<util::ToGenericMemrefOp>(loc, rewriter.getI8PtrType(), smemPtr);
                  mlir::func::FuncOp getSmemPtrFunc; // workaround for DynamicSharedMemoryOp lowering
                  rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
                     getSmemPtrFunc = rewriter.create<mlir::func::FuncOp>(loc, "getSmemPtr", mlir::FunctionType::get(ctxt, {}, {rewriter.getI8PtrType()}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
                  });
                  uint32_t scracthPadShift{powerOfTwo(16 * 1024 / sizeof(uint8_t*))};
                  mlir::Value scratchPadSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI32Type(), 1 << scracthPadShift));
                  Value res = rewriter.create<mlir::func::CallOp>(loc, getSmemPtrFunc, mlir::ValueRange{}).getResult(0);
                  rewriter.create<mlir::func::CallOp>(loc, initByPtrDeviceFunc, mlir::ValueRange{initStateFuncBodyArg, typeSizeAsI32, res, scratchPadSize});
               } else {
                  llvm::dbgs() << "Unknown type " << wrappedReturnType << "\n";
                  assert(0 && "Can't handle this type");
               }
               rewriter.create<mlir::func::ReturnOp>(loc);
            }
         });

         // [Wrapper] : allocate enough states
         rewriter.atStartOf(&gpuStepWrapperFuncOp.front(), [&](SubOpRewriter& rewriter) {
            mlir::Value stateSize;
            if (mlir::isa<subop::SimpleStateType>(wrappedReturnType)) {
               stateSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), wrappedReturnLoweredRefType.getElementType());
            } else if (mlir::isa<subop::PreAggrHtFragmentType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::PreAggregationHashtableFragment));
            } else if (mlir::isa<subop::BufferType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::GrowingBuffer));
            } else if (mlir::isa<subop::PreAggrHtType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::PreAggregationHashtable));
            } else {
               assert(0 && "Unknown/unsupported state type");
               // return failure();
            }
            mlir::Value numStates;
            auto localityLvl = createKernelLocalOp.getLocality();
            if (localityLvl == subop::KernelLocalityLvl::thread) {
               numStates = rewriter.create<mlir::arith::MulIOp>(loc, gridDimX, blockDimX);
            } else if (localityLvl == subop::KernelLocalityLvl::warp) {
               mlir::Value warpSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 32));
               mlir::Value atLeastStates = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 1));
               numStates = rewriter.create<mlir::arith::MulIOp>(loc, gridDimX, blockDimX);
               numStates = rewriter.create<mlir::arith::DivUIOp>(loc, numStates, warpSize);
               numStates = rewriter.create<mlir::arith::MaxUIOp>(loc, numStates, atLeastStates); // e.g., one-thread debugging won't use one whole warp
            } else if (localityLvl == subop::KernelLocalityLvl::thread_block) {
               numStates = gridDimX;
            } else {
               assert(0 && "Unknown kernel locality level");
            }
            stateSize = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), stateSize);
            mlir::Value allocSize = rewriter.create<mlir::arith::MulIOp>(loc, numStates, stateSize);
            devicePtr = rt::DeviceMemoryFuncs::getPtrForArray(rewriter, loc)({allocSize})[0];
            mlir::Value deviceLLVMPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getLLVMPtrType(), devicePtr).getResult(0);
            rewriter.wrapperKernelLocals.push_back(deviceLLVMPtr); // will be used later to create kernel call
         });
         {
            // Free kernel local after kernel call (end of wrapper)
            assert(devicePtr && "We expect a valid device ptr to kernel locals");
            mlir::OpBuilder::InsertionGuard guard(rewriter.operator mlir::OpBuilder&());
            rewriter.operator mlir::OpBuilder&().setInsertionPoint(&gpuStepWrapperFuncOp.front().back());
            auto op = util::FunctionHelper::call(rewriter, loc, rt::DeviceMemoryFuncs::freePtrForArray, {devicePtr});
            if (!rewriter.firstKernelLocalFreeInWrapper) {
               rewriter.firstKernelLocalFreeInWrapper = op;
            }
         }

         // [Kernel] : find the state by index, initialize on an approprieate locality level
         mlir::Block* kernelBody = rewriter.getGPUKernelBody();
         auto kernelLocalArrayLLVMPtr = kernelBody->addArgument(rewriter.getLLVMPtrType(), loc);
         mlir::Value threadStateRef;
         rewriter.atStartOf(kernelBody, [&](SubOpRewriter& rewriter) { // Each thread must know its state
            mlir::Value kernelLocalArrayI8Ref = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI8PtrType(), kernelLocalArrayLLVMPtr).getResult(0);
            mlir::Value blockIdxX = rewriter.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
            mlir::Value threadIdxGlobal = rewriter.create<mlir::gpu::GlobalIdOp>(loc, mlir::gpu::Dimension::x);
            mlir::Value stateSize;
            if (mlir::isa<subop::SimpleStateType>(wrappedReturnType)) {
               stateSize = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), wrappedReturnLoweredRefType.getElementType());
            } else if (mlir::isa<subop::PreAggrHtFragmentType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::PreAggregationHashtableFragment));
            } else if (mlir::isa<subop::BufferType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::GrowingBuffer));
            } else if (mlir::isa<subop::PreAggrHtType>(wrappedReturnType)) {
               stateSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizeof(cudaRT::PreAggregationHashtable));
            } else {
               assert(0 && "Unknown/unsupported state type");
               // return failure();
            }

            mlir::Value stateIdx;
            auto localityLvl = createKernelLocalOp.getLocality();
            if (localityLvl == subop::KernelLocalityLvl::thread) {
               stateIdx = threadIdxGlobal;
            } else if (localityLvl == subop::KernelLocalityLvl::warp) {
               mlir::Value warpSize = rewriter.create<arith::ConstantIndexOp>(loc, 32);
               stateIdx = rewriter.create<mlir::arith::DivUIOp>(loc, threadIdxGlobal, warpSize);
            } else if (localityLvl == subop::KernelLocalityLvl::thread_block) {
               stateIdx = blockIdxX;
            } else {
               assert(0 && "Unknown kernel locality level");
            }
            mlir::Value stateOffset = rewriter.create<mlir::arith::MulIOp>(loc, stateSize, stateIdx);
            mlir::Value stateI8Ptr = rewriter.create<util::ArrayElementPtrOp>(loc, rewriter.getI8PtrType(), kernelLocalArrayI8Ref, stateOffset);
            threadStateRef = rewriter.create<util::GenericMemrefCastOp>(loc, wrappedReturnLoweredRefType, stateI8Ptr);
            // Worker for initialization performs initialization
            // We should wrap initialization (locality levels other than "thread" should have this condition, so that only one worker does init)
            if (localityLvl != subop::KernelLocalityLvl::thread) {
               // Determine worker for initialization : it is the first of warp lanes/block threads
               mlir::Value initilizeWorkerId = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
               mlir::Value currentWorkerId;
               if (localityLvl == subop::KernelLocalityLvl::warp) {
                  currentWorkerId = rewriter.create<mlir::gpu::LaneIdOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 32));
               } else if (localityLvl == subop::KernelLocalityLvl::thread_block) {
                  currentWorkerId = rewriter.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
               }
               mlir::Value isInitilizeWorker = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, initilizeWorkerId, currentWorkerId);
               rewriter.create<mlir::scf::IfOp>(
                  loc, isInitilizeWorker, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
                     builder1.create<mlir::func::CallOp>(loc, initStateFuncOp, ValueRange{stateI8Ptr});
                     builder1.create<mlir::scf::YieldOp>(loc);
                  });
            } else {
               rewriter.create<mlir::func::CallOp>(loc, initStateFuncOp, ValueRange{stateI8Ptr});
            }

            // Locality level workers wait until the worker for initialization completes
            if (localityLvl == subop::KernelLocalityLvl::warp) { // __syncwarp();
               mlir::func::FuncOp syncWarpFuncOp;
               rewriter.atStartOf(gpuModule.getBody(), [&](SubOpRewriter& rewriter) {
                  syncWarpFuncOp = rewriter.create<mlir::func::FuncOp>(loc, "syncWarp", mlir::FunctionType::get(ctxt, {}, {}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
               });
               rewriter.create<mlir::func::CallOp>(loc, syncWarpFuncOp, mlir::ValueRange{}); // GPU dialect does not have an op for warp sync
            } else if (localityLvl == subop::KernelLocalityLvl::thread_block) { // __syncthreads();
               rewriter.create<mlir::gpu::BarrierOp>(loc);
            }
         });
         // llvm::dbgs() << "gpuModule" << "\n";
         // gpuModule.dump();
         rewriter.replaceOp(createKernelLocalOp, threadStateRef);
         return mlir::success();
      }
   };
   namespace {

   std::pair<mlir::Value, mlir::Value> getGPUScanStartStep(SubOpRewriter& rewriter, mlir::Location loc, bool batchToGPU = false) {
      mlir::Value blockDimX = rewriter.create<mlir::gpu::BlockDimOp>(loc, mlir::gpu::Dimension::x);
      mlir::Value threadIdxX = rewriter.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);

      mlir::Value start, step;
      if (batchToGPU) { // one batch maps to the entire "GPU": start=globalTID, step=numThreads
         mlir::Value blockIdxX = rewriter.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
         mlir::Value gridDimX = rewriter.create<mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::x);
         start = rewriter.create<mlir::arith::MulIOp>(loc, blockIdxX, blockDimX);
         start = rewriter.create<mlir::arith::AddIOp>(loc, start, threadIdxX);
         step = rewriter.create<mlir::arith::MulIOp>(loc, gridDimX, blockDimX);
      } else { // one batch per thread block (i.e., SM)
         start = threadIdxX;
         step = blockDimX;
      }
      return {start, step};
   }

   std::pair<mlir::Value, mlir::Value> getGPUBatchStartStep(SubOpRewriter& rewriter, mlir::Location loc, bool batchToGPU = false) {
      mlir::Value batchLoopStart, batchLoopStep;
      if (batchToGPU) { // each batch is processed by all threads
         batchLoopStart = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         batchLoopStep = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      } else { // batches are processed by blocks, so if we SOMEHOW (shouldn't normally happen) got a kernel config with gridDimX < numBatches, do a safe option of a block visiting >1 batches.
         batchLoopStart = rewriter.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
         batchLoopStep = rewriter.create<mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::x);
      }
      return {batchLoopStart, batchLoopStep};
   }

   class ScanRefsTableLoweringGPU : public SubOpConversionPattern<subop::ScanRefsOp> {
      public:
      using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

      LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
         auto& memberManager = getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
         if (!mlir::isa<subop::TableType>(scanOp.getState().getType())) return failure();
         auto loc = scanOp->getLoc();
         auto* ctxt = scanOp->getContext();
         subop::TableEntryRefType refType = mlir::cast<subop::TableEntryRefType>(scanOp.getRef().getColumn().type);
         std::string memberMapping = "[";
         std::vector<mlir::Type> accessedColumnTypes;
         auto members = refType.getMembers();
         for (auto m : members.getMembers()) {
            auto type = memberManager.getType(m);
            auto name = memberManager.getName(m);
            accessedColumnTypes.push_back(type);
            if (memberMapping.length() > 1) {
               memberMapping += ",";
            }
            memberMapping += "\"" + name + "\"";
         }
         memberMapping += "]";
         //bool parallel = scanOp->hasAttr("parallel");

         mlir::Value memberMappingValue = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(ctxt), memberMapping);
         mlir::Value dataSourceIterator = rt::DataSourceIteration::init(rewriter, loc)({adaptor.getState(), memberMappingValue})[0];
         ColumnMapping mapping;
         auto baseTypes = [](mlir::TypeRange arr) {
            std::vector<Type> res;
            for (auto x : arr) { res.push_back(getBaseType(x)); }
            return res;
         };
         mlir::TupleType tupleType = mlir::TupleType::get(ctxt, baseTypes(accessedColumnTypes));
         auto i16T = mlir::IntegerType::get(ctxt, 16);
         mlir::Type recordBatchType = mlir::TupleType::get(ctxt, {rewriter.getIndexType(), rewriter.getIndexType(), util::RefType::get(i16T), util::RefType::get(arrow::ArrayType::get(ctxt))});

         // create device function in gpu module for iteration:

         // [Kernel]
         {
            // The tuple stream may end with a kernel-local state that lives inside the kernel body, but scanrefs (or Subops in general) has no knowledge of which context elements are localized.
            //  Hence, to avoid complex explicit mappings, lower scanrefs directly into the kernel body, so that the kernel-local state is implicitly captured in scanref's scope.
            //  Lowering should create ops after kernel-local initFn, for now assume there is nothing important between kernel-local initialization and scanrefs.

            // The kernel is expected to be called with args (kernel_local0, kernel_local1, ... , batchPtr, numBatches, contextPtr).
            //   kernel_local0 refers to some contextPtr element, the "mapping" is implicitly created by the parallelization pass
            //   and is not explicitly required during this lowering.

            mlir::Block* gpuKernelBody = rewriter.getGPUKernelBody();
            mlir::Value scanFuncBodyArgBatchLLVMPtr = gpuKernelBody->addArgument(rewriter.getLLVMPtrType(), loc);
            mlir::Value scanFuncBodyArgNumBatches = gpuKernelBody->addArgument(rewriter.getI64Type(), loc);
            mlir::Value scanFuncBodyArgContextLLVMPtr = gpuKernelBody->addArgument(rewriter.getLLVMPtrType(), loc);

            // Start inserting at the end of the kernel (i.e., after kernel-locals)
            mlir::OpBuilder::InsertionGuard guard(rewriter.operator mlir::OpBuilder&());
            rewriter.operator mlir::OpBuilder&().setInsertionPoint(&gpuKernelBody->back());
            // rewriter.create<mlir::gpu::PrintfOp>(loc, "YA", mlir::ValueRange{}); // Print something

            mlir::Value scanFuncBodyArgContextPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI8PtrType(), scanFuncBodyArgContextLLVMPtr).getResult(0);
            rewriter.loadStepRequirements(scanFuncBodyArgContextPtr, typeConverter);

            mlir::Value batchI8Ptr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI8PtrType(), scanFuncBodyArgBatchLLVMPtr).getResult(0);

            auto [startIdx, step] = getGPUScanStartStep(rewriter, loc);
            auto [batchLoopStartIdx, batchLoopStep] = getGPUBatchStartStep(rewriter, loc);
            mlir::Value numBatches = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), scanFuncBodyArgNumBatches);
            // mlir::Value numBatches = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 301);

            rewriter.create<mlir::scf::ForOp>(loc, batchLoopStartIdx, numBatches, batchLoopStep, mlir::ValueRange{}, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value batchLoopIdx, mlir::ValueRange vr) {
               mlir::Value batchPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(recordBatchType), batchI8Ptr);
               mlir::Value recordBatchPointer = rewriter.create<util::ArrayElementPtrOp>(loc, batchPtr.getType(), batchPtr, batchLoopIdx);
               mlir::Value lenRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getIndexType()), recordBatchPointer, 0);
               mlir::Value offsetRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getIndexType()), recordBatchPointer, 1);
               //mlir::Value selVecRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(util::RefType::get(i16T)), recordBatchPointer, 2);
               mlir::Value ptrRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(util::RefType::get(arrow::ArrayType::get(ctxt))), recordBatchPointer, 3);
               mlir::Value ptrToColumns = rewriter.create<util::LoadOp>(loc, ptrRef);
               std::vector<mlir::Value> arrays;
               for (size_t i = 0; i < accessedColumnTypes.size(); i++) {
                  auto ci = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
                  auto array = rewriter.create<util::LoadOp>(loc, ptrToColumns, ci);
                  arrays.push_back(array);
               }
               auto arraysVal = rewriter.create<util::PackOp>(loc, arrays);
               auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
               auto end = rewriter.create<util::LoadOp>(loc, lenRef);
               auto globalOffset = rewriter.create<util::LoadOp>(loc, offsetRef);
               auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
               auto forOp2 = rewriter.create<mlir::scf::ForOp>(loc, startIdx, end, step, mlir::ValueRange{});
               rewriter.atStartOf(forOp2.getBody(), [&](SubOpRewriter& rewriter) {
                  auto withOffset = rewriter.create<mlir::arith::AddIOp>(loc, forOp2.getInductionVar(), globalOffset);
                  auto currentRecord = rewriter.create<util::PackOp>(loc, mlir::ValueRange{withOffset, arraysVal});
                  mapping.define(scanOp.getRef(), currentRecord);
                  rewriter.replaceTupleStream(scanOp, mapping);
               });
               b.create<mlir::scf::YieldOp>(loc);
            });
         }

         // The kernel will no longer receive any new arguments (all kernel locals are added before scan and scan added its batch and context ptr)
         //   We should finalize the kernel (wrap kernel body in a gpu func).
         mlir::gpu::GPUFuncOp kernel = rewriter.finalizeGPUKernel(); // manages insertion pos on its own

         // [Wrapper]
         //    With the finalized kernel, we can finally create a launch op.
         auto wrapper = rewriter.getGPUStepWrapper();
         mlir::Block* wrapperBody = &wrapper.getBody().front();
         {
            // Start inserting at the end of the kernel (i.e., after kernel-locals creation, but before kernel locals free)
            mlir::OpBuilder::InsertionGuard guard(rewriter.operator mlir::OpBuilder&());
            rewriter.operator mlir::OpBuilder&().setInsertionPoint(rewriter.firstKernelLocalFreeInWrapper);
            mlir::Value recordBatchesPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getLLVMPtrType(), wrapper.getArgument(0)).getResult(0);
            mlir::Value contextPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getLLVMPtrType(), wrapper.getArgument(2)).getResult(0);

            mlir::Value launchGridDimX = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), wrapperBody->getArgument(3));
            mlir::Value launchBlockDimX = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), wrapperBody->getArgument(4));
            mlir::gpu::AsyncTokenType tokenType = mlir::gpu::AsyncTokenType::get(ctxt);
            mlir::Value launchStream = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, tokenType, wrapperBody->getArgument(5)).getResult(0);
            mlir::Value forUnusedLaunchDims = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
            mlir::Value dynamicSharedMemSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI32Type(), 0));

            std::vector<mlir::Value> kernelLaunchArgumentsInOrder = rewriter.wrapperKernelLocals;
            kernelLaunchArgumentsInOrder.push_back(recordBatchesPtr);
            kernelLaunchArgumentsInOrder.push_back(wrapper.getArgument(1));
            kernelLaunchArgumentsInOrder.push_back(contextPtr);
            const auto blk = mlir::gpu::KernelDim3{.x = launchBlockDimX, .y = forUnusedLaunchDims, .z = forUnusedLaunchDims};
            const auto grd = mlir::gpu::KernelDim3{.x = launchGridDimX, .y = forUnusedLaunchDims, .z = forUnusedLaunchDims};
            // The kernel is launched as <<<{launchGridDimX,1,1},{launchGridDimX,1,1}>>>(kernel_local0*, kernel_local1*, ... , batch*, numBatches, context*).
            mlir::Value streamPtr = rewriter.create<mlir::gpu::LaunchFuncOp>(loc, kernel, grd, blk, dynamicSharedMemSize, kernelLaunchArgumentsInOrder, tokenType, ValueRange{launchStream}).getAsyncToken();
            assert(streamPtr && "Expected a valid stream ptr!");
            streamPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI8PtrType(), streamPtr).getResult(0);
            rt::DeviceMemoryFuncs::syncStream(rewriter, loc)({streamPtr});
         }
         // [Main]
         //    Get context tuple loaded on GPU and call data source iteration.
         auto gpuContextPtr = rewriter.storeStepRequirementsGPU();
         mlir::Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, wrapper.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(wrapper.getSymName())));
         // Runtime will use context tuple for the wrapper callback, additionally supplying wrapper with batch, kernel launch params and stream.
         rt::DataSourceIteration::iterateGPU(rewriter, loc)({dataSourceIterator, functionPointer, gpuContextPtr});
         return success();
      }
   };
} // end namepsace
namespace {
   PatternList getGPUPatternList(TypeConverter& typeConverter, mlir::MLIRContext* ctxt) {
      
      PatternList patterns;

      patterns.insertPattern<MapLowering>(typeConverter, ctxt);
   patterns.insertPattern<FilterLowering>(typeConverter, ctxt);
   patterns.insertPattern<RenameLowering>(typeConverter, ctxt);
   //external
   patterns.insertPattern<GetExternalTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<GetExternalHashIndexLowering>(typeConverter, ctxt);
   //ResultTable
   patterns.insertPattern<CreateTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<MaterializeTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<CreateFromResultTableLowering>(typeConverter, ctxt);
   //SimpleState
   patterns.insertPattern<CreateSimpleStateLoweringGPU>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsSimpleStateLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupSimpleStateLowering>(typeConverter, ctxt);
   //Table
   patterns.insertPattern<ScanRefsTableLoweringGPU>(typeConverter, ctxt);
   //patterns.insertPattern<ScanRefsLocalTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<TableRefGatherOpLowering>(typeConverter, ctxt);
   //Buffer
   patterns.insertPattern<CreateBufferLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsVectorLowering>(typeConverter, ctxt);
   patterns.insertPattern<MaterializeVectorLowering>(typeConverter, ctxt);

   //Hashmap
   patterns.insertPattern<CreateHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<PureLookupHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<HashMapRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanHashMapListLowering>(typeConverter, ctxt);
   //patterns.insertPattern<LockLowering>(typeConverter, ctxt);

   //HashMultiMap
   patterns.insertPattern<CreateHashMultiMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<InsertMultiMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupHashMultiMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanMultiMapListLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanHashMultiMap>(typeConverter, ctxt);
   patterns.insertPattern<HashMultiMapRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<HashMultiMapScatterOp>(typeConverter, ctxt);

   // ExternalHashIndex
   patterns.insertPattern<ScanExternalHashIndexListLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupExternalHashIndexLowering>(typeConverter, ctxt);
   patterns.insertPattern<ExternalHashIndexRefGatherOpLowering>(typeConverter, ctxt);

   //SortedView
   patterns.insertPattern<SortLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsSortedViewLowering>(typeConverter, ctxt);
   //HashIndexedView
   patterns.insertPattern<CreateHashIndexedViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupHashIndexedViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanListLowering>(typeConverter, ctxt);
   //ContinuousView
   patterns.insertPattern<CreateContinuousViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsContinuousViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<ContinuousRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ContinuousRefScatterOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ReduceContinuousRefLowering>(typeConverter, ctxt);
   patterns.insertPattern<ReduceContinuousRefAtomicLowering>(typeConverter, ctxt);

   //Heap
   patterns.insertPattern<CreateHeapLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsHeapLowering>(typeConverter, ctxt);
   patterns.insertPattern<MaterializeHeapLowering>(typeConverter, ctxt);
   //SegmentTreeView
   patterns.insertPattern<CreateSegmentTreeViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupSegmentTreeViewLowering>(typeConverter, ctxt);
   //Array
   patterns.insertPattern<CreateArrayLowering>(typeConverter, ctxt);
   //ThreadLocal
   patterns.insertPattern<CreateThreadLocalLowering>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalResultTable>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalBuffer>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalHeap>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalSimpleState>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalHashMap>(typeConverter, ctxt);
   patterns.insertPattern<CreateOpenHtFragmentLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupPreAggrHtFragment>(typeConverter, ctxt);
   patterns.insertPattern<MergePreAggrHashMap>(typeConverter, ctxt);
   patterns.insertPattern<ScanPreAggrHtLowering>(typeConverter, ctxt);
   patterns.insertPattern<PreAggrHtRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<PureLookupPreAggregationHtLowering>(typeConverter, ctxt);
   patterns.insertPattern<UnwrapOptionalPreAggregationHtRefLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanPreAggregationHtListLowering>(typeConverter, ctxt);

   patterns.insertPattern<DefaultGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScatterOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ReduceOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<NestedMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<UnrealizedConversionCastLowering>(typeConverter, ctxt);
   patterns.insertPattern<UnwrapOptionalHashmapRefLowering>(typeConverter, ctxt);
   patterns.insertPattern<OffsetReferenceByLowering>(typeConverter, ctxt);
   patterns.insertPattern<GetBeginLowering>(typeConverter, ctxt);
   patterns.insertPattern<GetEndLowering>(typeConverter, ctxt);
   patterns.insertPattern<EntriesBetweenLowering>(typeConverter, ctxt);
   patterns.insertPattern<InFlightLowering>(typeConverter, ctxt);
   patterns.insertPattern<GenerateLowering>(typeConverter, ctxt);
   patterns.insertPattern<LoopLowering>(typeConverter, ctxt);
   patterns.insertPattern<NestedExecutionGroupLowering>(typeConverter, ctxt);
   //patterns.insertPattern<GetSingleValLowering>(typeConverter, ctxt);
   //patterns.insertPattern<SetTrackedCountLowering>(typeConverter, ctxt);
   //patterns.insertPattern<SimpleStateGetScalarLowering>(typeConverter, ctxt);
   patterns.insertPattern<CreateKernelLocalOpLoweringGPU>(typeConverter, ctxt);
   patterns.insertPattern<MergeIntoOpLoweringGPU>(typeConverter, ctxt);
   patterns.insertPattern<MergeOneToOneOpLoweringGPU>(typeConverter, ctxt);
      return patterns;
   }
void handleExecutionStepGPU(PatternList& patterns, subop::ExecutionStepOp step, subop::ExecutionGroupOp executionGroup, mlir::IRMapping& mapping, mlir::TypeConverter& typeConverter) {
   // llvm::dbgs() << "[GPU] HANDLING STEP " << step << "\n";
   static int stepCounter{0};
   auto* ctxt = step->getContext();
   ModuleOp parentModule = step->getParentOfType<ModuleOp>();
   if (!parentModule->hasAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName())) {
      parentModule->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(), UnitAttr::get(ctxt));
   }
   SubOpRewriter rewriter(patterns, step, mapping);
   // Setup wrapper func (runtime callback) with args: (batch*, numBatches, context*, gridDim.x, blockDim.x, stream)
   mlir::func::FuncOp stepWrapper;
   auto wrapperFuncTy = mlir::FunctionType::get(ctxt, TypeRange{rewriter.getI8PtrType(), rewriter.getI64Type(), rewriter.getI8PtrType(), rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI8PtrType()}, TypeRange());
   rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
      stepWrapper = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "gpu_step_" + std::to_string(stepCounter++), wrapperFuncTy);
      auto* stepWrapperBody = new Block;
      stepWrapperBody->addArgument(rewriter.getI8PtrType(), parentModule.getLoc()); // batch*
      stepWrapperBody->addArgument(rewriter.getI64Type(), parentModule.getLoc()); // numBatches
      stepWrapperBody->addArgument(rewriter.getI8PtrType(), parentModule.getLoc()); // context*
      stepWrapperBody->addArgument(rewriter.getI64Type(), parentModule.getLoc()); // gridDim.x
      stepWrapperBody->addArgument(rewriter.getI64Type(), parentModule.getLoc()); // blockDim.x
      stepWrapperBody->addArgument(rewriter.getI8PtrType(), parentModule.getLoc()); // stream
      rewriter.atStartOf(stepWrapperBody, [&](SubOpRewriter& rewriter) {
         rewriter.create<mlir::func::ReturnOp>(parentModule.getLoc()); // we have to insert to the end, right before return op.
      });
      stepWrapper.getBody().push_back(stepWrapperBody);
   });

   // Setup GPU module for this step's rewriter
   mlir::gpu::GPUModuleOp gpuModule;
   if (mlir::isa<mlir::gpu::GPUModuleOp>(parentModule.getBody()->front())) {
      gpuModule = mlir::cast<mlir::gpu::GPUModuleOp>(parentModule.getBody()->front());
   } else {
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         gpuModule = rewriter.create<mlir::gpu::GPUModuleOp>(parentModule.getLoc(), "GPU_MODULE_" + std::to_string(stepCounter));
      });
   }
   rewriter.setGPUModule(gpuModule);

   mlir::Block* kernelBody = new mlir::Block;
   rewriter.atStartOf(kernelBody, [&](SubOpRewriter& rewriter) {
      rewriter.create<mlir::gpu::ReturnOp>(gpuModule.getLoc());
   });
   rewriter.setGPUKernelBody(kernelBody);

   rewriter.setGPUStepWrapper(stepWrapper);

   

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
      rewriter.rewrite(op, executionGroup);
   }
   auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
   for (auto [i, o] : llvm::zip(returnOp.getInputs(), step.getResults())) {
      mapping.map(o, rewriter.getMapped(i));
   }
   rewriter.cleanup();
   // llvm::dbgs() << "STEP READY!" << "\n";
   // ModuleOp parentModule1 = step->getParentOfType<ModuleOp>();
   // parentModule1.dump();
}

PatternList getCPUPatternList(TypeConverter& typeConverter, mlir::MLIRContext* ctxt) {
   PatternList patterns;
   patterns.insertPattern<MapLowering>(typeConverter, ctxt);
   patterns.insertPattern<FilterLowering>(typeConverter, ctxt);
   patterns.insertPattern<RenameLowering>(typeConverter, ctxt);
   //external
   patterns.insertPattern<GetExternalTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<GetExternalHashIndexLowering>(typeConverter, ctxt);
   //ResultTable
   patterns.insertPattern<CreateTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<MaterializeTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<CreateFromResultTableLowering>(typeConverter, ctxt);
   //SimpleState
   patterns.insertPattern<CreateSimpleStateLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsSimpleStateLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupSimpleStateLowering>(typeConverter, ctxt);
   //Table
   patterns.insertPattern<ScanRefsTableLowering>(typeConverter, ctxt);
   //rewriter.insertPattern<ScanRefsLocalTableLowering>(typeConverter, ctxt);
   patterns.insertPattern<TableRefGatherOpLowering>(typeConverter, ctxt);
   //Buffer
   patterns.insertPattern<CreateBufferLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsVectorLowering>(typeConverter, ctxt);
   patterns.insertPattern<MaterializeVectorLowering>(typeConverter, ctxt);

   //Hashmap
   patterns.insertPattern<CreateHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<PureLookupHashMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<HashMapRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanHashMapListLowering>(typeConverter, ctxt);
   patterns.insertPattern<LockLowering>(typeConverter, ctxt);

   //HashMultiMap
   patterns.insertPattern<CreateHashMultiMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<InsertMultiMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupHashMultiMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanMultiMapListLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanHashMultiMap>(typeConverter, ctxt);
   patterns.insertPattern<HashMultiMapRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<HashMultiMapScatterOp>(typeConverter, ctxt);

   // ExternalHashIndex
   patterns.insertPattern<ScanExternalHashIndexListLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupExternalHashIndexLowering>(typeConverter, ctxt);
   patterns.insertPattern<ExternalHashIndexRefGatherOpLowering>(typeConverter, ctxt);

   //SortedView
   patterns.insertPattern<SortLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsSortedViewLowering>(typeConverter, ctxt);
   //HashIndexedView
   patterns.insertPattern<CreateHashIndexedViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupHashIndexedViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanListLowering>(typeConverter, ctxt);
   //ContinuousView
   patterns.insertPattern<CreateContinuousViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsContinuousViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<ContinuousRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ContinuousRefScatterOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ReduceContinuousRefLowering>(typeConverter, ctxt);
   patterns.insertPattern<ReduceContinuousRefAtomicLowering>(typeConverter, ctxt);

   //Heap
   patterns.insertPattern<CreateHeapLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanRefsHeapLowering>(typeConverter, ctxt);
   patterns.insertPattern<MaterializeHeapLowering>(typeConverter, ctxt);
   //SegmentTreeView
   patterns.insertPattern<CreateSegmentTreeViewLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupSegmentTreeViewLowering>(typeConverter, ctxt);
   //Array
   patterns.insertPattern<CreateArrayLowering>(typeConverter, ctxt);
   //ThreadLocal
   patterns.insertPattern<CreateThreadLocalLowering>(typeConverter, ctxt);
   patterns.insertPattern<CreateThreadLocalBufferLowering>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalResultTable>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalBuffer>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalHeap>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalSimpleState>(typeConverter, ctxt);
   patterns.insertPattern<MergeThreadLocalHashMap>(typeConverter, ctxt);
   patterns.insertPattern<CreateOpenHtFragmentLowering>(typeConverter, ctxt);
   patterns.insertPattern<LookupPreAggrHtFragment>(typeConverter, ctxt);
   patterns.insertPattern<MergePreAggrHashMap>(typeConverter, ctxt);
   patterns.insertPattern<ScanPreAggrHtLowering>(typeConverter, ctxt);
   patterns.insertPattern<PreAggrHtRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<PureLookupPreAggregationHtLowering>(typeConverter, ctxt);
   patterns.insertPattern<UnwrapOptionalPreAggregationHtRefLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScanPreAggregationHtListLowering>(typeConverter, ctxt);

   patterns.insertPattern<DefaultGatherOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ScatterOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<ReduceOpLowering>(typeConverter, ctxt);
   patterns.insertPattern<NestedMapLowering>(typeConverter, ctxt);
   patterns.insertPattern<UnrealizedConversionCastLowering>(typeConverter, ctxt);
   patterns.insertPattern<UnwrapOptionalHashmapRefLowering>(typeConverter, ctxt);
   patterns.insertPattern<OffsetReferenceByLowering>(typeConverter, ctxt);
   patterns.insertPattern<GetBeginLowering>(typeConverter, ctxt);
   patterns.insertPattern<GetEndLowering>(typeConverter, ctxt);
   patterns.insertPattern<EntriesBetweenLowering>(typeConverter, ctxt);
   patterns.insertPattern<InFlightLowering>(typeConverter, ctxt);
   patterns.insertPattern<GenerateLowering>(typeConverter, ctxt);
   patterns.insertPattern<LoopLowering>(typeConverter, ctxt);
   patterns.insertPattern<NestedExecutionGroupLowering>(typeConverter, ctxt);
   //rewriter.insertPattern<GetSingleValLowering>(typeConverter, ctxt);
   patterns.insertPattern<SetTrackedCountLowering>(typeConverter, ctxt);
   //rewriter.insertPattern<SimpleStateGetScalarLowering>(typeConverter, ctxt);
   // GPU state movement
   patterns.insertPattern<MoveSimpleStateToFromCPU>(typeConverter, ctxt);
   return patterns;
}
void handleExecutionStepCPU(PatternList& patternList, subop::ExecutionStepOp step, subop::ExecutionGroupOp executionGroup, mlir::IRMapping& mapping, mlir::TypeConverter& typeConverter) {
   // llvm::dbgs() << "[CPU] HANDLING STEP " << step << "\n";
   SubOpRewriter rewriter(patternList, step, mapping);

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
   llvm::SmallVector<mlir::Operation*> ops;
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
      llvm::SmallVector<mlir::Type> types;
      for (size_t i = 0; i < t.getMembers().getMembers().size(); i++) {
         types.push_back(util::RefType::get(mlir::IntegerType::get(t.getContext(), 8)));
      }
      return util::RefType::get(mlir::TupleType::get(ctxt, types));
   });
   typeConverter.addConversion([&](subop::BufferType t) -> Type {
      return util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](subop::SortedViewType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr, t.getBasedOn().getMembers(), t.hasLock(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::ArrayType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr, t.getMembers(), t.hasLock(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::ContinuousViewType t) -> Type {
      return util::BufferType::get(t.getContext(), EntryStorageHelper(nullptr, t.getBasedOn().getMembers(), t.hasLock(), &typeConverter).getStorageType());
   });
   typeConverter.addConversion([&](subop::SimpleStateType t) -> Type {
      return util::RefType::get(t.getContext(), EntryStorageHelper(nullptr, t.getMembers(), t.hasLock(), &typeConverter).getStorageType());
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
   auto cpuPatternList = getCPUPatternList(typeConverter, ctxt);
      auto gpuPatternList = getGPUPatternList(typeConverter, ctxt);
   llvm::SmallVector<mlir::Operation*> toRemove;
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
            if (step.isOnGPU()) {
               // llvm::dbgs() << "Running step on GPU : " << step << "\n";
               handleExecutionStepGPU(gpuPatternList, step, executionGroup, mapping, typeConverter);
            } else {
               // llvm::dbgs() << "Running step on CPU : " << step << "\n";
               handleExecutionStepCPU(cpuPatternList, step, executionGroup, mapping, typeConverter);
            }
#if TRACER
            {
               mlir::OpBuilder builder(executionGroup);
               rt::ExecutionStepTracing::end(builder, op.getLoc())({tracingStep});
            }
#endif
         }
      }
      auto returnOp = mlir::cast<subop::ExecutionGroupReturnOp>(executionGroup.getRegion().front().getTerminator());
      llvm::SmallVector<mlir::Value> results;
      for (auto i : returnOp.getInputs()) {
         results.push_back(mapping.lookup(i));
      }
      executionGroup.replaceAllUsesWith(results);
      toRemove.push_back(executionGroup);
      return mlir::WalkResult::skip();
   });
   getOperation()->walk([&](subop::SetResultOp setResultOp) {
      mlir::OpBuilder builder(setResultOp);
      mlir::Value idVal = builder.create<mlir::arith::ConstantIntOp>(setResultOp.getLoc(), setResultOp.getResultId(), mlir::IntegerType::get(builder.getContext(), 32));
      lingodb::compiler::runtime::ExecutionContext::setResult(builder, setResultOp->getLoc())({idVal, setResultOp.getState()});

      toRemove.push_back(setResultOp);
   });
   for (auto* op : toRemove) {
      op->dropAllReferences();
      op->dropAllDefinedValueUses();
      op->erase();
   }
   llvm::SmallVector<mlir::Operation*> defs;
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
   //pm.addPass(subop::createGlobalOptPass());
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
