#ifndef MLIR_CONVERSION_RELALGTODB_PIPELINE_H
#define MLIR_CONVERSION_RELALGTODB_PIPELINE_H
#include <functional>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/SmallBitVector.h>

namespace mlir {
namespace relalg {

class PipelineDependency {
   size_t pipline;
   size_t idx;
   mlir::Type t;
   size_t funcId;


   public:
   PipelineDependency(size_t pipline, size_t idx, const Type& t,size_t funcId) : pipline(pipline), idx(idx), t(t),funcId(funcId) {}
   PipelineDependency(): pipline(-1),idx(-1),t(),funcId(-1){}
   const Type& getT() const {
      return t;
   }
   size_t getIdx() const {
      return idx;
   }
   size_t getPipline() const {
      return pipline;
   }
   size_t getFuncId() const {
      return funcId;
   }

};
class Pipeline;
class PipelineManager{
   std::vector<std::shared_ptr<Pipeline>> pipelines;
   std::shared_ptr<Pipeline> currentPipeline;
   std::unordered_map<size_t, std::vector<mlir::Value>> pipelineResults;
   public:
   static constexpr size_t MAX_PIPELINES=64;

   void addPipeline(std::shared_ptr<Pipeline> pipeline){
      pipelines.push_back(pipeline);
   }

   const std::shared_ptr<Pipeline>& getCurrentPipeline() const {
      return currentPipeline;
   }
   void setCurrentPipeline(const std::shared_ptr<Pipeline>& currentPipeline) {
      PipelineManager::currentPipeline = currentPipeline;
   }
   void execute(mlir::OpBuilder& builder);
   std::vector<mlir::Value> getResultsFromPipeline(std::shared_ptr<Pipeline> pipeline);

};

class Pipeline {
   size_t pipelineId;
   OpBuilder builder;
   ModuleOp parentModule;
   std::vector<FuncOp> initFns;
   std::vector<FuncOp> deinitFns;
   mlir::FuncOp finalizeFn;
   FuncOp mainFn;
   mlir::Value flag;
   std::vector<Type> mainArgTypes;
   std::vector<PipelineDependency> dependencies;
   llvm::SmallBitVector dependsOn;

   std::unordered_map<std::string, size_t> avoidDupNames;
   std::string createName(std::string base) {
      size_t num = avoidDupNames[base]++;
      return "pipeline_fn_" + std::to_string(pipelineId) + "_" + base + "_" + std::to_string(num);
   }

   public:
   Pipeline(ModuleOp parentModule) : builder(parentModule.getContext()), parentModule(parentModule),dependsOn(PipelineManager::MAX_PIPELINES) {
      static size_t id = 0;
      pipelineId = id++;
      OpBuilder moduleBuilder(parentModule->getContext());
      moduleBuilder.setInsertionPointToStart(parentModule.getBody());
      mainFn = moduleBuilder.create<FuncOp>(parentModule->getLoc(), createName("main"), builder.getFunctionType({}, {}));
      mlir::Block* functionBlock = new mlir::Block;
      mainFn.body().push_back(functionBlock);
      builder.setInsertionPointToStart(functionBlock);
      builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
      builder.setInsertionPointToStart(functionBlock);
   }
   void finishMainFunction(std::vector<Value> values);

   Value addDependency(PipelineDependency dep);

   std::vector<PipelineDependency> addInitFn(std::function<std::vector<Value>(OpBuilder&)> fn);
   std::vector<mlir::relalg::PipelineDependency> addFinalizeFn(std::function<std::vector<Value>(OpBuilder&,mlir::ValueRange args)> fn);

   ResultRange call(OpBuilder& builder,std::unordered_map<size_t, std::vector<mlir::Value>>& pipelineResults);
   //void addFinalizeFn(std::function<> fn);
   //void addDeInitFn(std::function<...> fn);
   OpBuilder& getBuilder() {
      return builder;
   }
   size_t getPipelineId() const {
      return pipelineId;
   }
   const Value& getFlag() const {
      return flag;
   }
   void setFlag(const Value& flag) {
      Pipeline::flag = flag;
   }
   const llvm::SmallBitVector& getDependsOn() const {
      return dependsOn;
   }
};
} // end namespace relalg
} // end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_PIPELINE_H
