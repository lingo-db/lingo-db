#include "mlir/Conversion/RelAlgToDB/Pipeline.h"

void mlir::relalg::PipelineManager::execute(mlir::OpBuilder& builder) {
   llvm::SmallBitVector done(PipelineManager::maxPipelines);

   while (done.find_first_unset() != -1 && ((size_t) done.find_first_unset()) < pipelines.size()) {
      for (auto p : pipelines) {
         if (done.test(p->getPipelineId())) continue;
         if ((p->getDependsOn() & ~done).none()) {
            p->call(builder, pipelineResults);
            done.set(p->getPipelineId());
         }
      }
   }
}

std::vector<mlir::Value> mlir::relalg::PipelineManager::getResultsFromPipeline(std::shared_ptr<Pipeline> pipeline) {
   return pipelineResults[pipeline->getPipelineId()];
}
std::vector<mlir::relalg::PipelineDependency> mlir::relalg::Pipeline::addInitFn(std::function<std::vector<Value>(OpBuilder&)> fn) {
   size_t initFnId = 1 + initFns.size();
   OpBuilder moduleBuilder(parentModule->getContext());
   moduleBuilder.setInsertionPointToStart(parentModule.getBody());
   std::vector<Type> resTypes;
   FuncOp initFn = moduleBuilder.create<FuncOp>(parentModule->getLoc(), createName("init"), builder.getFunctionType({}, resTypes));

   initFn->setAttr("passthrough", moduleBuilder.getArrayAttr({moduleBuilder.getStringAttr("noinline"), moduleBuilder.getStringAttr("optnone")}));
   mlir::Block* functionBlock = new mlir::Block;
   initFn.body().push_back(functionBlock);
   OpBuilder fnBuilder(parentModule.getContext());
   fnBuilder.setInsertionPointToStart(functionBlock);
   fnBuilder.setInsertionPointToStart(functionBlock);
   auto values = fn(fnBuilder);
   std::vector<PipelineDependency> res;
   size_t i = 0;
   for (auto v : values) {
      res.push_back({pipelineId, i++, v.getType(), initFnId});
      resTypes.push_back(v.getType());
   }
   fnBuilder.create<mlir::ReturnOp>(fnBuilder.getUnknownLoc(), values);
   initFn.setType(fnBuilder.getFunctionType({}, resTypes));
   initFns.push_back(initFn);
   return res;
}
std::vector<mlir::relalg::PipelineDependency> mlir::relalg::Pipeline::addFinalizeFn(std::function<std::vector<Value>(OpBuilder&, mlir::ValueRange)> fn) {
   OpBuilder moduleBuilder(parentModule->getContext());
   moduleBuilder.setInsertionPointToStart(parentModule.getBody());
   std::vector<Type> resTypes;
   finalizeFn = moduleBuilder.create<FuncOp>(parentModule->getLoc(), createName("finalize"), builder.getFunctionType({}, resTypes));
   finalizeFn->setAttr("passthrough", moduleBuilder.getArrayAttr({moduleBuilder.getStringAttr("noinline"), moduleBuilder.getStringAttr("optnone")}));

   mlir::Block* functionBlock = new mlir::Block;
   finalizeFn.body().push_back(functionBlock);
   OpBuilder fnBuilder(parentModule.getContext());
   fnBuilder.setInsertionPointToStart(functionBlock);
   fnBuilder.setInsertionPointToStart(functionBlock);
   for (auto argT : mainFn.getType().getResults()) {
      functionBlock->addArgument(argT, fnBuilder.getUnknownLoc());
   }
   auto values = fn(fnBuilder, functionBlock->getArguments());
   size_t i = 0;
   std::vector<mlir::relalg::PipelineDependency> res;
   for (auto v : values) {
      res.push_back({pipelineId, i++, v.getType(), 0});
      resTypes.push_back(v.getType());
   }
   fnBuilder.create<mlir::ReturnOp>(fnBuilder.getUnknownLoc(), values);
   finalizeFn.setType(fnBuilder.getFunctionType(mainFn.getType().getResults(), resTypes));
   return res;
}
mlir::Value mlir::relalg::Pipeline::addDependency(PipelineDependency dep) {
   if (dep.getPipline() != pipelineId) {
      dependsOn.set(dep.getPipline());
   }
   dependencies.push_back(dep);
   mainArgTypes.push_back(dep.getT());
   //todo: persist dependency
   auto& mainFnBlock = (*mainFn.body().begin());
   return mainFnBlock.addArgument(dep.getT(), OpBuilder(mainFn.getContext()).getUnknownLoc());
}

void mlir::relalg::Pipeline::finishMainFunction(std::vector<Value> values) {
   auto mainFnBlock = mainFn.body().begin();
   auto* terminator = mainFnBlock->getTerminator();
   OpBuilder b(terminator);
   //mainFn->setAttr("passthrough", b.getArrayAttr({b.getStringAttr("noinline"), b.getStringAttr("optnone")}));
   b.create<mlir::ReturnOp>(b.getUnknownLoc(), values);
   terminator->erase();
   std::vector<Type> resTypes;
   for (auto v : values) {
      resTypes.push_back(v.getType());
   }
   mainFn.setType(b.getFunctionType(mainArgTypes, resTypes));
}

mlir::ResultRange mlir::relalg::Pipeline::call(OpBuilder& builder, std::unordered_map<size_t, std::vector<mlir::Value>>& pipelineResults) {
   std::vector<CallOp> initFnCalls;
   for (size_t i = 0; i < initFns.size(); i++) {
      initFnCalls.push_back(builder.create<mlir::CallOp>(builder.getUnknownLoc(), initFns[i], ValueRange()));
   }
   std::vector<Value> mainParams;
   for (auto dep : dependencies) {
      auto fid = dep.getFuncId();
      if (fid == 0) {
         assert(pipelineResults[dep.getPipline()].size() > dep.getIdx());
         mainParams.push_back(pipelineResults[dep.getPipline()][dep.getIdx()]);
      } else {
         mainParams.push_back(initFnCalls[fid - 1].getResult(dep.getIdx()));
      }
   }
   CallOp mainCall = builder.create<mlir::CallOp>(builder.getUnknownLoc(), mainFn, mainParams);
   if (finalizeFn) {
      CallOp finalizeCall = builder.create<mlir::CallOp>(builder.getUnknownLoc(), finalizeFn, mainCall->getResults());
      pipelineResults[pipelineId] = std::vector<mlir::Value>(finalizeCall->getResults().begin(), finalizeCall->getResults().end());
      return finalizeCall->getResults();
   } else {
      pipelineResults[pipelineId] = std::vector<mlir::Value>(mainCall->getResults().begin(), mainCall->getResults().end());
      return mainCall->getResults();
   }
}
