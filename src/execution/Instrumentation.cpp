#include "lingodb/execution/Instrumentation.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/utility/Setting.h"
#include "lingodb/utility/Tracer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/LocationSnapshot.h"

#include "json.h"

#include <fstream>

namespace {
utility::GlobalSetting<bool> snapshotSetting("system.snapshot_passes", false);

utility::GlobalSetting<std::string> snapshotDir("system.snapshot_dir", ".");
//full: default, but only available if activated
//important
utility::GlobalSetting<std::string> snapshotLevel("system.snapshot_level", "full");
} // end anonymous namespace
namespace {
utility::Tracer::StringMetaDataEvent runPass("Compilation", "Pass", "PassName");
utility::Tracer::Event snapshotEvent("Ignore", "Snapshot");
using namespace lingodb::compiler::dialect;
class SnapshotAfterEachPass : public mlir::PassInstrumentation {
   std::shared_ptr<lingodb::execution::SnapshotState> serializationState;

   public:
   SnapshotAfterEachPass(std::shared_ptr<lingodb::execution::SnapshotState> serializationState) : serializationState(serializationState) {}
   virtual void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override {
      utility::Tracer::Trace trace(snapshotEvent);
      std::ofstream file("last-pass.json");
      nlohmann::json j({{"name", pass->getName().str()}, {"description", pass->getDescription().str()}, {"argument", pass->getArgument().str()}});
      file << j.dump(4) << std::endl;
   }
   virtual void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
      utility::Tracer::Trace trace(snapshotEvent);
      if (serializationState->serialize) {
         if (pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
            return;
         }
         auto id = serializationState->counter++;
         mlir::ModuleOp module = mlir::dyn_cast_or_null<mlir::ModuleOp>(op);
         if (!module) {
            module = op->getParentOfType<mlir::ModuleOp>();
         }
         if (module) {
            llvm::DenseMap<mlir::Operation*, relalg::TableMetaDataAttr> cached;
            module.walk([&](relalg::BaseTableOp tableOp) {
               if (auto tableMetaData = tableOp.getMetaAttr()) {
                  if (tableMetaData.getMeta()->isPresent()) {
                     cached[tableOp] = tableMetaData;
                     tableOp.setMetaAttr(relalg::TableMetaDataAttr::get(module.getContext(), std::make_shared<lingodb::runtime::TableMetaData>()));
                  }
               }
            });
            mlir::PassManager pm(module->getContext());
            auto fileName = snapshotDir.getValue() + "/detailed-snapshot-" + std::to_string(id) + ".mlir";
            pm.addPass(mlir::createLocationSnapshot(mlir::LocationSnapshotOptions{.fileName=fileName,.tag={},.enableDebugInfo=true,.printGenericOpForm=false,.printPrettyDebugInfo=false}));
            serializationState->passes.push_back({fileName, pass->getName().str(), pass->getDescription().str(), pass->getArgument().str()});
            if (pm.run(module).failed()) {
               llvm::dbgs() << "Failed to run location snapshot pass\n";
            }
            for (auto [op, attr] : cached) {
               mlir::cast<relalg::BaseTableOp>(op).setMetaAttr(attr);
            }
            std::ofstream file(snapshotDir.getValue() + "/detailed-snapshot-info.json");
            nlohmann::json j;
            for (auto id = 0ull; id < serializationState->passes.size(); id++) {
               auto [fileName, passName, passDescription, passArgument] = serializationState->passes[id];
               j.push_back({{"name", passName}, {"description", passDescription}, {"argument", passArgument}, {"file", fileName}});
            }
            file << j.dump(4) << std::endl;
         }
      }
   }
};
class TracePass : public mlir::PassInstrumentation {
   std::unique_ptr<utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>> trace;

   public:
   virtual void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override {
      if (pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
         return;
      }
      trace = std::make_unique<utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>>(runPass, pass->getArgument().str());
   }
   virtual void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
      if (trace)
         trace->stop();
   }
};
} // end anonymous namespace

void lingodb::execution::addLingoDBInstrumentation(mlir::PassManager& pm, std::shared_ptr<SnapshotState> serializationState) {
   if (snapshotSetting.getValue() && snapshotLevel.getValue() == "full") {
      pm.addInstrumentation(std::make_unique<SnapshotAfterEachPass>(serializationState));
   }
#ifdef TRACER
   pm.addInstrumentation(std::make_unique<TracePass>());
#endif
}

void lingodb::execution::snapshotImportantStep(std::string shortName, mlir::ModuleOp& moduleOp, std::shared_ptr<SnapshotState> serializationState) {
   if (snapshotSetting.getValue() && snapshotLevel.getValue() == "important") {
      llvm::DenseMap<mlir::Operation*, relalg::TableMetaDataAttr> cached;
      moduleOp.walk([&](relalg::BaseTableOp tableOp) {
         if (auto tableMetaData = tableOp.getMetaAttr()) {
            if (tableMetaData.getMeta()->isPresent()) {
               cached[tableOp] = tableMetaData;
               tableOp.setMetaAttr(relalg::TableMetaDataAttr::get(moduleOp.getContext(), std::make_shared<runtime::TableMetaData>()));
            }
         }
      });
      mlir::PassManager pm(moduleOp->getContext());
      auto fileName = snapshotDir.getValue() + "/important-snapshot-" + shortName + ".mlir";
      pm.addPass(mlir::createLocationSnapshot(mlir::LocationSnapshotOptions{.fileName=fileName,.tag={},.enableDebugInfo=true,.printGenericOpForm=false,.printPrettyDebugInfo=false}));
      serializationState->passes.push_back({fileName, "", "", ""});
      if (pm.run(moduleOp).failed()) {
         llvm::dbgs() << "Failed to run location snapshot pass\n";
      }
      for (auto [op, attr] : cached) {
         mlir::cast<relalg::BaseTableOp>(op).setMetaAttr(attr);
      }
      std::ofstream file(snapshotDir.getValue() + "/important-snapshot-info.json");
      nlohmann::json j;
      for (auto id = 0ull; id < serializationState->passes.size(); id++) {
         auto [fileName, passName, passDescription, passArgument] = serializationState->passes[id];
         j.push_back({{"name", passName}, {"description", passDescription}, {"argument", passArgument}, {"file", fileName}});
      }
      file << j.dump(4) << std::endl;
   }
}

std::string lingodb::execution::getSnapshotDir() {
   return snapshotDir.getValue();
}