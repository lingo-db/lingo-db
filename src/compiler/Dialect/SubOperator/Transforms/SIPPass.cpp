#include "json.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/helper.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/Support/Debug.h>
#include <algorithm>
#include <lingodb/compiler/Dialect/DB/IR/DBOps.h.inc>
#include <queue>
#include <string>
#include <mlir/Dialect/Arith/IR/Arith.h>

#define DEBUG_TYPE "subop-sip"
namespace {
using namespace lingodb::compiler::dialect;

static inline std::string trim(const std::string& s) {
   size_t a = s.find_first_not_of(" \t\n\r");
   if (a == std::string::npos) return "";
   size_t b = s.find_last_not_of(" \t\n\r");
   return s.substr(a, b - a + 1);
}

class SIPPass : public mlir::PassWrapper<SIPPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SIPPass)
   virtual llvm::StringRef getArgument() const override { return "subop-sip-pass"; }

   // Generic helper to identify build/probe sides of a hash join
   struct HashJoinInfo {
      mlir::Value buildSideRoot; // Root scan/table operation for build side
      mlir::Value probeSideRoot; // Root scan/table operation for probe side
      subop::CreateHashIndexedView hashView; // The hash indexed view
      subop::LookupOp lookupOp; // The lookup operation
      llvm::SmallVector<std::string> buildKeyColumnNames; // Columns used for hash key on build side
      llvm::SmallVector<std::string> probeKeyColumnsNames; // Columns used for hash key on probe side
      subop::GetExternalOp externalProbeOp;
      bool isPositiveJoin; // True for IN/all_true joins, false for NOT IN/none_true joins
   };

   std::pair<mlir::Operation*, std::vector<subop::Member>> walkToFindSourceScan(mlir::Operation* op, std::vector<tuples::ColumnRefAttr> keys, bool debug = false) {
      /* mlir::Operation* current = op;
      while (current->getNumOperands() > 0) {
         if (debug) {
            if (debug) {
               std::cerr << "Walking op: " << current->getName().getStringRef().str() << std::endl;
               current->dump();
            }
         }
         if (auto scan = mlir::dyn_cast_or_null<subop::ScanOp>(current)) {
            return scan;
         } else if (auto mat = mlir::dyn_cast_or_null<subop::MaterializeOp>(current)) {
            current = mat->getOperand(0).getDefiningOp();
         } else if (auto lookup = mlir::dyn_cast_or_null<subop::LookupOp>(current)) {
            if (debug)
               current->dump();
            current = lookup->getOperand(0).getDefiningOp();
            return nullptr;
         } else {
            if (current->getNumOperands() > 1) {
               if (debug) {
                  std::cerr << "should not happen" << std::endl;
                  std::cerr << "No scan for found last current: " << std::endl;
                  current->dump();
               }
            }
            current = current->getOperand(0).getDefiningOp();
         }
      }
      return nullptr;*/
      //Implement a breath first search using fifo que
      assert(op);
      assert(!keys.empty());
      if (debug) {
         std::cerr << "--------------walkToFindSourceScan--------------\n";
      }
      std::queue<mlir::Operation*> queue;
      queue.push(op);
      while (!queue.empty()) {
         auto current = queue.front();
         queue.pop();
         if (debug) {
            std::cerr << "Current " << std::endl;
            current->dump();
         }
         if (auto scan = mlir::dyn_cast_or_null<subop::ScanOp>(current)) {
            std::vector<subop::Member> members;
            members.reserve(keys.size());
            bool found = false;
            // For ScanOp, check if any of the keys match the columns defined by the scan, to decide if we have the correct scan operator
            for (auto mappingAttr : scan.getMapping().getMappingList()) {
               auto defAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(mappingAttr.second);
               if (defAttr) {
                  // Check if this column definition matches any of the keys
                  //TODO is this correct
                  for (const auto& key : keys) {
                     if (&defAttr.getColumn() == &key.getColumn()) {
                        members.push_back(mappingAttr.first);
                        break;
                     }
                  }
               }
            }
            if (members.empty()) {
               if (debug)
                  std::cerr << "No matching key found\n";
               continue;
            }

            if (debug) {
               std::cerr << "Found: \n";
               scan.dump();
               std::cerr << "--------------walkToFindSourceScan--------------\n";
            }
            return {scan, members};
         }
         std::vector<mlir::Operation*> toAdd;
         for (auto operand : current->getOperands()) {
            if (auto* defOp = operand.getDefiningOp()) {
               queue.push(defOp);
            }
         }
      }

      if (debug) {
         std::cerr << "None found \n";
         std::cerr << "--------------walkToFindSourceScan--------------\n";
      }

      return {nullptr, {}};
   }

   //Finds source scan for the build side
   std::pair<mlir::Operation*, std::vector<subop::Member>> findSourceScanFromCreateHashIndexedView(subop::CreateHashIndexedView createHashIndexedView, bool debug) {
      auto users = createHashIndexedView.getOperand().getUsers();
      for (auto* user : users) {
         if (auto mat = mlir::dyn_cast_or_null<subop::MaterializeOp>(user)) {
            mlir::Value streamBeforeMat = mat->getOperand(0);
            std::vector<tuples::ColumnRefAttr> buildKeyColumns;
            if (auto mapOp = mlir::dyn_cast_or_null<subop::MapOp>(streamBeforeMat.getDefiningOp())) {
               // Extract the input columns attribute from the map operation
               auto inputAttr = mapOp.getInputColsAttr();
               for (auto input : inputAttr) {
                  if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(input)) {
                     buildKeyColumns.push_back(colRef);
                  }
               }
            }

            if (debug) {
               std::cerr << "--------------findSourceScanFromCreateHashIndexedView--------------\n";
               std::cerr << "find table scan for: " << std::endl;
               mat->dump();
               std::cerr << " using keys:\n";
               for (auto keyCol : buildKeyColumns) {
                  keyCol.dump();
               }
               std::cerr << "-------------findSourceScanFromCreateHashIndexedView---------------\n";
            }
            if (buildKeyColumns.size() > 1) {
               std::cerr << "Unsupported key size: " << buildKeyColumns.size() << std::endl;
            }
            return walkToFindSourceScan(mat, buildKeyColumns, debug);
         }
      }

      return {nullptr, {}};
   }

   //Finds source scan for the probe side
   std::pair<mlir::Operation*, std::vector<subop::Member>> findSourceScanFromMap(mlir::Operation* op, bool debug) {
      if (debug) {
         std::cerr << "--------------findSourceScanFromMap--------------\n";
         std::cerr << "Search probe side scan with start: \n";
         op->dump();
      }
      if (auto map = mlir::dyn_cast_or_null<subop::MapOp>(op)) {
         std::vector<tuples::ColumnRefAttr> keys;
         for (auto attr : map.getInputColsAttr()) {
            if (debug) {
               std::cerr << "Using key column: \n";
               attr.dump();
            }
            if (auto columRef = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(attr)) {
               keys.push_back(columRef);
            }
         }
         return walkToFindSourceScan(map, keys, debug);
      } else {
         if (debug) {
            std::cerr << "Is no map at start\n";
         }
      }

      if (debug) {
         std::cerr << "--------------findSourceScanFromMap--------------\n";
      }

      return {nullptr, {}};
   }

   std::optional<HashJoinInfo> identifyHashJoinTables(subop::LookupOp lookupOp, bool debug = false) {
      mlir::Value stateValue = lookupOp.getState();
      auto* stateDefOp = stateValue.getDefiningOp();
      auto createHashView = mlir::dyn_cast_or_null<subop::CreateHashIndexedView>(stateDefOp);
      if (!createHashView) return std::nullopt;
      //First get Build/Input side
      auto [buildScan, buildKeyMembers] = findSourceScanFromCreateHashIndexedView(createHashView, debug);
      if (debug) {
         std::cerr << "found buildscan " << std::endl;
         if (buildScan)
            buildScan->dump();
         else
            std::cerr << "nullptr\n";
      }
      if (!buildScan) {
         return std::nullopt;
      }
      //Get probe side
      if (debug) {
         lookupOp.getStream().getDefiningOp()->dump();
      }
      auto [probeScan, probeKeyMembers] = findSourceScanFromMap(lookupOp.getStream().getDefiningOp(), debug);
      if (debug) {
         if (probeScan)
            probeScan->dump();
      }
      if (!probeScan) {
         return std::nullopt;
      }

      //Get get_external op from probe scan
      subop::GetExternalOp externalOp;
      if (auto scanOp = mlir::dyn_cast_or_null<subop::ScanOp>(probeScan)) {
         externalOp = mlir::dyn_cast_or_null<subop::GetExternalOp>(scanOp.getState().getDefiningOp());
         if (!externalOp) return std::nullopt;
      }

      // Extract key columns from build side: find the MapOp that feeds into materialize

      // Extract key columns from probe side: find the MapOp before lookup

      llvm::SmallVector<std::string> probeKeyColumnsNames;
      for (auto probeKeyMember : probeKeyMembers) {
         std::string name = probeKeyMember.internal->name;
         name.erase(name.end() - 2, name.end());
         probeKeyColumnsNames.push_back(name);
      }
      llvm::SmallVector<std::string> buildKeyNames;
      for (auto buildKeyMember : buildKeyMembers) {
         std::string name = buildKeyMember.internal->name;
         name.erase(name.end() - 2, name.end());
         buildKeyNames.push_back(name);
      }

      // Detect if this is a positive (IN/all_true) or negative (NOT IN/none_true) join
      //TODO detection may be hardly simplified
      //TODO einmal selber machen
      bool isPositiveJoin = true;
      for (auto* user : lookupOp->getUsers()) {
         if (auto nestedMap = mlir::dyn_cast_or_null<subop::NestedMapOp>(user)) {
            nestedMap.walk([&](subop::FilterOp filter) {
               if (filter.getFilterSemantic() == subop::FilterSemantic::none_true) {
                  isPositiveJoin = false;
               }
            });
         }
      }
      if (!isPositiveJoin) {
         return std::nullopt;
      }

      return HashJoinInfo{.buildSideRoot = buildScan->getResult(0),
                          .probeSideRoot = probeScan->getResult(0),
                          .hashView = createHashView,
                          .lookupOp = lookupOp,
                          .buildKeyColumnNames = buildKeyNames,
                          .probeKeyColumnsNames = probeKeyColumnsNames,
                          .externalProbeOp = externalOp,
                          .isPositiveJoin = isPositiveJoin};
   }
   std::string gen_random(const int len) {
      static const char alphanum[] =
         "0123456789"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "abcdefghijklmnopqrstuvwxyz";
      std::string tmp_s;
      tmp_s.reserve(len);

      for (int i = 0; i < len; ++i) {
         tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
      }

      return tmp_s;
   }
   void runOnOperation() override {
      auto& created = getAnalysis<subop::ColumnCreationAnalysis>();
      auto& used = getAnalysis<subop::ColumnUsageAnalysis>();
      std::optional<HashJoinInfo> joinInfo;
      getOperation()->walk([&](subop::LookupOp lookupOp) {
         static int count = 0;
         count++;

         joinInfo = identifyHashJoinTables(lookupOp, std::getenv("LINGODB_SIP_DEBUG"));
         if (joinInfo) {
#if 0
            llvm::dbgs() << "Hash Join Found:\n";
            llvm::dbgs() << "  Build Side Root: " << joinInfo->buildSideRoot << "\n";
            llvm::dbgs() << "  Build Side get external" << joinInfo->externalProbeOp << "\n";
            llvm::dbgs() << "  Probe Side Root: " << joinInfo->probeSideRoot << "\n";
            llvm::dbgs() << "  Hash View: " << joinInfo->probeKeyColumns.size() << "\n";
            llvm::dbgs() << "Build side key inputs: \n";
            for (auto& in : joinInfo->buildKeyColumns) {
               llvm::dbgs() << " - ";
               in.dump();
            }
            llvm::dbgs() << "Probe side key inputs: \n";
            for (auto& in : joinInfo->probeKeyColumns) {
               llvm::dbgs() << " - ";
               in.dump();
            }

            if (joinInfo->probeKeyColumns.size() != 1) {
               return;
            }
#endif
            if (joinInfo->probeKeyColumnsNames.size() != 1 || joinInfo->buildKeyColumnNames.size() != 1) {
               return;
            }

            //Do SIP
            auto module = getOperation();
            mlir::Location loc = joinInfo->hashView->getLoc();
            mlir::OpBuilder b(joinInfo->hashView);
            b.setInsertionPointAfter(joinInfo->hashView);
            // Create a SIP filter named "test" for testing purposes.

            std::string descrRaw = joinInfo->externalProbeOp.getDescr().str();
            auto externalDataSourceProp = lingodb::utility::deserializeFromHexString<lingodb::runtime::ExternalDatasourceProperty>(descrRaw);
            std::string sipName = gen_random(10);
            auto probeColRef = joinInfo->probeKeyColumnsNames[0];
            auto buildColRef = joinInfo->buildKeyColumnNames[0];

            lingodb::runtime::FilterDescription filterDesc{.columnName = probeColRef, .op = lingodb::runtime::FilterOp::SIP, .value = sipName};
            externalDataSourceProp.filterDescriptions.push_back(filterDesc);

            {
               if (std::getenv("LINGODB_SIP_DEBUG")) {
                  std::cerr << "----------------SIP Info----------------\n";
                  std::cerr << "Buildside: " << std::endl;
                  for (auto& in : joinInfo->buildKeyColumnNames) {
                     std::cerr << " - " << in;
                  }


                  std::cerr << "\n\nProbeSide: \n";
                  for (auto& in : joinInfo->probeKeyColumnsNames) {
                     std::cerr << " - " << in;
                  }
                  std::cerr << "\nSIP Name: " << sipName << std::endl;
                  std::cerr << "----------------SIP Info----------------\n";

                  std::cerr << "Probe: " << std::endl;
                  joinInfo->externalProbeOp->dump();
               }
            }

            //descr to string
            std::string updatedDescr = lingodb::utility::serializeToHexString(externalDataSourceProp);

            if (auto tableType = mlir::dyn_cast<subop::TableType>(joinInfo->externalProbeOp.getResult().getType()) ) {
               auto filteredTableType = subop::TableType::get(tableType.getContext(), tableType.getMembers(), true);
               joinInfo->externalProbeOp.getResult().setType(filteredTableType);
               joinInfo->externalProbeOp.setDescr(b.getStringAttr(updatedDescr));
            }

            b.create<subop::CreateSIPFilterOp>(loc, joinInfo->hashView.getResult(), b.getStringAttr(sipName));
         }
      });
   }
};

} // namespace
std::unique_ptr<mlir::Pass>
subop::createSIPPass() { return std::make_unique<SIPPass>(); }
