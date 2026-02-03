#include "json.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/helper.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <queue>
#include <string>

#define DEBUG_TYPE "subop-sip"
namespace {
using namespace lingodb::compiler::dialect;

class SIPPass : public mlir::PassWrapper<SIPPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SIPPass)
   virtual llvm::StringRef getArgument() const override { return "subop-sip-pass"; }

   // Generic helper to identify build/probe sides of a hash join
   struct HashJoinInfo {
      subop::ScanOp buildSideScan; // Root scan/table operation for build side
      mlir::Value probeSideRoot; // Root scan/table operation for probe side
      subop::CreateHashIndexedView hashView; // The hash indexed view
      subop::LookupOp lookupOp; // The lookup operation
      llvm::SmallVector<std::string> buildKeyColumnNames; // Columns used for hash key on build side
      llvm::SmallVector<std::string> probeKeyColumnsNames; // Columns used for hash key on probe side
      subop::GetExternalOp externalProbeOp;
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
         auto* current = queue.front();
         queue.pop();
         if (debug) {
            std::cerr << "Current " << std::endl;
            current->dump();
         }
         if (auto scan = mlir::dyn_cast_or_null<subop::ScanOp>(current)) {
            std::vector<subop::Member> members;
            members.reserve(keys.size());
            // For ScanOp, check if any of the keys match the columns defined by the scan, to decide if we have the correct scan operator
            for (auto mappingAttr : scan.getMapping().getMappingList()) {
               auto defAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(mappingAttr.second);
               if (defAttr) {
                  // Check if this column definition matches any of the keys
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
      bool isValid = true;
      subop::MapOp m;
      for (auto* user : users) {
         if (auto mat = mlir::dyn_cast_or_null<subop::MaterializeOp>(user)) {
            mlir::Value streamBeforeMat = mat->getOperand(0);
            std::vector<tuples::ColumnRefAttr> buildKeyColumns;
            if (auto mapOp = mlir::dyn_cast_or_null<subop::MapOp>(streamBeforeMat.getDefiningOp())) {
               mapOp.getRegion().walk([&](db::Hash hashOp) {
                  if (hashOp.getVal().size() > 1) {
                     isValid = false;
                     m = mapOp;
                  }
               });

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
            } else if (!isValid) {
               return {nullptr, {}};
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
      bool isAntiJoin = false;
      bool hasMarkerState = false;
      bool markerIsFiltered = false;

      for (auto* user : lookupOp->getUsers()) {
         if (auto nestedMap = mlir::dyn_cast_or_null<subop::NestedMapOp>(user)) {
            //Check if is anti join
            nestedMap.walk([&](subop::FilterOp filter) {
               if (filter.getFilterSemantic() == subop::FilterSemantic::none_true) {
                  isAntiJoin = true;
               }
            });

            // Check if markers are created
            nestedMap.walk([&](subop::CreateSimpleStateOp createState) {
               hasMarkerState = true;
            });

            // Walk the nested_map to find if the marker is filtered after the scan, if not we cannot use SIP
            if (hasMarkerState) {
               nestedMap.walk([&](subop::ScanOp scanOp) {
                  if (auto stateType = mlir::dyn_cast_or_null<subop::SimpleStateType>(scanOp.getState().getType())) {
                     for (auto* scanUser : scanOp->getUsers()) {
                        if (auto filterOp = mlir::dyn_cast_or_null<subop::FilterOp>(scanUser)) {
                           markerIsFiltered = true;
                           break;
                        }
                     }
                  }
               });
            }
         }
      }
      //AntiJoins cannot use current SIP implementions
      if (isAntiJoin) {
         return std::nullopt;
      }
      // Correlated subqueries (with outer references in predicates) cannot use SIP
      if (hasMarkerState && !markerIsFiltered) {
         return std::nullopt;
      }

      // For mark joins (EXISTS), only apply SIP if the marker is actually used for filtering
      // For regular joins, always apply SIP
      return HashJoinInfo{
         .buildSideScan = mlir::dyn_cast_or_null<subop::ScanOp>(buildScan),
         .probeSideRoot = probeScan->getResult(0),
         .hashView = createHashView,
         .lookupOp = lookupOp,
         .buildKeyColumnNames = buildKeyNames,
         .probeKeyColumnsNames = probeKeyColumnsNames,
         .externalProbeOp = externalOp,
      };
   }
   std::string genRandom(const int len) {
      static const char alphanum[] =
         "0123456789"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "abcdefghijklmnopqrstuvwxyz";
      std::string tmpS;
      tmpS.reserve(len);

      for (int i = 0; i < len; ++i) {
         tmpS += alphanum[rand() % (sizeof(alphanum) - 1)];
      }

      return tmpS;
   }

   protected:
   void runOnOperation() override {
      std::optional<HashJoinInfo> joinInfo;
      getOperation()->walk([&](subop::LookupOp lookupOp) {
         joinInfo = identifyHashJoinTables(lookupOp, std::getenv("LINGODB_SIP_DEBUG"));
         if (joinInfo) {
#if 0
            llvm::dbgs() << "Hash Join Found:\n";
            llvm::dbgs() << "  Build Side Root: " << joinInfo->buildSideScan << "\n";
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
            mlir::Location loc = joinInfo->hashView->getLoc();
            mlir::OpBuilder b(joinInfo->hashView);
            b.setInsertionPointAfter(joinInfo->hashView);
            // Create a SIP filter named "test" for testing purposes.

            std::string descrRaw = joinInfo->externalProbeOp.getDescr().str();
            auto externalDataSourceProp = lingodb::utility::deserializeFromHexString<lingodb::runtime::ExternalDatasourceProperty>(descrRaw);
            std::string sipName = genRandom(10);
            auto probeColRef = joinInfo->probeKeyColumnsNames[0];
            auto buildColRef = joinInfo->buildKeyColumnNames[0];
            static int64_t count = 0;
            count++;
            lingodb::runtime::FilterDescription filterDesc{.columnName = probeColRef, .op = lingodb::runtime::FilterOp::SIP, .value = count};

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

            if (auto tableType = mlir::dyn_cast<subop::TableType>(joinInfo->externalProbeOp.getResult().getType())) {
               auto filteredTableType = subop::TableType::get(tableType.getContext(), tableType.getMembers(), true);
               joinInfo->externalProbeOp.getResult().setType(filteredTableType);
               joinInfo->externalProbeOp.setDescr(b.getStringAttr(updatedDescr));
            }

            auto sipFilter = b.create<subop::CreateSIPFilterOp>(loc, joinInfo->hashView.getResult().getType(), joinInfo->hashView.getResult(), b.getI8IntegerAttr(count));
            // Update lookup to use the SIP-filtered view, ensuring the filter is built before lookup execution
            joinInfo->lookupOp.setOperand(1, sipFilter.getResult());
         }
      });
   }
};

} // namespace
std::unique_ptr<mlir::Pass>
subop::createSIPPass() { return std::make_unique<SIPPass>(); }
