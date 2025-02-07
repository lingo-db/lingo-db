#include "json.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
namespace {
using namespace lingodb::compiler::dialect;

class GlobalOptPass : public mlir::PassWrapper<GlobalOptPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GlobalOptPass)
   virtual llvm::StringRef getArgument() const override { return "subop-global-opt"; }
   static std::optional<std::string> fingerprint(mlir::Operation* op, std::unordered_map<tuples::Column*, std::string>& columnFingerPrints) {
      if (!op) return {};
      return llvm::TypeSwitch<mlir::Operation*, std::optional<std::string>>(op)
         //todo: rework
         .Case([&](tuples::GetColumnOp getColumnOp) -> std::optional<std::string> {
            if (!columnFingerPrints.contains(&getColumnOp.getAttr().getColumn())) return {};
            return columnFingerPrints.at(&getColumnOp.getAttr().getColumn());
         })
         .Default([&](mlir::Operation* op) -> std::optional<std::string> {
            std::string result = "{";
            llvm::raw_string_ostream nameStream(result);
            op->getName().print(nameStream);
            result += "(";
            for (auto operand : op->getOperands()) {
               if (auto* opDef = operand.getDefiningOp()) {
                  auto operandFp = fingerprint(opDef, columnFingerPrints);
                  if (operandFp) {
                     result += operandFp.value() + ",";
                  } else {
                     return {};
                  }
               } else {
                  return {};
               }
            }
            result += ") with ";
            llvm::raw_string_ostream attributeStringStream(result);
            op->getAttrDictionary().print(attributeStringStream);
            result += "}";
            return result;
         });
   }
   static void handleScanUserRec(mlir::Operation* op, std::unordered_map<tuples::Column*, std::string>& columnFingerPrints, std::vector<subop::MaterializeOp>& materializeOps,
                                 std::vector<subop::FilterOp>& filterOps) {
      llvm::TypeSwitch<mlir::Operation*>(op)
         .Case([&](subop::RenamingOp renamingOp) {
            for (auto computed : renamingOp.getColumns()) {
               auto colDefAttr = mlir::cast<tuples::ColumnDefAttr>(computed);
               auto* colDefAttrCol = &colDefAttr.getColumn();
               auto* colRefAttrCol = &mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(colDefAttr.getFromExisting())[0]).getColumn();
               auto fp = columnFingerPrints.contains(colRefAttrCol);
               if (fp) {
                  columnFingerPrints.insert({colDefAttrCol, columnFingerPrints.at(colRefAttrCol)});
               }
            }
            for (auto* user : renamingOp->getUsers()) {
               handleScanUserRec(user, columnFingerPrints, materializeOps, filterOps);
            }
         })
         .Case([&](subop::MapOp mapOp) {
            auto returnOp = mlir::cast<tuples::ReturnOp>(mapOp.getFn().front().getTerminator());
            for (auto computed : llvm::zip(mapOp.getComputedCols(), returnOp.getResults())) {
               auto* col = &(mlir::cast<tuples::ColumnDefAttr>(std::get<0>(computed)).getColumn());
               auto* op = std::get<1>(computed).getDefiningOp();
               auto fp = fingerprint(op, columnFingerPrints);
               if (fp) {
                  columnFingerPrints.insert({col, fp.value()});
               }
            }
            for (auto* user : mapOp->getUsers()) {
               handleScanUserRec(user, columnFingerPrints, materializeOps, filterOps);
            }
         })
         .Case([&](subop::FilterOp filterOp) {
            filterOps.push_back(filterOp);
            for (auto* user : filterOp->getUsers()) {
               handleScanUserRec(user, columnFingerPrints, materializeOps, filterOps);
            }
         })
         .Case([&](subop::MaterializeOp materializeOp) {
            materializeOps.push_back(materializeOp);
         });
   }

   void runOnOperation() override {
      auto subOpDependencyAnalysis = getAnalysis<subop::SubOpDependencyAnalysis>();
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();
      std::unordered_map<std::string, std::vector<subop::GetExternalOp>> externalOpByTableName;
      getOperation().walk([&](subop::GetExternalOp op) {
         if (auto tableType = mlir::dyn_cast_or_null<subop::TableType>(op.getType())) {
            auto json = nlohmann::json::parse(op.getDescr().str());
            externalOpByTableName[json["table"]].push_back(op);
         }
      });
      mlir::OpBuilder builder(&getContext());
      for (auto t : externalOpByTableName) {
         auto first = t.second[0];
         auto firstJson = nlohmann::json::parse(first.getDescr().str());
         std::unordered_map<std::string, std::string> columnToFirstMember;
         for (auto m : firstJson["mapping"].get<nlohmann::json::object_t>()) {
            columnToFirstMember.insert({m.second.get<std::string>(), m.first});
         }
         for (size_t i = 1; i < t.second.size(); i++) {
            auto other = t.second[i];
            auto otherJson = nlohmann::json::parse(other.getDescr().str());
            std::unordered_map<std::string, std::string> otherMemberToFirstMember;
            for (auto m : otherJson["mapping"].get<nlohmann::json::object_t>()) {
               otherMemberToFirstMember.insert({m.first, columnToFirstMember.at(m.second.get<std::string>())});
            }
            if (other->getBlock() != first->getBlock()) continue;
            std::vector<mlir::Value> replaceUses;
            for (auto* user : other->getUsers()) {
               if (auto scanOp = mlir::dyn_cast_or_null<subop::ScanOp>(user)) {
                  std::vector<mlir::NamedAttribute> newMapping;
                  for (auto m : scanOp.getMapping()) {
                     auto newMember = otherMemberToFirstMember.at(m.getName().str());
                     newMapping.push_back(builder.getNamedAttr(newMember, m.getValue()));
                  }
                  scanOp.setMappingAttr(builder.getDictionaryAttr(newMapping));
                  replaceUses.push_back(scanOp.getState());
               }
            }
            for (auto r : replaceUses) {
               r.replaceAllUsesWith(first);
            }
         }
      }
      auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      std::unordered_map<mlir::Operation*, std::vector<subop::ScanOp>> scanOpsByState;
      getOperation().walk([&](subop::ScanOp op) {
         if (auto* stateCreationOp = op.getState().getDefiningOp()) {
            scanOpsByState[stateCreationOp].push_back(op);
         }
      });
      std::unordered_set<mlir::Operation*> toDelete;
      for (auto t : scanOpsByState) {
         auto first = t.second[0];
         std::unordered_map<std::string, mlir::Attribute> mapping;
         for (auto m : first.getMapping()) {
            mapping.insert({m.getName().str(), m.getValue()});
         }

         for (size_t i = 1; i < t.second.size(); i++) {
            auto other = t.second[i];
            if (subOpDependencyAnalysis.areIndependent(first.getOperation(), other.getOperation()) && first->getBlock() == other->getBlock()) {
               std::vector<mlir::Attribute> mappedCols;

               for (auto x : other.getMapping()) {
                  if (mapping.contains(x.getName().str())) {
                     auto fromRef = colManager.createRef(&mlir::cast<tuples::ColumnDefAttr>(mapping.at(x.getName().str())).getColumn());
                     auto toDef = colManager.createDef(&mlir::cast<tuples::ColumnDefAttr>(x.getValue()).getColumn(), mlir::ArrayAttr::get(&getContext(), fromRef));
                     mappedCols.push_back(toDef);
                  } else {
                     mapping.insert({x.getName().str(), x.getValue()});
                  }
               }
               mlir::Value newScan = first;
               if (!mappedCols.empty()) {
                  mlir::OpBuilder builder(other.getOperation());
                  auto mapOp = builder.create<subop::RenamingOp>(other.getLoc(), newScan, builder.getArrayAttr(mappedCols));
                  newScan = mapOp.getResult();
               }
               other.getRes().replaceAllUsesWith(newScan);
               subOpDependencyAnalysis.addToRoot(first, other);
               toDelete.insert(other.getOperation());
            }
         }
         std::vector<mlir::NamedAttribute> newMapping;
         for (auto m : mapping) {
            newMapping.push_back(builder.getNamedAttr(m.first, m.second));
         }
         first.setMappingAttr(builder.getDictionaryAttr(newMapping));
      }
      for(auto* op : toDelete) {
         op->erase();
      }
      std::unordered_map<std::string, std::unordered_set<mlir::Operation*>> writtenToMember;
      getOperation().walk([&](subop::SubOperator op) {
         for (auto writtenMember : op.getWrittenMembers()) {
            writtenToMember[writtenMember].insert(op);
         }
      });
      auto isWrittenTo = [&](std::string member, mlir::Operation* excl) {
         if (writtenToMember.contains(member)) {
            for (auto* op : writtenToMember[member]) {
               if (excl != op) {
                  return true;
               }
            }
         }
         return false;
      };
      getOperation().walk([&](subop::ScanOp op) {
         std::unordered_map<tuples::Column*, std::string> columnFingerPrints;
         for (auto m : op.getMapping()) {
            auto* col = &mlir::cast<tuples::ColumnDefAttr>(m.getValue()).getColumn();
            columnFingerPrints.insert({col, m.getName().str()});
         }
         std::unordered_map<mlir::Operation*, std::vector<subop::MaterializeOp>> materializeOps;
         std::unordered_map<mlir::Operation*, std::vector<subop::FilterOp>> filterOps;
         std::vector<mlir::Operation*> relevantUsers;
         for (auto* user : op->getUsers()) {
            handleScanUserRec(user, columnFingerPrints, materializeOps[user], filterOps[user]);
            if (materializeOps[user].size() == 1 && mlir::isa<subop::BufferType, subop::MultiMapType>(materializeOps[user][0].getState().getType())) {
               relevantUsers.push_back(user);
            }
         }
         std::sort(relevantUsers.begin(), relevantUsers.end(), [&](mlir::Operation* left, mlir::Operation* right) {
            return materializeOps[left][0]->isBeforeInBlock(materializeOps[right][0]);
         });
         if (relevantUsers.size() > 1) {
            auto* first = relevantUsers.front();
            std::unordered_map<std::string, std::string> fingerprintMapping;
            auto firstState = materializeOps[first][0].getState();
            for (auto c : materializeOps[first][0].getMapping()) {
               auto* col = &mlir::cast<tuples::ColumnRefAttr>(c.getValue()).getColumn();
               if (columnFingerPrints.contains(col)) {
                  fingerprintMapping.insert({columnFingerPrints[col], c.getName().str()});
               } else {
                  return;
               }
            }
            std::unordered_set<std::string> firstFilters;
            for (auto f : filterOps[first]) {
               for (auto p : f.getConditions()) {
                  auto* col = &mlir::cast<tuples::ColumnRefAttr>(p).getColumn();
                  if (columnFingerPrints.contains(col)) {
                     firstFilters.insert(columnFingerPrints[col]);
                  } else {
                     return;
                  }
               }
            }

            for (size_t i = 1; i < relevantUsers.size(); i++) {
               auto* other = relevantUsers[i];
               std::vector<std::string> extraMembers;
               bool failed = false;
               auto otherState = materializeOps[other][0].getState();

               for (auto c : materializeOps[other][0].getMapping()) {
                  auto* col = &mlir::cast<tuples::ColumnRefAttr>(c.getValue()).getColumn();
                  if (columnFingerPrints.contains(col)) {
                     auto fingerprint = columnFingerPrints[col];
                     if (!fingerprintMapping.contains(fingerprint) || isWrittenTo(c.getName().str(), materializeOps[other][0].getOperation())) {
                        extraMembers.push_back(c.getName().str());
                     }
                  } else {
                     failed = true;
                     break;
                  }
               }
               std::unordered_set<std::string> currentFilters;
               for (auto f : filterOps[other]) {
                  for (auto p : f.getConditions()) {
                     auto* col = &mlir::cast<tuples::ColumnRefAttr>(p).getColumn();
                     if (columnFingerPrints.contains(col)) {
                        currentFilters.insert(columnFingerPrints[col]);
                     } else {
                        failed = true;
                        break;
                     }
                  }
               }
               failed |= firstFilters.size() != currentFilters.size();
               for (auto cf : currentFilters) {
                  failed |= !firstFilters.contains(cf);
               }
               if (failed) continue;
               if (!extraMembers.empty()) continue;
               std::unordered_map<std::string, std::string> memberMapping;
               for (auto c : materializeOps[other][0].getMapping()) {
                  memberMapping.insert({c.getName().str(), fingerprintMapping.at(columnFingerPrints.at(&mlir::cast<tuples::ColumnRefAttr>(c.getValue()).getColumn()))});
               }
               if (auto firstMultiMapType = mlir::dyn_cast_or_null<subop::MultiMapType>(firstState.getType())) {
                  if (auto otherMultiMapType = mlir::dyn_cast_or_null<subop::MultiMapType>(otherState.getType())) {
                     bool keysEqual = true;
                     if (firstMultiMapType.getKeyMembers().getNames().size() != otherMultiMapType.getKeyMembers().getNames().size()) {
                        continue;
                     }
                     for (auto z : llvm::zip(firstMultiMapType.getKeyMembers().getNames(), otherMultiMapType.getKeyMembers().getNames())) {
                        auto firstKey = mlir::cast<mlir::StringAttr>(std::get<0>(z)).str();
                        auto otherKey = mlir::cast<mlir::StringAttr>(std::get<1>(z)).str();
                        keysEqual |= memberMapping.at(otherKey) == firstKey;
                     }
                     if (keysEqual) {
                        subop::SubOpStateUsageTransformer transformer(columnUsageAnalysis, &getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
                           if (auto listType = mlir::dyn_cast_or_null<subop::ListType>(type)) {
                              if (mlir::isa<subop::MultiMapEntryRefType>(listType.getT())) {
                                 return subop::ListType::get(&getContext(), subop::MultiMapEntryRefType::get(&getContext(), firstMultiMapType));
                              }
                           }
                           if (mlir::isa<subop::MultiMapEntryRefType>(type)) {
                              return subop::MultiMapEntryRefType::get(&getContext(), firstMultiMapType);
                           }
                           assert(false && "not supported yet");
                           return mlir::Type();
                        });
                        transformer.mapMembers(memberMapping);
                        materializeOps[other][0].erase(); //todo: only now, because filters do match -> same tuples
                        transformer.updateValue(otherState, firstState.getType());
                        otherState.replaceAllUsesWith(firstState);
                     }
                  } else {
                     continue;
                  }

               } else {
                  //todo: implement for buffer
                  continue;
               }
            }
         }
      });

      getOperation().walk([&](subop::ScanOp op) {
         std::unordered_map<tuples::Column*, std::string> columnFingerPrints;
         for (auto m : op.getMapping()) {
            auto* col = &mlir::cast<tuples::ColumnDefAttr>(m.getValue()).getColumn();
            columnFingerPrints.insert({col, m.getName().str()});
         }
         std::unordered_map<mlir::Operation*, std::vector<subop::MaterializeOp>> materializeOps;
         std::unordered_map<mlir::Operation*, std::vector<subop::FilterOp>> filterOps;
         std::unordered_map<std::string, std::vector<mlir::Operation*>> operationsByCommonFilters;
         for (auto* user : op->getUsers()) {
            handleScanUserRec(user, columnFingerPrints, materializeOps[user], filterOps[user]);
            std::string filterFingerPrint;
            bool failed = false;
            for (auto f : filterOps[user]) {
               for (auto p : f.getConditions()) {
                  auto* col = &mlir::cast<tuples::ColumnRefAttr>(p).getColumn();
                  if (columnFingerPrints.contains(col)) {
                     filterFingerPrint += columnFingerPrints[col];
                  } else {
                     failed = true;
                  }
               }
            }
            if (!failed && !filterFingerPrint.empty()) {
               operationsByCommonFilters[filterFingerPrint].push_back(user);
            }
         }
         for (auto& group : operationsByCommonFilters) {
            if (group.second.size() < 2) continue;
            //llvm::dbgs() << group.first << "\n";
            std::sort(group.second.begin(), group.second.end(), [&](mlir::Operation* left, mlir::Operation* right) {
               return filterOps[left][filterOps[left].size() - 1]->isBeforeInBlock(filterOps[right][filterOps[right].size() - 1]);
            });
            for (size_t i = 1; i < group.second.size(); i++) {
               auto replaceWith = filterOps[group.second[0]][filterOps[group.second[0]].size() - 1];
               op.getRes().replaceUsesWithIf(replaceWith, [&](auto& use) { return use.getOwner() == group.second[i]; });
               mlir::Operation* moveAfter = replaceWith.getOperation();
               mlir::Operation* current = group.second[i];
               while (current && current->isBeforeInBlock(moveAfter)) {
                  current->moveAfter(moveAfter);
                  moveAfter = current;
                  current = current->getUsers().empty() ? nullptr : *current->getUsers().begin();
               }
               for (auto filterOp : filterOps[group.second[i]]) {
                  filterOp.replaceAllUsesWith(filterOp.getStream());
               }
            }
         }
      });

      //Q28
      static size_t uniqueId = 0;
      getOperation().walk([&](subop::ScanOp scanOp) {
         std::unordered_map<std::string, std::tuple<size_t, double, std::vector<mlir::Operation*>>> groupByMembers;
         std::unordered_map<mlir::Operation*, size_t> groupForOp;
         std::unordered_map<tuples::Column*, std::string> columnFingerPrints;
         for (auto m : scanOp.getMapping()) {
            auto* col = &mlir::cast<tuples::ColumnDefAttr>(m.getValue()).getColumn();
            columnFingerPrints.insert({col, m.getName().str()});
         }
         std::unordered_map<mlir::Operation*, std::vector<subop::MaterializeOp>> materializeOps;
         std::unordered_map<mlir::Operation*, std::vector<subop::FilterOp>> filterOps;
         std::unordered_map<std::string, std::vector<mlir::Operation*>> operationsByCommonFilters;
         size_t counter = 0;
         for (auto* user : scanOp->getUsers()) {
            handleScanUserRec(user, columnFingerPrints, materializeOps[user], filterOps[user]);
            std::string filterFingerPrint;
            bool failed = false;
            if (filterOps[user].empty()) continue;
            auto f = filterOps[user][0];
            if (!f->hasAttr("selectivity")) continue;
            double sel = f->getAttrOfType<mlir::FloatAttr>("selectivity").getValueAsDouble();
            for (auto p : f.getConditions()) {
               auto* col = &mlir::cast<tuples::ColumnRefAttr>(p).getColumn();
               if (columnFingerPrints.contains(col)) {
                  filterFingerPrint += columnFingerPrints[col];
               } else {
                  failed = true;
               }
            }
            if (failed) continue;
            std::string requiredMembers;
            for (auto m : scanOp.getMapping()) {
               auto members = m.getName().str();
               if (filterFingerPrint.find(members) != std::string::npos) {
                  requiredMembers += "," + members;
               }
            }
            if (requiredMembers.empty()) continue;
            while (true) {
               if (groupByMembers.contains(requiredMembers)) {
                  auto& entry = groupByMembers[requiredMembers];
                  if (std::get<1>(entry) + sel < 0.3) {
                     std::get<2>(entry).push_back(user);
                     break;
                  }
               } else {
                  groupByMembers.insert({requiredMembers, {counter++, sel, {user}}});
                  break;
               }
               requiredMembers += "//";
            }
         }
         bool first = true;

         for (auto& e : groupByMembers) {
            if (first) {
               first = false;
               //no rewriting necessary
            } else {
               auto& groupUsers = std::get<2>(e.second);
               std::sort(groupUsers.begin(), groupUsers.end(), [&](mlir::Operation* left, mlir::Operation* right) {
                  return left->isBeforeInBlock(right);
               });
               auto *firstGroupUser = groupUsers[0];
               mlir::OpBuilder builder(firstGroupUser->getContext());
               builder.setInsertionPoint(firstGroupUser);
               auto* cloned = builder.clone(*scanOp);
               cloned->setAttr("scanNr" + std::to_string(uniqueId++), builder.getUnitAttr());
               auto clonedScan = mlir::cast<subop::ScanOp>(cloned);

               for (auto* op : std::get<2>(e.second)) {
                  scanOp.getRes().replaceUsesWithIf(clonedScan.getRes(), [&](auto& use) { return use.getOwner() == op; });
               }
            }
         }
      });

   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createGlobalOptPass() { return std::make_unique<GlobalOptPass>(); }