#include "json.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
namespace {

class GlobalOptPass : public mlir::PassWrapper<GlobalOptPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GlobalOptPass)
   virtual llvm::StringRef getArgument() const override { return "subop-global-opt"; }
   static std::optional<std::string> fingerprint(mlir::Operation* op, std::unordered_map<mlir::tuples::Column*, std::string>& columnFingerPrints) {
      if (!op) return {};
      return llvm::TypeSwitch<mlir::Operation*, std::optional<std::string>>(op)
         .Case([&](mlir::tuples::GetColumnOp getColumnOp) -> std::optional<std::string> {
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
   static void handleScanUserRec(mlir::Operation* op, std::unordered_map<mlir::tuples::Column*, std::string>& columnFingerPrints, std::vector<mlir::subop::MaterializeOp>& materializeOps,
                                 std::vector<mlir::subop::FilterOp>& filterOps) {
      llvm::TypeSwitch<mlir::Operation*>(op)
         .Case([&](mlir::subop::MapOp mapOp) {
            auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(mapOp.getFn().front().getTerminator());
            for (auto computed : llvm::zip(mapOp.getComputedCols(), returnOp.getResults())) {
               auto* col = &(std::get<0>(computed).cast<mlir::tuples::ColumnDefAttr>().getColumn());
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
         .Case([&](mlir::subop::FilterOp filterOp) {
            filterOps.push_back(filterOp);
            for (auto* user : filterOp->getUsers()) {
               handleScanUserRec(user, columnFingerPrints, materializeOps, filterOps);
            }
         })
         .Case([&](mlir::subop::MaterializeOp materializeOp) {
            materializeOps.push_back(materializeOp);
         });
   }

   void runOnOperation() override {
      auto subOpDependencyAnalysis = getAnalysis<mlir::subop::SubOpDependencyAnalysis>();
      auto columnUsageAnalysis = getAnalysis<mlir::subop::ColumnUsageAnalysis>();
      std::unordered_map<std::string, std::vector<mlir::subop::GetExternalOp>> externalOpByTableName;
      getOperation().walk([&](mlir::subop::GetExternalOp op) {
         if (auto tableType = op.getType().dyn_cast_or_null<mlir::subop::TableType>()) {
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
               if (auto scanOp = mlir::dyn_cast_or_null<mlir::subop::ScanOp>(user)) {
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
      auto& colManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      std::unordered_map<mlir::Operation*, std::vector<mlir::subop::ScanOp>> scanOpsByState;
      getOperation().walk([&](mlir::subop::ScanOp op) {
         if (auto* stateCreationOp = op.getState().getDefiningOp()) {
            scanOpsByState[stateCreationOp].push_back(op);
         }
      });
      for (auto t : scanOpsByState) {
         auto first = t.second[0];
         std::unordered_map<std::string, mlir::Attribute> mapping;
         for (auto m : first.getMapping()) {
            mapping.insert({m.getName().str(), m.getValue()});
         }

         for (size_t i = 1; i < t.second.size(); i++) {
            auto other = t.second[i];
            if (subOpDependencyAnalysis.areIndependent(first.getOperation(), other.getOperation())) {
               std::vector<mlir::Attribute> mappedCols;
               std::vector<mlir::Value> mappedColVals;
               auto* mapBlock = new mlir::Block;
               auto tuple = mapBlock->addArgument(mlir::tuples::TupleType::get(&getContext()), other.getLoc());
               mlir::OpBuilder mapBuilder(&getContext());
               mapBuilder.setInsertionPointToStart(mapBlock);

               for (auto x : other.getMapping()) {
                  if (mapping.contains(x.getName().str())) {
                     mappedCols.push_back(x.getValue());
                     auto colRef = colManager.createRef(&mapping.at(x.getName().str()).cast<mlir::tuples::ColumnDefAttr>().getColumn());
                     mappedColVals.push_back(mapBuilder.create<mlir::tuples::GetColumnOp>(other.getLoc(), colRef.getColumn().type, colRef, tuple));
                  } else {
                     mapping.insert({x.getName().str(), x.getValue()});
                  }
               }
               mlir::Value newScan = first;
               if (!mappedCols.empty()) {
                  mapBuilder.create<mlir::tuples::ReturnOp>(other.getLoc(), mappedColVals);
                  mlir::OpBuilder builder(other.getOperation());
                  auto mapOp = builder.create<mlir::subop::MapOp>(other.getLoc(), newScan, builder.getArrayAttr(mappedCols));
                  mapOp.getFn().push_back(mapBlock);
                  newScan = mapOp.getResult();
               }
               other.getRes().replaceAllUsesWith(newScan);
               other->erase();
               subOpDependencyAnalysis.addToRoot(first, other);
            }
         }
         std::vector<mlir::NamedAttribute> newMapping;
         for (auto m : mapping) {
            newMapping.push_back(builder.getNamedAttr(m.first, m.second));
         }
         first.setMappingAttr(builder.getDictionaryAttr(newMapping));
      }
      std::unordered_map<std::string, std::unordered_set<mlir::Operation*>> writtenToMember;
      getOperation().walk([&](mlir::subop::SubOperator op) {
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
      getOperation().walk([&](mlir::subop::ScanOp op) {
         std::unordered_map<mlir::tuples::Column*, std::string> columnFingerPrints;
         for (auto m : op.getMapping()) {
            auto* col = &m.getValue().cast<mlir::tuples::ColumnDefAttr>().getColumn();
            columnFingerPrints.insert({col, m.getName().str()});
         }
         std::unordered_map<mlir::Operation*, std::vector<mlir::subop::MaterializeOp>> materializeOps;
         std::unordered_map<mlir::Operation*, std::vector<mlir::subop::FilterOp>> filterOps;
         std::vector<mlir::Operation*> relevantUsers;
         for (auto* user : op->getUsers()) {
            handleScanUserRec(user, columnFingerPrints, materializeOps[user], filterOps[user]);
            if (materializeOps[user].size() == 1 && materializeOps[user][0].getState().getType().isa<mlir::subop::BufferType, mlir::subop::MultiMapType>()) {
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
               auto* col = &c.getValue().cast<mlir::tuples::ColumnRefAttr>().getColumn();
               if (columnFingerPrints.contains(col)) {
                  fingerprintMapping.insert({columnFingerPrints[col], c.getName().str()});
               } else {
                  return;
               }
            }
            std::unordered_set<std::string> firstFilters;
            for (auto f : filterOps[first]) {
               for (auto p : f.getConditions()) {
                  auto* col = &p.cast<mlir::tuples::ColumnRefAttr>().getColumn();
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
                  auto* col = &c.getValue().cast<mlir::tuples::ColumnRefAttr>().getColumn();
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
                     auto* col = &p.cast<mlir::tuples::ColumnRefAttr>().getColumn();
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
               /*
               //todo: compute overlaping
               llvm::dbgs() << "first:";
               materializeOps[first][0].dump();
               materializeOps[first][0].getLoc().dump();
               llvm::dbgs() << "replacement candidate:";
               materializeOps[other][0].dump();
               materializeOps[other][0].getLoc().dump();
               llvm::dbgs() << " extraMembers: ";
               for (auto eM : extraMembers) {
                  llvm::dbgs() << eM << ",";
               }
               llvm::dbgs() << "\n";
               llvm::dbgs() << "used by:";
               for (auto* user : materializeOps[other][0].getState().getUsers()) {
                  user->dump();
               }
               llvm::dbgs() << "\n\n\n\n\n";
*/
               if (!extraMembers.empty()) continue;
               std::unordered_map<std::string, std::string> memberMapping;
               for (auto c : materializeOps[other][0].getMapping()) {
                  memberMapping.insert({c.getName().str(), fingerprintMapping.at(columnFingerPrints.at(&c.getValue().cast<mlir::tuples::ColumnRefAttr>().getColumn()))});
               }
               if (auto firstMultiMapType = firstState.getType().dyn_cast_or_null<mlir::subop::MultiMapType>()) {
                  if (auto otherMultiMapType = otherState.getType().dyn_cast_or_null<mlir::subop::MultiMapType>()) {
                     bool keysEqual = true;
                     if (firstMultiMapType.getKeyMembers().getNames().size() != otherMultiMapType.getKeyMembers().getNames().size()) {
                        continue;
                     }
                     for (auto z : llvm::zip(firstMultiMapType.getKeyMembers().getNames(), otherMultiMapType.getKeyMembers().getNames())) {
                        auto firstKey = std::get<0>(z).cast<mlir::StringAttr>().str();
                        auto otherKey = std::get<1>(z).cast<mlir::StringAttr>().str();
                        keysEqual |= memberMapping.at(otherKey) == firstKey;
                     }
                     if (keysEqual) {
                        mlir::subop::SubOpStateUsageTransformer transformer(columnUsageAnalysis, &getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
                           if (auto listType = type.dyn_cast_or_null<mlir::subop::ListType>()) {
                              if (listType.getT().isa<mlir::subop::MultiMapEntryRefType>()) {
                                 return mlir::subop::ListType::get(&getContext(), mlir::subop::MultiMapEntryRefType::get(&getContext(), firstMultiMapType));
                              }
                           }
                           if (type.isa<mlir::subop::MultiMapEntryRefType>()) {
                              return mlir::subop::MultiMapEntryRefType::get(&getContext(), firstMultiMapType);
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

               //steps:
               //1. create new state type for first materialization
               //2. adapt previous uses of state to the new type
               //2.1 create
               //2.2 materialization
               //2.3 create_hash_indexed_view ->
               //3. fix module => map members
               //4. local optimizations to allow for further effects
               //4.1 create_hash_indexed_view [link member is not first one]
               //    => replace create_hash_indexed_view with a duplicate one (using the same hash)
            }
         }
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createGlobalOptPass() { return std::make_unique<GlobalOptPass>(); }