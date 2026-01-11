#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

// Toggle to 1 to enable debug output
#define CSE_DEBUG 0

namespace {
using namespace lingodb::compiler::dialect;

class CommonSubtreeElimination : public mlir::PassWrapper<CommonSubtreeElimination, mlir::OperationPass<mlir::func::FuncOp>> {
   struct LeaderInfo {
      std::shared_ptr<tuples::Column> col;
      mlir::SymbolRefAttr name;
   };

   llvm::DenseMap<const tuples::Column*, LeaderInfo> colMapping;
   llvm::DenseMap<const tuples::Column*, llvm::SmallVector<LeaderInfo, 1>> inverseColMapping;
   llvm::DenseMap<mlir::Value, mlir::Value> equivalenceMap;

   size_t successfulPhysicalMerges = 0;
   size_t successfulVirtualMerges = 0;
   size_t failedPhysicalMerges = 0;

   // Helper to extract local definitions from ANY operation.
   llvm::DenseMap<const tuples::Column*, std::string> getLocalDefs(mlir::Operation* op) const {
      llvm::DenseMap<const tuples::Column*, std::string> localDefs;
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> defs;

      llvm::SmallSetVector<mlir::StringAttr, 4> names;
      if (auto info = op->getRegisteredInfo())
         for (auto n : info->getAttributeNames()) names.insert(n);
      for (auto n : op->getAttrs()) names.insert(n.getName());

      std::vector<mlir::StringAttr> sortedNames(names.begin(), names.end());
      llvm::sort(sortedNames, [](mlir::StringAttr a, mlir::StringAttr b) { return a.strref() < b.strref(); });

      for (auto n : sortedNames) {
         if (auto val = op->getAttr(n)) collectDefs(val, defs);
      }

      for (const auto& p : defs) {
         if (auto ptr = p.second.getColumnPtr()) {
            localDefs[ptr.get()] = p.first;
         }
      }
      return localDefs;
   }

   // Locates the operation defining a specific column within the subtree rooted at 'root'.
   // Returns the defining operation and the path of operand indices from 'root' to it.
   std::pair<mlir::Operation*, std::vector<unsigned>> findDefiningOp(mlir::Operation* root, const tuples::Column* col) const {
      auto localDefs = getLocalDefs(root);
      if (localDefs.count(col)) return {root, {}};

      for (unsigned i = 0; i < root->getNumOperands(); ++i) {
         auto opd = root->getOperand(i);
         if (auto defOp = opd.getDefiningOp()) {
            if (defOp->getDialect()->getNamespace() == "relalg") {
               auto res = findDefiningOp(defOp, col);
               if (res.first) {
                  res.second.push_back(i);
                  return res;
               }
            }
         }
      }
      return {nullptr, {}};
   }

   // Follows a path of operand indices from 'root' to find the corresponding operation in an equivalent subtree.
   mlir::Operation* followPath(mlir::Operation* root, const std::vector<unsigned>& path) const {
      mlir::Operation* curr = root;
      for (auto it = path.rbegin(); it != path.rend(); ++it) {
         if (*it >= curr->getNumOperands()) return nullptr;
         auto opd = curr->getOperand(*it);
         curr = opd.getDefiningOp();
         if (!curr) return nullptr;
      }
      return curr;
   }

   mlir::Attribute getCanonicalAttr(mlir::Attribute attr, const llvm::DenseMap<const tuples::Column*, std::string>* localDefs = nullptr) const {
      if (!attr) return {};

      if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
         auto ptrWrapper = colRef.getColumnPtr();
         if (!ptrWrapper) return attr;
         const auto* ptr = ptrWrapper.get();

         // 1. Check Global Mapping (Virtual Merge)
         auto it = colMapping.find(ptr);
         if (it != colMapping.end()) {
            return tuples::ColumnRefAttr::get(attr.getContext(), it->second.name, it->second.col);
         }

         // 2. Check Local Definitions (Structure Match)
         if (localDefs) {
            auto locIt = localDefs->find(ptr);
            if (locIt != localDefs->end()) {
               return mlir::StringAttr::get(attr.getContext(), "LocalColRef::" + locIt->second);
            }
         }
         return attr;
      }

      if (auto colDef = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
         if (auto from = colDef.getFromExisting()) {
            return getCanonicalAttr(from, localDefs);
         }
         return mlir::StringAttr::get(attr.getContext(), "ColDef");
      }

      if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
         llvm::SmallVector<mlir::Attribute> newElems;
         newElems.reserve(arr.size());
         bool changed = false;
         for (auto e : arr) {
            auto c = getCanonicalAttr(e, localDefs);
            newElems.push_back(c);
            if (c != e) changed = true;
         }
         return changed ? mlir::ArrayAttr::get(attr.getContext(), newElems) : attr;
      }

      if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
         llvm::SmallVector<mlir::NamedAttribute> newAttrs;
         newAttrs.reserve(dict.size());
         bool changed = false;
         for (auto named : dict) {
            auto newVal = getCanonicalAttr(named.getValue(), localDefs);
            newAttrs.emplace_back(named.getName(), newVal);
            if (newVal != named.getValue()) changed = true;
         }
         return changed ? mlir::DictionaryAttr::get(attr.getContext(), newAttrs) : attr;
      }
      return attr;
   }

   struct EquivalenceContext {
      llvm::DenseMap<mlir::Value, mlir::Value> candidateToLeaderVal;
      CommonSubtreeElimination* pass;
      int depth = 0;

      explicit EquivalenceContext(CommonSubtreeElimination* pass) : pass(pass) {}

      bool checkOps(mlir::Operation* leader, mlir::Operation* candidate) {
         if (leader->getName() != candidate->getName()) return false;
         if (leader->getNumOperands() != candidate->getNumOperands()) return false;
         if (leader->getNumRegions() != candidate->getNumRegions()) return false;
         if (leader->getNumResults() != candidate->getNumResults()) return false;

         for (unsigned i = 0; i < leader->getNumOperands(); ++i) {
            mlir::Value lVal = leader->getOperand(i);
            mlir::Value cVal = candidate->getOperand(i);

            if (auto it = candidateToLeaderVal.find(cVal); it != candidateToLeaderVal.end()) {
               if (it->second != lVal) {
                  return false;
               }
            } else {
               if (pass->resolveValue(cVal) != pass->resolveValue(lVal)) {
                  return false;
               }
            }
         }

         if (!checkAttributes(leader, candidate)) return false;

         for (unsigned i = 0; i < leader->getNumResults(); ++i) {
            candidateToLeaderVal[candidate->getResult(i)] = leader->getResult(i);
         }

         for (unsigned i = 0; i < leader->getNumRegions(); ++i) {
            if (!checkRegions(leader->getRegion(i), candidate->getRegion(i))) return false;
         }
         return true;
      }

      bool checkAttributes(mlir::Operation* leader, mlir::Operation* candidate) const {
         llvm::SmallSetVector<mlir::StringAttr, 4> names;
         if (auto info = leader->getRegisteredInfo()) {
            for (auto name : info->getAttributeNames()) names.insert(name);
         }
         for (auto named : leader->getAttrs()) names.insert(named.getName());

         auto leaderLocals = pass->getLocalDefs(leader);
         auto candidateLocals = pass->getLocalDefs(candidate);

         for (auto name : names) {
            if (pass->shouldIgnoreAttr(name)) continue;
            // BaseTableOp: ignore "columns" to allow merging different column sets
            if (mlir::isa<relalg::BaseTableOp>(leader) && name == "columns") continue;

            auto lVal = leader->getAttr(name);
            auto cVal = candidate->getAttr(name);

            if (!!lVal != !!cVal) return false;
            if (!lVal) continue;

            auto canonL = pass->getCanonicalAttr(lVal, &leaderLocals);
            auto canonC = pass->getCanonicalAttr(cVal, &candidateLocals);

            if (canonL != canonC) {
               return false;
            }
         }
         return true;
      }

      bool checkRegions(mlir::Region& r1, mlir::Region& r2) {
         if (r1.empty() != r2.empty()) return false;
         if (r1.empty()) return true;
         if (r1.getBlocks().size() != r2.getBlocks().size()) return false;

         auto it1 = r1.begin();
         auto it2 = r2.begin();

         while (it1 != r1.end()) {
            mlir::Block& b1 = *it1++;
            mlir::Block& b2 = *it2++;

            if (b1.getNumArguments() != b2.getNumArguments()) return false;
            for (unsigned i = 0; i < b1.getNumArguments(); ++i) {
               candidateToLeaderVal[b2.getArgument(i)] = b1.getArgument(i);
            }

            auto opIt1 = b1.begin();
            auto opIt2 = b2.begin();
            while (opIt1 != b1.end() && opIt2 != b2.end()) {
               depth++;
               if (!checkOps(&*opIt1, &*opIt2)) {
                  depth--;
                  return false;
               }
               depth--;
               ++opIt1;
               ++opIt2;
            }
            if (opIt1 != b1.end() || opIt2 != b2.end()) return false;
         }
         return true;
      }
   };

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonSubtreeElimination)
   llvm::StringRef getArgument() const override { return "relalg-cse"; }

   protected:
   void runOnOperation() override {
      auto funcOp = getOperation();
      colMapping.clear();
      inverseColMapping.clear();
      equivalenceMap.clear();
      successfulPhysicalMerges = 0;
      failedPhysicalMerges = 0;
      successfulVirtualMerges = 0;

      if (funcOp.getBody().empty()) return;

      mlir::DominanceInfo domInfo(funcOp);
      llvm::DenseMap<llvm::hash_code, llvm::SmallVector<mlir::Operation*, 2>> candidates;

      std::function<void(mlir::Region&)> traverseRegion;

      auto processBlock = [&](mlir::Block* block) {
         for (auto& op : llvm::make_early_inc_range(*block)) {
            if (op.getDialect()->getNamespace() != "relalg") continue;

            bool debugHash = CSE_DEBUG && (mlir::isa<relalg::BaseTableOp>(op));
            auto hash = computeHash(&op, debugHash);
            bool merged = false;

            if (CSE_DEBUG) {
               llvm::errs() << "Visiting: " << op.getName() << " (" << &op << ") Hash: " << hash << "\n";
               if (debugHash) op.dump();
            }

            if (auto it = candidates.find(hash); it != candidates.end()) {
               for (auto* leader : it->second) {
                  if (CSE_DEBUG) {
                     llvm::errs() << "  Comparing with leader: " << leader->getName() << " (" << leader << ")\n";
                  }

                  if (domInfo.properlyDominates(leader, &op) && areEquivalent(leader, &op)) {
                     if (!isSafeCrossRegionMerge(leader, &op)) continue;
                     if (mlir::isa<relalg::AggrFuncOp>(op)) continue;

                     if (mlir::dyn_cast<relalg::BaseTableOp>(leader)) {
                        mapVirtualBaseTableOpMerge(leader, &op);
                        merged = true;
                        break;
                     }

                     bool isCandidateForPhysical = !mlir::isa<
                        relalg::RenamingOp,
                        relalg::GroupJoinOp,
                        relalg::WindowOp,
                        relalg::ConstRelationOp>(op);

                     if (isCandidateForPhysical) {
                        if (tryMergeOperations(leader, &op)) {
                           merged = true;
                           break;
                        } else {
                           if (CSE_DEBUG) llvm::errs() << "    -> Physical merge failed, trying virtual\n";
                           mapVirtualMerge(leader, &op);
                           merged = true;
                           break;
                        }
                     } else {
                        mapVirtualMerge(leader, &op);
                        merged = true;
                        break;
                     }
                  }
               }
            }

            if (!merged) {
               candidates[hash].push_back(&op);
               for (auto& region : op.getRegions()) {
                  traverseRegion(region);
               }
            }
         }
      };

      std::function<void(llvm::DomTreeNodeBase<mlir::Block>*)> walkDomTree =
         [&](llvm::DomTreeNodeBase<mlir::Block>* node) {
            if (!node) return;
            processBlock(node->getBlock());
            for (auto* child : node->children()) {
               walkDomTree(child);
            }
         };

      traverseRegion = [&](mlir::Region& region) {
         if (region.empty()) return;
         if (region.hasOneBlock()) {
            processBlock(&region.front());
         } else {
            if (auto* root = domInfo.getRootNode(&region)) {
               walkDomTree(root);
            } else {
               for (auto& block : region) processBlock(&block);
            }
         }
      };

      for (auto& region : funcOp->getRegions()) {
         traverseRegion(region);
      }

      if (CSE_DEBUG && (successfulPhysicalMerges > 0 || successfulVirtualMerges > 0 || failedPhysicalMerges > 0)) {
         llvm::errs() << "=========================================================\n";
         llvm::errs() << "CSE Pass Summary\n";
         llvm::errs() << "  Successful Physical Merges: " << successfulPhysicalMerges << "\n";
         if (failedPhysicalMerges > 0) llvm::errs() << "  Failed Physical Merges:     " << failedPhysicalMerges << "\n";
         llvm::errs() << "  Successful Virtual Merges: " << successfulVirtualMerges << "\n";
         llvm::errs() << "=========================================================\n";
      }
   }

   private:
   mlir::Value resolveValue(mlir::Value val) const {
      auto it = equivalenceMap.find(val);
      while (it != equivalenceMap.end()) {
         val = it->second;
         it = equivalenceMap.find(val);
      }
      return val;
   }

   static bool shouldIgnoreAttr(llvm::StringRef name) {
      return name == "rows" || name == "total_rows";
   }

   llvm::hash_code computeHash(mlir::Operation* op, bool debug = false) const {
      auto hash = llvm::hash_value(op->getName().getAsOpaquePointer());
      if (debug) llvm::errs() << "  Hashing " << op->getName() << ":\n";

      for (auto opd : op->getOperands()) {
         auto resolved = resolveValue(opd);
         hash = llvm::hash_combine(hash, resolved);
         if (debug) llvm::errs() << "    Operand: " << resolved << "\n";
      }
      hash = llvm::hash_combine(hash, op->getNumRegions());

      llvm::SmallVector<mlir::NamedAttribute, 4> sortedAttrs;
      llvm::SmallSetVector<mlir::StringAttr, 4> attrNames;
      if (auto info = op->getRegisteredInfo()) {
         for (auto name : info->getAttributeNames()) attrNames.insert(name);
      }
      for (auto named : op->getAttrs()) attrNames.insert(named.getName());

      auto localDefs = getLocalDefs(op);

      for (auto name : attrNames) {
         if (shouldIgnoreAttr(name)) continue;
         // Special case: Ignore "columns" for BaseTableOp to ensure hash collision for same table
         if (mlir::isa<relalg::BaseTableOp>(op) && name == "columns") continue;

         if (auto val = op->getAttr(name)) {
            sortedAttrs.emplace_back(name, getCanonicalAttr(val, &localDefs));
         }
      }
      llvm::sort(sortedAttrs, [](const auto& a, const auto& b) { return a.getName().strref() < b.getName().strref(); });

      for (const auto& attr : sortedAttrs) {
         if (debug) llvm::errs() << "    Attr " << attr.getName().strref() << ": " << attr.getValue() << "\n";
         hash = llvm::hash_combine(hash, attr.getName(), attr.getValue());
      }

      for (auto& region : op->getRegions()) {
         for (auto& block : region) {
            hash = llvm::hash_combine(hash, block.getNumArguments());
            for (auto& nestedOp : block) {
               hash = llvm::hash_combine(hash, nestedOp.getName().getAsOpaquePointer());
            }
         }
      }
      return hash;
   }

   bool areEquivalent(mlir::Operation* leader, mlir::Operation* candidate) {
      EquivalenceContext ctx(this);
      return ctx.checkOps(leader, candidate);
   }

   static bool isSafeCrossRegionMerge(mlir::Operation* leader, mlir::Operation* duplicate) {
      mlir::Operation* ancestor = duplicate;
      while (ancestor && ancestor->getBlock() != leader->getBlock()) {
         ancestor = ancestor->getParentOp();
      }
      if (!ancestor) return false;

      if (ancestor == leader) return false;

      if (ancestor == duplicate) return true;

      for (auto operand : ancestor->getOperands()) {
         if (operand == leader->getResult(0)) return false;
      }
      return true;
   }

   static void collectDefs(mlir::Attribute attr, llvm::SmallVectorImpl<std::pair<std::string, tuples::ColumnDefAttr>>& defs) {
      if (auto def = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
         if (!def.getName()) return;
         defs.emplace_back(def.getName().getLeafReference().getValue().str(), def);
      } else if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
         for (auto e : arr) collectDefs(e, defs);
      } else if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
         for (auto e : dict) collectDefs(e.getValue(), defs);
      }
   }

   static void getAvailableColumns(mlir::Operation* op, llvm::DenseSet<const tuples::Column*>& out, llvm::SmallPtrSet<mlir::Operation*, 4>& visited) {
      if (!visited.insert(op).second) return;

      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> localDefs;
      llvm::SmallSetVector<mlir::StringAttr, 4> names;
      if (auto info = op->getRegisteredInfo())
         for (auto n : info->getAttributeNames()) names.insert(n);
      for (auto n : op->getAttrs()) names.insert(n.getName());

      std::vector<mlir::StringAttr> sortedNames(names.begin(), names.end());
      llvm::sort(sortedNames, [](mlir::StringAttr a, mlir::StringAttr b) { return a.strref() < b.strref(); });

      for (auto n : sortedNames)
         if (auto val = op->getAttr(n)) collectDefs(val, localDefs);

      for (auto& p : localDefs) {
         if (auto ptr = p.second.getColumnPtr()) out.insert(ptr.get());
      }

      if (mlir::isa<relalg::BaseTableOp>(op)) return;
      if (mlir::isa<relalg::RenamingOp>(op)) return;

      if (auto agg = mlir::dyn_cast<relalg::AggregationOp>(op)) {
         for (auto attr : agg.getGroupByCols()) {
            if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
               if (auto ptr = colRef.getColumnPtr()) out.insert(ptr.get());
            }
         }
         return;
      }

      if (auto proj = mlir::dyn_cast<relalg::ProjectionOp>(op)) {
         for (auto attr : proj.getCols()) {
            if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
               if (auto ptr = colRef.getColumnPtr()) out.insert(ptr.get());
            }
         }
         return;
      }

      if (mlir::isa<
             relalg::SemiJoinOp,
             relalg::AntiSemiJoinOp,
             relalg::MarkJoinOp,
             relalg::IntersectOp,
             relalg::ExceptOp>(op)) {
         if (op->getNumOperands() > 0) {
            if (auto* defOp = op->getOperand(0).getDefiningOp()) {
               if (defOp->getDialect()->getNamespace() == "relalg")
                  getAvailableColumns(defOp, out, visited);
            }
         }
         return;
      }

      if (mlir::isa<relalg::GroupJoinOp>(op)) {
         if (op->getNumOperands() > 0) {
            if (auto* defOp = op->getOperand(0).getDefiningOp()) {
               if (defOp->getDialect()->getNamespace() == "relalg")
                  getAvailableColumns(defOp, out, visited);
            }
         }
         return;
      }

      for (auto operand : op->getOperands()) {
         if (auto* defOp = operand.getDefiningOp()) {
            if (defOp->getDialect()->getNamespace() == "relalg") {
               getAvailableColumns(defOp, out, visited);
            }
         }
      }
   }

   static void collectRecursiveDefs(mlir::Operation* op, llvm::SmallVectorImpl<std::pair<std::string, tuples::ColumnDefAttr>>& defs, llvm::SmallPtrSetImpl<mlir::Operation*>& visited) {
      if (!visited.insert(op).second) return;

      llvm::SmallSetVector<mlir::StringAttr, 4> names;
      if (auto info = op->getRegisteredInfo())
         for (auto n : info->getAttributeNames()) names.insert(n);
      for (auto n : op->getAttrs()) names.insert(n.getName());

      std::vector<mlir::StringAttr> sortedNames(names.begin(), names.end());
      llvm::sort(sortedNames, [](mlir::StringAttr a, mlir::StringAttr b) { return a.strref() < b.strref(); });

      for (auto n : sortedNames) {
         if (auto val = op->getAttr(n)) collectDefs(val, defs);
      }

      for (auto operand : op->getOperands()) {
         if (auto* definingOp = operand.getDefiningOp()) {
            if (definingOp->getDialect()->getNamespace() == "relalg") {
               collectRecursiveDefs(definingOp, defs, visited);
            }
         }
      }
   }

   void mapVirtualBaseTableOpMerge(mlir::Operation* leader, mlir::Operation* duplicate) {
      auto leaderBase = mlir::dyn_cast<relalg::BaseTableOp>(leader);
      auto dupBase = mlir::cast<relalg::BaseTableOp>(duplicate);

      auto lColsAttr = leaderBase->getAttrOfType<mlir::DictionaryAttr>("columns");
      auto dColsAttr = dupBase->getAttrOfType<mlir::DictionaryAttr>("columns");

      llvm::DenseMap<mlir::StringAttr, mlir::Attribute> leaderCols;
      if (lColsAttr) {
         for (auto named : lColsAttr) leaderCols[named.getName()] = named.getValue();
      }

      bool leaderChanged = false;
      if (dColsAttr) {
         for (auto named : dColsAttr) {
            auto physName = named.getName();
            auto dDef = mlir::cast<tuples::ColumnDefAttr>(named.getValue());
            tuples::ColumnDefAttr lDef;
            if (leaderCols.count(physName)) {
               lDef = mlir::cast<tuples::ColumnDefAttr>(leaderCols[physName]);
            } else {
               leaderCols[physName] = dDef;
               leaderChanged = true;
               lDef = dDef;
            }

            if (dDef.getColumnPtr() && lDef.getColumnPtr() && dDef.getColumnPtr() != lDef.getColumnPtr()) {
               colMapping[dDef.getColumnPtr().get()] = {lDef.getColumnPtr(), lDef.getName()};
               inverseColMapping[lDef.getColumnPtr().get()].push_back({dDef.getColumnPtr(), dDef.getName()});
            }
         }
      }

      if (leaderChanged) {
         llvm::SmallVector<mlir::NamedAttribute> newLeaderCols;
         for (auto it : leaderCols) newLeaderCols.emplace_back(it.first, it.second);
         llvm::sort(newLeaderCols, [](const mlir::NamedAttribute& a, const mlir::NamedAttribute& b) {
            return a.getName().strref() < b.getName().strref();
         });
         leaderBase->setAttr("columns", mlir::DictionaryAttr::get(leader->getContext(), newLeaderCols));
      }

      if (duplicate->getNumResults() > 0 && leader->getNumResults() > 0) {
         equivalenceMap[duplicate->getResult(0)] = leader->getResult(0);
      }

      successfulVirtualMerges++;
      if (CSE_DEBUG) {
         llvm::errs() << "[CSE virtual REPLACE] " << duplicate->getName() << " -> " << leader->getName() << "\n";
         leader->dump();
         duplicate->dump();
      }
   }
   void mapVirtualMerge(mlir::Operation* leader, mlir::Operation* duplicate) {
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> lDefs, dDefs;

      auto collectLocal = [](mlir::Operation* op, auto& out) {
         llvm::SmallSetVector<mlir::StringAttr, 4> names;
         if (auto info = op->getRegisteredInfo())
            for (auto n : info->getAttributeNames()) names.insert(n);
         for (auto n : op->getAttrs()) names.insert(n.getName());

         std::vector<mlir::Attribute> sortedNames;
         for (auto n : names) sortedNames.push_back(n);
         llvm::sort(sortedNames, [](mlir::Attribute a, mlir::Attribute b) {
            return mlir::cast<mlir::StringAttr>(a).getValue() < mlir::cast<mlir::StringAttr>(b).getValue();
         });
         for (auto nAttr : sortedNames) {
            auto n = mlir::cast<mlir::StringAttr>(nAttr);
            if (auto val = op->getAttr(n)) collectDefs(val, out);
         }
      };

      collectLocal(leader, lDefs);
      collectLocal(duplicate, dDefs);

      if (lDefs.size() == dDefs.size()) {
         for (size_t i = 0; i < lDefs.size(); ++i) {
            auto lDef = lDefs[i].second;
            auto dDef = dDefs[i].second;

            auto lPtr = lDef.getColumnPtr();
            auto dPtr = dDef.getColumnPtr();
            if (lPtr && dPtr && lPtr != dPtr) {
               colMapping[dPtr.get()] = {lPtr, lDef.getName()};
               inverseColMapping[lPtr.get()].push_back({dPtr, dDef.getName()});
            }
         }
      }

      if (duplicate->getNumResults() > 0 && leader->getNumResults() > 0) {
         equivalenceMap[duplicate->getResult(0)] = leader->getResult(0);
      }

      if (CSE_DEBUG) {
         llvm::errs() << "[CSE VIRTUAL] " << duplicate->getName() << " -> " << leader->getName() << "\n";
         leader->dump();
         duplicate->dump();
      }
      successfulVirtualMerges++;
   }

   bool tryMergeOperations(mlir::Operation* leader, mlir::Operation* duplicate) {
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> lDefs, dDefs;
      auto collectLocal = [](mlir::Operation* op, auto& out) {
         llvm::SmallSetVector<mlir::StringAttr, 4> names;
         if (auto info = op->getRegisteredInfo())
            for (auto n : info->getAttributeNames()) names.insert(n);
         for (auto n : op->getAttrs()) names.insert(n.getName());

         std::vector<mlir::Attribute> sortedNames;
         for (auto n : names) sortedNames.push_back(n);
         llvm::sort(sortedNames, [](mlir::Attribute a, mlir::Attribute b) {
            return mlir::cast<mlir::StringAttr>(a).getValue() < mlir::cast<mlir::StringAttr>(b).getValue();
         });

         for (auto nAttr : sortedNames) {
            auto n = mlir::cast<mlir::StringAttr>(nAttr);
            if (auto val = op->getAttr(n)) collectDefs(val, out);
         }
      };
      collectLocal(leader, lDefs);
      collectLocal(duplicate, dDefs);

      if (lDefs.size() == dDefs.size()) {
         for (size_t i = 0; i < lDefs.size(); ++i) {
            auto lDef = lDefs[i].second;
            auto dDef = dDefs[i].second;

            auto lPtr = lDef.getColumnPtr();
            auto dPtr = dDef.getColumnPtr();

            if (lPtr && dPtr && lPtr != dPtr) {
               colMapping[dPtr.get()] = {lPtr, lDef.getName()};
            }
         }
      }

      llvm::DenseSet<const tuples::Column*> availableLeaderCols;
      {
         llvm::SmallPtrSet<mlir::Operation*, 4> visited;
         getAvailableColumns(leader, availableLeaderCols, visited);
      }
      llvm::DenseSet<const tuples::Column*> availableDuplicateCols;
      {
         llvm::SmallPtrSet<mlir::Operation*, 4> visited;
         getAvailableColumns(duplicate, availableDuplicateCols, visited);
      }
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> allDupDefs;
      {
         llvm::SmallPtrSet<mlir::Operation*, 8> visited;
         collectRecursiveDefs(duplicate, allDupDefs, visited);
      }
      llvm::sort(allDupDefs, [](const auto& a, const auto& b) { return a.first < b.first; });

      mlir::OpBuilder builder(duplicate);
      llvm::SmallVector<mlir::Attribute> renamingAttrs;
      llvm::DenseSet<const tuples::Column*> processed;

      for (const auto& [key, dDef] : allDupDefs) {
         auto dPtrWrapper = dDef.getColumnPtr();
         if (!dPtrWrapper) continue;
         const auto* dPtr = dPtrWrapper.get();
         if (processed.count(dPtr)) continue;
         if (!availableDuplicateCols.count(dPtr)) continue;

         processed.insert(dPtr);

         std::shared_ptr<tuples::Column> lCol = nullptr;
         mlir::SymbolRefAttr lName;

         if (auto it = colMapping.find(dPtr); it != colMapping.end()) {
            lCol = it->second.col;
            lName = it->second.name;
         } else if (availableLeaderCols.count(dPtr)) {
            lCol = dPtrWrapper;
            lName = dDef.getName();
         } else {
            if (auto it = inverseColMapping.find(dPtr); it != inverseColMapping.end()) {
               for (auto& info : it->second) {
                  if (availableLeaderCols.count(info.col.get())) {
                     lCol = info.col;
                     lName = info.name;
                     break;
                  }
               }
            }
         }

         if (!lCol || !availableLeaderCols.count(lCol.get())) {
            // Lazy fix: Try to find the column in the leader's BaseTable equivalent
            auto [dSourceOp, path] = findDefiningOp(duplicate, dPtr);
            if (dSourceOp && mlir::isa<relalg::BaseTableOp>(dSourceOp)) {
               if (auto* lSourceOp = followPath(leader, path)) {
                  if (auto baseTable = mlir::dyn_cast<relalg::BaseTableOp>(lSourceOp)) {
                     auto dBaseTable = mlir::cast<relalg::BaseTableOp>(dSourceOp);
                     auto dCols = dBaseTable->getAttrOfType<mlir::DictionaryAttr>("columns");
                     mlir::StringAttr physName;
                     if (dCols) {
                        for (auto named : dCols) {
                           auto val = mlir::dyn_cast<tuples::ColumnDefAttr>(named.getValue());
                           if (val && val.getColumnPtr() == dDef.getColumnPtr()) {
                              physName = named.getName();
                              break;
                           }
                        }
                     }

                     if (physName) {
                        auto cols = baseTable->getAttrOfType<mlir::DictionaryAttr>("columns");
                        llvm::SmallVector<mlir::NamedAttribute> newCols;
                        if (cols) newCols.append(cols.begin(), cols.end());

                        tuples::ColumnDefAttr existingLDef;
                        bool exists = false;
                        for (auto& named : newCols) {
                           if (named.getName() == physName) {
                              exists = true;
                              existingLDef = mlir::cast<tuples::ColumnDefAttr>(named.getValue());
                              break;
                           }
                        }

                        if (!exists) {
                           newCols.emplace_back(physName, dDef);
                           llvm::sort(newCols, [](const auto& a, const auto& b) { return a.getName().strref() < b.getName().strref(); });
                           baseTable->setAttr("columns", mlir::DictionaryAttr::get(baseTable.getContext(), newCols));

                           // We added dDef, so leader now has dDef's column
                           lCol = dDef.getColumnPtr();
                           lName = dDef.getName();
                           availableLeaderCols.insert(lCol.get());
                        } else {
                           // Leader already has it, but maybe we missed it?
                           lCol = existingLDef.getColumnPtr();
                           lName = existingLDef.getName();
                           availableLeaderCols.insert(lCol.get());
                        }
                     }
                  }
               }
            }
         }

         if (lCol) {
            bool available = availableLeaderCols.count(lCol.get());
            if (!available && lCol == dDef.getColumnPtr()) available = true;

            if (available) {
               auto lRef = tuples::ColumnRefAttr::get(builder.getContext(), lName, lCol);
               auto newDef = tuples::ColumnDefAttr::get(
                  builder.getContext(),
                  dDef.getName(),
                  dDef.getColumnPtr(),
                  builder.getArrayAttr({lRef}));
               renamingAttrs.push_back(newDef);
            } else {
               if (CSE_DEBUG) {
                  llvm::errs() << "[CSE FAIL] Physical merge failed: Leader missing column for " << lName << "\n";
                  leader->dump();
                  duplicate->dump();
               }
               failedPhysicalMerges++;
               return false;
            }
         } else {
            // FIXED: If we cannot resolve a column that is available in the duplicate, we must abort the merge.
            // Otherwise, we create a RenamingOp that hides this column, causing downstream crashes.
            if (CSE_DEBUG) {
               llvm::errs() << "[CSE FAIL] Physical merge failed: Could not resolve duplicate column " << dDef.getName() << "\n";
               leader->dump();
               duplicate->dump();
            }
            failedPhysicalMerges++;
            return false;
         }
      }

      mlir::Value replacement = leader->getResult(0);
      if (!renamingAttrs.empty()) {
         auto renamingOp = builder.create<relalg::RenamingOp>(
            duplicate->getLoc(),
            leader->getResult(0),
            builder.getArrayAttr(renamingAttrs));
         replacement = renamingOp.getResult();
      }

      if (replacement != leader->getResult(0)) {
         equivalenceMap[replacement] = leader->getResult(0);
      }

      duplicate->getResult(0).replaceAllUsesWith(replacement);
      if (CSE_DEBUG) {
         llvm::errs() << "[CSE REPLACE] " << duplicate->getName() << " -> " << leader->getName() << "\n";
         leader->dump();
         duplicate->dump();
         if (auto* r = replacement.getDefiningOp()) r->dump();
      }
      duplicate->erase();
      successfulPhysicalMerges++;

      return true;
   }
};
} // namespace

std::unique_ptr<mlir::Pass> relalg::createCommonSubtreeEliminationPass() {
   return std::make_unique<CommonSubtreeElimination>();
}