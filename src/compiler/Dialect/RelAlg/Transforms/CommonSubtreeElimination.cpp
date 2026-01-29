#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#define CSE_DEBUG 0
#define DEBUG_TYPE "relalg-cse"

using namespace lingodb::compiler::dialect;

namespace {

// Helper to manage column equivalence mapping
struct ColumnMappingContext {
   struct LeaderInfo {
      std::shared_ptr<tuples::Column> col;
      mlir::SymbolRefAttr name;
   };

   llvm::DenseMap<const tuples::Column*, LeaderInfo> colMapping;
   llvm::DenseMap<const tuples::Column*, llvm::SmallVector<LeaderInfo, 1>> inverseColMapping;

   void clear() {
      colMapping.clear();
      inverseColMapping.clear();
   }

   void addMapping(const tuples::Column* dupCol, const std::shared_ptr<tuples::Column>& leaderCol, mlir::SymbolRefAttr leaderName) {
      if (dupCol && leaderCol && dupCol != leaderCol.get()) {
         colMapping[dupCol] = {leaderCol, leaderName};
         inverseColMapping[leaderCol.get()].push_back({leaderCol, leaderName});
      }
   }
};

static void collectDefs(mlir::Attribute attr, llvm::SmallVectorImpl<std::pair<std::string, tuples::ColumnDefAttr>>& defs) {
   if (auto def = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
      if (def.getName()) {
         defs.emplace_back(def.getName().getLeafReference().getValue().str(), def);
      }
   } else if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      for (auto e : arr) collectDefs(e, defs);
   } else if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
      for (auto e : dict) collectDefs(e.getValue(), defs);
   }
}

static void collectSortedLocalDefs(mlir::Operation* op, llvm::SmallVectorImpl<std::pair<std::string, tuples::ColumnDefAttr>>& defs) {
   llvm::SmallSetVector<mlir::StringAttr, 4> names;
   if (auto info = op->getRegisteredInfo())
      for (auto n : info->getAttributeNames()) names.insert(n);
   for (auto n : op->getAttrs()) names.insert(n.getName());

   llvm::SmallVector<mlir::StringAttr, 4> sortedNames(names.begin(), names.end());
   llvm::sort(sortedNames, [](mlir::StringAttr a, mlir::StringAttr b) { return a.strref() < b.strref(); });

   for (auto n : sortedNames) {
      if (auto val = op->getAttr(n)) collectDefs(val, defs);
   }
}

static llvm::DenseMap<const tuples::Column*, std::string> getLocalDefsMap(mlir::Operation* op) {
   llvm::DenseMap<const tuples::Column*, std::string> localDefs;
   llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> defs;
   collectSortedLocalDefs(op, defs);

   for (const auto& p : defs) {
      if (auto ptr = p.second.getColumnPtr()) {
         localDefs[ptr.get()] = p.first;
      }
   }
   return localDefs;
}

static void getAvailableColumns(mlir::Operation* op, llvm::DenseSet<const tuples::Column*>& out, llvm::SmallPtrSet<mlir::Operation*, 4>& visited) {
   if (!visited.insert(op).second) return;

   llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> localDefs;
   collectSortedLocalDefs(op, localDefs);

   for (auto& p : localDefs) {
      if (auto ptr = p.second.getColumnPtr()) out.insert(ptr.get());
   }

   if (mlir::isa<relalg::BaseTableOp>(op)) return;

   if (auto renameOp = mlir::dyn_cast<relalg::RenamingOp>(op)) {
      llvm::DenseSet<const tuples::Column*> overwritten;
      if (auto colsAttr = renameOp.getColumns()) {
         for (auto attr : colsAttr) {
            if (auto def = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
               if (auto from = def.getFromExisting()) {
                  if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(from)) {
                     for (auto ref : arr) {
                        if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(ref))
                           if (auto ptr = colRef.getColumnPtr()) overwritten.insert(ptr.get());
                     }
                  }
               }
            }
         }
      }

      if (op->getNumOperands() > 0) {
         if (auto* defOp = op->getOperand(0).getDefiningOp()) {
            if (defOp->getDialect()->getNamespace() == "relalg") {
               llvm::DenseSet<const tuples::Column*> childCols;
               getAvailableColumns(defOp, childCols, visited);
               for (auto* c : childCols) {
                  if (!overwritten.count(c)) out.insert(c);
               }
            }
         }
      }
      return;
   }

   if (auto agg = mlir::dyn_cast<relalg::AggregationOp>(op)) {
      for (auto attr : agg.getGroupByCols()) {
         if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
            if (auto ptr = colRef.getColumnPtr()) out.insert(ptr.get());
         }
      }
      return;
   }

   if (mlir::isa<relalg::SemiJoinOp, relalg::AntiSemiJoinOp, relalg::MarkJoinOp,
                 relalg::IntersectOp, relalg::ExceptOp, relalg::GroupJoinOp>(op)) {
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

   collectSortedLocalDefs(op, defs);

   for (auto operand : op->getOperands()) {
      if (auto* definingOp = operand.getDefiningOp()) {
         if (definingOp->getDialect()->getNamespace() == "relalg") {
            collectRecursiveDefs(definingOp, defs, visited);
         }
      }
   }
}

static std::pair<mlir::Operation*, std::vector<unsigned>> findDefiningOp(mlir::Operation* root, const tuples::Column* col) {
   auto localDefs = getLocalDefsMap(root);
   if (localDefs.count(col)) return {root, {}};

   for (unsigned i = 0; i < root->getNumOperands(); ++i) {
      auto opd = root->getOperand(i);
      if (auto* defOp = opd.getDefiningOp()) {
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

static mlir::Operation* followPath(mlir::Operation* root, const std::vector<unsigned>& path) {
   mlir::Operation* curr = root;
   for (auto it = path.rbegin(); it != path.rend(); ++it) {
      if (*it >= curr->getNumOperands()) return nullptr;
      auto opd = curr->getOperand(*it);
      curr = opd.getDefiningOp();
      if (!curr) return nullptr;
   }
   return curr;
}

class CommonSubtreeElimination : public mlir::PassWrapper<CommonSubtreeElimination, mlir::OperationPass<mlir::func::FuncOp>> {
   ColumnMappingContext colContext;
   llvm::DenseMap<mlir::Value, mlir::Value> valueEquivalenceMap;

   size_t successfulPhysicalMerges = 0;
   size_t successfulVirtualMerges = 0;

   class EquivalenceChecker {
      llvm::DenseMap<mlir::Value, mlir::Value> candidateToLeaderVal;
      const CommonSubtreeElimination& pass;

      public:
      explicit EquivalenceChecker(const CommonSubtreeElimination& pass) : pass(pass) {}

      bool checkOps(mlir::Operation* leader, mlir::Operation* candidate) {
         if (leader->getName() != candidate->getName()) return false;
         if (leader->getNumOperands() != candidate->getNumOperands()) return false;
         if (leader->getNumRegions() != candidate->getNumRegions()) return false;
         if (leader->getNumResults() != candidate->getNumResults()) return false;

         for (unsigned i = 0; i < leader->getNumOperands(); ++i) {
            mlir::Value lVal = leader->getOperand(i);
            mlir::Value cVal = candidate->getOperand(i);

            if (auto it = candidateToLeaderVal.find(cVal); it != candidateToLeaderVal.end()) {
               if (it->second != lVal) return false;
            } else {
               if (pass.resolveValue(cVal) != pass.resolveValue(lVal)) return false;
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

      private:
      bool checkAttributes(mlir::Operation* leader, mlir::Operation* candidate) const {
         llvm::SmallSetVector<mlir::StringAttr, 4> names;
         if (auto info = leader->getRegisteredInfo()) {
            for (auto name : info->getAttributeNames()) names.insert(name);
         }
         for (auto named : leader->getAttrs()) names.insert(named.getName());

         auto leaderLocals = getLocalDefsMap(leader);
         auto candidateLocals = getLocalDefsMap(candidate);

         for (auto name : names) {
            if (mlir::isa<relalg::BaseTableOp>(leader) && name == "columns") continue;

            auto lVal = leader->getAttr(name);
            auto cVal = candidate->getAttr(name);

            if (!!lVal != !!cVal) return false;
            if (!lVal) continue;

            auto canonL = pass.getCanonicalAttr(lVal, &leaderLocals);
            auto canonC = pass.getCanonicalAttr(cVal, &candidateLocals);

            if (canonL != canonC) return false;
         }

         if (auto leaderBase = mlir::dyn_cast<relalg::BaseTableOp>(leader)) {
            auto candidateBase = mlir::cast<relalg::BaseTableOp>(candidate);
            if (leaderBase.getRestriction() != candidateBase.getRestriction()) return false;
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
               if (!checkOps(&*opIt1, &*opIt2)) {
                  return false;
               }
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

   void runOnOperation() override {
      auto funcOp = getOperation();
      colContext.clear();
      valueEquivalenceMap.clear();
      successfulPhysicalMerges = 0;
      successfulVirtualMerges = 0;

      if (funcOp.getBody().empty()) return;

      mlir::DominanceInfo domInfo(funcOp);
      llvm::DenseMap<llvm::hash_code, llvm::SmallVector<mlir::Operation*, 2>> candidates;

      std::function<void(mlir::Region&)> traverseRegion;

      std::function<void(llvm::DomTreeNodeBase<mlir::Block>*)> walkDomTree =
         [&](llvm::DomTreeNodeBase<mlir::Block>* node) {
            if (!node) return;
            processBlock(node->getBlock(), domInfo, candidates, traverseRegion);
            for (auto* child : node->children()) {
               walkDomTree(child);
            }
         };

      traverseRegion = [&](mlir::Region& region) {
         if (region.empty()) return;
         if (region.hasOneBlock()) {
            processBlock(&region.front(), domInfo, candidates, traverseRegion);
         } else {
            if (auto* root = domInfo.getRootNode(&region)) {
               walkDomTree(root);
            } else {
               for (auto& block : region)
                  processBlock(&block, domInfo, candidates, traverseRegion);
            }
         }
      };

      for (auto& region : funcOp->getRegions()) {
         traverseRegion(region);
      }

      if ((CSE_DEBUG | 1) && (successfulPhysicalMerges > 0 || successfulVirtualMerges > 0)) {
         llvm::errs() << "=========================================================\n";
         llvm::errs() << "CSE Pass Summary\n";
         llvm::errs() << "  Successful Physical Merges: " << successfulPhysicalMerges << "\n";
         llvm::errs() << "  Successful Virtual Merges: " << successfulVirtualMerges << "\n";
         llvm::errs() << "=========================================================\n";
      }
   }

   mlir::Value resolveValue(mlir::Value val) const {
      auto it = valueEquivalenceMap.find(val);
      while (it != valueEquivalenceMap.end()) {
         val = it->second;
         it = valueEquivalenceMap.find(val);
      }
      return val;
   }

   mlir::Attribute getCanonicalAttr(mlir::Attribute attr, const llvm::DenseMap<const tuples::Column*, std::string>* localDefs = nullptr) const {
      if (!attr) return {};

      if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
         auto ptrWrapper = colRef.getColumnPtr();
         if (!ptrWrapper) return attr;
         const auto* ptr = ptrWrapper.get();

         auto it = colContext.colMapping.find(ptr);
         if (it != colContext.colMapping.end()) {
            return tuples::ColumnRefAttr::get(attr.getContext(), it->second.name, it->second.col);
         }

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
      return attr;
   }

   private:
   llvm::hash_code computeHash(mlir::Operation* op) const {
      auto hash = llvm::hash_value(op->getName().getAsOpaquePointer());

      LLVM_DEBUG(llvm::dbgs() << " Hashing " << op->getName() << ":\n");

      for (auto opd : op->getOperands()) {
         auto resolved = resolveValue(opd);
         hash = llvm::hash_combine(hash, resolved);
         LLVM_DEBUG(llvm::dbgs() << "    Operand: " << resolved << "\n");
      }
      hash = llvm::hash_combine(hash, op->getNumRegions());

      llvm::SmallVector<mlir::NamedAttribute, 4> sortedAttrs;
      llvm::SmallSetVector<mlir::StringAttr, 4> attrNames;
      if (auto info = op->getRegisteredInfo()) {
         for (auto name : info->getAttributeNames()) attrNames.insert(name);
      }
      for (auto named : op->getAttrs()) attrNames.insert(named.getName());

      auto localDefs = getLocalDefsMap(op);

      for (auto name : attrNames) {
         if (mlir::isa<relalg::BaseTableOp>(op) && name == "columns") continue;

         if (auto val = op->getAttr(name)) {
            sortedAttrs.emplace_back(name, getCanonicalAttr(val, &localDefs));
         }
      }
      llvm::sort(sortedAttrs, [](const auto& a, const auto& b) { return a.getName().strref() < b.getName().strref(); });

      for (const auto& attr : sortedAttrs) {
         LLVM_DEBUG(llvm::dbgs() << "    Attr " << attr.getName().strref() << ": " << attr.getValue() << "\n");
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

   bool areEquivalent(mlir::Operation* leader, mlir::Operation* candidate) const {
      EquivalenceChecker ctx(*this);
      return ctx.checkOps(leader, candidate);
   }

   void processBlock(mlir::Block* block, mlir::DominanceInfo& domInfo,
                     llvm::DenseMap<llvm::hash_code, llvm::SmallVector<mlir::Operation*, 2>>& candidates,
                     llvm::function_ref<void(mlir::Region&)> traverseRegion) {
      for (auto& op : llvm::make_early_inc_range(*block)) {
         if (op.getDialect()->getNamespace() != "relalg") continue;

         auto hash = computeHash(&op);
         LLVM_DEBUG({
            llvm::dbgs() << "Visiting: " << op.getName() << " (" << &op << ") Hash: " << hash << "\n";
            if (mlir::isa<relalg::BaseTableOp>(op)) op.dump();
         });

         bool merged = false;
         if (auto it = candidates.find(hash); it != candidates.end()) {
            for (auto* leader : it->second) {
               LLVM_DEBUG(llvm::dbgs() << "  Comparing with leader: " << leader->getName() << " (" << leader << ")\n");

               if (domInfo.properlyDominates(leader, &op) && areEquivalent(leader, &op)) {
                  // if (!isSafeCrossRegionMerge(leader, &op)) continue;

                  if (mlir::isa<relalg::BaseTableOp>(leader)) {
                     mapVirtualBaseTableOpMerge(leader, &op);
                  } else if (mlir::isa<relalg::RenamingOp>(leader) || mlir::isa<relalg::AggrFuncOp>(leader)) {
                     mapVirtualMerge(leader, &op);
                  } else {
                     physicallyMergeOps(leader, &op);
                  }
                  merged = true;
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
   }

   // Merge Strategies
   void mapVirtualBaseTableOpMerge(mlir::Operation* leader, mlir::Operation* duplicate) {
      auto leaderBase = mlir::cast<relalg::BaseTableOp>(leader);
      auto dupBase = mlir::cast<relalg::BaseTableOp>(duplicate);

      auto lColsAttr = leaderBase.getColumnsAttr();
      auto dColsAttr = dupBase.getColumnsAttr();

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
            if (auto it = leaderCols.find(physName); it != leaderCols.end()) {
               lDef = mlir::cast<tuples::ColumnDefAttr>(it->second);
            } else {
               leaderCols[physName] = dDef;
               leaderChanged = true;
               lDef = dDef;
            }

            if (dDef.getColumnPtr() && lDef.getColumnPtr() && dDef.getColumnPtr() != lDef.getColumnPtr()) {
               colContext.addMapping(dDef.getColumnPtr().get(), lDef.getColumnPtr(), lDef.getName());
            }
         }
      }

      if (leaderChanged) {
         llvm::SmallVector<mlir::NamedAttribute> newLeaderCols;
         for (auto it : leaderCols) newLeaderCols.emplace_back(it.first, it.second);
         llvm::sort(newLeaderCols, [](const mlir::NamedAttribute& a, const mlir::NamedAttribute& b) {
            return a.getName().strref() < b.getName().strref();
         });
         leaderBase.setColumnsAttr(mlir::DictionaryAttr::get(leader->getContext(), newLeaderCols));
      }

      if (duplicate->getNumResults() > 0 && leader->getNumResults() > 0) {
         valueEquivalenceMap[duplicate->getResult(0)] = leader->getResult(0);
      }

      successfulVirtualMerges++;
      LLVM_DEBUG({
         llvm::dbgs() << "[CSE virtual REPLACE] " << duplicate->getName() << " -> " << leader->getName() << "\n";
         leader->dump();
         duplicate->dump();
      });
   }

   void mapVirtualMerge(mlir::Operation* leader, mlir::Operation* duplicate) {
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> lDefs, dDefs;
      collectSortedLocalDefs(leader, lDefs);
      collectSortedLocalDefs(duplicate, dDefs);

      if (lDefs.size() == dDefs.size()) {
         for (size_t i = 0; i < lDefs.size(); ++i) {
            auto lDef = lDefs[i].second;
            auto dDef = dDefs[i].second;

            auto lPtr = lDef.getColumnPtr();
            auto dPtr = dDef.getColumnPtr();
            if (lPtr && dPtr && lPtr != dPtr) {
               colContext.addMapping(dPtr.get(), lPtr, lDef.getName());
            }
         }
      }

      if (duplicate->getNumResults() > 0 && leader->getNumResults() > 0) {
         valueEquivalenceMap[duplicate->getResult(0)] = leader->getResult(0);
      }

      LLVM_DEBUG({
         llvm::dbgs() << "[CSE VIRTUAL] " << duplicate->getName() << " -> " << leader->getName() << "\n";
         leader->dump();
         duplicate->dump();
      });
      successfulVirtualMerges++;
   }

   void physicallyMergeOps(mlir::Operation* leader, mlir::Operation* duplicate) {
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> lDefs, dDefs;
      collectSortedLocalDefs(leader, lDefs);
      collectSortedLocalDefs(duplicate, dDefs);

      if (lDefs.size() == dDefs.size()) {
         for (size_t i = 0; i < lDefs.size(); ++i) {
            auto lDef = lDefs[i].second;
            auto dDef = dDefs[i].second;
            auto lPtr = lDef.getColumnPtr();
            auto dPtr = dDef.getColumnPtr();

            if (lPtr && dPtr && lPtr != dPtr) {
               colContext.colMapping[dPtr.get()] = {lPtr, lDef.getName()};
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

         if (auto it = colContext.colMapping.find(dPtr); it != colContext.colMapping.end()) {
            lCol = it->second.col;
            lName = it->second.name;
         } else if (availableLeaderCols.count(dPtr)) {
            lCol = dPtrWrapper;
            lName = dDef.getName();
         } else {
            if (auto it = colContext.inverseColMapping.find(dPtr); it != colContext.inverseColMapping.end()) {
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
            resolveLeaderColumn(leader, duplicate, dPtr, dDef, lCol, lName, availableLeaderCols);
         }

         if (lCol) {
            bool available = availableLeaderCols.count(lCol.get());
            if (!available && lCol == dDef.getColumnPtr()) available = true;

            if (available) {
               if (lCol != dDef.getColumnPtr()) {
                  auto lRef = tuples::ColumnRefAttr::get(builder.getContext(), lName, lCol);
                  auto newDef = tuples::ColumnDefAttr::get(
                     builder.getContext(),
                     dDef.getName(),
                     dDef.getColumnPtr(),
                     builder.getArrayAttr({lRef}));
                  renamingAttrs.push_back(newDef);
               }
            }
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
         valueEquivalenceMap[replacement] = leader->getResult(0);
      }

      duplicate->getResult(0).replaceAllUsesWith(replacement);

      LLVM_DEBUG({
         llvm::dbgs() << "[CSE REPLACE] " << duplicate->getName() << " -> " << leader->getName() << "\n";
         leader->dump();
         duplicate->dump();
         if (auto* r = replacement.getDefiningOp()) r->dump();
      });
      duplicate->erase();
      successfulPhysicalMerges++;
   }

   void resolveLeaderColumn(mlir::Operation* leader, mlir::Operation* duplicate,
                            const tuples::Column* dPtr, const tuples::ColumnDefAttr& dDef,
                            std::shared_ptr<tuples::Column>& lCol, mlir::SymbolRefAttr& lName,
                            llvm::DenseSet<const tuples::Column*>& availableLeaderCols) {
      auto [dSourceOp, path] = findDefiningOp(duplicate, dPtr);
      if (!dSourceOp) return;

      auto* lSourceOp = followPath(leader, path);
      if (!lSourceOp) return;

      if (auto dBase = mlir::dyn_cast<relalg::BaseTableOp>(dSourceOp)) {
         if (auto lBase = mlir::dyn_cast<relalg::BaseTableOp>(lSourceOp)) {
            auto dCols = dBase.getColumnsAttr();
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
               auto lCols = lBase.getColumnsAttr();
               llvm::SmallVector<mlir::NamedAttribute> newCols;
               if (lCols) newCols.append(lCols.begin(), lCols.end());

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
                  lBase.setColumnsAttr(mlir::DictionaryAttr::get(lBase.getContext(), newCols));
                  lCol = dDef.getColumnPtr();
                  lName = dDef.getName();
               } else {
                  lCol = existingLDef.getColumnPtr();
                  lName = existingLDef.getName();
               }
               availableLeaderCols.insert(lCol.get());
               return;
            }
         }
      }

      if (auto dRename = mlir::dyn_cast<relalg::RenamingOp>(dSourceOp)) {
         if (!mlir::isa<relalg::RenamingOp>(lSourceOp)) {
            if (auto colsAttr = dRename.getColumns()) {
               for (auto attr : colsAttr) {
                  if (auto def = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
                     if (def.getColumnPtr().get() == dPtr) {
                        if (auto from = def.getFromExisting()) {
                           if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(from)) {
                              if (arr.size() == 1) {
                                 if (auto ref = mlir::dyn_cast<tuples::ColumnRefAttr>(arr[0])) {
                                    if (auto innerPtr = ref.getColumnPtr()) {
                                       lCol = innerPtr;
                                       lName = ref.getName();
                                       availableLeaderCols.insert(lCol.get());
                                       return;
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
};

} // namespace

std::unique_ptr<mlir::Pass> relalg::createCommonSubtreeEliminationPass() {
   return std::make_unique<CommonSubtreeElimination>();
}