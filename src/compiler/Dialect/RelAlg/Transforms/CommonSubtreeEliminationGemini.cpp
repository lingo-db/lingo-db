#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

// Toggle to 1 to enable debug output
#define CSE_DEBUG 0

namespace {

using namespace lingodb::compiler::dialect;

/// Utilities for column introspection and structural traversal of RelAlg operators.
struct ColumnUtils {
   /// Collects column definitions defined locally by an operation.
   /// Returns a map from Column pointer to its leaf name.
   static llvm::DenseMap<const tuples::Column*, std::string> getLocalDefs(mlir::Operation* op) {
      llvm::DenseMap<const tuples::Column*, std::string> localDefs;
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> defs;

      collectSortedDefs(op, defs);

      for (const auto& [name, attr] : defs) {
         if (auto ptr = attr.getColumnPtr()) {
            localDefs[ptr.get()] = name;
         }
      }
      return localDefs;
   }

   /// Recursively collects all column definitions in the subtree rooted at `op`.
   static void collectRecursiveDefs(mlir::Operation* op,
                                    llvm::SmallVectorImpl<std::pair<std::string, tuples::ColumnDefAttr>>& out,
                                    llvm::SmallPtrSetImpl<mlir::Operation*>& visited) {
      if (!visited.insert(op).second) return;

      collectSortedDefs(op, out);

      for (auto operand : op->getOperands()) {
         if (auto* defOp = operand.getDefiningOp()) {
            if (defOp->getDialect()->getNamespace() == "relalg") {
               collectRecursiveDefs(defOp, out, visited);
            }
         }
      }
   }

   /// Computes the set of columns visible/available at the result of `op`.
   /// This respects projection barriers, join semantics, and renamings.
   static llvm::DenseSet<const tuples::Column*> getAvailableColumns(mlir::Operation* op) {
      llvm::DenseSet<const tuples::Column*> available;
      llvm::SmallPtrSet<mlir::Operation*, 8> visited;
      collectAvailableColumnsImpl(op, available, visited);
      return available;
   }

   private:
   static void collectDefsFromAttr(mlir::Attribute attr, llvm::SmallVectorImpl<std::pair<std::string, tuples::ColumnDefAttr>>& out) {
      llvm::TypeSwitch<mlir::Attribute>(attr)
         .Case<tuples::ColumnDefAttr>([&](auto def) {
            if (def.getName())
               out.emplace_back(def.getName().getLeafReference().getValue().str(), def);
         })
         .Case<mlir::ArrayAttr>([&](auto arr) {
            for (auto e : arr) collectDefsFromAttr(e, out);
         })
         .Case<mlir::DictionaryAttr>([&](auto dict) {
            for (auto e : dict) collectDefsFromAttr(e.getValue(), out);
         });
   }

   static void collectAvailableColumnsImpl(mlir::Operation* op, llvm::DenseSet<const tuples::Column*>& out, llvm::SmallPtrSetImpl<mlir::Operation*>& visited) {
      if (!visited.insert(op).second) return;

      // 1. Local definitions are always available (unless masked by the op itself, handled by logic below)
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> localDefs;
      collectSortedDefs(op, localDefs);
      for (const auto& p : localDefs) {
         if (auto ptr = p.second.getColumnPtr()) out.insert(ptr.get());
      }

      if (mlir::isa<relalg::BaseTableOp>(op)) return;

      // 2. Renaming: Recurse but filter out shadowed columns
      if (auto renameOp = mlir::dyn_cast<relalg::RenamingOp>(op)) {
         llvm::DenseSet<const tuples::Column*> shadowed;
         if (auto colsAttr = renameOp.getColumns()) {
            for (auto attr : colsAttr) {
               if (auto def = mlir::dyn_cast<tuples::ColumnDefAttr>(attr))
                  if (auto from = def.getFromExisting())
                     if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(from))
                        for (auto ref : arr)
                           if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(ref))
                              if (auto ptr = colRef.getColumnPtr()) shadowed.insert(ptr.get());
            }
         }
         if (op->getNumOperands() > 0)
            if (auto* defOp = op->getOperand(0).getDefiningOp()) {
               llvm::DenseSet<const tuples::Column*> childCols;
               collectAvailableColumnsImpl(defOp, childCols, visited);
               for (auto* c : childCols)
                  if (!shadowed.count(c)) out.insert(c);
            }
         return;
      }

      // 3. Operators that block child columns (Aggregation, Projection)
      if (mlir::isa<relalg::AggregationOp, relalg::ProjectionOp>(op)) return;

      // 4. Semi/Anti joins: Only left child propagates
      bool recurseChildren = true;
      llvm::TypeSwitch<mlir::Operation*>(op)
         .Case<relalg::SemiJoinOp, relalg::AntiSemiJoinOp, relalg::MarkJoinOp, relalg::IntersectOp, relalg::ExceptOp>([&](auto) {
            if (op->getNumOperands() > 0) {
               if (auto* defOp = op->getOperand(0).getDefiningOp())
                  collectAvailableColumnsImpl(defOp, out, visited);
            }
            recurseChildren = false;
         })
         .Case<relalg::GroupJoinOp>([&](auto) {
            // GroupJoin propagates left + aggregated columns (local defs)
            if (op->getNumOperands() > 0) {
               if (auto* defOp = op->getOperand(0).getDefiningOp())
                  collectAvailableColumnsImpl(defOp, out, visited);
            }
            recurseChildren = false;
         });

      if (recurseChildren) {
         for (auto operand : op->getOperands()) {
            if (auto* defOp = operand.getDefiningOp()) {
               if (defOp->getDialect()->getNamespace() == "relalg")
                  collectAvailableColumnsImpl(defOp, out, visited);
            }
         }
      }
   }

   public:
   static void collectSortedDefs(mlir::Operation* op, llvm::SmallVectorImpl<std::pair<std::string, tuples::ColumnDefAttr>>& out) {
      // Sort attributes by name to ensure deterministic order for structural matching
      llvm::SmallVector<mlir::NamedAttribute> attrs(op->getAttrs().begin(), op->getAttrs().end());
      llvm::sort(attrs, [](const mlir::NamedAttribute& a, const mlir::NamedAttribute& b) {
         return a.getName().strref() < b.getName().strref();
      });

      for (const auto& attr : attrs) {
         collectDefsFromAttr(attr.getValue(), out);
      }
   }
};

class CommonSubtreeElimination : public mlir::PassWrapper<CommonSubtreeElimination, mlir::OperationPass<mlir::func::FuncOp>> {
   struct LeaderInfo {
      std::shared_ptr<tuples::Column> col;
      mlir::SymbolRefAttr name;
   };

   // Maps DuplicateColumn -> LeaderColumn (for virtual merges)
   llvm::DenseMap<const tuples::Column*, LeaderInfo> colMapping;
   // Maps LeaderColumn -> List of DuplicateColumns (for inverse lookup)
   llvm::DenseMap<const tuples::Column*, llvm::SmallVector<LeaderInfo, 1>> inverseColMapping;
   // Maps value equivalence (Duplicate Result -> Leader Result)
   llvm::DenseMap<mlir::Value, mlir::Value> equivalenceMap;

   size_t successfulPhysicalMerges = 0;
   size_t successfulVirtualMerges = 0;
   size_t failedPhysicalMerges = 0;

   mlir::Value resolveValue(mlir::Value val) const {
      while (true) {
         auto it = equivalenceMap.find(val);
         if (it == equivalenceMap.end()) return val;
         val = it->second;
      }
   }

   /// Canonicalizes attributes for structural comparison.
   /// - Maps duplicate column pointers to leader column pointers.
   /// - Abstracts local column definitions to "LocalColRef::name".
   mlir::Attribute getCanonicalAttr(mlir::Attribute attr, const llvm::DenseMap<const tuples::Column*, std::string>* localDefs = nullptr) const {
      if (!attr) return {};

      return llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(attr)
         .Case<tuples::ColumnRefAttr>([&](auto colRef) -> mlir::Attribute {
            auto ptrWrapper = colRef.getColumnPtr();
            if (!ptrWrapper) return attr;
            const auto* ptr = ptrWrapper.get();

            // 1. Virtual Merge Mapping
            if (auto it = colMapping.find(ptr); it != colMapping.end()) {
               return tuples::ColumnRefAttr::get(attr.getContext(), it->second.name, it->second.col);
            }

            // 2. Local Structure Abstraction
            if (localDefs) {
               if (auto it = localDefs->find(ptr); it != localDefs->end()) {
                  return mlir::StringAttr::get(attr.getContext(), "LocalColRef::" + it->second);
               }
            }
            return attr;
         })
         .Case<tuples::ColumnDefAttr>([&](auto colDef) -> mlir::Attribute {
            if (auto from = colDef.getFromExisting()) {
               return getCanonicalAttr(from, localDefs);
            }
            return mlir::StringAttr::get(attr.getContext(), "ColDef");
         })
         .Case<mlir::ArrayAttr>([&](auto arr) -> mlir::Attribute {
            llvm::SmallVector<mlir::Attribute> newElems;
            bool changed = false;
            for (auto e : arr) {
               auto c = getCanonicalAttr(e, localDefs);
               newElems.push_back(c);
               if (c != e) changed = true;
            }
            return changed ? mlir::ArrayAttr::get(attr.getContext(), newElems) : attr;
         })
         .Case<mlir::DictionaryAttr>([&](auto dict) -> mlir::Attribute {
            llvm::SmallVector<mlir::NamedAttribute> newAttrs;
            bool changed = false;
            for (auto named : dict) {
               auto newVal = getCanonicalAttr(named.getValue(), localDefs);
               newAttrs.emplace_back(named.getName(), newVal);
               if (newVal != named.getValue()) changed = true;
            }
            return changed ? mlir::DictionaryAttr::get(attr.getContext(), newAttrs) : attr;
         })
         .Default([](auto a) { return a; });
   }

   llvm::hash_code computeHash(mlir::Operation* op) const {
      auto hash = llvm::hash_value(op->getName().getAsOpaquePointer());

      for (auto opd : op->getOperands()) {
         hash = llvm::hash_combine(hash, resolveValue(opd));
      }
      hash = llvm::hash_combine(hash, op->getNumRegions());

      auto localDefs = ColumnUtils::getLocalDefs(op);
      llvm::SmallVector<mlir::NamedAttribute> sortedAttrs(op->getAttrs().begin(), op->getAttrs().end());
      llvm::sort(sortedAttrs, [](const mlir::NamedAttribute& a, const mlir::NamedAttribute& b) {
         return a.getName().strref() < b.getName().strref();
      });

      for (const auto& attr : sortedAttrs) {
         llvm::StringRef name = attr.getName().strref();
         if (name == "rows" || name == "total_rows") continue;
         if (mlir::isa<relalg::BaseTableOp>(op) && name == "columns") continue;

         auto canonical = getCanonicalAttr(attr.getValue(), &localDefs);
         hash = llvm::hash_combine(hash, name, canonical);
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

   struct EquivalenceChecker {
      CommonSubtreeElimination& parent;
      llvm::DenseMap<mlir::Value, mlir::Value> candidateToLeaderVal;

      EquivalenceChecker(CommonSubtreeElimination& parent) : parent(parent) {}

      bool check(mlir::Operation* leader, mlir::Operation* candidate) {
         if (leader->getName() != candidate->getName()) return false;
         if (leader->getNumOperands() != candidate->getNumOperands()) return false;
         if (leader->getNumRegions() != candidate->getNumRegions()) return false;
         if (leader->getNumResults() != candidate->getNumResults()) return false;

         for (auto [lOp, cOp] : llvm::zip(leader->getOperands(), candidate->getOperands())) {
            mlir::Value cResolved = candidateToLeaderVal.count(cOp) ? candidateToLeaderVal.lookup(cOp) : parent.resolveValue(cOp);
            mlir::Value lResolved = parent.resolveValue(lOp);
            if (cResolved != lResolved) return false;
         }

         if (!checkAttributes(leader, candidate)) return false;

         for (auto [lRes, cRes] : llvm::zip(leader->getResults(), candidate->getResults())) {
            candidateToLeaderVal[cRes] = lRes;
         }

         for (auto [lReg, cReg] : llvm::zip(leader->getRegions(), candidate->getRegions())) {
            if (!checkRegion(lReg, cReg)) return false;
         }
         return true;
      }

      private:
      bool checkAttributes(mlir::Operation* leader, mlir::Operation* candidate) const {
         auto lLocals = ColumnUtils::getLocalDefs(leader);
         auto cLocals = ColumnUtils::getLocalDefs(candidate);

         llvm::SmallSetVector<mlir::StringAttr, 4> allNames;
         auto addNames = [&](mlir::Operation* op) {
            for (auto attr : op->getAttrs()) allNames.insert(attr.getName());
         };
         addNames(leader);
         addNames(candidate);

         for (auto name : allNames) {
            llvm::StringRef nameRef = name.strref();
            if (nameRef == "rows" || nameRef == "total_rows") continue;
            if (mlir::isa<relalg::BaseTableOp>(leader) && nameRef == "columns") continue;

            auto lVal = leader->getAttr(name);
            auto cVal = candidate->getAttr(name);
            if (!lVal || !cVal) return false;

            if (parent.getCanonicalAttr(lVal, &lLocals) != parent.getCanonicalAttr(cVal, &cLocals))
               return false;
         }
         return true;
      }

      bool checkRegion(mlir::Region& r1, mlir::Region& r2) {
         if (r1.empty() != r2.empty()) return false;
         if (r1.empty()) return true;

         auto it1 = r1.begin(), it2 = r2.begin();
         while (it1 != r1.end() && it2 != r2.end()) {
            mlir::Block& b1 = *it1++;
            mlir::Block& b2 = *it2++;

            if (b1.getNumArguments() != b2.getNumArguments()) return false;
            for (auto [arg1, arg2] : llvm::zip(b1.getArguments(), b2.getArguments())) {
               candidateToLeaderVal[arg2] = arg1;
            }

            auto opIt1 = b1.begin(), opIt2 = b2.begin();
            while (opIt1 != b1.end() && opIt2 != b2.end()) {
               if (!check(&*opIt1, &*opIt2)) return false;
               ++opIt1;
               ++opIt2;
            }
            if (opIt1 != b1.end() || opIt2 != b2.end()) return false;
         }
         return it1 == r1.end() && it2 == r2.end();
      }
   };

   void updateColumnMappings(mlir::Operation* leader, mlir::Operation* duplicate) {
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> lDefs, dDefs;
      ColumnUtils::collectSortedDefs(leader, lDefs);
      ColumnUtils::collectSortedDefs(duplicate, dDefs);

      if (lDefs.size() != dDefs.size()) return;

      for (auto [lPair, dPair] : llvm::zip(lDefs, dDefs)) {
         auto lPtr = lPair.second.getColumnPtr();
         auto dPtr = dPair.second.getColumnPtr();

         if (lPtr && dPtr && lPtr != dPtr) {
            colMapping[dPtr.get()] = {lPtr, lPair.second.getName()};
            inverseColMapping[lPtr.get()].push_back({dPtr, dPair.second.getName()});
         }
      }
   }

   void mergeBaseTables(relalg::BaseTableOp leader, relalg::BaseTableOp duplicate) {
      auto lCols = leader.getColumnsAttr();
      auto dCols = duplicate.getColumnsAttr();

      llvm::DenseMap<mlir::StringAttr, mlir::Attribute> mergedCols;
      if (lCols) {
         for (auto named : lCols) mergedCols[named.getName()] = named.getValue();
      }

      bool leaderModified = false;
      if (dCols) {
         for (auto named : dCols) {
            auto physName = named.getName();
            auto dDef = mlir::cast<tuples::ColumnDefAttr>(named.getValue());

            tuples::ColumnDefAttr lDef;
            if (mergedCols.count(physName)) {
               lDef = mlir::cast<tuples::ColumnDefAttr>(mergedCols[physName]);
            } else {
               mergedCols[physName] = dDef;
               leaderModified = true;
               lDef = dDef;
            }

            if (dDef.getColumnPtr() != lDef.getColumnPtr()) {
               colMapping[dDef.getColumnPtr().get()] = {lDef.getColumnPtr(), lDef.getName()};
               inverseColMapping[lDef.getColumnPtr().get()].push_back({dDef.getColumnPtr(), dDef.getName()});
            }
         }
      }

      if (leaderModified) {
         llvm::SmallVector<mlir::NamedAttribute> newCols;
         for (auto& [name, attr] : mergedCols) newCols.emplace_back(name, attr);
         llvm::sort(newCols, [](const mlir::NamedAttribute& a, const mlir::NamedAttribute& b) {
            return a.getName().strref() < b.getName().strref();
         });
         leader.setColumnsAttr(mlir::DictionaryAttr::get(leader.getContext(), newCols));
      }

      equivalenceMap[duplicate.getResult()] = leader.getResult();
      successfulVirtualMerges++;
      if (CSE_DEBUG) llvm::errs() << "MERGE (Virtual BaseTable): " << duplicate->getName() << " -> " << leader->getName() << "\n";
   }

   std::optional<LeaderInfo> resolveLeaderColumn(
      mlir::Operation* leader, mlir::Operation* duplicate,
      const tuples::Column* targetPtr, const tuples::ColumnDefAttr& targetDef) {
      // DFS to find path in duplicate subtree (top-down definition path)
      auto findPath = [&](mlir::Operation* root, const tuples::Column* ptr) -> std::pair<mlir::Operation*, std::vector<unsigned>> {
         std::vector<std::pair<mlir::Operation*, std::vector<unsigned>>> stack;
         stack.push_back({root, {}});

         while (!stack.empty()) {
            auto [curr, path] = stack.back();
            stack.pop_back();

            auto local = ColumnUtils::getLocalDefs(curr);
            if (local.count(ptr)) return {curr, path};

            for (size_t i = 0; i < curr->getNumOperands(); ++i) {
               if (auto defOp = curr->getOperand(i).getDefiningOp()) {
                  if (defOp->getDialect()->getNamespace() == "relalg") {
                     auto newPath = path;
                     newPath.push_back(i);
                     stack.push_back({defOp, newPath});
                  }
               }
            }
         }
         return {nullptr, {}};
      };

      auto [dSourceOp, path] = findPath(duplicate, targetPtr);
      if (!dSourceOp) return std::nullopt;

      // Trace same path in Leader
      mlir::Operation* lSourceOp = leader;
      for (unsigned opIdx : path) {
         if (opIdx >= lSourceOp->getNumOperands()) return std::nullopt;
         lSourceOp = lSourceOp->getOperand(opIdx).getDefiningOp();
         if (!lSourceOp) return std::nullopt;
      }

      // Case 1: BaseTable (Schema Merge)
      if (auto dBase = mlir::dyn_cast<relalg::BaseTableOp>(dSourceOp)) {
         if (auto lBase = mlir::dyn_cast<relalg::BaseTableOp>(lSourceOp)) {
            auto dCols = dBase.getColumnsAttr();
            mlir::StringAttr physName;
            for (auto named : dCols) {
               if (mlir::cast<tuples::ColumnDefAttr>(named.getValue()).getColumnPtr() == targetDef.getColumnPtr()) {
                  physName = named.getName();
                  break;
               }
            }
            if (physName) {
               // If physical names match, we can equate them.
               // We only inject into Leader BaseTable if leader == lBase (we are processing the BaseTable itself)
               // OR if we know the schema modification won't be blocked.
               auto lCols = lBase.getColumnsAttr();
               auto lColVal = lCols ? lCols.get(physName) : nullptr;
               if (lColVal) {
                  auto existing = mlir::cast<tuples::ColumnDefAttr>(lColVal);
                  return LeaderInfo{existing.getColumnPtr(), existing.getName()};
               } else {
                  // If we are deep inside a tree, modifying the BaseTable schema is risky unless we know it propagates.
                  // For safety, we only modify if the leader op *is* the BaseTable, or we rely on the caller to check availability.
                  llvm::SmallVector<mlir::NamedAttribute> newCols(lCols ? lCols.getValue() : llvm::ArrayRef<mlir::NamedAttribute>{});
                  newCols.emplace_back(physName, targetDef);
                  llvm::sort(newCols, [](const mlir::NamedAttribute& a, const mlir::NamedAttribute& b) {
                     return a.getName().strref() < b.getName().strref();
                  });
                  lBase.setColumnsAttr(mlir::DictionaryAttr::get(lBase.getContext(), newCols));
                  return LeaderInfo{targetDef.getColumnPtr(), targetDef.getName()};
               }
            }
         }
      }

      // Case 2: Structural Match
      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> lDefs, dDefs;
      ColumnUtils::collectSortedDefs(lSourceOp, lDefs);
      ColumnUtils::collectSortedDefs(dSourceOp, dDefs);

      if (lDefs.size() == dDefs.size()) {
         for (size_t i = 0; i < dDefs.size(); ++i) {
            if (dDefs[i].second.getColumnPtr() == targetDef.getColumnPtr()) {
               return LeaderInfo{lDefs[i].second.getColumnPtr(), lDefs[i].second.getName()};
            }
         }
      }
      return std::nullopt;
   }

   bool tryPhysicalMerge(mlir::Operation* leader, mlir::Operation* duplicate) {
      updateColumnMappings(leader, duplicate);

      auto availableLeaderCols = ColumnUtils::getAvailableColumns(leader);
      auto availableDupCols = ColumnUtils::getAvailableColumns(duplicate);

      llvm::SmallVector<std::pair<std::string, tuples::ColumnDefAttr>> allDupDefs;
      llvm::SmallPtrSet<mlir::Operation*, 8> visited;
      ColumnUtils::collectRecursiveDefs(duplicate, allDupDefs, visited);
      // Sort for deterministic renaming generation
      llvm::sort(allDupDefs, [](const auto& a, const auto& b) { return a.first < b.first; });

      mlir::OpBuilder builder(duplicate);
      llvm::SmallVector<mlir::Attribute> renamingAttrs;
      llvm::DenseSet<const tuples::Column*> processed;

      for (const auto& [key, dDef] : allDupDefs) {
         auto dPtrWrapper = dDef.getColumnPtr();
         if (!dPtrWrapper || processed.count(dPtrWrapper.get())) continue;
         if (!availableDupCols.count(dPtrWrapper.get())) continue;

         processed.insert(dPtrWrapper.get());
         const tuples::Column* dPtr = dPtrWrapper.get();

         std::shared_ptr<tuples::Column> lCol = nullptr;
         mlir::SymbolRefAttr lName;

         // 1. Direct Lookup
         if (auto it = colMapping.find(dPtr); it != colMapping.end()) {
            lCol = it->second.col;
            lName = it->second.name;
         }
         // 2. Identity (Already available in leader scope, e.g. shared subtree)
         else if (availableLeaderCols.count(dPtr)) {
            lCol = dPtrWrapper;
            lName = dDef.getName();
         }
         // 3. Inverse Lookup
         else if (auto it = inverseColMapping.find(dPtr); it != inverseColMapping.end()) {
            for (const auto& info : it->second) {
               if (availableLeaderCols.count(info.col.get())) {
                  lCol = info.col;
                  lName = info.name;
                  break;
               }
            }
         }

         // 4. Structural Resolution (Recursive search)
         if (!lCol || !availableLeaderCols.count(lCol.get())) {
            if (auto resolved = resolveLeaderColumn(leader, duplicate, dPtr, dDef)) {
               lCol = resolved->col;
               lName = resolved->name;
            }
         }

         // Strict Verification: The resolved column MUST be available at the Leader's result.
         if (lCol && (availableLeaderCols.count(lCol.get()) || (lCol == dDef.getColumnPtr() && availableLeaderCols.count(lCol.get())))) {
            // Only create renaming if pointer differs
            if (lCol != dDef.getColumnPtr()) {
               auto lRef = tuples::ColumnRefAttr::get(builder.getContext(), lName, lCol);
               auto newDef = tuples::ColumnDefAttr::get(builder.getContext(), dDef.getName(), dDef.getColumnPtr(), builder.getArrayAttr({lRef}));
               renamingAttrs.push_back(newDef);
            }
         } else {
            // If we can't find a valid replacement for an exposed column, we cannot merge physically.
            if (CSE_DEBUG) llvm::errs() << "FAIL MERGE: Could not resolve column " << dDef.getName() << "\n";
            failedPhysicalMerges++;
            return false;
         }
      }

      mlir::Value replacement = leader->getResult(0);
      if (!renamingAttrs.empty()) {
         replacement = builder.create<relalg::RenamingOp>(duplicate->getLoc(), replacement, builder.getArrayAttr(renamingAttrs));
      }

      if (replacement != leader->getResult(0)) {
         equivalenceMap[replacement] = leader->getResult(0);
      }

      duplicate->getResult(0).replaceAllUsesWith(replacement);
      duplicate->erase();
      successfulPhysicalMerges++;
      if (CSE_DEBUG) llvm::errs() << "MERGE (Physical): " << duplicate->getName() << " -> " << leader->getName() << "\n";
      return true;
   }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonSubtreeElimination)
   llvm::StringRef getArgument() const override { return "relalg-cse"; }

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

      auto traverse = [&](auto&& self, mlir::Region& region) -> void {
         if (region.empty()) return;

         auto processBlock = [&](mlir::Block* block) {
            for (auto& op : llvm::make_early_inc_range(*block)) {
               if (op.getDialect()->getNamespace() != "relalg") continue;

               auto hash = computeHash(&op);
               bool merged = false;

               if (auto it = candidates.find(hash); it != candidates.end()) {
                  for (auto* leader : it->second) {
                     if (domInfo.properlyDominates(leader, &op) && EquivalenceChecker(*this).check(leader, &op)) {
                        // Safety check for cross-region data flow
                        if (auto ancestor = op.getParentOp(); ancestor && ancestor != leader && ancestor != op.getParentOp()) {
                           bool safe = true;
                           mlir::Operation* curr = &op;
                           while (curr && curr->getBlock() != leader->getBlock()) {
                              curr = curr->getParentOp();
                              if (curr == leader) {
                                 safe = false;
                                 break;
                              }
                              for (auto opd : curr->getOperands())
                                 if (opd == leader->getResult(0)) {
                                    safe = false;
                                    break;
                                 }
                           }
                           if (!safe) continue;
                        }

                        if (mlir::isa<relalg::AggrFuncOp>(op)) continue;

                        if (auto lTable = mlir::dyn_cast<relalg::BaseTableOp>(leader)) {
                           mergeBaseTables(lTable, mlir::cast<relalg::BaseTableOp>(op));
                           merged = true;
                           break;
                        }

                        bool isPhysicalCandidate = !mlir::isa<relalg::RenamingOp, relalg::GroupJoinOp, relalg::WindowOp, relalg::ConstRelationOp>(op);

                        if (isPhysicalCandidate && tryPhysicalMerge(leader, &op)) {
                           merged = true;
                           break;
                        } else {
                           updateColumnMappings(leader, &op);
                           if (op.getNumResults() > 0) equivalenceMap[op.getResult(0)] = leader->getResult(0);
                           successfulVirtualMerges++;
                           if (CSE_DEBUG) llvm::errs() << "MERGE (Virtual): " << op.getName() << " -> " << leader->getName() << "\n";
                           merged = true;
                           break;
                        }
                     }
                  }
               }

               if (!merged) {
                  candidates[hash].push_back(&op);
                  for (auto& r : op.getRegions()) self(self, r);
               }
            }
         };

         // Fast path for single-block regions (avoids DominanceInfo crashes on predicates)
         if (region.hasOneBlock()) {
            processBlock(&region.front());
         } else if (auto* root = domInfo.getRootNode(&region)) {
            std::function<void(llvm::DomTreeNodeBase<mlir::Block>*)> walk =
               [&](auto* node) {
                  processBlock(node->getBlock());
                  for (auto* child : node->children()) walk(child);
               };
            walk(root);
         } else {
            for (auto& block : region) processBlock(&block);
         }
      };

      for (auto& region : funcOp->getRegions()) traverse(traverse, region);

      if ((CSE_DEBUG | 1) && (successfulPhysicalMerges > 0 || successfulVirtualMerges > 0 || failedPhysicalMerges > 0)) {
         llvm::errs() << "=========================================================\n";
         llvm::errs() << "CSE Pass Summary\n";
         llvm::errs() << "  Successful Physical Merges: " << successfulPhysicalMerges << "\n";
         if (failedPhysicalMerges > 0) llvm::errs() << "  Failed Physical Merges:     " << failedPhysicalMerges << "\n";
         llvm::errs() << "  Successful Virtual Merges: " << successfulVirtualMerges << "\n";
         llvm::errs() << "=========================================================\n";
      }
   }
};
} // namespace

std::unique_ptr<mlir::Pass> relalg::createCommonSubtreeEliminationPass() {
   return std::make_unique<CommonSubtreeElimination>();
}