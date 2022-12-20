#ifndef MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
#define MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
namespace mlir::relalg {

class FunctionalDependencies {
   std::vector<std::pair<mlir::relalg::ColumnSet, mlir::relalg::ColumnSet>> fds;

   public:
   void insert(const FunctionalDependencies& other) {
      fds.insert(fds.end(), other.fds.begin(), other.fds.end());
   }
   void insert(const ColumnSet& left, const ColumnSet& right) {
      fds.push_back({left, right});
   }
   void dump(mlir::MLIRContext* context) {
      for (auto fd : fds) {
         fd.first.dump(context);
         llvm::dbgs() << "->";
         fd.second.dump(context);
         llvm::dbgs() << "\n";
      }
   }
   ColumnSet expand(const ColumnSet& available) {
      ColumnSet result = available;
      bool didChange = false;
      do {
         ColumnSet local = result;
         for (auto fd : fds) {
            if (fd.first.isSubsetOf(local)) {
               local.insert(fd.second);
            }
         }
         didChange = local.size() > result.size();
         result = local;
      } while (didChange);
      return result;
   }
   ColumnSet reduce(const ColumnSet& keys) {
      ColumnSet res = keys;
      for (auto* k : keys) {
         ColumnSet local = res;
         local.remove(ColumnSet::from(k));
         if (expand(local).intersect(keys).size() == keys.size()) {
            res.remove(ColumnSet::from(k));
         }
      }
      return res;
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H