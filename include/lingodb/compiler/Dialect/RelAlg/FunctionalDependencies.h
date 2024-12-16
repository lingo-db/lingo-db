#ifndef LINGODB_COMPILER_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
#define LINGODB_COMPILER_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
namespace lingodb::compiler::dialect::relalg {

class FunctionalDependencies {
   std::vector<std::pair<lingodb::compiler::dialect::relalg::ColumnSet, lingodb::compiler::dialect::relalg::ColumnSet>> fds;
   /*
    * if set, these columns not only determine all other values functionally,
    * but are also unique (i.e. no other tuple with the same values will
    * occur in the tuple stream
    */
   std::optional<lingodb::compiler::dialect::relalg::ColumnSet> key;

   public:
   FunctionalDependencies() : fds(), key() {}

   void setKey(dialect::relalg::ColumnSet key) {
      this->key = key;
   }
   const std::optional<lingodb::compiler::dialect::relalg::ColumnSet>& getKey() {
      return key;
   }
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
      bool didChange;
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
   bool isDuplicateFreeKey(const ColumnSet& candidate) {
      if (key) {
         return key.value().isSubsetOf(expand(candidate));
      } else {
         return false;
      }
   }
};
} // namespace lingodb::compiler::dialect::relalg
#endif //LINGODB_COMPILER_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H