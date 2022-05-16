#ifndef MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
#define MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H
namespace mlir::relalg {

class FunctionalDependencies {
   std::vector<std::pair<mlir::relalg::ColumnSet,mlir::relalg::ColumnSet>> fds;
   public:
   void insert(const FunctionalDependencies& other){
      fds.insert(fds.end(),other.fds.begin(),other.fds.end());
   }
   void insert(const ColumnSet& left, const ColumnSet& right){
      fds.push_back({left,right});
   }
   ColumnSet reduce(const ColumnSet& keys){
      ColumnSet res=keys;
      ColumnSet remove;
      for(auto fd:fds){
         if(fd.first.isSubsetOf(keys)){
            remove.insert(fd.second);
         }
      }
      res.remove(remove);
      return res;
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_FUNCTIONALDEPENDENCIES_H