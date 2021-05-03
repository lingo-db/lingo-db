#ifndef MLIR_DIALECT_RELALG_IR_SORTSPECIFICATIONATTRSTORAGE_H
#define MLIR_DIALECT_RELALG_IR_SORTSPECIFICATIONATTRSTORAGE_H
namespace mlir::relalg {
struct SortSpecificationAttrStorage : public AttributeStorage {
   SortSpecificationAttrStorage(RelationalAttributeRefAttr attr, SortSpec sortSpec)
      : attr(attr),sortSpec(sortSpec) {}

  using KeyTy = std::pair<RelationalAttributeRefAttr, SortSpec>;

  bool operator==(const KeyTy &key) const {
    return key.first == attr && key.second == sortSpec;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }
  static SortSpecificationAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<SortSpecificationAttrStorage>())
       SortSpecificationAttrStorage(key.first, key.second);
  }
  RelationalAttributeRefAttr attr;
  SortSpec sortSpec;
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_IR_SORTSPECIFICATIONATTRSTORAGE_H
