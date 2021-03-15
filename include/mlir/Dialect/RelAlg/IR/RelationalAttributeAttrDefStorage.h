//
// Created by michael on 13.03.21.
//

#ifndef MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEATTRDEFSTORAGE_H
#define MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEATTRDEFSTORAGE_H
namespace mlir::relalg {
struct RelationalAttributeAttrDefStorage : public AttributeStorage {
  RelationalAttributeAttrDefStorage(
      StringRef name, std::shared_ptr<RelationalAttribute> relationalAttribute)
      : name(name), relationalAttribute(relationalAttribute) {}

  using KeyTy = std::pair<std::string, std::shared_ptr<RelationalAttribute>>;

  bool operator==(const KeyTy &key) const {
    return key.first == name && key.second == relationalAttribute;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second.get());
  }
  static RelationalAttributeAttrDefStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RelationalAttributeAttrDefStorage>())
        RelationalAttributeAttrDefStorage(key.first, key.second);
  }
  std::string name;
  std::shared_ptr<RelationalAttribute> relationalAttribute;
};
} // namespace mlir::relalg
#endif // MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEATTRDEFSTORAGE_H
