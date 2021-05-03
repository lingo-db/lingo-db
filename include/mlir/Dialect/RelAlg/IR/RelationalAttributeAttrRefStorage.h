//
// Created by michael on 13.03.21.
//

#ifndef MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEATTRREFSTORAGE_H
#define MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEATTRREFSTORAGE_H
namespace mlir::relalg {
struct RelationalAttributeAttrRefStorage : public AttributeStorage {
  RelationalAttributeAttrRefStorage(
      SymbolRefAttr name, std::shared_ptr<RelationalAttribute> relationalAttribute)
      : name(name), relationalAttribute(relationalAttribute) {}

  using KeyTy = std::pair<SymbolRefAttr, std::shared_ptr<RelationalAttribute>>;

  bool operator==(const KeyTy &key) const {
    return key.first == name && key.second == relationalAttribute;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second.get());
  }
  static RelationalAttributeAttrRefStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RelationalAttributeAttrRefStorage>())
        RelationalAttributeAttrRefStorage(key.first, key.second);
  }
  SymbolRefAttr name;
  std::shared_ptr<RelationalAttribute> relationalAttribute;
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEATTRREFSTORAGE_H
