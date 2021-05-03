#ifndef MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEATTRDEFSTORAGE_H
#define MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEATTRDEFSTORAGE_H
#include "mlir/IR/Attributes.h"
namespace mlir::relalg {
struct RelationalAttributeAttrDefStorage : public AttributeStorage {
  RelationalAttributeAttrDefStorage(
      StringRef name, std::shared_ptr<RelationalAttribute> relationalAttribute,Attribute fromExisting)
      : name(name), relationalAttribute(relationalAttribute),fromExisting(fromExisting) {}

  using KeyTy = std::tuple<std::string, std::shared_ptr<RelationalAttribute>,Attribute>;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == name &&  std::get<1>(key) == relationalAttribute &&  std::get<2>(key)==fromExisting;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key),std::get<1>(key).get(),std::get<2>(key));
  }
  static RelationalAttributeAttrDefStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RelationalAttributeAttrDefStorage>())
        RelationalAttributeAttrDefStorage(std::get<0>(key),std::get<1>(key),std::get<2>(key));
  }
  std::string name;
  std::shared_ptr<RelationalAttribute> relationalAttribute;
   Attribute fromExisting;

};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEATTRDEFSTORAGE_H
