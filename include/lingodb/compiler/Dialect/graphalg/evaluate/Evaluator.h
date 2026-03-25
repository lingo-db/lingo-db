#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgCast.h"
#include "graphalg/GraphAlgTypes.h"

namespace graphalg {

/** Helper for reading elements of \c MatrixAttr. */
class MatrixAttrReader {
   private:
   MatrixType _type;
   std::size_t _rows;
   std::size_t _cols;
   llvm::ArrayRef<mlir::Attribute> _elems;

   public:
   MatrixAttrReader(MatrixAttr attr)
      : _type(llvm::cast<MatrixType>(attr.getType())),
        _rows(_type.getRows().getConcreteDim()),
        _cols(_type.getCols().getConcreteDim()),
        _elems(attr.getElems().getValue()) {}

   std::size_t nRows() const { return _rows; }
   std::size_t nCols() const { return _cols; }

   SemiringTypeInterface ring() const {
      return llvm::cast<SemiringTypeInterface>(_type.getSemiring());
   }

   mlir::TypedAttr at(std::size_t row, std::size_t col) const {
      assert(row < _rows);
      assert(col < _cols);
      return llvm::cast<mlir::TypedAttr>(_elems[row * _cols + col]);
   }
};

class MatrixAttrBuilder {
   private:
   MatrixType _type;
   SemiringTypeInterface _ring;
   std::size_t _rows;
   std::size_t _cols;
   llvm::SmallVector<mlir::Attribute> _elems;

   public:
   MatrixAttrBuilder(MatrixType type)
      : _type(type), _rows(_type.getRows().getConcreteDim()),
        _cols(_type.getCols().getConcreteDim()),
        _ring(llvm::cast<SemiringTypeInterface>(type.getSemiring())),
        _elems(_rows * _cols, _ring.addIdentity()) {}

   std::size_t nRows() const { return _rows; }
   std::size_t nCols() const { return _cols; }

   SemiringTypeInterface ring() const { return _ring; }

   void set(std::size_t row, std::size_t col, mlir::TypedAttr attr) {
      assert(row < _rows);
      assert(col < _cols);
      assert(attr.getType() == _ring);
      _elems[row * _cols + col] = attr;
   }

   MatrixAttr build() {
      auto* ctx = _type.getContext();
      return MatrixAttr::get(ctx, _type, mlir::ArrayAttr::get(ctx, _elems));
   }
};

MatrixAttr evaluate(mlir::func::FuncOp funcOp, llvm::ArrayRef<MatrixAttr> args);

} // namespace graphalg
