#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgCast.h"
#include "graphalg/GraphAlgDialect.h"
#include "graphalg/GraphAlgOps.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"
#include "graphalg/evaluate/Evaluator.h"

namespace graphalg {

namespace {

class Evaluator {
   private:
   llvm::DenseMap<mlir::Value, MatrixAttr> _values;

   mlir::LogicalResult evaluate(TransposeOp op);
   mlir::LogicalResult evaluate(DiagOp op);
   mlir::LogicalResult evaluate(MatMulOp op);
   mlir::LogicalResult evaluate(ReduceOp op);
   mlir::LogicalResult evaluate(BroadcastOp op);
   mlir::LogicalResult evaluate(ConstantMatrixOp op);
   mlir::LogicalResult evaluate(ForConstOp op);
   mlir::LogicalResult evaluate(ApplyOp op);
   mlir::LogicalResult evaluate(PickAnyOp op);
   mlir::LogicalResult evaluate(TrilOp op);
   mlir::LogicalResult evaluate(mlir::Operation* op);

   public:
   MatrixAttr evaluate(mlir::func::FuncOp funcOp,
                       llvm::ArrayRef<MatrixAttr>
                          args);
};

class ScalarEvaluator {
   private:
   llvm::SmallDenseMap<mlir::Value, mlir::TypedAttr> _values;

   mlir::LogicalResult evaluate(ConstantOp op);
   mlir::LogicalResult evaluate(mlir::arith::ConstantOp op);
   mlir::LogicalResult evaluate(AddOp op);
   mlir::LogicalResult evaluate(MulOp op);
   mlir::LogicalResult evaluate(CastScalarOp op);
   mlir::LogicalResult evaluate(EqOp op);
   mlir::LogicalResult evaluate(mlir::arith::DivFOp op);
   mlir::LogicalResult evaluate(mlir::arith::SubIOp op);
   mlir::LogicalResult evaluate(mlir::arith::SubFOp op);
   mlir::LogicalResult evaluate(mlir::Operation* op);

   public:
   mlir::TypedAttr evaluate(ApplyOp op, llvm::ArrayRef<mlir::TypedAttr> args);
};

} // namespace

mlir::LogicalResult Evaluator::evaluate(TransposeOp op) {
   MatrixAttrReader input(_values[op.getInput()]);
   MatrixAttrBuilder result(op.getType());
   for (auto row : llvm::seq(input.nRows())) {
      for (auto col : llvm::seq(input.nCols())) {
         result.set(col, row, input.at(row, col));
      }
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(DiagOp op) {
   MatrixAttrReader input(_values[op.getInput()]);
   MatrixAttrBuilder result(op.getType());

   for (auto row : llvm::seq(input.nRows())) {
      result.set(row, row, input.at(row, 0));
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(MatMulOp op) {
   MatrixAttrReader lhs(_values[op.getLhs()]);
   MatrixAttrReader rhs(_values[op.getRhs()]);
   MatrixAttrBuilder result(op.getType());

   auto ring = result.ring();
   // result[row, col] = SUM{i}(lhs[row, i] * rhs[i, col])
   for (auto row : llvm::seq(lhs.nRows())) {
      for (auto col : llvm::seq(rhs.nCols())) {
         auto value = ring.addIdentity();
         for (auto i : llvm::seq(lhs.nCols())) {
            value = ring.add(value, ring.mul(lhs.at(row, i), rhs.at(i, col)));
         }

         result.set(row, col, value);
      }
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(ReduceOp op) {
   MatrixAttrReader input(_values[op.getInput()]);
   MatrixAttrBuilder result(op.getType());

   auto ring = result.ring();
   if (op.getType().isScalar()) {
      // Reduce all to a single value.
      auto value = ring.addIdentity();
      for (auto row : llvm::seq(input.nRows())) {
         for (auto col : llvm::seq(input.nCols())) {
            value = ring.add(value, input.at(row, col));
         }
      }

      result.set(0, 0, value);
   } else if (op.getType().isColumnVector()) {
      // Per-row reduce.
      for (auto row : llvm::seq(input.nRows())) {
         auto value = ring.addIdentity();
         for (auto col : llvm::seq(input.nCols())) {
            value = ring.add(value, input.at(row, col));
         }

         result.set(row, 0, value);
      }
   } else if (op.getType().isRowVector()) {
      // Per-column reduce.
      for (auto col : llvm::seq(input.nCols())) {
         auto value = ring.addIdentity();
         for (auto row : llvm::seq(input.nRows())) {
            value = ring.add(value, input.at(row, col));
         }

         result.set(0, col, value);
      }
   } else {
      // Reduce nothing.
      return op.emitOpError("Not reducing along any dimension");
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(BroadcastOp op) {
   MatrixAttrReader input(_values[op.getInput()]);
   MatrixAttrBuilder result(op.getType());

   for (auto row : llvm::seq(result.nRows())) {
      for (auto col : llvm::seq(result.nCols())) {
         auto inRow = input.nRows() == 1 ? 0 : row;
         auto inCol = input.nCols() == 1 ? 0 : col;
         result.set(row, col, input.at(inRow, inCol));
      }
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(ConstantMatrixOp op) {
   MatrixAttrBuilder result(op.getType());

   for (auto row : llvm::seq(result.nRows())) {
      for (auto col : llvm::seq(result.nCols())) {
         result.set(row, col, op.getValue());
      }
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(ForConstOp op) {
   MatrixAttrReader rangeBeginMat(_values[op.getRangeBegin()]);
   MatrixAttrReader rangeEndMat(_values[op.getRangeEnd()]);
   auto rangeBegin =
      llvm::cast<mlir::IntegerAttr>(rangeBeginMat.at(0, 0)).getInt();
   auto rangeEnd = llvm::cast<mlir::IntegerAttr>(rangeEndMat.at(0, 0)).getInt();

   auto& body = op.getBody().front();
   auto* ctx = op.getContext();

   // Initialize block arguments
   for (auto [init, blockArg] :
        llvm::zip_equal(op.getInitArgs(), body.getArguments().drop_front())) {
      _values[blockArg] = _values[init];
   }

   for (auto i : llvm::seq(rangeBegin, rangeEnd)) {
      // Iteration variable.
      auto iterAttr = mlir::IntegerAttr::get(SemiringTypes::forInt(ctx), i);
      auto iterArg = body.getArgument(0);
      auto iterType = llvm::cast<MatrixType>(iterArg.getType());
      MatrixAttrBuilder iterBuilder(iterType);
      iterBuilder.set(0, 0, iterAttr);
      _values[body.getArgument(0)] = iterBuilder.build();

      for (auto& op : body) {
         if (auto yieldOp = llvm::dyn_cast<YieldOp>(op)) {
            // Update block arguments
            for (auto [value, blockArg] : llvm::zip_equal(
                    yieldOp.getInputs(), body.getArguments().drop_front())) {
               _values[blockArg] = _values[value];
            }
         } else if (mlir::failed(evaluate(&op))) {
            return mlir::failure();
         }
      }

      bool breakFromUntil = false;
      if (!op.getUntil().empty()) {
         // Have an until clause to evaluate.
         auto& until = op.getUntil().front();

         // Use current state of loop variables as input to until block.
         for (auto [bodyArg, untilArg] :
              llvm::zip_equal(body.getArguments(), until.getArguments())) {
            _values[untilArg] = _values[bodyArg];
         }

         for (auto& op : until) {
            if (auto yieldOp = llvm::dyn_cast<YieldOp>(op)) {
               // Check break condition
               assert(yieldOp->getNumOperands() == 1);
               MatrixAttrReader condMat(_values[yieldOp.getInputs().front()]);
               breakFromUntil =
                  llvm::cast<mlir::BoolAttr>(condMat.at(0, 0)).getValue();
            } else if (mlir::failed(evaluate(&op))) {
               return mlir::failure();
            }
         }
      }

      if (breakFromUntil) {
         break;
      }
   }

   // Set loop results.
   for (auto [value, result] :
        llvm::zip_equal(body.getArguments().drop_front(), op->getResults())) {
      _values[result] = _values[value];
   }

   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(ApplyOp op) {
   llvm::SmallVector<MatrixAttrReader> inputs;
   for (auto input : op.getInputs()) {
      inputs.emplace_back(_values[input]);
   }

   MatrixAttrBuilder result(op.getType());
   for (auto row : llvm::seq(result.nRows())) {
      for (auto col : llvm::seq(result.nCols())) {
         llvm::SmallVector<mlir::TypedAttr> args;
         for (const auto& input : inputs) {
            // Implicit broadcast.
            auto r = row < input.nRows() ? row : 0;
            auto c = col < input.nCols() ? col : 0;
            args.push_back(input.at(r, c));
         }

         ScalarEvaluator scalarEvaluator;
         auto value = scalarEvaluator.evaluate(op, args);
         if (!value) {
            return mlir::failure();
         }

         result.set(row, col, value);
      }
   }

   _values[op] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(PickAnyOp op) {
   MatrixAttrReader input(_values[op.getInput()]);
   MatrixAttrBuilder result(op.getType());

   for (auto row : llvm::seq(input.nRows())) {
      for (auto col : llvm::seq(input.nCols())) {
         auto value = input.at(row, col);
         if (value != result.ring().addIdentity()) {
            result.set(row, col, value);
            break;
         }
      }
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(TrilOp op) {
   MatrixAttrReader input(_values[op.getInput()]);
   MatrixAttrBuilder result(op.getType());

   for (auto row : llvm::seq(input.nRows())) {
      for (auto col : llvm::seq(input.nCols())) {
         if (col < row) {
            auto value = input.at(row, col);
            result.set(row, col, value);
         }
      }
   }

   _values[op.getResult()] = result.build();
   return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(mlir::Operation* op) {
   return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
#define GA_CASE(Op) .Case<Op>([&](Op op) { return evaluate(op); })
      GA_CASE(TransposeOp) GA_CASE(DiagOp) GA_CASE(MatMulOp) GA_CASE(ReduceOp)
         GA_CASE(BroadcastOp) GA_CASE(ConstantMatrixOp) GA_CASE(ForConstOp)
            GA_CASE(ApplyOp) GA_CASE(PickAnyOp) GA_CASE(TrilOp)
#undef GA_CASE
               .Default([](mlir::Operation* op) {
                  return op->emitOpError("unsupported op");
               });
}

MatrixAttr Evaluator::evaluate(mlir::func::FuncOp funcOp,
                               llvm::ArrayRef<MatrixAttr>
                                  args) {
   auto& body = funcOp.getFunctionBody().front();
   if (body.getNumArguments() != args.size()) {
      funcOp->emitOpError("function has ")
         << funcOp.getFunctionType().getNumInputs() << " inputs, got "
         << args.size() << "inputs";
      return nullptr;
   }

   for (auto [i, value] : llvm::enumerate(args)) {
      auto arg = body.getArgument(i);
      if (arg.getType() != value.getType()) {
         mlir::emitError(arg.getLoc())
            << "parameter " << i << " has type " << arg.getType()
            << ", but argument value has type " << value.getType();
         return nullptr;
      }

      _values[arg] = value;
   }

   for (auto& op : body) {
      if (auto retOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
         assert(retOp->getNumOperands() == 1);
         return _values[retOp->getOperand(0)];
      }

      if (mlir::failed(evaluate(&op))) {
         return nullptr;
      }
   }

   funcOp->emitOpError("missing return op");
   return nullptr;
}

mlir::LogicalResult ScalarEvaluator::evaluate(ConstantOp op) {
   _values[op] = op.getValue();
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(mlir::arith::ConstantOp op) {
   _values[op] = op.getValue();
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(AddOp op) {
   auto ring = llvm::cast<SemiringTypeInterface>(op.getType());
   _values[op] = ring.add(_values[op.getLhs()], _values[op.getRhs()]);
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(MulOp op) {
   auto ring = llvm::cast<SemiringTypeInterface>(op.getType());
   _values[op] = ring.mul(_values[op.getLhs()], _values[op.getRhs()]);
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(CastScalarOp op) {
   auto* dialect = op->getContext()->getLoadedDialect<GraphAlgDialect>();
   _values[op] = dialect->castAttribute(_values[op.getInput()], op.getType());
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(EqOp op) {
   auto ring = llvm::cast<SemiringTypeInterface>(op.getType());
   bool eq = _values[op.getLhs()] == _values[op.getRhs()];
   _values[op] = mlir::BoolAttr::get(op.getContext(), eq);
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(mlir::arith::DivFOp op) {
   auto lhs =
      llvm::cast<mlir::FloatAttr>(_values[op.getLhs()]).getValueAsDouble();
   auto rhs =
      llvm::cast<mlir::FloatAttr>(_values[op.getRhs()]).getValueAsDouble();
   double result = rhs == 0 ? 0 : lhs / rhs;
   _values[op] = mlir::FloatAttr::get(op.getType(), result);
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(mlir::arith::SubIOp op) {
   auto lhs = llvm::cast<mlir::IntegerAttr>(_values[op.getLhs()]).getInt();
   auto rhs = llvm::cast<mlir::IntegerAttr>(_values[op.getRhs()]).getInt();
   double result = lhs - rhs;
   _values[op] = mlir::IntegerAttr::get(op.getType(), result);
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(mlir::arith::SubFOp op) {
   auto lhs =
      llvm::cast<mlir::FloatAttr>(_values[op.getLhs()]).getValueAsDouble();
   auto rhs =
      llvm::cast<mlir::FloatAttr>(_values[op.getRhs()]).getValueAsDouble();
   double result = lhs - rhs;
   _values[op] = mlir::FloatAttr::get(op.getType(), result);
   return mlir::success();
}

mlir::LogicalResult ScalarEvaluator::evaluate(mlir::Operation* op) {
   return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
#define GA_CASE(Op) .Case<Op>([&](Op op) { return evaluate(op); })
      GA_CASE(ConstantOp) GA_CASE(mlir::arith::ConstantOp) GA_CASE(AddOp)
         GA_CASE(MulOp) GA_CASE(CastScalarOp) GA_CASE(EqOp)
            GA_CASE(mlir::arith::DivFOp) GA_CASE(mlir::arith::SubIOp)
               GA_CASE(mlir::arith::SubFOp)
#undef GA_CASE
                  .Default([](mlir::Operation* op) {
                     return op->emitOpError("unsupported op");
                  });
}

mlir::TypedAttr
ScalarEvaluator::evaluate(ApplyOp op, llvm::ArrayRef<mlir::TypedAttr> args) {
   auto& block = op.getBody().front();
   for (auto [blockArg, value] : llvm::zip_equal(block.getArguments(), args)) {
      _values[blockArg] = value;
   }

   for (auto& op : block) {
      if (auto retOp = llvm::dyn_cast<ApplyReturnOp>(op)) {
         return _values[retOp.getValue()];
      } else if (mlir::failed(evaluate(&op))) {
         return nullptr;
      }
   }

   op->emitOpError("missing return op");
   return nullptr;
}

MatrixAttr evaluate(mlir::func::FuncOp funcOp,
                    llvm::ArrayRef<MatrixAttr>
                       args) {
   Evaluator evaluator;
   return evaluator.evaluate(funcOp, args);
}

} // namespace graphalg
