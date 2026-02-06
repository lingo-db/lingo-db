#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

template <typename Analysis, typename Op = mlir::ModuleOp>
class PrintDataFlowLatticeTestPass : public mlir::OperationPass<Op> {
private:
  using Base = mlir::OperationPass<Op>;

  void runOnOperation() override {
    auto &analysis = Base::template getAnalysis<Analysis>();
    if (mlir::failed(analysis.status())) {
      Base::getOperation().emitError("The analysis failed");
      return;
    }

    std::string buf;
    llvm::raw_string_ostream bufAdapter{buf};

    mlir::Operation *op = Base::getOperation();
    op->walk([&](mlir::Operation *op) {
      auto tag = op->getAttrOfType<mlir::StringAttr>("tag");
      if (!tag) {
        return;
      }

      mlir::InFlightDiagnostic pending =
          mlir::emitRemark(op->getLoc(), tag.getValue());
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        const auto *lattice = analysis.getFor(operand);
        mlir::Diagnostic &note = pending.attachNote();
        note << " operand #" << index << ": ";
        if (lattice) {
          buf.clear();
          lattice->print(bufAdapter);
          note << buf;
        } else {
          note << "<NULL>";
        }
      }
      for (auto [index, operand] : llvm::enumerate(op->getResults())) {
        const auto *lattice = analysis.getFor(operand);

        mlir::Diagnostic &note = pending.attachNote();
        note << " result #" << index << ": ";
        if (lattice) {
          buf.clear();
          lattice->print(bufAdapter);
          note << buf;
        } else {
          note << "<NULL>";
        }
      }
      for (auto [rIdx, region] : llvm::enumerate(op->getRegions())) {
        for (auto [bIdx, block] : llvm::enumerate(region.getBlocks())) {
          for (auto [aIdx, arg] : llvm::enumerate(block.getArguments())) {
            const auto *lattice = analysis.getFor(arg);

            mlir::Diagnostic &note = pending.attachNote(arg.getLoc());
            note << " arg #" << rIdx << ":" << bIdx << ":" << aIdx << ": ";
            if (lattice) {
              buf.clear();
              lattice->print(bufAdapter);
              note << buf;
            } else {
              note << "<NULL>";
            }
          }
        }
      }
    });
  }

public:
  using Base::Base;
};
