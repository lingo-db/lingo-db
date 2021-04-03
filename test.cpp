#include "mlir/Dialect/RelAlg/queryopt/QueryGraph.h"
#include "mlir/Dialect/RelAlg/queryopt/DPhyp.h"

#include <iostream>
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include <llvm/Support/ErrorOr.h>
#include <mlir/Dialect/DB/IR/DBDialect.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

class JoinOrder {
    using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute *, 8>;
    using node_set = mlir::relalg::QueryGraph::node_set;
    using NodeResolver = mlir::relalg::QueryGraph::NodeResolver;

    mlir::MLIRContext *context;
    mlir::ModuleOp moduleOp;
    std::unordered_set<mlir::Operation *> already_optimized;
public:
    JoinOrder(mlir::MLIRContext *context, mlir::ModuleOp moduleOp) : context(context), moduleOp(moduleOp) {}


    size_t countCreatingOperators(Operator op) {
        size_t res = 0;
        auto children = op.getChildren();
        auto used = op.getUsedAttributes();
        auto created = op.getCreatedAttributes();
        if (already_optimized.count(op.getOperation())) {
            res += 1;
            return res;
        }
        for (auto child:children) {
            res += countCreatingOperators(child);
        }

        if (mlir::isa<mlir::relalg::CrossProductOp>(op.getOperation())) {
            //do not construct crossproducts in the querygraph
        } else if (mlir::isa<Join>(op.getOperation())) {

        } else if (created.size()) {
            res += 1;

        } else if (mlir::isa<mlir::relalg::SelectionOp>(op.getOperation())) {
        } else {
            assert(false && " should not happen");
        }
        return res;
    }

    NodeResolver populateQueryGraph(Operator op, mlir::relalg::QueryGraph &qg) {
        auto children = op.getChildren();
        auto used = op.getUsedAttributes();
        auto created = op.getCreatedAttributes();
        NodeResolver resolver(qg);
        if (already_optimized.count(op.getOperation())) {
            size_t new_node = qg.addNode(op);
            for (auto attr:op.getAvailableAttributes()) {
                resolver.add(attr, new_node);
            }
            return resolver;
        }
        for (auto child:children) {
            resolver.merge(populateQueryGraph(child, qg));
        }
        if (mlir::isa<mlir::relalg::CrossProductOp>(op.getOperation())) {
            //do not construct crossproducts in the querygraph
        } else if (mlir::isa<Join>(op.getOperation())) {
            //add join edges into the query graph
            node_set TES = qg.calcTES(op, resolver);
            node_set left_TES = qg.calcT(children[0], resolver) & TES;
            node_set right_TES = qg.calcT(children[1], resolver) & TES;
            qg.addEdge(qg.expand(left_TES), qg.expand(right_TES), op, false);
        } else if (created.size()) {
            //add node for operators that create attributes
            size_t new_node = qg.addNode(op);
            for (auto attr:op.getCreatedAttributes()) {
                resolver.add(attr, new_node);
            }
            if (children.size() == 1) {
                //if operator has one child e.g. aggregation/renaming/map
                // -> create "implicit" hyperedge
                node_set TES = qg.calcTES(op, resolver);
                qg.nodes[new_node].dependencies = qg.expand(TES);
                qg.addEdge(qg.expand(TES), qg.single(new_node), op, true);
            }
        } else if (mlir::isa<mlir::relalg::SelectionOp>(op.getOperation())) {
            node_set SES = qg.calcSES(op, resolver);

            if (SES.count() == 1) {
                //if selection is only based on one node -> add selection to node
                auto node_id = SES.find_first();
                qg.nodes[node_id].additional_predicates.push_back(op);
            } else {
                qg.iterateSubsets(SES, [&](node_set left) {
                    node_set right = SES & ~left;
                    if (left < right) {
                        left = qg.expand(left);
                        right = qg.expand(right) & ~left;

                        qg.addEdge(left, right, op, false);
                    }
                });
            }
        } else {
            assert(false && " should not happen");
        }
        return resolver;
    }

    bool isUnsupportedOp(mlir::Operation *op) {
        return ::llvm::TypeSwitch<mlir::Operation *, bool>(op)
                .Case<mlir::relalg::CrossProductOp, Join, mlir::relalg::SelectionOp, mlir::relalg::AggregationOp, mlir::relalg::MapOp, mlir::relalg::RenamingOp>(
                        [&](mlir::Operation *op) {
                            return false;
                        })
                .Default([&](auto x) {
                    return true;
                });
    }

    bool isOptimizationRoot(mlir::Operation *op) {
        //reason one: used by multiple parent operators (DAG)
        auto users = op->getUsers();
        if (!users.empty() && ++users.begin() != users.end()) {
            return true;
        }
        //reason two: result of operation is accessed by non-operator
        if (llvm::any_of(op->getUsers(),
                         [](mlir::OpOperand user) { return !mlir::isa<Operator>(user.getOwner()); })) {
            return true;
        }
        return isUnsupportedOp(op);
    }

    Operator optimize(Operator op) {
        if (already_optimized.count(op.getOperation())) {
            return op;
        }
        if (isUnsupportedOp(op)) {
            auto children = op.getChildren();
            for (size_t i = 0; i < children.size(); i++) {
                children[i] = optimize(children[i]);
            }
            op.setChildren(children);
            already_optimized.insert(op.getOperation());
            return op;
        } else {
            llvm::outs() << "optimize:";
            op->print(llvm::outs());
            mlir::relalg::QueryGraph qg(countCreatingOperators(op),already_optimized);
            populateQueryGraph(op, qg);
            qg.dump();
            mlir::relalg::CostFunction cf;
            mlir::relalg::DPHyp solver(qg, cf);
            solver.solve();
            already_optimized.insert(op.getOperation());
            return op;
        }
    }

    void run() {
        mlir::FuncOp func = mlir::dyn_cast_or_null<mlir::FuncOp>(&moduleOp.getRegion().front().front());
        func.walk([&](Operator op) {
                      if (isOptimizationRoot(op.getOperation())) {
                          Operator optimized = optimize(op);
                          if (optimized != op) {
                              op->replaceAllUsesWith(optimized);
                          }
                      }
                  }
        );

    }

};

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
    mlir::DialectRegistry registry;
    registry.insert<mlir::relalg::RelAlgDialect>();
    registry.insert<mlir::db::DBDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    mlir::MLIRContext context;
    context.appendDialectRegistry(registry);
    mlir::OwningModuleRef module;
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    if (int error = loadMLIR(context, module))
        return error;
    JoinOrder(&context, module.get()).run();

}