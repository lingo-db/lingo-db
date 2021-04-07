#include "mlir/Dialect/RelAlg/queryopt/DPhyp.h"
#include "mlir/Dialect/RelAlg/queryopt/QueryGraph.h"
#include "mlir/Dialect/RelAlg/queryopt/QueryGraphBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include <llvm/Support/ErrorOr.h>
#include <iostream>
#include <mlir/Dialect/DB/IR/DBDialect.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
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
   mlir::MLIRContext* context;
   mlir::ModuleOp moduleOp;
   std::unordered_set<mlir::Operation*> already_optimized;

   public:
   JoinOrder(mlir::MLIRContext* context, mlir::ModuleOp moduleOp) : context(context), moduleOp(moduleOp) {}

   bool isUnsupportedOp(mlir::Operation* op) {
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op)
         .Case<mlir::relalg::CrossProductOp, Join, mlir::relalg::SelectionOp, mlir::relalg::AggregationOp, mlir::relalg::MapOp, mlir::relalg::RenamingOp>(
            [&](mlir::Operation* op) {
               return false;
            })
         .Default([&](auto x) {
            return true;
         });
   }

   bool isOptimizationRoot(mlir::Operation* op) {
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
   std::string printPlanOp(Operator op) {
      static size_t nodeid = 0;
      std::string opstr;
      llvm::raw_string_ostream strstream(opstr);
      if (op) {
         op.print(strstream);
      } else {
         strstream << "crossproduct";
      }

      std::string nodename = "n" + std::to_string(nodeid++);
      std::string nodelabel = strstream.str();

      std::cout << " node [label=" << std::quoted(nodelabel) << "] " << nodename << ";" << std::endl;
      return nodename;
   }
   std::string printPlanNode(std::shared_ptr<mlir::relalg::Plan> plan) {
      std::string firstNodeName;
      std::string lastNodeName;
      for (auto op : plan->additional_ops) {
         std::string nodename = printPlanOp(op);
         if (!lastNodeName.empty()) {
            std::cout << lastNodeName << " -> " << nodename << ";" << std::endl;
         }
         lastNodeName = nodename;
         if (firstNodeName.empty()) {
            firstNodeName = nodename;
         }
      }
      std::string nodename = printPlanOp(plan->op);
      if (!lastNodeName.empty()) {
         std::cout << lastNodeName << " -> " << nodename << ";" << std::endl;
      }
      if (firstNodeName.empty()) {
         firstNodeName = nodename;
      }
      for (auto subplan : plan->subplans) {
         std::string subnode = printPlanNode(subplan);
         std::cout << nodename << " -> " << subnode << ";" << std::endl;
      }
      return firstNodeName;
   }

   void printPlan(std::shared_ptr<mlir::relalg::Plan> plan) {
      std::cout << "digraph{" << std::endl;
      printPlanNode(plan);
      std::cout << "}" << std::endl;
   }
   void moveTreeBefore(Operator tree, mlir::Operation* before) {
      auto children = tree.getChildren();
      tree->moveBefore(before);
      for (auto child : tree.getChildren()) {
         moveTreeBefore(child, tree.getOperation());
      }
   }
   void fix(Operator tree) {
      for (auto child : tree.getChildren()) {
         moveTreeBefore(child, tree);
         fix(child);
      }
   }
   Operator realizePlan(std::shared_ptr<mlir::relalg::Plan> plan) {
      Operator tree = realizePlan_(plan);
      fix(tree);
      return tree;
   }
   Operator realizePlan_(std::shared_ptr<mlir::relalg::Plan> plan) {
      bool isLeaf = plan->subplans.empty();
      Operator firstNode{};
      Operator lastNode{};
      for (auto op : plan->additional_ops) {
         if (lastNode) {
            lastNode.setChildren({op});
         }
         lastNode = op;
         if (!firstNode) {
            firstNode = op;
         }
      }
      llvm::SmallVector<Operator, 4> children;
      for (auto subplan : plan->subplans) {
         auto subop = realizePlan(subplan);
         children.push_back(subop);
      }
      auto currop = plan->op;
      if (!isLeaf) {
         if (currop) {
            if (mlir::isa<mlir::relalg::SelectionOp>(currop.getOperation()) && children.size() == 2) {
               auto selop = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(currop.getOperation());
               mlir::OpBuilder builder(currop.getOperation());
               auto x = builder.create<mlir::relalg::InnerJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), children[0]->getResult(0), children[1]->getResult(0));
               x.predicate().push_back(new mlir::Block);
               x.getLambdaBlock().addArgument(mlir::relalg::TupleType::get(builder.getContext()));
               selop.getLambdaArgument().replaceAllUsesWith(x.getLambdaArgument());
               x.getLambdaBlock().getOperations().splice(x.getLambdaBlock().end(), selop.getLambdaBlock().getOperations());
               //selop.replaceAllUsesWith(x.getOperation());
               currop = x;
            } else if (mlir::isa<mlir::relalg::InnerJoinOp>(currop.getOperation()) && children.size() == 1) {
               assert(false && "need to implement Join -> Selection transition");
            }
         } else if (!currop && children.size() == 2) {
            mlir::OpBuilder builder(children[0].getOperation());
            currop = builder.create<mlir::relalg::CrossProductOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), children[0]->getResult(0), children[1]->getResult(0));
         } else if (!currop && children.size() == 1) {
            if (lastNode) {
               lastNode.setChildren({children[0]});
            }
            if (!firstNode) {
               firstNode = children[0];
            }
            return firstNode;
         } else {
            assert(false && "should not happen");
         }
      }
      if (lastNode) {
         lastNode.setChildren({currop});
      }
      if (!firstNode) {
         firstNode = currop;
      }
      if (!isLeaf) {
         currop.setChildren(children);
      }
      return firstNode;
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
         llvm::SmallPtrSet<mlir::Operation*, 8> prev_users{op->user_begin(), op->user_end()};

         llvm::SmallVector<Operator, 4> before = op.getAllSubOperators();
         mlir::relalg::QueryGraphBuilder queryGraphBuilder(op,already_optimized);
         queryGraphBuilder.generate();
         mlir::relalg::CostFunction cf;
         mlir::relalg::DPHyp solver(queryGraphBuilder.getQueryGraph(), cf);
         auto solution = solver.solve();
         //std::cout << "plan descr:" << solution->descr << std::endl;
         //printPlan(solution);
         Operator realized = realizePlan(solution);
         llvm::SmallVector<Operator, 4> after = realized.getAllSubOperators();
         llvm::SmallPtrSet<mlir::Operation*, 8> after_ht;
         for (auto op : after) {
            after_ht.insert(op.getOperation());
         }
         if (realized != op) {
            op->getResult(0).replaceUsesWithIf(realized->getResult(0), [&](mlir::OpOperand& operand) {
               return prev_users.contains(operand.getOwner());
            });
         }
         for (auto op : before) {
            if (!after_ht.contains(op.getOperation())) {
               op->dropAllUses();
               op->remove();
               //op->destroy();
            }
         }

         already_optimized.insert(realized);
         return realized;
      }
   }

   void run() {
      mlir::FuncOp func = mlir::dyn_cast_or_null<mlir::FuncOp>(&moduleOp.getRegion().front().front());
      func.walk([&](Operator op) {
         if (isOptimizationRoot(op.getOperation())) {
            optimize(op);
         }
      });
   }
};

int main(int argc, char** argv) {
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
   module->print(llvm::outs());
}