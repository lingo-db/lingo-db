#include "mlir/Dialect/RelAlg/Transforms/queryopt/utils.h"
namespace mlir::relalg {
void NodeSet::iterateSubsets(const std::function<void(NodeSet)>& fn) const {
   if (!storage.any()) return;
   NodeSet s = *this;
   auto s1 = s & s.negate();
   while (s1 != s) {
      fn(s1);
      auto s1flipped = s1.flip();
      auto s2 = s & s1flipped;
      s1 = s & s2.negate();
   }
   fn(s);
}
NodeSet NodeSet::negate() const {
   NodeSet res = *this;
   size_t pos = res.findFirst();
   size_t flipLen = res.storage.size() - pos - 1;
   if (flipLen) {
      llvm::SmallBitVector flipVector(res.storage.size());
      flipVector.set(pos + 1, res.storage.size());
      res.storage ^= flipVector;
   }
   return res;
}
static std::string printPlanOp(Operator op) {
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
   std::stringstream sstream;
   sstream << std::quoted(nodelabel);
   llvm::dbgs() << " node [label=" << sstream.str() << "] " << nodename << ";\n";
   return nodename;
}
std::string Plan::dumpNode() {
   std::string firstNodeName;
   std::string lastNodeName;
   for (auto op : additionalOps) {
      std::string nodename = printPlanOp(op);
      if (!lastNodeName.empty()) {
         llvm::dbgs() << lastNodeName << " -> " << nodename << ";\n";
      }
      lastNodeName = nodename;
      if (firstNodeName.empty()) {
         firstNodeName = nodename;
      }
   }
   std::string nodename = printPlanOp(op);
   if (!lastNodeName.empty()) {
      llvm::dbgs() << lastNodeName << " -> " << nodename << ";\n";
   }
   if (firstNodeName.empty()) {
      firstNodeName = nodename;
   }
   for (auto subplan : subplans) {
      std::string subnode = subplan->dumpNode();
      llvm::dbgs() << nodename << " -> " << subnode << ";\n";
   }
   return firstNodeName;
}

void Plan::dump() {
   llvm::dbgs() << "digraph{\n";
   dumpNode();
   llvm::dbgs() << "}\n";
}

static void fix(Operator tree) {
   for (auto child : tree.getChildren()) {
      child.moveSubTreeBefore(tree);
      fix(child);
   }
}
Operator Plan::realizePlan() {
   Operator tree = realizePlanRec();
   fix(tree);
   return tree;
}
Operator Plan::realizePlanRec() {
   bool isLeaf = subplans.empty();
   Operator firstNode{};
   Operator lastNode{};
   for (auto op : additionalOps) {
      op->setAttr("cost", mlir::FloatAttr::get(mlir::FloatType::getF64(op.getContext()), cost));
      op->setAttr("rows", mlir::FloatAttr::get(mlir::FloatType::getF64(op.getContext()), rows));
      if (lastNode) {
         lastNode.setChildren({op});
      }
      lastNode = op;
      if (!firstNode) {
         firstNode = op;
      }
   }
   llvm::SmallVector<Operator, 4> children;
   for (auto subplan : subplans) {
      auto subop = subplan->realizePlan();
      children.push_back(subop);
   }
   auto currop = op;
   if (!isLeaf) {
      if (currop) {
         if (mlir::isa<mlir::relalg::SelectionOp>(currop.getOperation()) && children.size() == 2) {
            auto selop = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(currop.getOperation());
            mlir::OpBuilder builder(currop.getOperation());
            auto x = builder.create<mlir::relalg::InnerJoinOp>(selop.getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), children[0]->getResult(0), children[1]->getResult(0));
            x.predicate().push_back(new mlir::Block);
            x.getLambdaBlock().addArgument(mlir::relalg::TupleType::get(builder.getContext()));
            selop.getLambdaArgument().replaceAllUsesWith(x.getLambdaArgument());
            x.getLambdaBlock().getOperations().splice(x.getLambdaBlock().end(), selop.getLambdaBlock().getOperations());
            //selop.replaceAllUsesWith(x.getOperation());
            currop = x;
         } else if (mlir::isa<mlir::relalg::InnerJoinOp>(currop.getOperation()) && children.size() == 1) {
            assert(false && "need to implement Join -> Selection transition");
         }
         currop->setAttr("cost", mlir::FloatAttr::get(mlir::FloatType::getF64(op.getContext()), cost));
         currop->setAttr("rows", mlir::FloatAttr::get(mlir::FloatType::getF64(op.getContext()), rows));
      } else if (!currop && children.size() == 2) {
         mlir::OpBuilder builder(children[0].getOperation());
         currop = builder.create<mlir::relalg::CrossProductOp>(children[0].getOperation()->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), children[0]->getResult(0), children[1]->getResult(0));
         //currop->setAttr("cost",mlir::FloatAttr::get(mlir::FloatType::getF64(op.getContext()),cost));
         //currop->setAttr("rows",mlir::FloatAttr::get(mlir::FloatType::getF64(op.getContext()),rows));
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
double Plan::getCost() const {
   return cost;
}
void Plan::setDescription(const std::string& descr) {
   Plan::description = descr;
}
const std::string& Plan::getDescription() const {
   return description;
}
} // namespace mlir::relalg