#include "mlir/Dialect/RelAlg/Transforms/queryopt/utils.h"
#include "mlir/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"
#include <unordered_set>
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
      if (!child->isBeforeInBlock(tree)) {
         child.moveSubTreeBefore(tree);
      }
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
            x.getLambdaBlock().addArgument(mlir::relalg::TupleType::get(builder.getContext()), selop->getLoc());
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

std::shared_ptr<Plan> Plan::joinPlans(NodeSet s1, NodeSet s2, std::shared_ptr<Plan> p1, std::shared_ptr<Plan> p2, QueryGraph& queryGraph, NodeSet& s) {
   s = s1 | s2;

   struct HashOp {
      size_t operator()(const Operator& op) const {
         return (size_t) op.operator mlir::Operation*();
      }
   };
   std::unordered_set<Operator, HashOp> predicates;

   Operator specialJoin{};
   double totalSelectivity = 1;

   llvm::EquivalenceClasses<const mlir::relalg::Column*> equivalentColumns;
   for (auto& edge : queryGraph.joins) {
      if (!edge.connects(s1, s2) && edge.left.isSubsetOf(s) && edge.right.isSubsetOf(s) && edge.equality) {
         equivalentColumns.unionSets(edge.equality->first, edge.equality->second);
      }
   }
   for (auto& edge : queryGraph.joins) {
      if (edge.connects(s1, s2)) {
         if (!edge.op) {
            //special case: forced cross product
            //do nothing
         } else if (!mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
            specialJoin = edge.op;
            if (!edge.left.isSubsetOf(s1)) {
               std::swap(s1, s2);
               std::swap(p1, p2);
            }
            totalSelectivity *= edge.selectivity;
         } else {
            bool useSelection = true;
            if (edge.equality && equivalentColumns.isEquivalent(edge.equality->first, edge.equality->second)) {
               useSelection = false;
            }
            if (useSelection) {
               totalSelectivity *= edge.selectivity;
               predicates.insert(edge.op);
            }
         }
         if (edge.equality) {
            equivalentColumns.unionSets(edge.equality->first, edge.equality->second);
         }
         if (edge.createdNode) {
            s |= NodeSet::single(queryGraph.numNodes, edge.createdNode.getValue());
         }
      }
   }
   for (auto& edge : queryGraph.selections) {
      if (edge.connects2(s, s1, s2)) {
         totalSelectivity *= queryGraph.calculateSelectivity(edge, s1, s2);
         predicates.insert(edge.op);
      }
   }
   std::shared_ptr<Plan> currPlan;

   if (specialJoin) {
      double estimatedResultSize;
      switch (mlir::relalg::detail::getBinaryOperatorType(specialJoin)) {
         case detail::BinaryOperatorType::AntiSemiJoin: estimatedResultSize = p1->getRows() - std::min(p1->getRows(), p1->getRows() * p2->getRows() * totalSelectivity); break;
         case detail::BinaryOperatorType::SemiJoin: estimatedResultSize = std::min(p1->getRows(), p1->getRows() * p2->getRows() * totalSelectivity); break;
         case detail::BinaryOperatorType::CollectionJoin:
         case detail::BinaryOperatorType::OuterJoin: //todo: not really correct, but we would need to split singlejoin/outerjoin
         case detail::BinaryOperatorType::MarkJoin: estimatedResultSize = p1->getRows(); break;
         default: estimatedResultSize = p1->getRows() * p2->getRows() * totalSelectivity;
      }
      currPlan = std::make_shared<Plan>(specialJoin, std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(predicates.begin(), predicates.end()), estimatedResultSize);
   } else if (!predicates.empty()) {
      auto estimatedResultSize = p1->getRows() * p2->getRows() * totalSelectivity;
      if (p1->getRows() > p2->getRows()) {
         std::swap(p1, p2);
      }
      currPlan = std::make_shared<Plan>(*predicates.begin(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(++predicates.begin(), predicates.end()), estimatedResultSize);
   } else {
      auto estimatedResultSize = p1->getRows() * p2->getRows() * totalSelectivity;
      currPlan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>({}), estimatedResultSize);
   }
   currPlan->setDescription("(" + p1->getDescription() + ") join (" + p2->getDescription() + ")");
   return currPlan;
}
} // namespace mlir::relalg