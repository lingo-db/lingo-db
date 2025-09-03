#include "../../../../include/lingodb/compiler/frontend/ast/pipe_operator.h"

#include "../../../../include/lingodb/compiler/frontend/ast/query_node.h"
namespace lingodb::ast {
PipeOperator::PipeOperator(PipeOperatorType pipeOpType, std::shared_ptr<AstNode> node) : TableProducer(NodeType::PIPE_OP), node(node), pipeOpType(pipeOpType) {
}

std::string PipeOperator::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   if (input) {
      dot += "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))) +
         " -> node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(input.get()))) + "[label=\"has input\"];\n";
      dot += input->toDotGraph(depth + 1, idGen);
   }

   dot += "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))) +
      " [label=\"Pipe Operator\\n";
   dot += "<<";
   switch (pipeOpType) {
      case PipeOperatorType::SELECT:
         dot += "SELECT";
         break;
      case PipeOperatorType::AGGREGATE:
         dot += "AGGREGATE";
         break;
      case PipeOperatorType::RESULT_MODIFIER:
         dot += "RESULT MODIFIER";
         break;
      case PipeOperatorType::WHERE:
         dot += "WHERE";
         break;
      case PipeOperatorType::EXTEND:
         dot += "EXTEND";
         break;
      default:
         dot += "UNKNOWN " + std::to_string(static_cast<int>(pipeOpType));
   }
   dot += ">>";
   dot += "\"];\n";

   dot += "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))) +
      " -> node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(node.get()))) + ";\n";

   dot += node->toDotGraph(depth + 1, idGen);

   return dot;
}

} // namespace lingodb::ast