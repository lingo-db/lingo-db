#include "lingodb/compiler/frontend/ast/bound/bound_resultmodifier.h"
namespace lingodb::ast {
std::string BoundResultModifier::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   if (input) {
      dot += "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))) +
         " -> node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(input.get()))) + "[label=\"has input\"];\n";
      dot += input->toDotGraph(depth + 1, idGen);
   }
   return dot;
}
std::string BoundOrderByModifier::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   if (input) {
      dot += "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))) +
         " -> node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(input.get()))) + "[label=\"has input\"];\n";
      dot += input->toDotGraph(depth + 1, idGen);
   }
   dot += "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))) +
      " [label=\"BoundOrderByModifier\\n\"]";
   return dot;
}

std::string BoundLimitModifier::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast