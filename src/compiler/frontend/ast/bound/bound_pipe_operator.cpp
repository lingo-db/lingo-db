#include "lingodb/compiler/frontend/ast/bound/bound_pipe_operator.h"
namespace lingodb::ast {
BoundSetPipeOperator::BoundSetPipeOperator(PipeOperatorType pipeOpType, std::shared_ptr<TableProducer> node, std::shared_ptr<TableProducer> input) : PipeOperator(pipeOpType, node), node(node) {
   this->input = input;
}
std::string BoundSetPipeOperator::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
}