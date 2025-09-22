#pragma once

#include "lingodb/compiler/frontend/ast/pipe_operator.h"
#include "lingodb/compiler/frontend/column_semantic.h"
#include "lingodb/compiler/frontend/sql_context.h"

#include <memory>
namespace lingodb::ast {

class BoundSetPipeOperator : public PipeOperator {
   public:
   BoundSetPipeOperator(PipeOperatorType pipeOpType, std::shared_ptr<TableProducer> node, std::shared_ptr<TableProducer> input) : PipeOperator(pipeOpType, node), node(node) {
      this->input = input;
   }
   PipeOperatorType pipeOpType;
   std::shared_ptr<TableProducer> node;

   std::shared_ptr<analyzer::SQLScope> leftScope;

   std::shared_ptr<analyzer::SQLScope> rightScope;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override {
      return "";
   };
};
} // namespace lingodb::ast