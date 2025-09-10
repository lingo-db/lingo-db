#pragma once
#include "lingodb/compiler/frontend/sql-parser/gen/location.hh"
#include <cstdint>
#include <unordered_map>
namespace lingodb::ast {
enum class NodeType : uint8_t {
   QUERY_NODE = 1,
   EXPRESSION = 2,
   TABLE_REF = 3,
   PIPE_OP = 4,
   GROUP_BY = 5,
   AGGREGATION = 6,
   RESULT_MODIFIER = 7,
   EXTEND_NODE = 8,
   CREATE_NODE = 9,
   INSERT_NODE = 10,
   SET_NODE = 11,
   COPY_NODE = 12,

   //Bound
   BOUND_TABLE_REF = 20,
   BOUND_EXPRESSION = 22,
   BOUND_GROUP_BY = 25,
   BOUND_AGGREGATION = 26,
   BOUND_RESULT_MODIFIER = 27,
   BOUND_EXTEND_NODE = 28,
   BOUND_INSERT_NODE = 30,
};
//used for getting unique ids when printing the AST for debugging purposes as Dot graph
class NodeIdGenerator {
   public:
   void reset() {
      currentId = 0;
      idMap.clear();
   }

   uint64_t getId(uintptr_t id) {
      if (!idMap.contains(id)) {
         idMap[id] = currentId;
         currentId++;
      }
      return idMap[id];
   }

   void setId(uint64_t id, uint64_t setID) {
      idMap[id] = setID;
   }

   private:
   uint64_t currentId;
   std::unordered_map<uint64_t, uint64_t> idMap;
};
class AstNode {
   public:
   explicit AstNode(NodeType type) : nodeType(type) {}
   virtual ~AstNode() = default;

   NodeType nodeType;
   location loc;

   virtual std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) = 0;
};
} // namespace lingodb::ast
