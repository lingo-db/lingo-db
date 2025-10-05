#ifndef LINGODB_COMPILER_FRONTEND_NODE_FACTORY_H
#define LINGODB_COMPILER_FRONTEND_NODE_FACTORY_H

#include "lingodb/compiler/frontend/generated/location.hh"
#include <memory>
#include <vector>
namespace lingodb::ast {
class NodeFactory {
   public:
   template <class T, class... Args>
   std::shared_ptr<T> node(lingodb::location loc, Args... args) {
      auto node = std::make_shared<T>(std::forward<Args>(args)...);
      node->loc = loc;
      return std::move(node);
   }
   template <class T, class... Args>
   std::vector<std::shared_ptr<T>> listShared() {
      std::vector<std::shared_ptr<T>> result{};
      return result;
   }

   template <class T, class... Args>
   std::vector<T> list() {
      std::vector<T> result{};
      return result;
   }
};
} // namespace lingodb::ast
#endif
