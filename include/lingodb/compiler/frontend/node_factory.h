#pragma once
#include "lingodb/compiler/frontend/sql-parser/gen/location.hh"
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