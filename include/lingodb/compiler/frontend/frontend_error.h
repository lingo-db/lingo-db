#pragma once
#include "lingodb/compiler/frontend/sql-parser/gen/location.hh"

#include <stdexcept>
namespace lingodb {
class frontend_error : public std::runtime_error {
   public:
   frontend_error();
   explicit frontend_error(const std::string& message);
   frontend_error(const std::string& message, const location& loc);
   const char* what() const noexcept override;
   private:
   location loc;
   std::string errorMsg;
};

class syntax_error : public frontend_error {
   public:
   syntax_error();
   explicit syntax_error(const std::string& message);
   syntax_error(const std::string& message, const location& loc);


};
} // namespace lingodb