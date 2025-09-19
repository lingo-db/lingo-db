#ifndef LINGODB_COMPILER_FRONTEND_FRONTEND_ERROR_H
#define LINGODB_COMPILER_FRONTEND_FRONTEND_ERROR_H

#include "lingodb/compiler/frontend/sql-parser/gen/location.hh"

#include <stdexcept>
namespace lingodb {
class FrontendError : public std::runtime_error {
   public:
   FrontendError();
   explicit FrontendError(const std::string& message);
   FrontendError(const std::string& message, const location& loc);
   const char* what() const noexcept override;
   private:
   location loc;
   std::string errorMsg;
};

class SyntaxError : public FrontendError {
   public:
   SyntaxError();
   explicit SyntaxError(const std::string& message);
   SyntaxError(const std::string& message, const location& loc);


};
} // namespace lingodb
#endif
