#include "lingodb/compiler/frontend/frontend_error.h"
#include <sstream>
namespace lingodb {
frontend_error::frontend_error() : std::runtime_error("FrontendError") {
}
frontend_error::frontend_error(const std::string& message) : std::runtime_error(message) {
}
frontend_error::frontend_error(const std::string& message, const location& loc) : std::runtime_error(message), loc(loc) {
   std::ostringstream s{};
   s << runtime_error::what() << " at " << loc;
   errorMsg = s.str();

}
const char* frontend_error::what() const noexcept {
   return errorMsg.c_str();
}

syntax_error::syntax_error() {
}
syntax_error::syntax_error(const std::string& message) : frontend_error(message) {
}
syntax_error::syntax_error(const std::string& message, const location& loc) : frontend_error(message, loc) {
}
} // namespace lingodb