#include "lingodb/compiler/frontend/frontend_error.h"
#include <sstream>
namespace lingodb {
FrontendError::FrontendError() : std::runtime_error("FrontendError") {
}
FrontendError::FrontendError(const std::string& message) : std::runtime_error(message) {
}
FrontendError::FrontendError(const std::string& message, const location& loc) : std::runtime_error(message), loc(loc) {
   std::ostringstream s{};
   s << runtime_error::what() << " at " << loc;
   errorMsg = s.str();

}
const char* FrontendError::what() const noexcept {
   return errorMsg.c_str();
}

SyntaxError::SyntaxError() {
}
SyntaxError::SyntaxError(const std::string& message) : FrontendError(message) {
}
SyntaxError::SyntaxError(const std::string& message, const location& loc) : FrontendError(message, loc) {
}
} // namespace lingodb