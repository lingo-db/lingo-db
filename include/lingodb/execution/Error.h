#ifndef LINGODB_EXECUTION_ERROR_H
#define LINGODB_EXECUTION_ERROR_H
#include <sstream>

namespace lingodb::execution {
class Error {
   bool present = false;
   std::stringstream message;

   public:
   std::string getMessage() { return message.str(); }
   operator bool() const {
      return present;
   }
   std::stringstream& emit() {
      present = true;
      return message;
   }
};
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_ERROR_H
