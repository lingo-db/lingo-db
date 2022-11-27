#ifndef EXECUTION_ERROR_H
#define EXECUTION_ERROR_H
#include <sstream>

namespace execution {
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
} // namespace execution
#endif //EXECUTION_ERROR_H
