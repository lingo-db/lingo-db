#ifndef LINGODB_EXECUTION_ERROR_H
#define LINGODB_EXECUTION_ERROR_H
#include <sstream>

namespace lingodb::execution {
class Error {
   public:
   enum ErrorPhase {
      frontend,
      optimizer,
      tuple_tracking,
      lowering,
      backend,
      unknown
   };

   private:
   bool present = false;
   std::stringstream message;
   ErrorPhase errorPhase = unknown;

   public:
   std::string getMessage() { return message.str(); }
   ErrorPhase getErrorPhase() const { return errorPhase; }
   void setErrorPhase(ErrorPhase type) { this->errorPhase = type; }
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
