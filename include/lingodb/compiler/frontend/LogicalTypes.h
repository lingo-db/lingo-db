#ifndef LINGODB_COMPILER_FRONTEND_LOGICALTYPES_H
#define LINGODB_COMPILER_FRONTEND_LOGICALTYPES_H
#include "mlir/IR/Types.h"

namespace lingodb::catalog {
class IntTypeInfo;
class DecimalTypeInfo;
class StringTypeInfo;
class CharTypeInfo;
class TimestampTypeInfo;
class DateTypeInfo;
class IntervalTypeInfo;
}
namespace lingodb::compiler::frontend {

class MLIRTypeCreator {
   public:
   virtual mlir::Type createType(mlir::MLIRContext* context) = 0;
   ~MLIRTypeCreator() = default;
};
std::shared_ptr<MLIRTypeCreator> createBoolTypeCreator();
std::shared_ptr<MLIRTypeCreator> createIntTypeCreator(std::shared_ptr<catalog::IntTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createFloatTypeCreator();
std::shared_ptr<MLIRTypeCreator> createDoubleTypeCreator();
std::shared_ptr<MLIRTypeCreator> createDecimalTypeCreator(std::shared_ptr<catalog::DecimalTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createDateTypeCreator(std::shared_ptr<catalog::DateTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createTimestampTypeCreator(std::shared_ptr<catalog::TimestampTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createIntervalTypeCreator(std::shared_ptr<catalog::IntervalTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createCharTypeCreator(std::shared_ptr<catalog::CharTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createStringTypeCreator(std::shared_ptr<catalog::StringTypeInfo> info);
}

#endif //LINGODB_COMPILER_FRONTEND_LOGICALTYPES_H
