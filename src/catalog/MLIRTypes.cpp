#include "lingodb/catalog/MLIRTypes.h"
#include "../../include/lingodb/catalog/Types.h"
#ifndef MLIR_DISABLED
#include "../../include/lingodb/compiler/Dialect/DB/IR/DBTypes.h"
namespace {
class BoolTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   mlir::Type createType(mlir::MLIRContext* context) override {
      return mlir::IntegerType::get(context, 1);
   }
};
class IntTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   explicit IntTypeCreator(std::shared_ptr<lingodb::catalog::IntTypeInfo> info) : info(info) {}
   mlir::Type createType(mlir::MLIRContext* context) override {
      return mlir::IntegerType::get(context, info->getBitWidth()); //todo: sign
   }

   private:
   std::shared_ptr<lingodb::catalog::IntTypeInfo> info;
};
template <class T>
class GenericTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   mlir::Type createType(mlir::MLIRContext* context) override {
      return T::get(context);
   }
};
class DecimalTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   explicit DecimalTypeCreator(std::shared_ptr<lingodb::catalog::DecimalTypeInfo> info) : info(info) {}
   mlir::Type createType(mlir::MLIRContext* context) override {
      return lingodb::compiler::dialect::db::DecimalType::get(context, info->getPrecision(), info->getScale());
   }

   private:
   std::shared_ptr<lingodb::catalog::DecimalTypeInfo> info;
};
class TimestampTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   explicit TimestampTypeCreator(std::shared_ptr<lingodb::catalog::TimestampTypeInfo> info) : info(info) {}
   mlir::Type createType(mlir::MLIRContext* context) override {
      return lingodb::compiler::dialect::db::TimestampType::get(context, lingodb::compiler::dialect::db::symbolizeTimeUnitAttr(static_cast<size_t>(info->getUnit())).value());
   }

   private:
   std::shared_ptr<lingodb::catalog::TimestampTypeInfo> info;
};
class DateTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   DateTypeCreator(std::shared_ptr<lingodb::catalog::DateTypeInfo> info) : info(info) {}
   mlir::Type createType(mlir::MLIRContext* context) override {
      return lingodb::compiler::dialect::db::DateType::get(context, lingodb::compiler::dialect::db::symbolizeDateUnitAttr(static_cast<size_t>(info->getUnit())).value());
   }

   private:
   std::shared_ptr<lingodb::catalog::DateTypeInfo> info;
};

class IntervalTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   explicit IntervalTypeCreator(std::shared_ptr<lingodb::catalog::IntervalTypeInfo> info) : info(info) {}
   mlir::Type createType(mlir::MLIRContext* context) override {
      return lingodb::compiler::dialect::db::IntervalType::get(context, lingodb::compiler::dialect::db::symbolizeIntervalUnitAttr(static_cast<size_t>(info->getUnit())).value());
   }

   private:
   std::shared_ptr<lingodb::catalog::IntervalTypeInfo> info;
};
class CharTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   explicit CharTypeCreator(std::shared_ptr<lingodb::catalog::CharTypeInfo> info) : info(info) {}
   mlir::Type createType(mlir::MLIRContext* context) override {
      return lingodb::compiler::dialect::db::CharType::get(context, info->getLength());
   }

   private:
   std::shared_ptr<lingodb::catalog::CharTypeInfo> info;
};
class NoneTypeCreator : public lingodb::catalog::MLIRTypeCreator {
   public:
   mlir::Type createType(mlir::MLIRContext* context) override {
      return mlir::NoneType::get(context);
   }
};
} // namespace
namespace lingodb::catalog {
std::shared_ptr<MLIRTypeCreator> createBoolTypeCreator() {
   return std::make_shared<BoolTypeCreator>();
}
std::shared_ptr<MLIRTypeCreator> createIntTypeCreator(std::shared_ptr<catalog::IntTypeInfo> info) {
   return std::make_shared<IntTypeCreator>(info);
}
std::shared_ptr<MLIRTypeCreator> createFloatTypeCreator() {
   return std::make_shared<GenericTypeCreator<mlir::Float32Type>>();
}
std::shared_ptr<MLIRTypeCreator> createDoubleTypeCreator() {
   return std::make_shared<GenericTypeCreator<mlir::Float64Type>>();
}
std::shared_ptr<MLIRTypeCreator> createDecimalTypeCreator(std::shared_ptr<catalog::DecimalTypeInfo> info) {
   return std::make_shared<DecimalTypeCreator>(info);
}
std::shared_ptr<MLIRTypeCreator> createDateTypeCreator(std::shared_ptr<catalog::DateTypeInfo> info) {
   return std::make_shared<DateTypeCreator>(info);
}
std::shared_ptr<MLIRTypeCreator> createTimestampTypeCreator(std::shared_ptr<catalog::TimestampTypeInfo> info) {
   return std::make_shared<TimestampTypeCreator>(info);
}
std::shared_ptr<MLIRTypeCreator> createIntervalTypeCreator(std::shared_ptr<catalog::IntervalTypeInfo> info) {
   return std::make_shared<IntervalTypeCreator>(info);
}
std::shared_ptr<MLIRTypeCreator> createCharTypeCreator(std::shared_ptr<catalog::CharTypeInfo> info) {
   return std::make_shared<CharTypeCreator>(info);
}
std::shared_ptr<MLIRTypeCreator> createStringTypeCreator(std::shared_ptr<catalog::StringTypeInfo> info) {
   return std::make_shared<GenericTypeCreator<lingodb::compiler::dialect::db::StringType>>();
}
std::shared_ptr<MLIRTypeCreator> createNoneTypeCreator() {
   return std::make_shared<NoneTypeCreator>();
}
std::shared_ptr<MLIRTypeCreator> createIndexTypeCreator() {
   return std::make_shared<GenericTypeCreator<mlir::IndexType>>();
}

} // namespace lingodb::catalog
#else
namespace lingodb::catalog {
std::shared_ptr<MLIRTypeCreator> createBoolTypeCreator() {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createIntTypeCreator(std::shared_ptr<catalog::IntTypeInfo> info) {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createFloatTypeCreator() {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createDoubleTypeCreator() {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createDecimalTypeCreator(std::shared_ptr<catalog::DecimalTypeInfo> info) {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createDateTypeCreator(std::shared_ptr<catalog::DateTypeInfo> info) {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createTimestampTypeCreator(std::shared_ptr<catalog::TimestampTypeInfo> info) {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createIntervalTypeCreator(std::shared_ptr<catalog::IntervalTypeInfo> info) {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createCharTypeCreator(std::shared_ptr<catalog::CharTypeInfo> info) {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createStringTypeCreator(std::shared_ptr<catalog::StringTypeInfo> info) {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createNoneTypeCreator() {
   return {};
}
std::shared_ptr<MLIRTypeCreator> createIndexTypeCreator() {
   return {};
}
}
#endif