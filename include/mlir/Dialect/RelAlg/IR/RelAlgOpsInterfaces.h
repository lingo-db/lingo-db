#ifndef RelAlg_RelAlgInterfaces
#define RelAlg_RelAlgInterfaces

#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <mlir/Dialect/RelAlg/Attributes.h>
#include <mlir/IR/SymbolTable.h>
#include "mlir/IR/Builders.h"


namespace mlir::relalg::detail {
Attributes getUsedAttributes(mlir::Operation* op);
Attributes getAvailableAttributes(mlir::Operation* op);
Attributes getFreeAttributes(mlir::Operation* op);
Attributes getCreatedAttributes(mlir::Operation* op);
bool isDependentJoin(mlir::Operation* op);

enum BinaryOperatorType {
   None,
   CP = 1,
   InnerJoin,
   SemiJoin,
   AntiSemiJoin,
   OuterJoin,
   FullOuterJoin,
   MarkJoin,
   LAST
};

extern const bool assoc[BinaryOperatorType::LAST][BinaryOperatorType::LAST];
extern const bool lAsscom[BinaryOperatorType::LAST][BinaryOperatorType::LAST];
extern const bool rAsscom[BinaryOperatorType::LAST][BinaryOperatorType::LAST];
BinaryOperatorType getBinaryOperatorType(Operation* op);
bool binaryOperatorIs(const bool (&table)[BinaryOperatorType::LAST][BinaryOperatorType::LAST], Operation* a, Operation* b);
bool isJoin(Operation* op);

void addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)> predicateProducer);
void initPredicate(mlir::Operation* op);
}
class Operator;
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h.inc"

#endif // RelAlg_RelAlgInterfaces
