#include "lingodb/compiler/frontend/translation_context.h"
namespace lingodb::translator {

TupleScope::TupleScope(TranslationContext* context) : context(context) {
   context->currTuple.push(context->currTuple.top());
}
TupleScope::~TupleScope() {
   context->currTuple.pop();
}

TranslationContext::TranslationContext() : currTuple() {
   currTuple.push(mlir::Value());
}
mlir::Value TranslationContext::getCurrentTuple() {
   return currTuple.top();
}
void TranslationContext::setCurrentTuple(mlir::Value v) {
   currTuple.top() = v;
}

} // namespace lingodb::translator