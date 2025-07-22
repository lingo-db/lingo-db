#pragma once
#include "ast/bound/bound_insert_node.h"
#include "ast/bound/bound_pipe_operator.h"
#include "ast/create_node.h"
#include "ast/insert_node.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include "lingodb/compiler/frontend/ast/bound/bound_aggregation.h"
#include "lingodb/compiler/frontend/ast/bound/bound_resultmodifier.h"
#include "lingodb/compiler/frontend/ast/bound/bound_tableref.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
#include "lingodb/compiler/frontend/ast/select_node.h"
#include "lingodb/compiler/frontend/sql_context.h"
#include "lingodb/compiler/frontend/translation_context.h"

#include <memory>

#include "mlir/IR/BuiltinOps.h"

#include <lingodb/compiler/Dialect/util/UtilOps.h>

namespace lingodb::ast {
class BoundSetOperationNode;
}
namespace lingodb::translator {
#define error(message, loc)                                         \
   {                                                                \
      std::ostringstream s{};                                       \
      s << "Translator: " << message << " at " << loc << std::endl; \
      throw std::runtime_error(s.str());                            \
   }
class SQLMlirTranslator {
   public:
   SQLMlirTranslator(mlir::ModuleOp moduleOp, std::shared_ptr<catalog::Catalog> catalog);
   mlir::ModuleOp moduleOp;
   compiler::dialect::tuples::ColumnManager& attrManager;

   std::optional<mlir::Value> translateStart(mlir::OpBuilder& builder, std::shared_ptr<ast::AstNode> astNode, std::shared_ptr<analyzer::SQLContext> context);

   private:
   std::shared_ptr<TranslationContext> translationContext;
   std::shared_ptr<catalog::Catalog> catalog;

   mlir::Value translateTableProducer(mlir::OpBuilder& builder, std::shared_ptr<ast::TableProducer> tableProducer, std::shared_ptr<analyzer::SQLContext> context);

   void translateCreateNode(mlir::OpBuilder& builder, std::shared_ptr<ast::CreateNode> createNode, std::shared_ptr<analyzer::SQLContext> context);
   void translateInsertNode(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundInsertNode> insertNode, std::shared_ptr<analyzer::SQLContext> context);
   catalog::CreateTableDef translateTableElements(mlir::OpBuilder& builder, std::vector<std::shared_ptr<ast::TableElement>> tableElements, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translatePipeOperator(mlir::OpBuilder& builder, std::shared_ptr<ast::PipeOperator> pipeOperator, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);

   mlir::Value translateResultModifier(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundResultModifier> resultModifier, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);

   mlir::Value translateExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context);
   mlir::Value translateBinaryOperatorExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundOperatorExpression> expression, std::shared_ptr<analyzer::SQLContext> context, mlir::Value left, mlir::Value right);
   mlir::Value translateWhenCheks(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundCaseExpression> boundCase, std::optional<mlir::Value> caseExprTranslated, std::vector<ast::BoundCaseExpression::BoundCaseCheck> caseChecks, std::shared_ptr<ast::BoundExpression> elseExpr, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translateWhenCheck(mlir::OpBuilder& builder, ast::BoundCaseExpression::BoundCaseCheck whenCheck, std::shared_ptr<ast::BoundExpression> elseExpr, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translateTableRef(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundTableRef> tableRef, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translateSetOperation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundSetOperationNode> boundSetOp, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translateAggregation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundAggregationNode> aggregation, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);
   mlir::Value createMap(mlir::OpBuilder& builder, std::string mapName, std::vector<std::shared_ptr<ast::BoundExpression>> toMap, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);

   /*
    * Utility
   */

   mlir::Block* translatePredicate(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value createStringValue(mlir::OpBuilder& builder, std::string str) {
      return builder.create<compiler::dialect::util::CreateConstVarLen>(builder.getUnknownLoc(), compiler::dialect::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(str));
   }

   public:
   /*
   * Helper functions
   */

   mlir::Type createBaseTypeFromColumnType(mlir::MLIRContext* context, const catalog::Type& t);
};
} // namespace lingodb::translator