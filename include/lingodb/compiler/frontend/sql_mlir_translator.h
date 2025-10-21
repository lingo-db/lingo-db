#ifndef LINGODB_COMPILER_FRONTEND_SQL_MLIR_TRANSLATOR_H
#define LINGODB_COMPILER_FRONTEND_SQL_MLIR_TRANSLATOR_H

#include "ast/bound/bound_create_node.h"
#include "ast/bound/bound_insert_node.h"
#include "ast/bound/bound_pipe_operator.h"
#include "ast/create_node.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include "lingodb/compiler/frontend/ast/bound/bound_aggregation.h"
#include "lingodb/compiler/frontend/ast/bound/bound_resultmodifier.h"
#include "lingodb/compiler/frontend/ast/bound/bound_tableref.h"
#include "lingodb/compiler/frontend/ast/copy_node.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
#include "lingodb/compiler/frontend/ast/set_node.h"
#include "lingodb/compiler/frontend/frontend_error.h"
#include "lingodb/compiler/frontend/sql_context.h"
#include "lingodb/compiler/frontend/translation_context.h"

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include <lingodb/compiler/Dialect/util/UtilOps.h>

namespace lingodb::translator {
#define translatorError(message, loc)    \
   {                                     \
      std::ostringstream s{};            \
      s << message;                      \
      throw FrontendError(s.str(), loc); \
   }
class SQLMlirTranslator {
   public:
   SQLMlirTranslator(mlir::ModuleOp moduleOp, catalog::Catalog* catalog);
   mlir::ModuleOp moduleOp;
   catalog::Catalog* catalog;
   compiler::dialect::tuples::ColumnManager& attrManager;

   std::optional<mlir::Value> translateStart(mlir::OpBuilder& builder, std::shared_ptr<ast::AstNode> astNode, std::shared_ptr<analyzer::SQLContext> context);

   private:
   std::shared_ptr<TranslationContext> translationContext;

   mlir::Value translateTableProducer(mlir::OpBuilder& builder, std::shared_ptr<ast::TableProducer> tableProducer, std::shared_ptr<analyzer::SQLContext> context);

   void translateCreateNode(mlir::OpBuilder& builder, std::shared_ptr<ast::CreateNode> createNode, std::shared_ptr<analyzer::SQLContext> context);
   void translateCreateFunction(mlir::OpBuilder& builder, std::shared_ptr<ast::CreateNode> createNode, std::shared_ptr<ast::BoundCreateFunctionInfo> boundCreateFunctionInfo, std::shared_ptr<analyzer::SQLContext> context);
   void translateInsertNode(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundInsertNode> insertNode, std::shared_ptr<analyzer::SQLContext> context);
   void translateSetNode(mlir::OpBuilder& builder, std::shared_ptr<ast::SetNode> insertNode, std::shared_ptr<analyzer::SQLContext> context);
   void translateCopyNode(mlir::OpBuilder& builder, std::shared_ptr<ast::CopyNode> copyStmt, std::shared_ptr<analyzer::SQLContext> context);
   catalog::CreateTableDef translateTableElements(mlir::OpBuilder& builder, std::vector<std::shared_ptr<ast::TableElement>> tableElements, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translatePipeOperator(mlir::OpBuilder& builder, std::shared_ptr<ast::PipeOperator> pipeOperator, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);

   mlir::Value translateResultModifier(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundResultModifier> resultModifier, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);

   mlir::Value translateSubquery(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundSubqueryExpression> subquery, std::shared_ptr<analyzer::SQLContext> context);
   mlir::Value translateExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context);
   mlir::Value translateBinaryOperatorExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundOperatorExpression> expression, std::shared_ptr<analyzer::SQLContext> context, mlir::Value left, mlir::Value right);
   mlir::Value translateWindowExpression(mlir::OpBuilder& builder, mlir::Value tree, std::shared_ptr<ast::BoundWindowExpression> expression, std::shared_ptr<analyzer::SQLContext> context);
   mlir::Value translateWhenChecks(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundCaseExpression> boundCase, std::optional<mlir::Value> caseExprTranslated, std::vector<ast::BoundCaseExpression::BoundCaseCheck> caseChecks, std::shared_ptr<ast::BoundExpression> elseExpr, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translateCoalesceExpression(mlir::OpBuilder& builder, NullableType resultType, std::vector<std::shared_ptr<ast::BoundExpression>> expressions, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translateTableRef(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundTableRef> tableRef, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value translateSetOperation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundSetOperationNode> boundSetOp, std::shared_ptr<analyzer::SQLContext> context);
   mlir::Value translateAggregationFunction(mlir::OpBuilder& builder, std::string mapName, std::vector<mlir::Attribute> groupByAttrs, mlir::Value relation, mlir::OpBuilder functionBuilder, std::shared_ptr<ast::BoundFunctionExpression> aggrFunction, mlir::Value& expr, compiler::dialect::tuples::ColumnDefAttr& attrDef);
   mlir::Value translateGroupByAttributesAndAggregate(mlir::OpBuilder& builder, mlir::Value tree, mlir::Location loc, std::vector<std::shared_ptr<ast::ColumnReference>> groupColumnReferences, std::vector<std::shared_ptr<ast::BoundFunctionExpression>> aggregations, std::string mapName);

   mlir::Value translateAggregation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundAggregationNode> aggregation, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);
   mlir::Value createMap(mlir::OpBuilder& builder, mlir::Location loc, std::string mapName, std::vector<std::shared_ptr<ast::BoundExpression>> toMap, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree);

   /*
    * Utility
   */

   mlir::Block* translatePredicate(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context);

   mlir::Value createStringValue(mlir::OpBuilder& builder, std::string str) {
      return builder.create<compiler::dialect::util::CreateConstVarLen>(builder.getUnknownLoc(), compiler::dialect::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(str));
   }

   mlir::Location getLocationFromBison(const location& loc, mlir::MLIRContext* ctx);

   public:
};
} // namespace lingodb::translator
#endif
