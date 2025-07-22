#include "lingodb/compiler/frontend/sql_mlir_translator.h"

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/MLIRTypes.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/ast/bound/bound_create_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_extend_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_insert_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_query_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_tableref.h"
#include "lingodb/compiler/frontend/ast/create_node.h"
#include "lingodb/compiler/frontend/ast/cte_node.h"
#include "lingodb/compiler/frontend/ast/insert_node.h"
#include "lingodb/compiler/old-frontend/SQL/Parser.h"

#include "lingodb/utility/Serialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <lingodb/compiler/Dialect/util/UtilOps.h.inc>
#include <lingodb/compiler/runtime/ExecutionContext.h>
#include <lingodb/compiler/runtime/RelationHelper.h>
#include <mlir-c/IR.h>
namespace lingodb::translator {
using namespace lingodb::compiler::dialect;
SQLMlirTranslator::SQLMlirTranslator(mlir::ModuleOp moduleOp, std::shared_ptr<catalog::Catalog> catalog) : moduleOp(moduleOp),
                                                                                                           attrManager(moduleOp->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager()), catalog(catalog), translationContext(std::make_shared<TranslationContext>())

{
   moduleOp.getContext()->getLoadedDialect<util::UtilDialect>()->getFunctionHelper().setParentModule(moduleOp);
}
std::optional<mlir::Value> SQLMlirTranslator::translateStart(mlir::OpBuilder& builder, std::shared_ptr<ast::AstNode> astNode, std::shared_ptr<analyzer::SQLContext> context) {
   auto tableProducer = std::dynamic_pointer_cast<ast::TableProducer>(astNode);
   if (!tableProducer) {
      //Root node is not a TableProducer
      switch (astNode->nodeType) {
         case ast::NodeType::CREATE_NODE: {
            auto createNode = std::static_pointer_cast<ast::CreateNode>(astNode);
            translateCreateNode(builder, createNode, context);
            return std::nullopt;
         }
         case ast::NodeType::BOUND_INSERT_NODE: {
            auto insertNode = std::static_pointer_cast<ast::BoundInsertNode>(astNode);
            translateInsertNode(builder, insertNode, context);
            return std::nullopt;
         }
         default: error("Invalid root node type", astNode->loc);
      }

   } else {
      //Root node is a TableProducer
      mlir::Block* block = new mlir::Block();
      mlir::Type localTableType;
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(block);

         //Translate the cteNodes
         for (auto [name, cte] : context->ctes) {
            auto cteNode = cte.second;
            context->pushNewScope(std::make_shared<analyzer::SQLScope>(cteNode->subQueryScope));
            auto _tree = translateTableProducer(builder, cteNode->query, context);
            context->popCurrentScope();
            context->translatedCtes.insert({name, _tree});
         }

         auto tree = translateTableProducer(builder, tableProducer, context);

         context->currentScope->evalBefore.clear();
         std::vector<mlir::Attribute> attrs;
         std::vector<mlir::Attribute> names;
         std::vector<mlir::Attribute> colMemberNames;
         std::vector<mlir::Attribute> colTypes;
         auto& memberManager = builder.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

         for (auto& named : context->currentScope->targetInfo.targetColumns) {
            switch (named->type) {
               case ast::NamedResultType::Column: {
                  auto colMemberName = memberManager.getUniqueMember(named->displayName);
                  names.push_back(builder.getStringAttr(named->displayName));

                  colTypes.push_back(mlir::TypeAttr::get(named->resultType.toMlirType(builder.getContext())));
                  colMemberNames.push_back(builder.getStringAttr(colMemberName));
                  auto attrDef = named->createRef(builder, attrManager);
                  attrs.push_back(attrDef);
                  break;
               }
               case ast::NamedResultType::Function: {
                  auto functionInfo = std::static_pointer_cast<ast::FunctionInfo>(named);
                  names.push_back(builder.getStringAttr(functionInfo->displayName.empty() ? functionInfo->name : functionInfo->displayName));
                  auto colMemberName = memberManager.getUniqueMember(functionInfo->name);
                  colMemberNames.push_back(builder.getStringAttr(colMemberName));
                  colTypes.push_back(mlir::TypeAttr::get(functionInfo->resultType.toMlirType(builder.getContext())));
                  auto attrDef = functionInfo->createRef(builder, attrManager);
                  attrs.push_back(attrDef);
                  break;
               }
               case ast::NamedResultType::EXPRESSION: {
                  names.push_back(builder.getStringAttr(named->displayName));
                  auto colMemberName = memberManager.getUniqueMember(named->name);
                  colMemberNames.push_back(builder.getStringAttr(colMemberName));
                  colTypes.push_back(mlir::TypeAttr::get(named->resultType.toMlirType(builder.getContext())));

                  auto attrDef = named->createRef(builder, attrManager);
                  attrs.push_back(attrDef);
                  break;
               }
               default: {
                  error("Not implemented", tableProducer->loc);
               }
            }
         }

         localTableType = subop::LocalTableType::get(
            builder.getContext(),
            subop::StateMembersAttr::get(
               builder.getContext(),
               builder.getArrayAttr(colMemberNames),
               builder.getArrayAttr(colTypes)),
            builder.getArrayAttr(names));

         mlir::Value result = builder.create<relalg::MaterializeOp>(
            builder.getUnknownLoc(),
            localTableType,
            tree,
            builder.getArrayAttr(attrs),
            builder.getArrayAttr(names));

         // Use the materialized result in the QueryReturnOp instead of the input tree
         builder.create<relalg::QueryReturnOp>(builder.getUnknownLoc(), result);
      }
      relalg::QueryOp queryOp = builder.create<relalg::QueryOp>(builder.getUnknownLoc(), mlir::TypeRange{localTableType}, mlir::ValueRange{});
      queryOp.getQueryOps().getBlocks().clear();
      queryOp.getQueryOps().push_back(block);
      return queryOp.getResults()[0];
   }
}

mlir::Value SQLMlirTranslator::translateTableProducer(mlir::OpBuilder& builder, std::shared_ptr<ast::TableProducer> tableProducer, std::shared_ptr<analyzer::SQLContext> context) {
   mlir::Value tree;

   switch (tableProducer->nodeType) {
      case ast::NodeType::PIPE_OP: {
         auto pipeOperator = std::static_pointer_cast<ast::PipeOperator>(tableProducer);

         if (pipeOperator->input) {
            tree = translateTableProducer(builder, pipeOperator->input, context);
         }
         tree = translatePipeOperator(builder, pipeOperator, context, tree);
         break;
      }
      case ast::NodeType::BOUND_TABLE_REF: {
         auto tableRef = std::static_pointer_cast<ast::BoundTableRef>(tableProducer);
         tree = translateTableRef(builder, tableRef, context);

         break;
      }
      case ast::NodeType::BOUND_RESULT_MODIFIER: {
         auto resultModifier = std::static_pointer_cast<ast::BoundResultModifier>(tableProducer);
         if (resultModifier->input) {
            tree = translateTableProducer(builder, resultModifier->input, context);
         }
         tree = translateResultModifier(builder, resultModifier, context, tree);
         break;
      }
      case ast::NodeType::QUERY_NODE: {
         auto queryNode = std::static_pointer_cast<ast::QueryNode>(tableProducer);
         switch (queryNode->type) {
            case ast::QueryNodeType::CTE_NODE: {
               auto cteNode = std::static_pointer_cast<ast::CTENode>(queryNode);
               if (cteNode->child) {
                  tree = translateTableProducer(builder, cteNode->child, context);
               }

               return tree;
            }
            case ast::QueryNodeType::BOUND_SET_OPERATION_NODE: {
               return translateSetOperation(builder, std::static_pointer_cast<ast::BoundSetOperationNode>(tableProducer), context);
            }
            default: error("Not implemented", tableProducer->loc);
         }
      }
      default: error("Not implemented", tableProducer->loc);
   }

   return tree;
}

void SQLMlirTranslator::translateCreateNode(mlir::OpBuilder& builder, std::shared_ptr<ast::CreateNode> createNode, std::shared_ptr<analyzer::SQLContext> context) {
   switch (createNode->createInfo->type) {
      case ast::CatalogType::TABLE_ENTRY: {
         auto createTableInfo = std::static_pointer_cast<ast::CreateTableInfo>(createNode->createInfo);
         auto tableName = createTableInfo->tableName;
         auto tableDef = translateTableElements(builder, createTableInfo->tableElements, context);
         tableDef.name = tableName;
         auto descriptionValue = createStringValue(builder, utility::serializeToHexString(tableDef));
         compiler::runtime::RelationHelper::createTable(builder, builder.getUnknownLoc())(mlir::ValueRange({descriptionValue}));
         break;
      }
      default: error("CreateInfo type not implemented", createNode->loc);
   }
}

void SQLMlirTranslator::translateInsertNode(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundInsertNode> insertNode, std::shared_ptr<analyzer::SQLContext> context) {
   mlir::Block* block = new mlir::Block;
   mlir::Type localTableType;
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(block);
      auto tree = translateTableProducer(builder, insertNode->producer, context);
      auto rel = context->catalog->getTypedEntry<catalog::TableCatalogEntry>(insertNode->tableName).value();

      std::vector<mlir::Value> createdValues;
      std::unordered_map<std::string, mlir::Value> columnNameToCreatedValue;

      //Build map
      mlir::Block* block = new mlir::Block;
      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      auto tupleScope = translationContext->createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      translationContext->setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);

      std::unordered_map<std::string, mlir::Attribute> insertedCols;

      std::vector<mlir::Attribute> createdCols;
      auto mapName = attrManager.getUniqueScope("map");
      for (size_t i = 0; i < insertNode->columnsToInsert.size(); i++) {
         auto attrRef = context->currentScope->targetInfo.targetColumns[i]->createRef(builder, attrManager);
         auto currentType = context->currentScope->targetInfo.targetColumns[i]->resultType;
         auto tableType = insertNode->allColumnsAndTypes.at(insertNode->columnsToInsert[i]);
         mlir::Value expr = mapBuilder.create<tuples::GetColumnOp>(mapBuilder.getUnknownLoc(), attrRef.getColumn().type, attrRef, tuple);
         if (currentType != tableType) {
            auto attrDef = attrManager.createDef(mapName, std::string("inserted") + std::to_string(i));
            attrDef.getColumn().type = tableType.toMlirType(builder.getContext());

            createdCols.push_back(attrDef);
            mlir::Value casted = tableType.castValueToThisType(mapBuilder, expr, context->currentScope->targetInfo.targetColumns[i]->resultType.isNullable); // SQLTypeInference::castValueToType(mapBuilder, expr, tableType);

            createdValues.push_back(casted);
            columnNameToCreatedValue[insertNode->columnsToInsert[i]] = casted;
            insertedCols[insertNode->columnsToInsert[i]] = attrManager.createRef(&attrDef.getColumn());

         } else {
            columnNameToCreatedValue[insertNode->columnsToInsert[i]] = expr;
            insertedCols[insertNode->columnsToInsert[i]] = attrRef;
         }
      }

      auto mapOp = builder.create<relalg::MapOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);

      std::vector<mlir::Attribute> colMemberNames;
      std::vector<mlir::Attribute> orderedColNamesAttrs;
      std::vector<mlir::Attribute> orderedColAttrs;
      std::vector<mlir::Attribute> colTypes;
      auto& memberManager = builder.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

      for (auto x : rel->getColumnNames()) {
         colMemberNames.push_back(builder.getStringAttr(memberManager.getUniqueMember(x)));
         orderedColNamesAttrs.push_back(builder.getStringAttr(x));
         orderedColAttrs.push_back(insertedCols.at(x));
         colTypes.push_back(mlir::TypeAttr::get(mlir::cast<tuples::ColumnRefAttr>(insertedCols.at(x)).getColumn().type));
      }

      localTableType = subop::LocalTableType::get(builder.getContext(), subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr(colMemberNames), builder.getArrayAttr(colTypes)), builder.getArrayAttr(orderedColNamesAttrs));
      mlir::Value newRows = builder.create<relalg::MaterializeOp>(builder.getUnknownLoc(), localTableType, mapOp.getResult(), builder.getArrayAttr(orderedColAttrs), builder.getArrayAttr(orderedColNamesAttrs));
      builder.create<relalg::QueryReturnOp>(builder.getUnknownLoc(), newRows);
   }

   relalg::QueryOp queryOp = builder.create<relalg::QueryOp>(builder.getUnknownLoc(), mlir::TypeRange{localTableType}, mlir::ValueRange{});
   queryOp.getQueryOps().getBlocks().clear();
   queryOp.getQueryOps().push_back(block);
   auto tableNameValue = createStringValue(builder, insertNode->tableName);
   auto resultIdValue = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 0, builder.getI32Type());
   builder.create<subop::SetResultOp>(builder.getUnknownLoc(), 0, queryOp.getResults()[0]);

   compiler::runtime::RelationHelper::appendTableFromResult(builder, builder.getUnknownLoc())(mlir::ValueRange{tableNameValue, resultIdValue});
   compiler::runtime::ExecutionContext::clearResult(builder, builder.getUnknownLoc())({resultIdValue});
}

catalog::CreateTableDef SQLMlirTranslator::translateTableElements(mlir::OpBuilder& builder, std::vector<std::shared_ptr<ast::TableElement>> tableElements, std::shared_ptr<analyzer::SQLContext> context) {
   catalog::CreateTableDef tableDef{};
   for (auto tableElement : tableElements) {
      switch (tableElement->type) {
         case ast::TableElementType::COLUMN: {
            auto columnElement = std::static_pointer_cast<ast::BoundColumnElement>(tableElement);
            catalog::Column c{columnElement->name, columnElement->datatype.type, columnElement->datatype.isNullable};
            tableDef.columns.emplace_back(c);

            break;
         }
         case ast::TableElementType::CONSTRAINT: {
            auto tableConstraintElement = std::static_pointer_cast<ast::TableConstraintElement>(tableElement);
            switch (tableConstraintElement->constraint->type) {
               case ast::ConstraintType::UNIQUE: {
                  auto uniqueConstraint = std::static_pointer_cast<ast::UniqueConstraint>(tableConstraintElement->constraint);
                  if (uniqueConstraint->isPrimaryKey) {
                     std::ranges::copy(uniqueConstraint->columnNames, std::back_inserter(tableDef.primaryKey));
                  }

                  break;
               }
               default: error("TableElement constraint type not implemented", tableElement->loc);
            }
            break;
         }
         default: error("TableElement type not implemented", tableElement->loc);
      }
   }
   return tableDef;
}

mlir::Value SQLMlirTranslator::translatePipeOperator(mlir::OpBuilder& builder, std::shared_ptr<ast::PipeOperator> pipeOperator, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree) {
   switch (pipeOperator->pipeOpType) {
      case ast::PipeOperatorType::SELECT: {
         auto selectList = std::static_pointer_cast<ast::BoundTargetsExpression>(pipeOperator->node);
         if (selectList->distinctExpressions.has_value()) {
            if (!selectList->distinctExpressions.value().empty()) {
               error("Distinct clause with multiple elements not supported yet", selectList->distinctExpressions.value()[0]->loc);
            }
            std::vector<mlir::Attribute> columns;
            for (auto x : context->currentScope->targetInfo.targetColumns) {
               columns.push_back(x->createRef(builder, attrManager));
            }
            tree = builder.create<relalg::ProjectionOp>(builder.getUnknownLoc(), relalg::SetSemantic::distinct, tree, builder.getArrayAttr(columns));
         }
         return tree;
      }
      case ast::PipeOperatorType::WHERE: {
         assert(pipeOperator->node->nodeType == ast::NodeType::BOUND_EXPRESSION);
         auto whereClause = std::static_pointer_cast<ast::BoundExpression>(pipeOperator->node);
         //Make predicate
         auto* pred = new mlir::Block;
         mlir::OpBuilder predBuilder(builder.getContext());
         pred->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto tupleScope = translationContext->createTupleScope();
         translationContext->setCurrentTuple(pred->getArgument(0));

         predBuilder.setInsertionPointToStart(pred);
         auto expr = translateExpression(predBuilder, whereClause, context);
         predBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), expr);

         auto sel = builder.create<relalg::SelectionOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), tree);
         sel.getPredicate().push_back(pred);
         return sel.getResult();
      }
      case ast::PipeOperatorType::AGGREGATE: {
         assert(pipeOperator->node->nodeType == ast::NodeType::BOUND_AGGREGATION);
         auto aggregationNode = std::static_pointer_cast<ast::BoundAggregationNode>(pipeOperator->node);
         //TODO logic here
         tree = createMap(builder, attrManager.getUniqueScope("mapBeforeAggr"), aggregationNode->evalBeforeAggr, context, tree);

         tree = translateAggregation(builder, aggregationNode, context, tree);
         tree = createMap(builder, attrManager.getUniqueScope("map2"), context->currentScope->evalBefore, context, tree);
         context->currentScope->evalBefore.clear();
         return tree;
      }
      case ast::PipeOperatorType::EXTEND: {
         auto extendNode = std::static_pointer_cast<ast::BoundExtendNode>(pipeOperator->node);
         tree = createMap(builder, extendNode->mapName, extendNode->extensions, context, tree);
         return tree;
      }
      case ast::PipeOperatorType::UNION_ALL:
      case ast::PipeOperatorType::UNION: {
         error("Should not happen", pipeOperator->loc);
      }
      default: error("Not implememted", pipeOperator->loc);
   }
}

mlir::Value SQLMlirTranslator::translateResultModifier(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundResultModifier> resultModifier, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree) {
   switch (resultModifier->modifierType) {
      case ast::ResultModifierType::BOUND_ORDER_BY: {
         auto orderByModifier = std::static_pointer_cast<ast::BoundOrderByModifier>(resultModifier);
         std::vector<mlir::Attribute> mapping;
         for (auto orderByElement : orderByModifier->orderByElements) {
            relalg::SortSpec spec = relalg::SortSpec::asc;
            if (orderByElement->type == ast::OrderType::DESCENDING) {
               spec = relalg::SortSpec::desc;
            }
            assert(orderByElement->namedResult);
            auto namedResult = orderByElement->namedResult;
            auto attrDef = namedResult->createRef(builder, attrManager);
            mapping.push_back(relalg::SortSpecificationAttr::get(builder.getContext(), attrDef, spec));
         }
         return builder.create<relalg::SortOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(mapping));
      }
      case ast::ResultModifierType::BOUND_LIMIT: {
         auto limitModifier = std::static_pointer_cast<ast::BoundLimitModifier>(resultModifier);
         auto value = std::static_pointer_cast<ast::IntValue>(std::static_pointer_cast<ast::BoundConstantExpression>(limitModifier->limitExpression)->value);
         return builder.create<relalg::LimitOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), value->iVal, tree);
      }
      default: error("ResultModifier Not implemented", resultModifier->loc);
   }
}

mlir::Value SQLMlirTranslator::translateExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context) {
   assert(expression->resultType.has_value());
   switch (expression->exprClass) {
      case ast::ExpressionClass::BOUND_COLUMN_REF: {
         auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(expression);
         assert(columnRef->namedResult.has_value());
         auto nameResult = columnRef->namedResult.value();

         mlir::Type type = nameResult->resultType.toMlirType(builder.getContext());

         auto attrDef = nameResult->createRef(builder, attrManager);
         return builder.create<tuples::GetColumnOp>(
            builder.getUnknownLoc(),
            type, attrDef, translationContext->getCurrentTuple());
      }
      case ast::ExpressionClass::BOUND_CONSTANT: {
         auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(expression);
         switch (constExpr->value->type) {
            case ast::ConstantType::INT: {
               auto value = std::static_pointer_cast<ast::IntValue>(constExpr->value);
               return builder.create<db::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getI32IntegerAttr(value->iVal));
            }
            case ast::ConstantType::STRING: {
               auto value = std::static_pointer_cast<ast::StringValue>(constExpr->value);
               mlir::Type stringType = db::StringType::get(builder.getContext());
               if (value->sVal.size() <= 8 && value->sVal.size() > 0) {
                  stringType = db::CharType::get(builder.getContext(), value->sVal.size());
               };
               return builder.create<db::ConstantOp>(builder.getUnknownLoc(), stringType, builder.getStringAttr(value->sVal));
            }
            case ast::ConstantType::FLOAT: {
               auto value = std::static_pointer_cast<ast::FloatValue>(constExpr->value);
               assert(constExpr->resultType.has_value());
               //TODO support only decimal, without the need for string, see old parser
               return builder.create<db::ConstantOp>(builder.getUnknownLoc(), constExpr->resultType.value().type.getMLIRTypeCreator()->createType(builder.getContext()), builder.getStringAttr(value->fVal));
            }
            case ast::ConstantType::NULL_P: {
               assert(constExpr->resultType.has_value());
               return builder.create<db::NullOp>(builder.getUnknownLoc(), db::NullableType::get(builder.getContext(), builder.getNoneType()));
            }
            case ast::ConstantType::BOOLEAN: {
               auto value = std::static_pointer_cast<ast::BoolValue>(constExpr->value);
               return builder.create<db::ConstantOp>(builder.getUnknownLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI1Type(), value->bVal));
            }

            default: error("Not implemented", expression->loc);
         }
      }
      case ast::ExpressionClass::BOUND_COMPARISON: {
         auto comparisonExpr = std::static_pointer_cast<ast::BoundComparisonExpression>(expression);
         auto left = translateExpression(builder, comparisonExpr->left, context);
         //Handle in
         if (comparisonExpr->type == ast::ExpressionType::COMPARE_IN || comparisonExpr->type == ast::ExpressionType::COMPARE_NOT_IN) {
            std::vector<mlir::Value> values;
            assert(comparisonExpr->left->resultType.has_value());

            values.push_back(comparisonExpr->left->resultType->castValue(builder, left));

            for (auto& rightChild : comparisonExpr->rightChildren) {
               auto right = translateExpression(builder, rightChild, context);
               assert(rightChild->resultType.has_value());
               values.push_back(rightChild->resultType->castValue(builder, right));
            }

            auto oneOf = builder.create<db::OneOfOp>(builder.getUnknownLoc(), values);
            if (comparisonExpr->type == ast::ExpressionType::COMPARE_NOT_IN) {
               return builder.create<db::NotOp>(builder.getUnknownLoc(), oneOf);
            }
            return oneOf;
         }
         //Not IN
         assert(comparisonExpr->rightChildren.size() == 1);
         assert(comparisonExpr->resultType.has_value() && comparisonExpr->left->resultType.has_value() && comparisonExpr->rightChildren[0]->resultType.has_value());

         auto right = translateExpression(builder, comparisonExpr->rightChildren[0], context);
         auto ctLeft = comparisonExpr->left->resultType->castValue(builder, left);
         auto ctRight = comparisonExpr->rightChildren[0]->resultType->castValue(builder, right);
         if (comparisonExpr->type == ast::ExpressionType::COMPARE_LIKE || comparisonExpr->type == ast::ExpressionType::COMPARE_NOT_LIKE) {
            auto isNullable = mlir::isa<db::NullableType>(left.getType()) || mlir::isa<db::NullableType>(right.getType());
            mlir::Type resType = isNullable ? (mlir::Type) db::NullableType::get(builder.getContext(), builder.getI1Type()) : (mlir::Type) builder.getI1Type();
            auto like = builder.create<db::RuntimeCall>(builder.getUnknownLoc(), resType, "Like", mlir::ValueRange({ctLeft, ctRight})).getRes();
            return comparisonExpr->type == ast::ExpressionType::COMPARE_NOT_LIKE ? builder.create<db::NotOp>(builder.getUnknownLoc(), like) : like;
         }
         db::DBCmpPredicate pred;
         switch (expression->type) {
            case ast::ExpressionType::COMPARE_EQUAL:
               pred = db::DBCmpPredicate::eq;
               break;
            case ast::ExpressionType::COMPARE_NOTEQUAL:
               pred = db::DBCmpPredicate::neq;
               break;
            case ast::ExpressionType::COMPARE_LESSTHAN:
               pred = db::DBCmpPredicate::lt;
               break;
            case ast::ExpressionType::COMPARE_GREATERTHAN:
               pred = db::DBCmpPredicate::gt;
               break;
            case ast::ExpressionType::COMPARE_LESSTHANOREQUALTO:
               pred = db::DBCmpPredicate::lte;
               break;
            case ast::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
               pred = db::DBCmpPredicate::gte;
               break;
            default: throw std::runtime_error("not implemented");
         }
         //TODO replace
         //TODO discuss: move cast into analyzer
         //std::vector ct = {comparisonExpr->comparisonType.castValueToThisType(builder,left, comparisonExpr->left->resultType->isNullable), comparisonExpr->comparisonType.castValueToThisType(builder,right, comparisonExpr->rightChildren[0]->resultType->isNullable)};

         return builder.create<db::CmpOp>(builder.getUnknownLoc(), pred, ctLeft, ctRight);
      }
      case ast::ExpressionClass::BOUND_CONJUNCTION: {
         auto conjunction = std::static_pointer_cast<ast::BoundConjunctionExpression>(expression);
         std::vector<mlir::Value> values;
         for (auto exp : conjunction->children) {
            values.push_back(translateExpression(builder, exp, context));
         }
         switch (conjunction->type) {
            case ast::ExpressionType::CONJUNCTION_AND: {
               return builder.create<db::AndOp>(builder.getUnknownLoc(), values);
            }
            case ast::ExpressionType::CONJUNCTION_OR: {
               return builder.create<db::OrOp>(builder.getUnknownLoc(), values);
            }
            default: throw std::runtime_error("not implemented");
         }
      }
      case ast::ExpressionClass::BOUND_OPERATOR: {
         auto operatorExpr = std::static_pointer_cast<ast::BoundOperatorExpression>(expression);

         mlir::Value left, right;

         switch (operatorExpr->type) {
            case ast::ExpressionType::OPERATOR_MINUS:
            case ast::ExpressionType::OPERATOR_DIVIDE:
            case ast::ExpressionType::OPERATOR_MOD:
            case ast::ExpressionType::OPERATOR_TIMES:
            case ast::ExpressionType::OPERATOR_PLUS: {
               assert(operatorExpr->children.size() == 2);
               left = translateExpression(builder, operatorExpr->children[0], context);
               right = translateExpression(builder, operatorExpr->children[1], context);
               return translateBinaryOperatorExpression(builder, operatorExpr, context, left, right);
            }
            case ast::ExpressionType::OPERATOR_IS_NOT_NULL:
            case ast::ExpressionType::OPERATOR_IS_NULL: {
               assert(operatorExpr->children.size() == 1);
               auto childTree = translateExpression(builder, operatorExpr->children[0], context);
               mlir::Value isNull = builder.create<db::IsNullOp>(builder.getUnknownLoc(), childTree);
               if (operatorExpr->type == ast::ExpressionType::OPERATOR_IS_NOT_NULL) {
                  return builder.create<db::NotOp>(builder.getUnknownLoc(), isNull);
               } else {
                  return isNull;
               }
            }
            default: error("Not implemented", expression->loc);
         }
      }
      case ast::ExpressionClass::BOUND_CAST: {
         auto castExpr = std::static_pointer_cast<ast::BoundCastExpression>(expression);
         auto toCast = translateExpression(builder, castExpr->child, context);
         //TODO support for typeModifiers/ move to analyzer
         // Inspire From Parse.cpp;
         //auto resType = createType(typeName, getTypeModList(castNode->type_name_->typmods_)).getMLIRTypeCreator()->createType(builder.getContext());

         assert(castExpr->resultType.has_value());
         auto resType = castExpr->resultType->toMlirType(builder.getContext());
         if (auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(toCast.getDefiningOp())) {
            if (auto intervalType = mlir::dyn_cast<db::IntervalType>(resType)) {
               //TODO maybe create stringRepresentation at analyzer Level?
               auto stringRepresentation = castExpr->stringRepr;
               //!Shortcutted here, implement different interval types later
               /*
                * TODO
               * auto typeModList = getTypeModList(castNode->type_name_->typmods_);
                     if (typeModList.size() > 0 && std::get<size_t>(typeModList[0]) & 4) {
                        //interval in years
                        stringRepresentation = std::to_string(std::stol(stringRepresentation) * 12);
                     }
                */
               //TODO maybe create stringRepresentation at analyzer Level?
               constOp->setAttr("value", builder.getStringAttr(stringRepresentation));
            }
            constOp.getResult().setType(resType);
            return constOp;
         } else {
            return castExpr->resultType->castValueToThisType(builder, toCast, castExpr->resultType->isNullable);
         }
      }
      case ast::ExpressionClass::BOUND_BETWEEN: {
         auto boundBetween = std::static_pointer_cast<ast::BoundBetweenExpression>(expression);
         mlir::Value input, lower, upper;
         input = translateExpression(builder, boundBetween->input, context);
         lower = translateExpression(builder, boundBetween->lower, context);
         upper = translateExpression(builder, boundBetween->upper, context);
         auto ctInput = boundBetween->input->resultType->castValue(builder, input);
         auto ctLower = boundBetween->lower->resultType->castValue(builder, lower);
         auto ctUpper = boundBetween->upper->resultType->castValue(builder, upper);

         mlir::Value between = builder.create<db::BetweenOp>(builder.getUnknownLoc(), ctInput, ctLower, ctUpper, true, true);
         if (boundBetween->type == ast::ExpressionType::COMPARE_NOT_BETWEEN) {
            between = builder.create<db::NotOp>(builder.getUnknownLoc(), between);
         }
         return between;
      }
      case ast::ExpressionClass::BOUND_FUNCTION: {
         auto function = std::static_pointer_cast<ast::BoundFunctionExpression>(expression);
         if (function->functionName == "EXTRACT") {
            assert(function->arguments.size() == 2);
            auto part = translateExpression(builder, function->arguments[0], context);
            auto arg2 = translateExpression(builder, function->arguments[1], context);
            return builder.create<db::RuntimeCall>(builder.getUnknownLoc(), wrapNullableType(builder.getContext(), builder.getI64Type(), {part, arg2}), "ExtractFromDate", mlir::ValueRange({part, arg2})).getRes();
         }
         if (function->functionName == "SUBSTRING" || function->functionName == "SUBSTR") {
            auto str = translateExpression(builder, function->arguments[0], context);
            auto from = function->arguments[1] ? translateExpression(builder, function->arguments[1], context) : nullptr;
            auto to = function->arguments[2] ? translateExpression(builder, function->arguments[2], context) : nullptr;
            return builder.create<db::RuntimeCall>(builder.getUnknownLoc(), str.getType(), "Substring", mlir::ValueRange({str, from, to})).getRes();
         }
         if (function->functionName == "ROUND") {
            auto val = translateExpression(builder, function->arguments[0], context);
            auto scale = translateExpression(builder, function->arguments[1], context);
            return builder.create<db::RuntimeCall>(builder.getUnknownLoc(), val.getType(), getBaseType(val.getType()).isIntOrIndex() ? "RoundInt" + std::to_string(getBaseType(val.getType()).getIntOrFloatBitWidth()) : "RoundDecimal", mlir::ValueRange{val, scale}).getRes();
         }
         if (function->functionName == "UPPER") {
            auto val = translateExpression(builder, function->arguments[0], context);
            return builder.create<db::RuntimeCall>(builder.getUnknownLoc(), val.getType(), "ToUpper", val).getRes();
         }
         if (function->functionName == "ABS") {
            auto val = translateExpression(builder, function->arguments[0], context);
            //TODO move type logic to analyzer
            return builder.create<db::RuntimeCall>(builder.getUnknownLoc(), val.getType(), mlir::isa<db::DecimalType>(getBaseType(val.getType())) ? "AbsDecimal" : "AbsInt", val).getRes();
         }
         error("Function '" << function->functionName << "' not implemented", expression->loc);
      }
      case ast::ExpressionClass::BOUND_SUBQUERY: {
         auto subquery = std::static_pointer_cast<ast::BoundSubqueryExpression>(expression);
         assert(subquery->sqlScope);
         context->pushNewScope(subquery->sqlScope);
         auto translatedSubquery = translateTableProducer(builder, subquery->subquery, context);
         context->popCurrentScope();
         switch (subquery->subqueryType) {
            case ast::SubqueryType::SCALAR: {
               assert(subquery->resultType.has_value());
               mlir::Type resType = subquery->resultType->toMlirType(builder.getContext());
               if (!subquery->resultType->isNullable) {
                  resType = db::NullableType::get(builder.getContext(), resType);
                  subquery->resultType->isNullable = true;
               }
               assert(subquery->namedResult.has_value());

               //TODO use zero instead of null
               mlir::Value scalarValue = builder.create<relalg::GetScalarOp>(builder.getUnknownLoc(), resType, subquery->namedResult.value()->createRef(builder, attrManager), translatedSubquery);
               return scalarValue;
            }
            case ast::SubqueryType::ANY:
            case ast::SubqueryType::NOT_ANY: {
               auto* block = new mlir::Block;
               mlir::OpBuilder predBuilder(builder.getContext());
               block->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
               auto tupleScope = translationContext->createTupleScope();
               translationContext->setCurrentTuple(block->getArgument(0));

               predBuilder.setInsertionPointToStart(block);
               mlir::Value expr = translateExpression(predBuilder, subquery->testExpr, context);

               auto mlirType = subquery->namedResult.value()->resultType.toMlirType(builder.getContext());

               mlir::Value colVal = predBuilder.create<tuples::GetColumnOp>(predBuilder.getUnknownLoc(), mlirType, subquery->namedResult.value()->createRef(builder, attrManager), block->getArgument(0));
               //TODO remove
               auto ctCol = subquery->namedResult.value()->resultType.castValue(builder, colVal);
               auto ctExpr = subquery->testExpr->resultType->castValue(builder, expr);
               //TODO extract and remove hardcoded
               db::DBCmpPredicate dbCmpPred = db::DBCmpPredicate::eq;
               mlir::Value pred = predBuilder.create<db::CmpOp>(predBuilder.getUnknownLoc(), dbCmpPred, ctCol, ctExpr);
               ;
               predBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), pred);

               auto sel = builder.create<relalg::SelectionOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), translatedSubquery);
               sel.getPredicate().push_back(block);
               translatedSubquery = sel.getResult();
               auto existsOp = builder.create<relalg::ExistsOp>(builder.getUnknownLoc(), builder.getI1Type(), translatedSubquery);
               if (subquery->subqueryType == ast::SubqueryType::NOT_ANY) {
                  return builder.create<db::NotOp>(builder.getUnknownLoc(), existsOp);
               }
               return existsOp;
            }
            case ast::SubqueryType::EXISTS: {
               return builder.create<relalg::ExistsOp>(builder.getUnknownLoc(), builder.getI1Type(), translatedSubquery);
            }
            case ast::SubqueryType::NOT_EXISTS: {
               auto existsOp = builder.create<relalg::ExistsOp>(builder.getUnknownLoc(), builder.getI1Type(), translatedSubquery);
               return builder.create<db::NotOp>(builder.getUnknownLoc(), existsOp);
            }
            default: error("Subquery type not implemented", expression->loc);
         }
      }
      case ast::ExpressionClass::BOUND_CASE: {
         auto boundCase = std::static_pointer_cast<ast::BoundCaseExpression>(expression);
         //TODO translate arg
         std::optional<mlir::Value> caseExprTranslated = std::nullopt;
         if (boundCase->caseExpr.has_value()) {
            caseExprTranslated = translateExpression(builder, boundCase->caseExpr.value(), context);
         }
         return translateWhenCheks(builder, boundCase,caseExprTranslated, boundCase->caseChecks, boundCase->elseExpr, context);
      }

      default: error("Not implemented", expression->loc);
   }
}

mlir::Value SQLMlirTranslator::translateBinaryOperatorExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundOperatorExpression> expression, std::shared_ptr<analyzer::SQLContext> context, mlir::Value left, mlir::Value right) {
   auto loc = builder.getUnknownLoc();
   assert(expression->resultType.has_value() && expression->children.size() == 2 && expression->children[0]->resultType.has_value() && expression->children[1]->resultType.has_value());

   switch (expression->type) {
      case ast::ExpressionType::OPERATOR_PLUS: {
         if (mlir::isa<db::DateType>(getBaseType(left.getType())) && mlir::isa<db::IntervalType>(getBaseType(right.getType()))) {
            return builder.create<db::RuntimeCall>(loc, left.getType(), "DateAdd", mlir::ValueRange({left, right})).getRes();
         }
         std::vector ct = {expression->resultType->castValueToThisType(builder, left, expression->children[0]->resultType->isNullable), expression->resultType->castValueToThisType(builder, right, expression->children[1]->resultType->isNullable)};
         return builder.create<db::AddOp>(builder.getUnknownLoc(), ct);
      }
      case ast::ExpressionType::OPERATOR_MINUS: {
         if (mlir::isa<db::DateType>(left.getType()) && mlir::isa<db::IntervalType>(right.getType())) {
            return builder.create<db::RuntimeCall>(loc, left.getType(), "DateSubtract", mlir::ValueRange({left, right})).getRes();
         }

         std::vector ct = {expression->resultType->castValueToThisType(builder, left, expression->children[0]->resultType->isNullable), expression->resultType->castValueToThisType(builder, right, expression->children[1]->resultType->isNullable)};
         return builder.create<db::SubOp>(builder.getUnknownLoc(), ct);
      }
      case ast::ExpressionType::OPERATOR_TIMES: {
         auto ct = {expression->children[0]->resultType->castValue(builder, left), expression->children[1]->resultType->castValue(builder, right)};
         return builder.create<db::MulOp>(builder.getUnknownLoc(), ct);
      }
      case ast::ExpressionType::OPERATOR_DIVIDE: {
         auto ct = {expression->children[0]->resultType->castValue(builder, left), expression->children[1]->resultType->castValue(builder, right)};
         return builder.create<db::DivOp>(builder.getUnknownLoc(), ct);
      }
      case ast::ExpressionType::OPERATOR_MOD: {
         auto ct = {expression->children[0]->resultType->castValue(builder, left), expression->children[1]->resultType->castValue(builder, right)};
         return builder.create<db::ModOp>(builder.getUnknownLoc(), ct);
      }
      default: error("Not implemented", expression->loc);
   }
}

mlir::Value SQLMlirTranslator::translateWhenCheks(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundCaseExpression> boundCase, std::optional<mlir::Value> caseExprTranslated, std::vector<ast::BoundCaseExpression::BoundCaseCheck> caseChecks, std::shared_ptr<ast::BoundExpression> elseExpr, std::shared_ptr<analyzer::SQLContext> context) {
   if (caseChecks.empty()) {
      if (!elseExpr) {
         return builder.create<db::NullOp>(builder.getUnknownLoc(), db::NullableType::get(builder.getContext(), builder.getNoneType()));
      }
      return translateExpression(builder, elseExpr, context);
   }
   auto commonType = boundCase->resultType->toMlirType(builder.getContext());

   auto check = caseChecks[0];
   auto hasNextCheck = caseChecks.size() > 1;

   //TOOD arg

   auto condTranslated = translateExpression(builder, check.whenExpr, context);
   //condTranslated = check.whenExpr->resultType->castValue(builder, condTranslated);
   if (caseExprTranslated.has_value()) {
      caseExprTranslated = boundCase->caseExpr.value()->resultType->castValue(builder, caseExprTranslated.value());
      condTranslated = builder.create<db::CmpOp>(builder.getUnknownLoc(), db::DBCmpPredicate::eq, condTranslated, caseExprTranslated.value());
   }

   auto* whenBlock = new mlir::Block;
   auto* elseBlock = new mlir::Block;
   mlir::OpBuilder whenBuilder(builder.getContext());
   whenBuilder.setInsertionPointToStart(whenBlock);
   auto thenTranslated = translateExpression(whenBuilder, check.thenExpr, context);
   mlir::OpBuilder elseBuilder(builder.getContext());
   elseBuilder.setInsertionPointToStart(elseBlock);
   mlir::Value elseTranslated;
   if (hasNextCheck) {
      std::vector<ast::BoundCaseExpression::BoundCaseCheck> nextChecks{caseChecks.begin() + 1, caseChecks.end()};
      elseTranslated = translateWhenCheks(elseBuilder, boundCase, caseExprTranslated, nextChecks, elseExpr, context);
   } else {
      elseTranslated = translateExpression(elseBuilder, elseExpr, context);
   }


   thenTranslated = boundCase->resultType->castValueToThisType(whenBuilder, thenTranslated, check.thenExpr->resultType->isNullable);
   elseTranslated = boundCase->resultType->castValueToThisType(elseBuilder, elseTranslated, elseExpr->resultType->isNullable);

   whenBuilder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), thenTranslated);
   elseBuilder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), elseTranslated);
   condTranslated = builder.create<db::DeriveTruth>(builder.getUnknownLoc(), condTranslated);
   auto ifOp = builder.create<mlir::scf::IfOp>(builder.getUnknownLoc(), commonType, condTranslated, true);
   ifOp.getThenRegion().getBlocks().clear();
   ifOp.getElseRegion().getBlocks().clear();
   ifOp.getThenRegion().push_back(whenBlock);
   ifOp.getElseRegion().push_back(elseBlock);

   return ifOp.getResult(0);
   throw std::runtime_error("Should never reach here");
}

mlir::Value SQLMlirTranslator::translateWhenCheck(mlir::OpBuilder& builder, ast::BoundCaseExpression::BoundCaseCheck whenCheck, std::shared_ptr<ast::BoundExpression> elseExpr, std::shared_ptr<analyzer::SQLContext> context) {
}

mlir::Value SQLMlirTranslator::translateTableRef(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundTableRef> tableRef, std::shared_ptr<analyzer::SQLContext> context) {
   mlir::Value last;
   switch (tableRef->type) {
      case ast::TableReferenceType::BASE_TABLE: {
         auto baseTableRef = std::static_pointer_cast<ast::BoundBaseTableRef>(tableRef);
         std::string relation = baseTableRef->relationName;
         //std::string alias = relation;
         if (context->ctes.contains(relation)) {
            auto [cteInfo, cteNode] = context->ctes.at(relation);
            if (cteNode->query) {
               mlir::Value _tree;
               if (context->translatedCtes.contains(relation)) {
                  //Already translated
                  _tree = context->translatedCtes.at(relation);
               } else {
                  context->pushNewScope(std::make_shared<analyzer::SQLScope>(cteNode->subQueryScope));
                  _tree = translateTableProducer(builder, cteNode->query, context);
                  context->popCurrentScope();
                  context->translatedCtes.insert({relation, _tree});
               }
               std::vector<mlir::Attribute> renamingDefsAsAttr;
               assert(context->ctes.at(cteNode->alias).second->renamedResults.size() == baseTableRef->namedResultsEntries.size());
               for (size_t i = 0; i < context->ctes.at(cteNode->alias).second->renamedResults.size(); i++) {
                  auto [from, to] = context->ctes.at(cteNode->alias).second->renamedResults[i];
                  renamingDefsAsAttr.emplace_back(baseTableRef->namedResultsEntries[i]->createDef(builder, attrManager, builder.getArrayAttr(from->createRef(builder, attrManager))));
               }

               return builder.create<relalg::RenamingOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), _tree, builder.getArrayAttr(renamingDefsAsAttr));
            }
         }
         if (baseTableRef->namedResultsEntries.empty()) {
            error("Table " << baseTableRef->relationName << " not found", baseTableRef->loc);
         }
         auto rel = baseTableRef->namedResultsEntries;
         std::string uniqueScope = baseTableRef->mlirScope;

         std::vector<mlir::NamedAttribute> columns{};
         for (auto& info : rel) {
            auto attrDef = attrManager.createDef(uniqueScope, info->name);
            attrDef.getColumn().type = info->resultType.toMlirType(builder.getContext());
            columns.push_back(builder.getNamedAttr(info->name, attrDef));
         }
         return builder.create<relalg::BaseTableOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), relation, builder.getDictionaryAttr(columns));
      }
      case ast::TableReferenceType::JOIN: {
         auto boundJoin = std::static_pointer_cast<ast::BoundJoinRef>(tableRef);
         mlir::Value left, right;
         assert(boundJoin->leftScope && boundJoin->rightScope);
         context->pushNewScope(boundJoin->leftScope);
         left = translateTableProducer(builder, boundJoin->left, context);
         context->popCurrentScope();
         //Here the evalbefore gets lost!
         context->pushNewScope(boundJoin->rightScope);
         right = translateTableProducer(builder, boundJoin->right, context);
         context->popCurrentScope();
         switch (boundJoin->type) {
            case ast::JoinType::INNER: {
               mlir::Block* pred;
               if (!std::holds_alternative<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition)) {
                  error("Not implemented", tableRef->loc);
               }

               pred = translatePredicate(builder, std::get<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition), context);

               //TODO translate predicate
               auto joinOp = builder.create<relalg::InnerJoinOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), left, right);
               joinOp.getPredicate().push_back(pred);

               return joinOp;
            }
            case ast::JoinType::CROSS: {
               auto joinOp = builder.create<relalg::CrossProductOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), left, right);
               return joinOp;
            }
            case ast::JoinType::LEFT: {
               mlir::Block* pred;
               if (!std::holds_alternative<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition)) {
                  error("Not implemented", tableRef->loc);
               }

               pred = translatePredicate(builder, std::get<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition), context);

               std::vector<mlir::Attribute> outerJoinMapping{};
               static size_t i = 0;
               std::ranges::transform(boundJoin->outerJoinMapping, std::back_inserter(outerJoinMapping), [&](std::pair<std::string, std::shared_ptr<ast::NamedResult>> scopeAndNamedResult) {
                  i++;
                  std::cout << "Add to mapping: " << scopeAndNamedResult.second->scope << "," << scopeAndNamedResult.second->name << "from existing: " << scopeAndNamedResult.first << "," << scopeAndNamedResult.second->name << std::endl;

                  auto attrDef = scopeAndNamedResult.second->createDef(builder, attrManager, builder.getArrayAttr({attrManager.createRef(scopeAndNamedResult.first, scopeAndNamedResult.second->name)}));

                  return attrDef;
               });

               mlir::ArrayAttr mapping = builder.getArrayAttr(outerJoinMapping);

               auto join = builder.create<relalg::OuterJoinOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), left, right, mapping);
               join.getPredicate().push_back(pred);
               return join;
            }
            case ast::JoinType::OUTER: {
            }
            default: error("Not implemented", tableRef->loc);
         }

         right = translateTableProducer(builder, boundJoin->right, context);
         error("TODO hier weiter machen", tableRef->loc);
      }
      case ast::TableReferenceType::SUBQUERY: {
         auto subquery = std::static_pointer_cast<ast::BoundSubqueryRef>(tableRef);
         context->pushNewScope(subquery->sqlScope);
         mlir::Value subQuery;
         auto translated = translateTableProducer(builder, subquery->subSelect, context);
         context->popCurrentScope();

         return translated;
      }
      case ast::TableReferenceType::EXPRESSION_LIST: {
         auto expressionList = std::static_pointer_cast<ast::BoundExpressionListRef>(tableRef);
         std::vector<mlir::Attribute> rows;
         for (auto row : expressionList->values) {
            std::vector<mlir::Attribute> values;
            std::vector<mlir::Type> types;
            for (auto constExpr : row) {
               mlir::Attribute value;
               switch (constExpr->value->type) {
                  case ast::ConstantType::INT: {
                     auto iValue = std::static_pointer_cast<ast::IntValue>(constExpr->value);
                     value = builder.getI32IntegerAttr(iValue->iVal);
                     break;
                  }
                  case ast::ConstantType::STRING: {
                     auto sValue = std::static_pointer_cast<ast::StringValue>(constExpr->value);
                     value = builder.getStringAttr(sValue->sVal);
                     break;
                  }
                  case ast::ConstantType::FLOAT: {
                     auto fValue = std::static_pointer_cast<ast::FloatValue>(constExpr->value);
                     //TODO support only decimal
                     value = builder.getStringAttr(fValue->fVal);
                     assert(constExpr->resultType.has_value());
                     break;
                  }

                  default: error("Not implemented", constExpr->loc);
               }
               values.emplace_back(value);
               types.emplace_back(constExpr->resultType.value().toMlirType(builder.getContext()));
            }
            rows.push_back(builder.getArrayAttr(values));
         }

         std::vector<mlir::Attribute> attributes;
         for (auto namedResult : expressionList->namedResultsEntries) {
            auto attrDef = namedResult->createDef(builder, attrManager);
            attributes.push_back(attrDef);
         }

         auto translated = builder.create<relalg::ConstRelationOp>(builder.getUnknownLoc(), builder.getArrayAttr(attributes), builder.getArrayAttr(rows));
         return translated;
      }
      default:
         error("Not implemented", tableRef->loc);
   }
}

mlir::Value SQLMlirTranslator::translateSetOperation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundSetOperationNode> boundSetOp, std::shared_ptr<analyzer::SQLContext> context) {
   auto setSemantic = boundSetOp->setOpAll ? relalg::SetSemantic::all : relalg::SetSemantic::distinct;
   context->pushNewScope(boundSetOp->leftScope);
   auto leftTree = translateTableProducer(builder, std::static_pointer_cast<ast::TableProducer>(boundSetOp->boundLeft), context);
   context->popCurrentScope();
   context->pushNewScope(boundSetOp->rightScope);
   auto rightTree = translateTableProducer(builder, std::static_pointer_cast<ast::TableProducer>(boundSetOp->boundRight), context);
   context->popCurrentScope();

   std::vector<mlir::Attribute> attributes;

   mlir::Block* leftMapBlock = new mlir::Block;
   mlir::Block* rightMapBlock = new mlir::Block;
   mlir::OpBuilder leftMapBuilder(builder.getContext());
   mlir::OpBuilder rightMapBuilder(builder.getContext());
   leftMapBuilder.setInsertionPointToStart(leftMapBlock);
   rightMapBuilder.setInsertionPointToStart(rightMapBlock);
   leftMapBlock->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   rightMapBlock->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto leftMapScope = attrManager.getUniqueScope("map");
   auto rightMapScope = attrManager.getUniqueScope("map");
   std::vector<mlir::Attribute> createdColsLeft;
   std::vector<mlir::Attribute> createdColsRight;
   mlir::Value leftTuple = leftMapBlock->getArgument(0);
   mlir::Value rightTuple = rightMapBlock->getArgument(0);
   std::vector<mlir::Value> leftMapResults;
   std::vector<mlir::Value> rightMapResults;
   //TODO Move most logic to analyzer
   for (size_t i = 0; i < boundSetOp->leftScope->targetInfo.targetColumns.size(); i++) {
      auto leftResult = boundSetOp->leftScope->targetInfo.targetColumns[i];
      auto rightResult = boundSetOp->rightScope->targetInfo.targetColumns[i];
      auto commonType = context->currentScope->targetInfo.targetColumns[i]->resultType;
      //TODO what todo if left.type != right.type != context.result.type
      if (rightResult->resultType != commonType) {
         auto attrDef = attrManager.createDef(rightMapScope, std::string("set_op") + std::to_string(i));
         auto attrRef = rightResult->createRef(builder, attrManager);

         createdColsRight.push_back(attrDef);
         mlir::Value expr = rightMapBuilder.create<tuples::GetColumnOp>(rightMapBuilder.getUnknownLoc(), attrRef.getColumn().type, attrRef, rightTuple);
         rightMapResults.push_back(rightResult->resultType.castValue(rightMapBuilder, expr));
         rightResult->resultType = commonType;
         rightResult->scope = rightMapScope;
         rightResult->name = "set_op" + std::to_string(i);
      }
      if (leftResult->resultType != commonType) {
         auto attrDef = attrManager.createDef(leftMapScope, std::string("set_op") + std::to_string(i));
         auto attrRef = leftResult->createRef(builder, attrManager);

         createdColsLeft.push_back(attrDef);
         mlir::Value expr = leftMapBuilder.create<tuples::GetColumnOp>(leftMapBuilder.getUnknownLoc(), attrRef.getColumn().type, attrRef, leftTuple);
         leftMapResults.push_back(leftResult->resultType.castValue(leftMapBuilder, expr));
         leftResult->resultType = commonType;
         leftResult->scope = leftMapScope;
         leftResult->name = "set_op" + std::to_string(i);
      }

      auto newType = commonType.toMlirType(builder.getContext());
      auto newColName = leftResult->name;
      auto newColDef = context->currentScope->targetInfo.targetColumns[i]->createDef(builder, attrManager, builder.getArrayAttr({leftResult->createRef(builder, attrManager), rightResult->createRef(builder, attrManager)}));
      auto* newCol = &newColDef.getColumn();
      newCol->type = newType;
      attributes.push_back(newColDef);
   }

   if (!leftMapResults.empty()) {
      auto mapOp = builder.create<relalg::MapOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), leftTree, builder.getArrayAttr(createdColsLeft));
      mapOp.getPredicate().push_back(leftMapBlock);
      leftMapBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), leftMapResults);
      leftTree = mapOp.getResult();
   } else {
      delete leftMapBlock;
   }
   if (!rightMapResults.empty()) {
      auto mapOp = builder.create<relalg::MapOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), rightTree, builder.getArrayAttr(createdColsRight));
      mapOp.getPredicate().push_back(rightMapBlock);
      rightMapBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), rightMapResults);
      rightTree = mapOp.getResult();

   } else {
      delete rightMapBlock;
   }

   mlir::Value tree;
   switch (boundSetOp->setType) {
      case ast::SetOperationType::UNION: {
         tree = builder.create<relalg::UnionOp>(builder.getUnknownLoc(), lingodb::compiler::dialect::relalg::SetSemanticAttr::get(builder.getContext(), setSemantic), leftTree, rightTree, builder.getArrayAttr(attributes));
         break;
      }
      case ast::SetOperationType::INTERSECT: {
         tree = builder.create<relalg::IntersectOp>(builder.getUnknownLoc(), lingodb::compiler::dialect::relalg::SetSemanticAttr::get(builder.getContext(), setSemantic), leftTree, rightTree, builder.getArrayAttr(attributes));
         break;
      }
      case ast::SetOperationType::EXCEPT: {
         tree = builder.create<relalg::ExceptOp>(builder.getUnknownLoc(), lingodb::compiler::dialect::relalg::SetSemanticAttr::get(builder.getContext(), setSemantic), leftTree, rightTree, builder.getArrayAttr(attributes));
         break;
      }
      default: error("Set operation type not implemented", boundSetOp->loc);
   }

   return tree;
}

mlir::Value SQLMlirTranslator::translateAggregation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundAggregationNode> aggregation, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree) {
   //Ignore empty aggregations
   if ((!aggregation->groupByNode || aggregation->groupByNode->groupNamedResults.empty()) && aggregation->aggregations.empty()) {
      return tree;
   }
   //create map
   tree = createMap(builder, aggregation->mapName, aggregation->toMapExpressions, context, tree);

   //Translate group by Attributes
   std::vector<mlir::Attribute> groupByAttrs;
   std::unordered_map<std::string, mlir::Attribute> groupedExpressions;
   std::unordered_map<std::string, size_t> groupByAttrToPos;
   for (auto& groupByNamedResult : aggregation->groupByNode->groupNamedResults) {
      auto attrDef = groupByNamedResult->createRef(builder, attrManager);
      //TODO
      /*auto attrName = fieldsToString(columnRef->fields_);
      groupByAttrToPos[attrName] = i;*/
      groupByAttrs.push_back(attrDef);
   }

   //TODO maby split logic into two different methods!

   /*
    *Perform aggregation
   */

   //TODO rollup logic
   bool rollup = false;
   if (rollup) {
      error("Not implemented", aggregation->loc);
   } else {
      static size_t groupById = 0;
      auto tupleStreamType = tuples::TupleStreamType::get(builder.getContext());
      auto tupleType = tuples::TupleType::get(builder.getContext());

      std::string groupByName = "aggr" + std::to_string(groupById++);
      auto tupleScope = translationContext->createTupleScope();
      auto* block = new mlir::Block;
      block->addArgument(tupleStreamType, builder.getUnknownLoc());
      block->addArgument(tupleType, builder.getUnknownLoc());
      mlir::Value relation = block->getArgument(0);
      mlir::OpBuilder aggrBuilder(builder.getContext());
      aggrBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      std::unordered_map<std::string, tuples::Column*> mapping;

      //AggrFunctions
      for (auto aggrFunction : aggregation->aggregations) {
         mlir::Value expr; //TODO??
         auto aggrFuncName = aggrFunction->functionName;

         auto attrDef = attrManager.createDef(aggrFunction->scope, aggrFunction->aliasOrUniqueIdentifier);
         if (aggrFuncName == "count*") {
            expr = aggrBuilder.create<relalg::CountRowsOp>(builder.getUnknownLoc(), builder.getI64Type(), relation);
            //TODO not star
            //TODO use zero instead of null
            /*if (groupByAttrs.empty()) {
               context.useZeroInsteadNull.insert(&attrDef.getColumn());
            }*/
         } else {
            //TODO move logic to analyzer
            auto relalgAggrFunc = llvm::StringSwitch<relalg::AggrFunc>(aggrFuncName)
                                     .Case("sum", relalg::AggrFunc::sum)
                                     .Case("avg", relalg::AggrFunc::avg)
                                     .Case("min", relalg::AggrFunc::min)
                                     .Case("max", relalg::AggrFunc::max)
                                     .Case("count", relalg::AggrFunc::count)
                                     .Case("stddev_samp", relalg::AggrFunc::stddev_samp)
                                     .Default(relalg::AggrFunc::count);
            //TODO use zero instead of null
            /*if (relalgAggrFunc == relalg::AggrFunc::count) {
               error("Use zero instead of null", aggrFunction->loc);
               /*if (groupByAttrs.empty()) {
                  context.useZeroInsteadNull.insert(&attrDef.getColumn());
               }*/
            //}
            assert(aggrFunction->arguments.size() == 1);
            tuples::ColumnRefAttr refAttr;
            switch (aggrFunction->arguments[0]->type) {
               case ast::ExpressionType::BOUND_COLUMN_REF: {
                  auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(aggrFunction->arguments[0]);
                  assert(columnRef->namedResult.has_value());
                  auto namedResult = columnRef->namedResult.value();
                  refAttr = namedResult->createRef(builder, attrManager);
                  break;
               }
               default: {
                  //Is in map
                  refAttr = attrManager.createRef(aggregation->mapName, aggrFunction->arguments[0]->alias);

                  break;
               };
            }

            mlir::Value currRel = relation;
            //TODO distinct
            if (aggrFunction->distinct) {
               currRel = aggrBuilder.create<relalg::ProjectionOp>(builder.getUnknownLoc(), relalg::SetSemantic::distinct, currRel, builder.getArrayAttr({refAttr}));
            }
            mlir::Type aggrResultType;
            assert(aggrFunction->resultType.has_value());
            aggrResultType = aggrFunction->resultType->toMlirType(builder.getContext());

            if (aggrFunction->arguments[0]->type != ast::ExpressionType::BOUND_COLUMN_REF) {
               //TODO better, over context!!
               assert(aggrFunction->arguments[0]->namedResult.has_value());

               //aggrResultType = aggrFunction->arguments[0]->namedResult.value()->resultType.toMlirType(builder.getContext());
            }
            //TODO define type
            if (relalgAggrFunc == relalg::AggrFunc::avg) {
               /*auto baseType = getBaseType(aggrResultType);
               if (baseType.isIntOrFloat() && !baseType.isIntOrIndex()) {
                  //keep aggrResultType
               } else if (mlir::isa<db::DecimalType>(baseType)) {
                  mlir::OpBuilder b(builder.getContext());
                  mlir::Value x = b.create<db::ConstantOp>(b.getUnknownLoc(), baseType, b.getUnitAttr());
                  mlir::Value x2 = b.create<db::ConstantOp>(b.getUnknownLoc(), db::DecimalType::get(b.getContext(), 19, 0), b.getUnitAttr());
                  mlir::Value div = b.create<db::DivOp>(b.getUnknownLoc(), x, x2);
                  aggrResultType = div.getType();
                  div.getDefiningOp()->erase();
                  x2.getDefiningOp()->erase();
                  x.getDefiningOp()->erase();
               } else {
                  mlir::OpBuilder b(builder.getContext());
                  mlir::Value x = b.create<db::ConstantOp>(b.getUnknownLoc(), db::DecimalType::get(b.getContext(), 19, 0), b.getUnitAttr());
                  mlir::Value div = b.create<db::DivOp>(b.getUnknownLoc(), x, x);
                  aggrResultType = div.getType();
                  div.getDefiningOp()->erase();
                  x.getDefiningOp()->erase();
               }
               if (mlir::isa<db::NullableType>(refAttr.getColumn().type)) {
                  aggrResultType = db::NullableType::get(builder.getContext(), aggrResultType);
               }*/
               assert(aggrFunction->namedResult.has_value());
               aggrFunction->namedResult.value()->resultType.isNullable = true;
            }
            //TODO move to analyzer
            if (!mlir::isa<db::NullableType>(aggrResultType) && (groupByAttrs.empty()) && aggrFunction->functionName != "count") {
               aggrResultType = db::NullableType::get(builder.getContext(), aggrResultType);
            }
            if (mlir::isa<db::NullableType>(aggrResultType)) {
               assert(aggrFunction->namedResult.has_value());
               aggrFunction->namedResult.value()->resultType.isNullable = true;
            }

            expr = aggrBuilder.create<relalg::AggrFuncOp>(builder.getUnknownLoc(), aggrResultType, relalgAggrFunc, currRel, refAttr);
         }
         attrDef.getColumn().type = expr.getType();

         //TODO mapping.insert({12, "&attrDef.getColumn()"});
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);
      }

      aggrBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
      auto groupByOp = builder.create<relalg::AggregationOp>(builder.getUnknownLoc(), tupleStreamType, tree, builder.getArrayAttr(groupByAttrs), builder.getArrayAttr(createdCols));
      groupByOp.getAggrFunc().push_back(block);
      return groupByOp.getResult();
   }
   return tree;
}

mlir::Value SQLMlirTranslator::createMap(mlir::OpBuilder& builder, std::string mapName, std::vector<std::shared_ptr<ast::BoundExpression>> toMap, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree) {
   if (toMap.empty()) {
      return tree;
   }
   auto* block = new mlir::Block;
   static size_t mapId = 0;

   mlir::OpBuilder mapBuilder(builder.getContext());
   block->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto tupleScope = translationContext->createTupleScope();
   mlir::Value tuple = block->getArgument(0);
   translationContext->setCurrentTuple(tuple);

   mapBuilder.setInsertionPointToStart(block);
   std::vector<mlir::Value> createdValues;
   std::vector<mlir::Attribute> createdCols;

   for (auto p : toMap) {
      mlir::Value expr = translateExpression(mapBuilder, p, context);
      //TODO does the use of alias make sense here?
      auto attrDef = attrManager.createDef(mapName, p->alias);
      if (p->namedResult.has_value()) {
         p->namedResult.value()->scope = mapName;
         attrDef = p->namedResult.value()->createDef(builder, attrManager);
      }

      attrDef.getColumn().type = expr.getType();
      //TODO MAP context.mapAttribute(scope, p.first->colId, &attrDef.getColumn());
      createdCols.push_back(attrDef);
      createdValues.push_back(expr);
   }
   auto mapOp = builder.create<relalg::MapOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
   mapOp.getPredicate().push_back(block);
   mapBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
   return mapOp.getResult();
}

/*
 * Utility
 */

mlir::Block* SQLMlirTranslator::translatePredicate(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context) {
   auto* block = new mlir::Block;
   mlir::OpBuilder predBuilder(builder.getContext());
   block->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto tupleScope = translationContext->createTupleScope();
   translationContext->setCurrentTuple(block->getArgument(0));

   predBuilder.setInsertionPointToStart(block);
   mlir::Value expr = translateExpression(predBuilder, expression, context);
   predBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), expr);
   return block;
}

/*
 * Helper functions
 */

mlir::Type SQLMlirTranslator::createBaseTypeFromColumnType(mlir::MLIRContext* context, const catalog::Type& t) {
   return t.getMLIRTypeCreator()->createType(context);
}

} // namespace lingodb::translator