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
#include "lingodb/compiler/frontend/ast/copy_node.h"
#include "lingodb/compiler/frontend/ast/create_node.h"
#include "lingodb/compiler/frontend/ast/insert_node.h"
#include "lingodb/compiler/frontend/ast/set_node.h"

#include "lingodb/utility/Serialization.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <lingodb/compiler/runtime/ExecutionContext.h>
#include <lingodb/compiler/runtime/RelationHelper.h>
#include <mlir-c/IR.h>
namespace lingodb::translator {
using namespace lingodb::compiler::dialect;
SQLMlirTranslator::SQLMlirTranslator(mlir::ModuleOp moduleOp, catalog::Catalog* catalog) : moduleOp(moduleOp),
                                                                                                           attrManager(moduleOp->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager()), catalog(catalog), translationContext(std::make_shared<TranslationContext>())
{
   moduleOp.getContext()->getLoadedDialect<util::UtilDialect>()->getFunctionHelper().setParentModule(moduleOp);
}
std::optional<mlir::Value> SQLMlirTranslator::translateStart(mlir::OpBuilder& builder, std::shared_ptr<ast::AstNode> astNode, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(astNode->loc, mlirContext);

   auto startTranslate = std::chrono::high_resolution_clock::now();
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
         case ast::NodeType::SET_NODE: {
            auto setNode = std::static_pointer_cast<ast::SetNode>(astNode);
            translateSetNode(builder, setNode, context);
            return std::nullopt;
         }
         case ast::NodeType::COPY_NODE: {
            auto copyNode = std::static_pointer_cast<ast::CopyNode>(astNode);
            translateCopyNode(builder, copyNode, context);
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

         std::vector<mlir::Attribute> attrs;
         std::vector<mlir::Attribute> names;
         llvm::SmallVector<subop::Member> members;
         std::vector<mlir::Attribute> colTypes;
         auto& memberManager = mlirContext->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

         for (auto& named : context->currentScope->targetInfo.targetColumns) {
            names.push_back(builder.getStringAttr(named->displayName));

            members.push_back(memberManager.createMember(named->name.empty() ? "unnamed" : named->name, named->resultType.toMlirType(mlirContext)));
            colTypes.push_back(mlir::TypeAttr::get(named->resultType.toMlirType(mlirContext)));
            auto attrDef = named->createRef(builder, attrManager);
            attrs.push_back(attrDef);
         }

         localTableType = subop::LocalTableType::get(mlirContext, subop::StateMembersAttr::get(mlirContext, members), builder.getArrayAttr(names));
         mlir::Value result = builder.create<relalg::MaterializeOp>(
          location,
            localTableType,
            tree,
            builder.getArrayAttr(attrs),
            builder.getArrayAttr(names));

         // Use the materialized result in the QueryReturnOp instead of the input tree
         builder.create<relalg::QueryReturnOp>(location, result);
      }
      relalg::QueryOp queryOp = builder.create<relalg::QueryOp>(location, mlir::TypeRange{localTableType}, mlir::ValueRange{});
      queryOp.getQueryOps().getBlocks().clear();
      queryOp.getQueryOps().push_back(block);
      auto endTranslate = std::chrono::high_resolution_clock::now();
      this->timing = std::chrono::duration_cast<std::chrono::microseconds>(endTranslate - startTranslate).count() / 1000.0;
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
            case ast::QueryNodeType::BOUND_CTE_NODE: {
               auto cteNode = std::static_pointer_cast<ast::BoundCTENode>(queryNode);
               if (cteNode->child) {
                  tree = translateTableProducer(builder, cteNode->child, context);
               }

               return tree;
            }
            case ast::QueryNodeType::BOUND_SET_OPERATION_NODE: {
               return translateSetOperation(builder, std::static_pointer_cast<ast::BoundSetOperationNode>(tableProducer), context);
            }
            case ast::QueryNodeType::BOUND_VALUES: {
               auto boundValuesNode = std::static_pointer_cast<ast::BoundValuesQueryNode>(tableProducer);
               return translateTableRef(builder, boundValuesNode->expressionListRef, context);
            }
            default: error("QueryNode type not implemented", tableProducer->loc);
         }
      }
      default: error("Node type not implemented", tableProducer->loc);
   }

   return tree;
}

void SQLMlirTranslator::translateCreateNode(mlir::OpBuilder& builder, std::shared_ptr<ast::CreateNode> createNode, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(createNode->loc, mlirContext);
   switch (createNode->createInfo->type) {
      case ast::CatalogType::TABLE_ENTRY: {
         auto createTableInfo = std::static_pointer_cast<ast::CreateTableInfo>(createNode->createInfo);
         auto tableName = createTableInfo->tableName;
         auto tableDef = translateTableElements(builder, createTableInfo->tableElements, context);
         tableDef.name = tableName;
         auto descriptionValue = createStringValue(builder, utility::serializeToHexString(tableDef));
         compiler::runtime::RelationHelper::createTable(builder, location)(mlir::ValueRange({descriptionValue}));
         break;
      }
      default: error("CreateInfo type not implemented", createNode->loc);
   }
}

void SQLMlirTranslator::translateInsertNode(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundInsertNode> insertNode, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(insertNode->loc, mlirContext);
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
      mlir::OpBuilder mapBuilder(mlirContext);
      block->addArgument(tuples::TupleType::get(mlirContext), location);
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
         mlir::Value expr = mapBuilder.create<tuples::GetColumnOp>(location, attrRef.getColumn().type, attrRef, tuple);
         if (currentType != tableType) {
            auto attrDef = attrManager.createDef(mapName, std::string("inserted") + std::to_string(i));
            attrDef.getColumn().type = tableType.toMlirType(mlirContext);

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

      auto mapOp = builder.create<relalg::MapOp>(getLocationFromBison(insertNode->producer->loc, mlirContext), tuples::TupleStreamType::get(mlirContext), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<tuples::ReturnOp>(getLocationFromBison(insertNode->producer->loc, mlirContext), createdValues);

      llvm::SmallVector<subop::Member> members;
      std::vector<mlir::Attribute> orderedColNamesAttrs;
      std::vector<mlir::Attribute> orderedColAttrs;
      std::vector<mlir::Attribute> colTypes;
      auto& memberManager = mlirContext->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

      for (auto x : rel->getColumnNames()) {
         mlir::Type type = mlir::cast<tuples::ColumnRefAttr>(insertedCols.at(x)).getColumn().type;
         members.push_back(memberManager.createMember(x, type));
         orderedColNamesAttrs.push_back(builder.getStringAttr(x));
         orderedColAttrs.push_back(insertedCols.at(x));
         colTypes.push_back(mlir::TypeAttr::get(type));
      }

      localTableType = subop::LocalTableType::get(mlirContext, subop::StateMembersAttr::get(mlirContext, members), builder.getArrayAttr(orderedColNamesAttrs));
      mlir::Value newRows = builder.create<relalg::MaterializeOp>(location, localTableType, mapOp.getResult(), builder.getArrayAttr(orderedColAttrs), builder.getArrayAttr(orderedColNamesAttrs));
      builder.create<relalg::QueryReturnOp>(location, newRows);
   }

   relalg::QueryOp queryOp = builder.create<relalg::QueryOp>(location, mlir::TypeRange{localTableType}, mlir::ValueRange{});
   queryOp.getQueryOps().getBlocks().clear();
   queryOp.getQueryOps().push_back(block);
   auto tableNameValue = createStringValue(builder, insertNode->tableName);
   auto resultIdValue = builder.create<mlir::arith::ConstantIntOp>(location, 0, builder.getI32Type());
   builder.create<subop::SetResultOp>(location, 0, queryOp.getResults()[0]);

   compiler::runtime::RelationHelper::appendTableFromResult(builder, location)(mlir::ValueRange{tableNameValue, resultIdValue});
   compiler::runtime::ExecutionContext::clearResult(builder, location)({resultIdValue});
}

void SQLMlirTranslator::translateSetNode(mlir::OpBuilder& builder, std::shared_ptr<ast::SetNode> insertNode, std::shared_ptr<analyzer::SQLContext> context) {
   auto location = getLocationFromBison(insertNode->loc, builder.getContext());
   switch (insertNode->setType) {
      case ast::SetType::SET: {
         auto setVariableOperation = std::static_pointer_cast<ast::SetVariableStatement>(insertNode);
         if (setVariableOperation->name == "persist") {
            assert(setVariableOperation->values.size() == 1 && setVariableOperation->values[0]->nodeType == ast::NodeType::EXPRESSION && std::static_pointer_cast<ast::ParsedExpression>(setVariableOperation->values[0])->exprClass == ast::ExpressionClass::CONSTANT);
            auto constant = std::static_pointer_cast<ast::ConstantExpression>(setVariableOperation->values[0]);
            if (constant->value->type != ast::ConstantType::INT) {
               error("Persist value must be an integer", setVariableOperation->loc);
            }
            auto iValue= std::static_pointer_cast<ast::IntValue>(constant->value);
            auto persistValue = builder.create<mlir::arith::ConstantIntOp>(location, iValue->iVal, 1);
            compiler::runtime::RelationHelper::setPersist(builder, location)({persistValue});

         }
         break;
      }
      default: error("Could not set variable", insertNode->loc);
   }
}

void SQLMlirTranslator::translateCopyNode(mlir::OpBuilder& builder, std::shared_ptr<ast::CopyNode> copyStmt, std::shared_ptr<analyzer::SQLContext> context) {
   std::string fileName = copyStmt->copyInfo->fromFileName;
   std::string tableName = copyStmt->copyInfo->table;
   std::string delimiter = ",";
   std::string escape = "";
   for (auto [optionName, optionValue] : copyStmt->copyInfo->options) {
      if (optionName == "DELIMITER") {
         delimiter = optionValue;
      } else if (optionName == "ESCAPE") {
         escape = optionValue;
      } else if (optionName == "FORMAT") {
         std::string format = optionValue;
         if (format != "csv") {
            throw std::runtime_error("copy only supports csv");
         }

      } else if (optionName == "NULL") {
      } else {
         error(optionName <<  "option not implemented", copyStmt->loc);
      }
   }
   auto tableNameValue = createStringValue(builder, tableName);
   auto fileNameValue = createStringValue(builder, fileName);
   auto delimiterValue = createStringValue(builder, delimiter);
   auto escapeValue = createStringValue(builder, escape);
   compiler::runtime::RelationHelper::copyFromIntoTable(builder, builder.getUnknownLoc())(mlir::ValueRange{tableNameValue, fileNameValue, delimiterValue, escapeValue});
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
   assert(pipeOperator->node);
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(pipeOperator->node->loc, mlirContext);
   /*
    * handle the case where input is null, e.g., in the case of select without from
    */
   if (!tree) {
      auto dummyAttr = attrManager.createDef(attrManager.getUniqueScope("dummyScope"), "dummyName");
      dummyAttr.getColumn().type = builder.getI32Type();
      std::vector<mlir::Attribute> columns{dummyAttr};
      std::vector<mlir::Attribute> rows{builder.getArrayAttr({builder.getI64IntegerAttr(0)})};
      tree = builder.create<relalg::ConstRelationOp>(location, builder.getArrayAttr(columns), builder.getArrayAttr(rows));
   }
   switch (pipeOperator->pipeOpType) {
      case ast::PipeOperatorType::SELECT: {
         auto selectList = std::static_pointer_cast<ast::BoundTargetsExpression>(pipeOperator->node);
         if (selectList->distinct) {
            std::vector<mlir::Attribute> columns;
            for (auto x : context->currentScope->targetInfo.targetColumns) {
               columns.push_back(x->createRef(builder, attrManager));
            }
            tree = builder.create<relalg::ProjectionOp>(location, relalg::SetSemantic::distinct, tree, builder.getArrayAttr(columns));
         }
         return tree;
      }
      case ast::PipeOperatorType::WHERE: {
         assert(pipeOperator->node->nodeType == ast::NodeType::BOUND_EXPRESSION);
         auto whereClause = std::static_pointer_cast<ast::BoundExpression>(pipeOperator->node);
         //Make predicate
         auto* pred = new mlir::Block;
         mlir::OpBuilder predBuilder(mlirContext);
         pred->addArgument(tuples::TupleType::get(mlirContext), location);
         auto tupleScope = translationContext->createTupleScope();
         translationContext->setCurrentTuple(pred->getArgument(0));

         predBuilder.setInsertionPointToStart(pred);
         auto expr = translateExpression(predBuilder, whereClause, context);
         predBuilder.create<tuples::ReturnOp>(location, expr);

         auto sel = builder.create<relalg::SelectionOp>(location, tuples::TupleStreamType::get(mlirContext), tree);
         sel.getPredicate().push_back(pred);
         return sel.getResult();
      }
      case ast::PipeOperatorType::AGGREGATE: {
         assert(pipeOperator->node->nodeType == ast::NodeType::BOUND_AGGREGATION);
         auto aggregationNode = std::static_pointer_cast<ast::BoundAggregationNode>(pipeOperator->node);
         tree = translateAggregation(builder, aggregationNode, context, tree);
         return tree;
      }
      case ast::PipeOperatorType::EXTEND: {
         auto extendNode = std::static_pointer_cast<ast::BoundExtendNode>(pipeOperator->node);
         for (auto window: extendNode->windowExpressions) {
            tree = translateWindowExpression(builder, tree, window, context);
         }
         tree = createMap(builder, location, extendNode->mapName, extendNode->extensions, context, tree);

         return tree;
      }
      case ast::PipeOperatorType::DROP:
      case ast::PipeOperatorType::SET_OPERATION: {
         error("Should not happen", pipeOperator->loc);
      }
      case ast::PipeOperatorType::SET: {
         auto setNode = std::static_pointer_cast<ast::BoundSetColumnExpression>(pipeOperator->node);

         tree = createMap(builder, location, attrManager.getUniqueScope("set"), setNode->sets, context, tree);

         return tree;
      }


      default: error("Pipe operator not implemented", pipeOperator->loc);
   }
}

mlir::Value SQLMlirTranslator::translateResultModifier(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundResultModifier> resultModifier, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(resultModifier->loc, mlirContext);
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
            mapping.push_back(relalg::SortSpecificationAttr::get(mlirContext, attrDef, spec));
         }
         return builder.create<relalg::SortOp>(location, tuples::TupleStreamType::get(mlirContext), tree, builder.getArrayAttr(mapping));
      }
      case ast::ResultModifierType::BOUND_LIMIT: {
         auto limitModifier = std::static_pointer_cast<ast::BoundLimitModifier>(resultModifier);
         auto value = std::static_pointer_cast<ast::IntValue>(std::static_pointer_cast<ast::BoundConstantExpression>(limitModifier->limitExpression)->value);
         return builder.create<relalg::LimitOp>(location, tuples::TupleStreamType::get(mlirContext), value->iVal, tree);
      }
      default: error("ResultModifier Not implemented", resultModifier->loc);
   }
}

mlir::Value SQLMlirTranslator::translateExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto exprLocation = getLocationFromBison(expression->loc, mlirContext);
   assert(expression->resultType.has_value());
   switch (expression->exprClass) {
      case ast::ExpressionClass::BOUND_COLUMN_REF: {
         auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(expression);
         assert(columnRef->namedResult.has_value());
         auto nameResult = columnRef->namedResult.value();

         mlir::Type type = nameResult->resultType.toMlirType(mlirContext);

         auto attrDef = nameResult->createRef(builder, attrManager);
         return builder.create<tuples::GetColumnOp>(
            exprLocation,
            type, attrDef, translationContext->getCurrentTuple());
      }
      case ast::ExpressionClass::BOUND_CONSTANT: {
         auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(expression);
         switch (constExpr->value->type) {
            case ast::ConstantType::INT: {
               auto value = std::static_pointer_cast<ast::IntValue>(constExpr->value);
               return builder.create<db::ConstantOp>(exprLocation, builder.getI32Type(), builder.getI32IntegerAttr(value->iVal));
            }
            case ast::ConstantType::STRING: {
               auto value = std::static_pointer_cast<ast::StringValue>(constExpr->value);
               mlir::Type stringType = db::StringType::get(mlirContext);
               if (value->sVal.size() <= 8 && value->sVal.size() > 0) {
                  stringType = db::CharType::get(mlirContext, value->sVal.size());
               };
               return builder.create<db::ConstantOp>(exprLocation, stringType, builder.getStringAttr(value->sVal));
            }
            case ast::ConstantType::FLOAT: {
               auto value = std::static_pointer_cast<ast::FloatValue>(constExpr->value);
               assert(constExpr->resultType.has_value());
               return builder.create<db::ConstantOp>(exprLocation, constExpr->resultType.value().type.getMLIRTypeCreator()->createType(mlirContext), builder.getStringAttr(value->fVal));
            }
            case ast::ConstantType::NULL_P: {
               assert(constExpr->resultType.has_value());
               return builder.create<db::NullOp>(exprLocation, db::NullableType::get(mlirContext, builder.getNoneType()));
            }
            case ast::ConstantType::BOOLEAN: {
               auto value = std::static_pointer_cast<ast::BoolValue>(constExpr->value);
               return builder.create<db::ConstantOp>(exprLocation, builder.getI1Type(), builder.getIntegerAttr(builder.getI1Type(), value->bVal));
            }

            default: error("Constant type not implemented", expression->loc);
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

            auto oneOf = builder.create<db::OneOfOp>(exprLocation, values);
            if (comparisonExpr->type == ast::ExpressionType::COMPARE_NOT_IN) {
               return builder.create<db::NotOp>(exprLocation, oneOf);
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
            mlir::Type resType = isNullable ? (mlir::Type) db::NullableType::get(mlirContext, builder.getI1Type()) : (mlir::Type) builder.getI1Type();
            auto like = builder.create<db::RuntimeCall>(exprLocation, resType, "Like", mlir::ValueRange({ctLeft, ctRight})).getRes();
            return comparisonExpr->type == ast::ExpressionType::COMPARE_NOT_LIKE ? builder.create<db::NotOp>(exprLocation, like) : like;
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
            default: throw std::runtime_error("Compare not implemented");
         }

         return builder.create<db::CmpOp>(exprLocation, pred, ctLeft, ctRight);
      }
      case ast::ExpressionClass::BOUND_CONJUNCTION: {
         auto conjunction = std::static_pointer_cast<ast::BoundConjunctionExpression>(expression);
         std::vector<mlir::Value> values;
         for (auto exp : conjunction->children) {
            values.push_back(translateExpression(builder, exp, context));
         }
         switch (conjunction->type) {
            case ast::ExpressionType::CONJUNCTION_AND: {
               return builder.create<db::AndOp>(exprLocation, values);
            }
            case ast::ExpressionType::CONJUNCTION_OR: {
               return builder.create<db::OrOp>(exprLocation, values);
            }
            default: throw std::runtime_error("CONJUNCTION not implemented");
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
               mlir::Value isNull = builder.create<db::IsNullOp>(exprLocation, childTree);
               if (operatorExpr->type == ast::ExpressionType::OPERATOR_IS_NOT_NULL) {
                  return builder.create<db::NotOp>(exprLocation, isNull);
               } else {
                  return isNull;
               }
            }
            case ast::ExpressionType::OPERATOR_CONCAT: {
               left = translateExpression(builder, operatorExpr->children[0], context);
               right = translateExpression(builder, operatorExpr->children[1], context);
               //Cast both to string
               left = operatorExpr->resultType->castValueToThisType(builder, left, operatorExpr->children[0]->resultType->isNullable);
               right = operatorExpr->resultType->castValueToThisType(builder, right, operatorExpr->children[0]->resultType->isNullable);

               mlir::Type resType = operatorExpr->resultType->toMlirType(mlirContext);
               return builder.create<db::RuntimeCall>(exprLocation, resType, "Concatenate", mlir::ValueRange({left, right})).getRes();
            }
            case ast::ExpressionType::OPERATOR_NOT: {
               return builder.create<db::NotOp>(exprLocation, translateExpression(builder, operatorExpr->children[0], context) );
            }
            default: error("Operator not implemented", expression->loc);
         }
      }
      case ast::ExpressionClass::BOUND_CAST: {
         auto castExpr = std::static_pointer_cast<ast::BoundCastExpression>(expression);
         auto toCast = translateExpression(builder, castExpr->child, context);
         assert(castExpr->resultType.has_value());
         auto resType = castExpr->resultType->toMlirType(mlirContext);
         if (auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(toCast.getDefiningOp())) {
            if (auto intervalType = mlir::dyn_cast<db::IntervalType>(resType)) {
               auto stringRepresentation = castExpr->stringRepr;
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

         mlir::Value between = builder.create<db::BetweenOp>(exprLocation, ctInput, ctLower, ctUpper, true, true);
         if (boundBetween->type == ast::ExpressionType::COMPARE_NOT_BETWEEN) {
            between = builder.create<db::NotOp>(exprLocation, between);
         }
         return between;
      }
      case ast::ExpressionClass::BOUND_FUNCTION: {
         auto function = std::static_pointer_cast<ast::BoundFunctionExpression>(expression);
         if (function->functionName == "EXTRACT") {
            assert(function->arguments.size() == 2);
            auto part = translateExpression(builder, function->arguments[0], context);
            auto arg2 = translateExpression(builder, function->arguments[1], context);
            return builder.create<db::RuntimeCall>(exprLocation, wrapNullableType(mlirContext, builder.getI64Type(), {part, arg2}), "ExtractFromDate", mlir::ValueRange({part, arg2})).getRes();
         }
         if (function->functionName == "SUBSTRING" || function->functionName == "SUBSTR") {
            auto str = translateExpression(builder, function->arguments[0], context);
            auto from = function->arguments[1] ? translateExpression(builder, function->arguments[1], context) : nullptr;
            auto to = function->arguments[2] ? translateExpression(builder, function->arguments[2], context) : nullptr;
            return builder.create<db::RuntimeCall>(exprLocation, str.getType(), "Substring", mlir::ValueRange({str, from, to})).getRes();
         }
         if (function->functionName == "ROUND") {
            auto val = translateExpression(builder, function->arguments[0], context);
            auto scale = translateExpression(builder, function->arguments[1], context);
            return builder.create<db::RuntimeCall>(exprLocation, val.getType(), getBaseType(val.getType()).isIntOrIndex() ? "RoundInt" + std::to_string(getBaseType(val.getType()).getIntOrFloatBitWidth()) : "RoundDecimal", mlir::ValueRange{val, scale}).getRes();
         }
         if (function->functionName == "UPPER") {
            auto val = translateExpression(builder, function->arguments[0], context);
            return builder.create<db::RuntimeCall>(exprLocation, val.getType(), "ToUpper", val).getRes();
         }
         if (function->functionName == "ABS") {
            auto val = translateExpression(builder, function->arguments[0], context);
            //TODO move type logic to analyzer
            std::string typeString = function->arguments[0]->resultType.value().type.getTypeId() == catalog::LogicalTypeId::DECIMAL ? "AbsDecimal" : "AbsInt";
            return builder.create<db::RuntimeCall>(exprLocation, val.getType(), typeString, val).getRes();
         }
         if (function->functionName == "COALESCE") {
            return translateCoalesceExpression(builder,function->resultType.value(), function->arguments, context);

         }
         if (function->functionName == "LENGTH") {
            auto str = translateExpression(builder, function->arguments[0], context);
            return builder.create<db::RuntimeCall>(exprLocation, builder.getI64Type(), "StringLength", str).getRes();
         }
         if (function->functionName == "REGEXP_REPLACE") {
            auto text = translateExpression(builder, function->arguments[0], context);
            auto pattern = translateExpression(builder, function->arguments[1], context);
            auto replace = translateExpression(builder, function->arguments[2], context);
            return builder.create<db::RuntimeCall>(exprLocation, text.getType(), "RegexpReplace", mlir::ValueRange({text, pattern, replace})).getRes();
         }
         if (function->functionName  == "DATE_TRUNC") {
            auto part = translateExpression(builder, function->arguments[0], context);
            auto arg2 = translateExpression(builder, function->arguments[1], context);
            return  builder.create<db::RuntimeCall>(exprLocation, wrapNullableType(mlirContext, builder.getI64Type(), {part, arg2}), "DateTrunc", mlir::ValueRange({part, arg2})).getRes();
         }
         if (function->functionName == "HASH") {
            std::vector<mlir::Value> values;
            for (auto arg: function->arguments) {
               values.push_back(translateExpression(builder, arg, context));
            }

            auto packed = builder.create<util::PackOp>(exprLocation, values);
            return builder.create<db::Hash>(exprLocation, builder.getIndexType(), packed);
         }
         error("Fdunction '" << function->functionName << "' not implemented", expression->loc);
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
               mlir::Type resType = subquery->resultType->toMlirType(mlirContext);
               if (!subquery->resultType->isNullable) {
                  resType = db::NullableType::get(mlirContext, resType);
                  subquery->resultType->isNullable = true;
               }
               assert(subquery->namedResult.has_value());

               mlir::Value scalarValue = builder.create<relalg::GetScalarOp>(exprLocation, resType, subquery->namedResultForSubquery->createRef(builder, attrManager), translatedSubquery);
               if (subquery->namedResult.value()->resultType.useZeroInsteadOfNull) {
                  mlir::Value isNull = builder.create<db::IsNullOp>(exprLocation, scalarValue);
                  mlir::Value nonNullValue = builder.create<db::NullableGetVal>(exprLocation, scalarValue);
                  mlir::Value defaultValue = builder.create<db::ConstantOp>(exprLocation, getBaseType(scalarValue.getType()), builder.getIntegerAttr(getBaseType(scalarValue.getType()), 0));
                  return builder.create<mlir::arith::SelectOp>(exprLocation, isNull, defaultValue, nonNullValue);
               } else {
                  return scalarValue;
               }
               return scalarValue;
            }
            case ast::SubqueryType::ANY:
            case ast::SubqueryType::NOT_ANY: {
               auto* block = new mlir::Block;
               mlir::OpBuilder predBuilder(mlirContext);
               block->addArgument(tuples::TupleType::get(mlirContext), exprLocation);
               auto tupleScope = translationContext->createTupleScope();
               translationContext->setCurrentTuple(block->getArgument(0));

               predBuilder.setInsertionPointToStart(block);
               mlir::Value expr = translateExpression(predBuilder, subquery->testExpr, context);

               assert(subquery->namedResultForSubquery);
               auto mlirType = subquery->namedResultForSubquery->resultType.toMlirType(mlirContext);
               mlir::Value colVal = predBuilder.create<tuples::GetColumnOp>(exprLocation, mlirType, subquery->namedResultForSubquery->createRef(builder, attrManager), block->getArgument(0));

               auto ctCol = subquery->namedResult.value()->resultType.castValue(builder, colVal);
               auto ctExpr = subquery->testExpr->resultType->castValue(builder, expr);


               db::DBCmpPredicate dbCmpPred = db::DBCmpPredicate::eq;
               mlir::Value pred;
               if (subquery->comparisonType.has_value()) {
                  if (subquery->comparisonType.value() == ast::ExpressionType::COMPARE_LIKE || subquery->comparisonType.value() == ast::ExpressionType::COMPARE_NOT_LIKE) {
                     auto isNullable = mlir::isa<db::NullableType>(ctExpr.getType()) || mlir::isa<db::NullableType>(ctCol.getType());
                     mlir::Type resType = isNullable ? (mlir::Type) db::NullableType::get(predBuilder.getContext(), predBuilder.getI1Type()) : (mlir::Type) predBuilder.getI1Type();
                     auto like = predBuilder.create<db::RuntimeCall>(exprLocation, resType, "Like", mlir::ValueRange({ctExpr, ctCol})).getRes();
                     pred = subquery->comparisonType.value() == ast::ExpressionType::COMPARE_NOT_LIKE ? predBuilder.create<db::NotOp>(exprLocation, like) : like;
                  } else {
                     switch (subquery->comparisonType.value()) {
                        case ast::ExpressionType::COMPARE_EQUAL:
                           dbCmpPred = db::DBCmpPredicate::eq;
                           break;
                        case ast::ExpressionType::COMPARE_NOTEQUAL:
                           dbCmpPred = db::DBCmpPredicate::neq;
                           break;
                        case ast::ExpressionType::COMPARE_LESSTHAN:
                           dbCmpPred = db::DBCmpPredicate::lt;
                           break;
                        case ast::ExpressionType::COMPARE_GREATERTHAN:
                           dbCmpPred = db::DBCmpPredicate::gt;
                           break;
                        case ast::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                           dbCmpPred = db::DBCmpPredicate::lte;
                           break;
                        case ast::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                           dbCmpPred = db::DBCmpPredicate::gte;
                           break;

                        default: error("Invalid compare", expression->loc);
                     }

                     pred = predBuilder.create<db::CmpOp>(exprLocation, dbCmpPred, ctExpr, ctCol);

                  }


               } else {
                  pred = predBuilder.create<db::CmpOp>(exprLocation, dbCmpPred, ctExpr, ctCol);
               }


               predBuilder.create<tuples::ReturnOp>(exprLocation, pred);

               auto sel = builder.create<relalg::SelectionOp>(exprLocation, tuples::TupleStreamType::get(mlirContext), translatedSubquery);
               sel.getPredicate().push_back(block);
               translatedSubquery = sel.getResult();
               auto existsOp = builder.create<relalg::ExistsOp>(exprLocation, builder.getI1Type(), translatedSubquery);
               if (subquery->subqueryType == ast::SubqueryType::NOT_ANY) {
                  return builder.create<db::NotOp>(exprLocation, existsOp);
               }
               return existsOp;
            }
            /**
             * TODO:  Currently the location is set quite roughly. Maybe refine this later, for better debugging
             */
            case ast::SubqueryType::ALL: {
                 auto* block = new mlir::Block;
               mlir::OpBuilder predBuilder(mlirContext);
               block->addArgument(tuples::TupleType::get(mlirContext), exprLocation);
               auto tupleScope = translationContext->createTupleScope();
               translationContext->setCurrentTuple(block->getArgument(0));

               predBuilder.setInsertionPointToStart(block);
               mlir::Value expr = translateExpression(predBuilder, subquery->testExpr, context);

               assert(subquery->namedResultForSubquery);
               auto mlirType = subquery->namedResultForSubquery->resultType.toMlirType(mlirContext);
               mlir::Value colVal = predBuilder.create<tuples::GetColumnOp>(exprLocation, mlirType, subquery->namedResultForSubquery->createRef(builder, attrManager), block->getArgument(0));
               //TODO check if following line is correct, could break in some edge cases
               auto ctCol = subquery->namedResult.value()->resultType.castValue(builder, colVal);
               auto ctExpr = subquery->testExpr->resultType->castValue(builder, expr);

               db::DBCmpPredicate dbCmpPred = db::DBCmpPredicate::eq;
               mlir::Value pred;
               if (subquery->comparisonType.has_value()) {
                  if (subquery->comparisonType.value() == ast::ExpressionType::COMPARE_LIKE || subquery->comparisonType.value() == ast::ExpressionType::COMPARE_NOT_LIKE) {
                     auto isNullable = mlir::isa<db::NullableType>(ctExpr.getType()) || mlir::isa<db::NullableType>(ctCol.getType());
                     mlir::Type resType = isNullable ? (mlir::Type) db::NullableType::get(predBuilder.getContext(), predBuilder.getI1Type()) : (mlir::Type) predBuilder.getI1Type();
                     auto like = predBuilder.create<db::RuntimeCall>(exprLocation, resType, "Like", mlir::ValueRange({ctExpr, ctCol})).getRes();
                     pred = subquery->comparisonType.value() == ast::ExpressionType::COMPARE_NOT_LIKE ? predBuilder.create<db::NotOp>(exprLocation, like) : like;
                  } else {
                     switch (subquery->comparisonType.value()) {
                        case ast::ExpressionType::COMPARE_EQUAL:
                           dbCmpPred = db::DBCmpPredicate::eq;
                           break;
                        case ast::ExpressionType::COMPARE_NOTEQUAL:
                           dbCmpPred = db::DBCmpPredicate::neq;
                           break;
                        case ast::ExpressionType::COMPARE_LESSTHAN:
                           dbCmpPred = db::DBCmpPredicate::lt;
                           break;
                        case ast::ExpressionType::COMPARE_GREATERTHAN:
                           dbCmpPred = db::DBCmpPredicate::gt;
                           break;
                        case ast::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                           dbCmpPred = db::DBCmpPredicate::lte;
                           break;
                        case ast::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                           dbCmpPred = db::DBCmpPredicate::gte;
                           break;

                        default: error("Invalid compare", expression->loc);
                     }

                     pred = predBuilder.create<db::CmpOp>(exprLocation, dbCmpPred, ctExpr, ctCol);

                  }


               } else {
                  pred = predBuilder.create<db::CmpOp>(exprLocation, dbCmpPred, ctExpr, ctCol);
               }

               pred = predBuilder.create<db::NotOp>(exprLocation, pred);


               predBuilder.create<tuples::ReturnOp>(exprLocation, pred);

               auto sel = builder.create<relalg::SelectionOp>(exprLocation, tuples::TupleStreamType::get(mlirContext), translatedSubquery);
               sel.getPredicate().push_back(block);
               translatedSubquery = sel.getResult();
               auto existsOp = builder.create<relalg::ExistsOp>(exprLocation, builder.getI1Type(), translatedSubquery);
               return builder.create<db::NotOp>(exprLocation, existsOp);

            }
            case ast::SubqueryType::EXISTS: {
               return builder.create<relalg::ExistsOp>(exprLocation, builder.getI1Type(), translatedSubquery);
            }
            case ast::SubqueryType::NOT_EXISTS: {
               auto existsOp = builder.create<relalg::ExistsOp>(exprLocation, builder.getI1Type(), translatedSubquery);
               return builder.create<db::NotOp>(exprLocation, existsOp);
            }
            default: error("Subquery type not implemented", expression->loc);
         }
      }
      case ast::ExpressionClass::BOUND_CASE: {
         auto boundCase = std::static_pointer_cast<ast::BoundCaseExpression>(expression);
         std::optional<mlir::Value> caseExprTranslated = std::nullopt;
         if (boundCase->caseExpr.has_value()) {
            caseExprTranslated = translateExpression(builder, boundCase->caseExpr.value(), context);
         }
         return translateWhenChecks(builder, boundCase,caseExprTranslated, boundCase->caseChecks, boundCase->elseExpr, context);
      }


      default: error("Expression not implemented", expression->loc);
   }
}

mlir::Value SQLMlirTranslator::translateBinaryOperatorExpression(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundOperatorExpression> expression, std::shared_ptr<analyzer::SQLContext> context, mlir::Value left, mlir::Value right) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(expression->loc, mlirContext);
   assert(expression->resultType.has_value() && expression->children.size() == 2 && expression->children[0]->resultType.has_value() && expression->children[1]->resultType.has_value());

   switch (expression->type) {
      case ast::ExpressionType::OPERATOR_PLUS: {
         if (mlir::isa<db::DateType>(getBaseType(left.getType())) && mlir::isa<db::IntervalType>(getBaseType(right.getType()))) {
            return builder.create<db::RuntimeCall>(location, left.getType(), "DateAdd", mlir::ValueRange({left, right})).getRes();
         }
         std::vector ct = {expression->resultType->castValueToThisType(builder, left, expression->children[0]->resultType->isNullable), expression->resultType->castValueToThisType(builder, right, expression->children[1]->resultType->isNullable)};
         return builder.create<db::AddOp>(location, ct);
      }
      case ast::ExpressionType::OPERATOR_MINUS: {
         if (mlir::isa<db::DateType>(left.getType()) && mlir::isa<db::IntervalType>(right.getType())) {
            return builder.create<db::RuntimeCall>(location, left.getType(), "DateSubtract", mlir::ValueRange({left, right})).getRes();
         }

         std::vector ct = {expression->resultType->castValueToThisType(builder, left, expression->children[0]->resultType->isNullable), expression->resultType->castValueToThisType(builder, right, expression->children[1]->resultType->isNullable)};
         return builder.create<db::SubOp>(location, ct);
      }
      case ast::ExpressionType::OPERATOR_TIMES: {
         auto ct = {expression->children[0]->resultType->castValue(builder, left), expression->children[1]->resultType->castValue(builder, right)};
         return builder.create<db::MulOp>(location, ct);
      }
      case ast::ExpressionType::OPERATOR_DIVIDE: {
         auto ct = {expression->children[0]->resultType->castValue(builder, left), expression->children[1]->resultType->castValue(builder, right)};
         return builder.create<db::DivOp>(location, ct);
      }
      case ast::ExpressionType::OPERATOR_MOD: {
         auto ct = {expression->children[0]->resultType->castValue(builder, left), expression->children[1]->resultType->castValue(builder, right)};
         return builder.create<db::ModOp>(location, ct);
      }
      default: error("Binary operator not implemented", expression->loc);
   }
}

mlir::Value SQLMlirTranslator::translateWindowExpression(mlir::OpBuilder& builder, mlir::Value tree, std::shared_ptr<ast::BoundWindowExpression> expression, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(expression->loc, mlirContext);
   mlir::Value expr;
   tuples::ColumnDefAttr attrDef;
   auto tupleStreamType = tuples::TupleStreamType::get(mlirContext);
   auto tupleType = tuples::TupleType::get(mlirContext);

   auto tupleScope = translationContext->createTupleScope();
   translationContext->getCurrentTuple();
   auto* block = new mlir::Block;
   block->addArgument(tupleStreamType, location);
   block->addArgument(tupleType, location);
   mlir::Value relation = block->getArgument(0);
   mlir::OpBuilder windowBuilder(mlirContext);
   windowBuilder.setInsertionPointToStart(block);
   std::vector<mlir::Value> createdValues;
   std::vector<mlir::Attribute> createdCols;

   expr = translateAggregationFunction(builder, "windowMap", std::vector<mlir::Attribute>{}, relation, windowBuilder, expression->function, expr, attrDef) ;

   attrDef.getColumn().type = expr.getType();

   createdCols.push_back(attrDef);
   createdValues.push_back(expr);

   windowBuilder.create<tuples::ReturnOp>(location, createdValues);

   std::vector<mlir::Attribute> partitionByAttrs;
   std::vector<mlir::Attribute> orderBySpecs;
   if (expression->order.has_value()) {
      for (auto orderByElement : expression->order.value()->orderByElements) {
         tuples::ColumnRefAttr attr;
         assert(orderByElement->namedResult);
         attr = orderByElement->namedResult->createRef(builder, attrManager);

         orderBySpecs.push_back(relalg::SortSpecificationAttr::get(mlirContext, attr, orderByElement->type == ast::OrderType::DESCENDING ? relalg::SortSpec::desc : relalg::SortSpec::asc));

      }
   }

   for (auto& partition: expression->partitions) {
      assert(partition->type == ast::ExpressionType::BOUND_COLUMN_REF);
      partitionByAttrs.push_back(partition->namedResult.value()->createRef(builder, attrManager));
   }
   auto windowOp = builder.create<relalg::WindowOp>(location, tupleStreamType, tree, builder.getArrayAttr(partitionByAttrs), builder.getArrayAttr(orderBySpecs), builder.getArrayAttr(createdCols), expression->windowBoundary->start, expression->windowBoundary->end);
   windowOp.getAggrFunc().push_back(block);


   return windowOp.getResult() ;

}

mlir::Value SQLMlirTranslator::translateWhenChecks(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundCaseExpression> boundCase, std::optional<mlir::Value> caseExprTranslated, std::vector<ast::BoundCaseExpression::BoundCaseCheck> caseChecks, std::shared_ptr<ast::BoundExpression> elseExpr, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto boundCaseLocation = getLocationFromBison(boundCase->loc, mlirContext);
   if (caseChecks.empty()) {
      if (!elseExpr) {
         return builder.create<db::NullOp>(boundCaseLocation, db::NullableType::get(mlirContext, builder.getNoneType()));
      }
      return translateExpression(builder, elseExpr, context);
   }
   auto commonType = boundCase->resultType->toMlirType(mlirContext);

   auto check = caseChecks[0];
   auto hasNextCheck = caseChecks.size() > 1;

   auto condTranslated = translateExpression(builder, check.whenExpr, context);

   if (caseExprTranslated.has_value()) {
      condTranslated = check.whenExpr->resultType->castValue(builder, condTranslated);
      caseExprTranslated = boundCase->caseExpr.value()->resultType->castValue(builder, caseExprTranslated.value());
      condTranslated = builder.create<db::CmpOp>(getLocationFromBison(check.whenExpr->loc, mlirContext), db::DBCmpPredicate::eq, condTranslated, caseExprTranslated.value());
   }

   auto* whenBlock = new mlir::Block;
   auto* elseBlock = new mlir::Block;
   mlir::OpBuilder whenBuilder(mlirContext);
   whenBuilder.setInsertionPointToStart(whenBlock);
   auto thenTranslated = translateExpression(whenBuilder, check.thenExpr, context);
   mlir::OpBuilder elseBuilder(mlirContext);
   elseBuilder.setInsertionPointToStart(elseBlock);
   mlir::Value elseTranslated;
   if (hasNextCheck) {
      std::vector<ast::BoundCaseExpression::BoundCaseCheck> nextChecks{caseChecks.begin() + 1, caseChecks.end()};
      elseTranslated = translateWhenChecks(elseBuilder, boundCase, caseExprTranslated, nextChecks, elseExpr, context);
   } else {
      elseTranslated = translateExpression(elseBuilder, elseExpr, context);
   }

   thenTranslated = boundCase->resultType->castValueToThisType(whenBuilder, thenTranslated, check.thenExpr->resultType->isNullable);
   elseTranslated = boundCase->resultType->castValueToThisType(elseBuilder, elseTranslated, elseExpr->resultType->isNullable);

   whenBuilder.create<mlir::scf::YieldOp>(getLocationFromBison(check.thenExpr->loc, mlirContext), thenTranslated);
   elseBuilder.create<mlir::scf::YieldOp>(getLocationFromBison(elseExpr->loc, mlirContext), elseTranslated);
   condTranslated = builder.create<db::DeriveTruth>(boundCaseLocation, condTranslated);
   auto ifOp = builder.create<mlir::scf::IfOp>(boundCaseLocation, commonType, condTranslated, true);
   ifOp.getThenRegion().getBlocks().clear();
   ifOp.getElseRegion().getBlocks().clear();
   ifOp.getThenRegion().push_back(whenBlock);
   ifOp.getElseRegion().push_back(elseBlock);

   return ifOp.getResult(0);
}

mlir::Value SQLMlirTranslator::translateCoalesceExpression(mlir::OpBuilder& builder, NullableType resultType, std::vector<std::shared_ptr<ast::BoundExpression>> expressions, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();

   if (expressions.empty()) {
      auto value = builder.create<db::NullOp>(builder.getUnknownLoc(), db::NullableType::get(mlirContext, builder.getNoneType()));
      return resultType.castValueToThisType(builder, value, true);

   }
   auto currentExpression = expressions[0];
   auto location = getLocationFromBison(currentExpression->loc, mlirContext);
   mlir::Value value = translateExpression(builder, currentExpression, context);
   mlir::Value isNull = currentExpression->resultType->isNullable ? (mlir::Value) builder.create<db::IsNullOp>(location, value) : (mlir::Value) builder.create<mlir::arith::ConstantIntOp>(location, 0, builder.getI1Type());
   mlir::Value isNotNull = builder.create<db::NotOp>(location, isNull);

   value = resultType.castValueToThisType(builder, value, currentExpression->resultType->isNullable);

   auto* whenBlock = new mlir::Block;
   auto* elseBlock = new mlir::Block;
   mlir::OpBuilder whenBuilder(mlirContext);
   whenBuilder.setInsertionPointToStart(whenBlock);
   mlir::OpBuilder elseBuilder(mlirContext);
   elseBuilder.setInsertionPointToStart(elseBlock);
   auto elseResl = translateCoalesceExpression(elseBuilder, resultType, std::vector<std::shared_ptr<ast::BoundExpression>>{expressions.begin() + 1, expressions.end()}, context);

   whenBuilder.create<mlir::scf::YieldOp>(location, value);
   elseBuilder.create<mlir::scf::YieldOp>(location, elseResl);
   auto ifOp = builder.create<mlir::scf::IfOp>(location, resultType.toMlirType(mlirContext), isNotNull, true);
   ifOp.getThenRegion().getBlocks().clear();
   ifOp.getElseRegion().getBlocks().clear();
   ifOp.getThenRegion().push_back(whenBlock);
   ifOp.getElseRegion().push_back(elseBlock);
   return ifOp.getResult(0);

}

mlir::Value SQLMlirTranslator::translateTableRef(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundTableRef> tableRef, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(tableRef->loc, mlirContext);
   mlir::Value last;
   switch (tableRef->type) {
      case ast::TableReferenceType::BASE_TABLE: {
         auto baseTableRef = std::static_pointer_cast<ast::BoundBaseTableRef>(tableRef);
         std::string relation = baseTableRef->relationName;
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

               return builder.create<relalg::RenamingOp>(location, tuples::TupleStreamType::get(mlirContext), _tree, builder.getArrayAttr(renamingDefsAsAttr));
            }
         }
         if (baseTableRef->namedResultsEntries.empty()) {
            error("Table " << baseTableRef->relationName << " not found", baseTableRef->loc);
         }
         auto rel = baseTableRef->namedResultsEntries;
         std::string uniqueScope = baseTableRef->mlirScope;

         std::vector<mlir::NamedAttribute> columns{};
         for (auto& info : rel) {
            auto attrDef = info->createDef(builder, attrManager);
            attrDef.getColumn().type = info->resultType.toMlirType(mlirContext);
            columns.push_back(builder.getNamedAttr(info->name, attrDef));
         }
         return builder.create<relalg::BaseTableOp>(location, tuples::TupleStreamType::get(mlirContext), relation, builder.getDictionaryAttr(columns));
      }
      case ast::TableReferenceType::CROSS_PRODUCT: {
         auto crossProd = std::static_pointer_cast<ast::BoundCrossProductRef>(tableRef);
         mlir::Value current = translateTableProducer(builder, crossProd->boundTables[0], context);
         for (size_t i = 1; i < crossProd->boundTables.size(); i++) {
            current = builder.create<relalg::CrossProductOp>(location, tuples::TupleStreamType::get(mlirContext), current, translateTableProducer(builder, crossProd->boundTables[i], context));
         }

         return current;
      }
      case ast::TableReferenceType::JOIN: {
         auto boundJoin = std::static_pointer_cast<ast::BoundJoinRef>(tableRef);
         mlir::Value left, right;

         if (boundJoin->leftScope) {
            context->pushNewScope(boundJoin->leftScope);
         }
         left = translateTableProducer(builder, boundJoin->left, context);
         if (boundJoin->leftScope) {
            context->popCurrentScope();
         }
         //Here the evalbefore gets lost!
         if (boundJoin->rightScope) {
            context->pushNewScope(boundJoin->rightScope);
         }
         if (boundJoin->right) {
            right = translateTableProducer(builder, boundJoin->right, context);
         }

         if (boundJoin->leftScope) {
            context->popCurrentScope();
         }
         switch (boundJoin->type) {
            case ast::JoinType::INNER: {
               mlir::Block* pred;
               if (!std::holds_alternative<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition)) {
                  error("Invalid join condition", tableRef->loc);
               }

               pred = translatePredicate(builder, std::get<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition), context);

               auto joinOp = builder.create<relalg::InnerJoinOp>(location, tuples::TupleStreamType::get(mlirContext), left, right);
               joinOp.getPredicate().push_back(pred);

               return joinOp;
            }
            case ast::JoinType::LEFT: {
               mlir::Block* pred;
               if (!std::holds_alternative<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition)) {
                  error("Invalid join condition", tableRef->loc);
               }

               pred = translatePredicate(builder, std::get<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition), context);

               std::vector<mlir::Attribute> outerJoinMapping{};
               static size_t i = 0;
               std::ranges::transform(boundJoin->outerJoinMapping, std::back_inserter(outerJoinMapping), [&](std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>> scopeAndNamedResult) {
                  i++;
                  auto attrDef = scopeAndNamedResult.second->createDef(builder, attrManager, builder.getArrayAttr({scopeAndNamedResult.first->createRef(builder, attrManager)}));
                  return attrDef;
               });

               mlir::ArrayAttr mapping = builder.getArrayAttr(outerJoinMapping);

               auto join = builder.create<relalg::OuterJoinOp>(location, tuples::TupleStreamType::get(mlirContext), left, right, mapping);
               join.getPredicate().push_back(pred);
               return join;
            }
            case ast::JoinType::FULL: {
               mlir::Block* pred;
               if (!std::holds_alternative<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition)) {
                  error("Invalid join condition", tableRef->loc);
               }
               pred = translatePredicate(builder, std::get<std::shared_ptr<ast::BoundExpression>>(boundJoin->condition), context);

               std::vector<mlir::Attribute> outerJoinMapping{};
               static size_t i = 0;
               std::ranges::transform(boundJoin->outerJoinMapping, std::back_inserter(outerJoinMapping), [&](std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>> scopeAndNamedResult) {
                  i++;
                  auto attrDef = scopeAndNamedResult.second->createDef(builder, attrManager, builder.getArrayAttr({scopeAndNamedResult.first->createRef(builder, attrManager)}));

                  return attrDef;
               });

               mlir::ArrayAttr mapping = builder.getArrayAttr(outerJoinMapping);

               auto join = builder.create<relalg::FullOuterJoinOp>(location, tuples::TupleStreamType::get(mlirContext), left, right, mapping);
               join.getPredicate().push_back(pred);
               return join;

            }
            default: error("Invalid join type", tableRef->loc);
         }

         right = translateTableProducer(builder, boundJoin->right, context);
      }
      case ast::TableReferenceType::SUBQUERY: {
         auto subquery = std::static_pointer_cast<ast::BoundSubqueryRef>(tableRef);
         context->pushNewScope(subquery->sqlScope);
         mlir::Value subQuery;
         auto translated = translateTableProducer(builder, subquery->subSelect, context);
         context->popCurrentScope();

         return translated;
      }
      case ast::TableReferenceType::BOUND_EXPRESSION_LIST: {
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
                     value = builder.getStringAttr(fValue->fVal);
                     assert(constExpr->resultType.has_value());
                     break;
                  }
                  case ast::ConstantType::NULL_P: {
                     value = builder.getUnitAttr();
                     assert(constExpr->resultType.has_value());
                     break;
                  }


                  default: error("Invalid constant in expression list", constExpr->loc);
               }
               values.emplace_back(value);
               types.emplace_back(constExpr->resultType.value().toMlirType(mlirContext));
            }
            rows.push_back(builder.getArrayAttr(values));
         }

         std::vector<mlir::Attribute> attributes;
         for (auto namedResult : expressionList->namedResultsEntries) {
            auto attrDef = namedResult->createDef(builder, attrManager);
            attributes.push_back(attrDef);
         }

         auto translated = builder.create<relalg::ConstRelationOp>(location, builder.getArrayAttr(attributes), builder.getArrayAttr(rows));
         return translated;
      }

      default:
         error("Table reference not implemented", tableRef->loc);
   }
}

mlir::Value SQLMlirTranslator::translateSetOperation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundSetOperationNode> boundSetOp, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(boundSetOp->loc, mlirContext);
   auto rightLocation = getLocationFromBison(boundSetOp->boundRight->loc, mlirContext);
   auto leftLocation = getLocationFromBison(boundSetOp->boundLeft->loc, mlirContext);
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
   mlir::OpBuilder leftMapBuilder(mlirContext);
   mlir::OpBuilder rightMapBuilder(mlirContext);
   leftMapBuilder.setInsertionPointToStart(leftMapBlock);
   rightMapBuilder.setInsertionPointToStart(rightMapBlock);
   leftMapBlock->addArgument(tuples::TupleType::get(mlirContext), location);
   rightMapBlock->addArgument(tuples::TupleType::get(mlirContext), location);
   auto leftMapScope = context->getUniqueScope("setMap");
   auto rightMapScope = context->getUniqueScope("setMap");
   std::vector<mlir::Attribute> createdColsLeft;
   std::vector<mlir::Attribute> createdColsRight;
   mlir::Value leftTuple = leftMapBlock->getArgument(0);
   mlir::Value rightTuple = rightMapBlock->getArgument(0);
   std::vector<mlir::Value> leftMapResults;
   std::vector<mlir::Value> rightMapResults;
   for (size_t i = 0; i < boundSetOp->leftScope->targetInfo.targetColumns.size(); i++) {
      auto leftResult = boundSetOp->leftScope->targetInfo.targetColumns[i];
      auto rightResult = boundSetOp->rightScope->targetInfo.targetColumns[i];
      auto commonType = context->currentScope->targetInfo.targetColumns[i]->resultType;
      //TODO maybe move this logic into analyzer (could be tricky)
      if (rightResult->resultType != commonType) {
         auto attrDef = attrManager.createDef(rightMapScope, std::string("set_op") + std::to_string(i));
         auto attrRef = rightResult->createRef(builder, attrManager);

         createdColsRight.push_back(attrDef);
         mlir::Value expr = rightMapBuilder.create<tuples::GetColumnOp>(location, attrRef.getColumn().type, attrRef, rightTuple);
         rightMapResults.push_back(commonType.castValueToThisType(rightMapBuilder, expr,   rightResult->resultType.isNullable));

         rightResult->resultType = commonType;
         rightResult->scope = rightMapScope;
         rightResult->name = "set_op" + std::to_string(i);
      }
      //TODO maybe move this logic into analyzer (could be tricky)
      if (leftResult->resultType != commonType) {
         auto attrDef = attrManager.createDef(leftMapScope, std::string("set_op") + std::to_string(i));
         auto attrRef = leftResult->createRef(builder, attrManager);

         createdColsLeft.push_back(attrDef);
         mlir::Value expr = leftMapBuilder.create<tuples::GetColumnOp>(leftLocation, attrRef.getColumn().type, attrRef, leftTuple);
         leftMapResults.push_back(commonType.castValueToThisType(leftMapBuilder, expr,  leftResult->resultType.isNullable));

         leftResult->resultType = commonType;
         leftResult->scope = leftMapScope;
         leftResult->name = "set_op" + std::to_string(i);
      }

      auto newType = commonType.toMlirType(mlirContext);
      auto newColName = leftResult->name;
      auto newColDef = context->currentScope->targetInfo.targetColumns[i]->createDef(builder, attrManager, builder.getArrayAttr({leftResult->createRef(builder, attrManager), rightResult->createRef(builder, attrManager)}));
      auto* newCol = &newColDef.getColumn();
      newCol->type = newType;
      attributes.push_back(newColDef);
   }

   if (!leftMapResults.empty()) {
      auto mapOp = builder.create<relalg::MapOp>(location, tuples::TupleStreamType::get(mlirContext), leftTree, builder.getArrayAttr(createdColsLeft));
      mapOp.getPredicate().push_back(leftMapBlock);
      leftMapBuilder.create<tuples::ReturnOp>(leftLocation, leftMapResults);
      leftTree = mapOp.getResult();
   } else {
      delete leftMapBlock;
   }
   if (!rightMapResults.empty()) {
      auto mapOp = builder.create<relalg::MapOp>(rightLocation, tuples::TupleStreamType::get(mlirContext), rightTree, builder.getArrayAttr(createdColsRight));
      mapOp.getPredicate().push_back(rightMapBlock);
      rightMapBuilder.create<tuples::ReturnOp>(rightLocation, rightMapResults);
      rightTree = mapOp.getResult();

   } else {
      delete rightMapBlock;
   }

   mlir::Value tree;
   switch (boundSetOp->setType) {
      case ast::SetOperationType::UNION: {
         tree = builder.create<relalg::UnionOp>(location, lingodb::compiler::dialect::relalg::SetSemanticAttr::get(mlirContext, setSemantic), leftTree, rightTree, builder.getArrayAttr(attributes));
         break;
      }
      case ast::SetOperationType::INTERSECT: {
         tree = builder.create<relalg::IntersectOp>(location, lingodb::compiler::dialect::relalg::SetSemanticAttr::get(mlirContext, setSemantic), leftTree, rightTree, builder.getArrayAttr(attributes));
         break;
      }
      case ast::SetOperationType::EXCEPT: {
         tree = builder.create<relalg::ExceptOp>(location, lingodb::compiler::dialect::relalg::SetSemanticAttr::get(mlirContext, setSemantic), leftTree, rightTree, builder.getArrayAttr(attributes));
         break;
      }
      default: error("Set operation type not implemented", boundSetOp->loc);
   }

   return tree;
}

mlir::Value SQLMlirTranslator::translateAggregationFunction(mlir::OpBuilder& builder, std::string mapName, std::vector<mlir::Attribute> groupByAttrs, mlir::Value relation, mlir::OpBuilder functionBuilder, std::shared_ptr<ast::BoundFunctionExpression> aggrFunction, mlir::Value& expr, tuples::ColumnDefAttr& attrDef) {
   auto *mlirContext = builder.getContext();
   auto aggrFuncName = aggrFunction->functionName;
   auto location = getLocationFromBison(aggrFunction->loc, mlirContext);
   attrDef = aggrFunction->namedResult.value()->createDef(builder, attrManager);
   if (aggrFuncName == "COUNT*") {
      expr = functionBuilder.create<relalg::CountRowsOp>(location, builder.getI64Type(), relation);

   } else if (aggrFuncName == "RANK" || aggrFuncName == "ROW_NUMBER") {
      expr = functionBuilder.create<relalg::RankOp>(location, builder.getI64Type(), relation);
   } else {
      auto relalgAggrFunc = llvm::StringSwitch<relalg::AggrFunc>(aggrFuncName)
                               .Case("SUM", relalg::AggrFunc::sum)
                               .Case("AVG", relalg::AggrFunc::avg)
                               .Case("MIN", relalg::AggrFunc::min)
                               .Case("MAX", relalg::AggrFunc::max)
                               .Case("COUNT", relalg::AggrFunc::count)
                               .Case("STDDEV_SAMP", relalg::AggrFunc::stddev_samp)
                               .Default(relalg::AggrFunc::count);
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
            refAttr = aggrFunction->arguments[0]->namedResult.value()->createRef(builder, attrManager);

            break;
         };
      }

      mlir::Value currRel = relation;
      if (aggrFunction->distinct) {
         currRel = functionBuilder.create<relalg::ProjectionOp>(location, relalg::SetSemantic::distinct, currRel, builder.getArrayAttr({refAttr}));
      }
      mlir::Type aggrResultType;
      assert(aggrFunction->resultType.has_value());
      aggrResultType = aggrFunction->resultType->toMlirType(mlirContext);

      if (aggrFunction->arguments[0]->type != ast::ExpressionType::BOUND_COLUMN_REF) {
         assert(aggrFunction->arguments[0]->namedResult.has_value());

      }
      if (mlir::isa<db::NullableType>(aggrResultType)) {
         assert(aggrFunction->namedResult.has_value());
         aggrFunction->namedResult.value()->resultType.isNullable = true;
      }

      expr = functionBuilder.create<relalg::AggrFuncOp>(location, aggrResultType, relalgAggrFunc, currRel, refAttr);
   }
   attrDef.getColumn().type = expr.getType();
   return expr;
}
mlir::Value SQLMlirTranslator::translateGroupByAttributesAndAggregate(mlir::OpBuilder& builder, mlir::Value tree, mlir::Location loc, std::vector<std::shared_ptr<ast::NamedResult>> groupNamedResults, std::vector<std::shared_ptr<ast::BoundFunctionExpression>> aggregations, std::string mapName) {
   auto *mlirContext = builder.getContext();

   //Translate group by Attributes
   std::vector<mlir::Attribute> groupByAttrs;
   std::unordered_map<std::string, mlir::Attribute> groupedExpressions;
   std::unordered_map<std::string, size_t> groupByAttrToPos;
   for (auto& groupByNamedResult : groupNamedResults) {
      auto attrDef = groupByNamedResult->createRef(builder, attrManager);
      groupByAttrs.push_back(attrDef);
   }

   /*
       *Perform aggregation
      */
   static size_t groupById = 0;
   auto tupleStreamType = tuples::TupleStreamType::get(mlirContext);
   auto tupleType = tuples::TupleType::get(mlirContext);


   auto tupleScope = translationContext->createTupleScope();
   auto* block = new mlir::Block;
   block->addArgument(tupleStreamType, loc);
   block->addArgument(tupleType, loc);
   mlir::Value relation = block->getArgument(0);
   mlir::OpBuilder aggrBuilder(mlirContext);
   aggrBuilder.setInsertionPointToStart(block);
   std::vector<mlir::Value> createdValues;
   std::vector<mlir::Attribute> createdCols;
   std::unordered_map<std::string, tuples::Column*> mapping;

   //AggrFunctions
   for (auto aggrFunction : aggregations) {
      mlir::Value expr;
      tuples::ColumnDefAttr attrDef;
      expr = translateAggregationFunction(builder, mapName, groupByAttrs, relation, aggrBuilder, aggrFunction, expr, attrDef);
      createdCols.push_back(attrDef);
      createdValues.push_back(expr);
   }

   aggrBuilder.create<tuples::ReturnOp>(loc, createdValues);
   auto groupByOp = builder.create<relalg::AggregationOp>(loc, tupleStreamType, tree, builder.getArrayAttr(groupByAttrs), builder.getArrayAttr(createdCols));
   groupByOp.getAggrFunc().push_back(block);
   return groupByOp.getResult();
}
mlir::Value SQLMlirTranslator::translateAggregation(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundAggregationNode> aggregation, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(aggregation->loc, mlirContext);
   auto mapToNullable = [this, &location, &mlirContext](mlir::OpBuilder& builder, std::vector<std::shared_ptr<ast::NamedResult>> toMap, std::vector<std::shared_ptr<ast::NamedResult>> mapTo, mlir::Value tree) {
      if (toMap.empty()) return tree;
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(mlirContext);
      block->addArgument(tuples::TupleType::get(mlirContext), location);
      auto tupleScope = translationContext->createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      translationContext->setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      for (size_t i = 0; i < toMap.size(); i++) {
         auto colRef = mlir::cast<tuples::ColumnRefAttr>(toMap[i]->createRef(builder, attrManager));
         auto newColDef = mapTo[i]->createDef(builder, attrManager);
         mlir::Value expr = mapBuilder.create<tuples::GetColumnOp>(location, colRef.getColumn().type, colRef, tuple);
         if (colRef.getColumn().type != newColDef.getColumn().type) {
            expr = mapBuilder.create<db::AsNullableOp>(location, newColDef.getColumn().type, expr);
         }
         auto attrDef = mapTo[i]->createDef(builder, attrManager);
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);
      }
      auto mapOp = builder.create<relalg::MapOp>(location, tuples::TupleStreamType::get(mlirContext), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<tuples::ReturnOp>(location, createdValues);
      return mapOp.asRelation();
   };
   auto mapToNull = [this, &location, &mlirContext](mlir::OpBuilder& builder, std::vector<std::shared_ptr<ast::NamedResult>> toMap, mlir::Value tree) {
      if (toMap.empty()) return tree;
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(mlirContext);
      block->addArgument(tuples::TupleType::get(mlirContext),location);
      auto tupleScope = translationContext->createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      translationContext->setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      for (auto p : toMap) {
         mlir::Value expr = mapBuilder.create<db::NullOp>(location, p->resultType.toMlirType(mlirContext));
         auto attrDef = p->createDef(builder, attrManager);
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);
      }
      auto mapOp = builder.create<relalg::MapOp>(location, tuples::TupleStreamType::get(mlirContext), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<tuples::ReturnOp>(location, createdValues);
      return mapOp.asRelation();
   };

   //Ignore empty aggregations
   if ((!aggregation->groupByNode || aggregation->groupByNode->groupByNamedResults.empty()) && aggregation->aggregations.empty()) {
      return tree;
   }
   //create map
   tree = createMap(builder, location, aggregation->mapName, aggregation->toMapExpressions, context, tree);
   if(aggregation->groupByNode && !aggregation->groupByNode->localGroupByNamedResults.empty()) {
      auto asNullable = [](mlir::Type t) { return mlir::isa<db::NullableType>(t) ? t : db::NullableType::get(t.getContext(), t); };
      auto mapInt = [this, &location, &mlirContext](mlir::OpBuilder& builder, size_t intVal, std::shared_ptr<ast::NamedResult> namedResult, mlir::Value tree) -> mlir::Value {
         auto* block = new mlir::Block;
         static size_t mapId = 0;
         std::string mapName = "map" + std::to_string(mapId++);

         mlir::OpBuilder mapBuilder(mlirContext);
         block->addArgument(tuples::TupleType::get(mlirContext),location);
         auto tupleScope = translationContext->createTupleScope();
         mlir::Value tuple = block->getArgument(0);
         translationContext->setCurrentTuple(tuple);

         mapBuilder.setInsertionPointToStart(block);
         std::vector<mlir::Value> createdValues;
         std::vector<mlir::Attribute> createdCols;
         mlir::Value expr = mapBuilder.create<mlir::arith::ConstantIntOp>(location, intVal, mapBuilder.getI64Type());
         auto attrDef = namedResult->createDef(builder, attrManager);
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);

         auto mapOp = builder.create<relalg::MapOp>(location, tuples::TupleStreamType::get(mlirContext), tree, builder.getArrayAttr(createdCols));
         mapOp.getPredicate().push_back(block);
         mapBuilder.create<tuples::ReturnOp>(location, createdValues);
         return mapOp.getResult();
      };
      auto mapCheckBit = [this, &location, &mlirContext](mlir::OpBuilder& builder, size_t shift, std::shared_ptr<ast::NamedResult> namedResult, std::shared_ptr<ast::NamedResult> namedResultColumn, mlir::Value tree) -> mlir::Value {
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(mlirContext);
      block->addArgument(tuples::TupleType::get(mlirContext), location);
      auto tupleScope = translationContext->createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      translationContext->setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      auto colDef = namedResultColumn->createDef(builder, attrManager);
      auto colRef = namedResultColumn->createRef(builder, attrManager);
      mlir::Value shiftVal = mapBuilder.create<mlir::arith::ConstantIntOp>(location, shift, mapBuilder.getI64Type());
      mlir::Value colVal = mapBuilder.create<tuples::GetColumnOp>(location, colRef.getColumn().type, colRef, tuple);
      mlir::Value shifted = mapBuilder.create<mlir::arith::ShRUIOp>(location, colVal, shiftVal);
      mlir::Value one = mapBuilder.create<mlir::arith::ConstantIntOp>(location, 1, mapBuilder.getI64Type());
      mlir::Value expr = mapBuilder.create<mlir::arith::AndIOp>(location, shifted, one);
      auto attrDef = namedResult->createDef(builder, attrManager);
      attrDef.getColumn().type = mapBuilder.getI64Type();
      createdCols.push_back(attrDef);
      createdValues.push_back(expr);

      auto mapOp = builder.create<relalg::MapOp>(location, tuples::TupleStreamType::get(mlirContext), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<tuples::ReturnOp>(location, createdValues);
      return mapOp.getResult();
   };


      struct Part {
         mlir::Value tree;
         std::vector<std::shared_ptr<ast::NamedResult>> groupByCols;
         std::vector<std::shared_ptr<ast::NamedResult>> groupByCols2;
         std::vector<std::shared_ptr<ast::NamedResult>> computed;
         std::shared_ptr<ast::NamedResult> grouping;

      };
      std::vector<Part> parts;

      static size_t rollupId = 0;
      auto scopeName = "rollup_" + std::to_string(rollupId);
      rollupId++;

      for (size_t i = 0; i<aggregation->groupByNode->localGroupByNamedResults.size(); i++) {

         std::vector<std::shared_ptr<ast::NamedResult>> localGroupByAttrs = aggregation->groupByNode->localGroupByNamedResults.at(i);
         std::vector<std::shared_ptr<ast::NamedResult>> localGroupByAttrsNullable = aggregation->groupByNode->localMapToNullNamedResults.at(i);
         std::vector<std::shared_ptr<ast::NamedResult>> notAvailable = aggregation->groupByNode->localNotAvailableNamedResults.at(i);
         std::vector<std::shared_ptr<ast::NamedResult>> computed;

         for (size_t j = 0; j < aggregation->aggregations.size(); j++) {
            aggregation->aggregations.at(j)->namedResult = aggregation->groupByNode->localAggregationNamedResults.at(i).at(j);
            computed.emplace_back(aggregation->groupByNode->localAggregationNamedResults.at(i).at(j));
         }

         auto tree2 = translateGroupByAttributesAndAggregate(builder, tree, location, localGroupByAttrs, aggregation->aggregations, aggregation->mapName);
         tree2 = mapToNull(builder, notAvailable, tree2);
         tree2 = mapToNullable(builder, localGroupByAttrs, localGroupByAttrsNullable, tree2);

         tree2 = mapInt(builder, aggregation->groupByNode->localPresentIntval[i].first, aggregation->groupByNode->localPresentIntval[i].second, tree2);

         parts.push_back({ tree2, localGroupByAttrsNullable, notAvailable, computed, aggregation->groupByNode->localPresentIntval[i].second});

      }
      mlir::Value currTree = parts[0].tree;
      std::vector currentAttributes(parts[0].groupByCols.begin(), parts[0].groupByCols.end());
      currentAttributes.insert(currentAttributes.end(), parts[0].groupByCols2.begin(), parts[0].groupByCols2.end());
      currentAttributes.insert(currentAttributes.end(), parts[0].computed.begin(), parts[0].computed.end());
      currentAttributes.emplace_back(parts[0].grouping);

      for (size_t i = 1; i <= aggregation->groupByNode->unionNamedResults.size(); i++) {
         std::vector<std::shared_ptr<ast::NamedResult>> currentLocalAttributes(parts[i].groupByCols.begin(), parts[i].groupByCols.end());
         currentLocalAttributes.insert(currentLocalAttributes.end(), parts[i].groupByCols2.begin(), parts[i].groupByCols2.end());
         currentLocalAttributes.insert(currentLocalAttributes.end(), parts[i].computed.begin(), parts[i].computed.end());
         currentLocalAttributes.emplace_back(parts[i].grouping);
         auto unionNamedResults = aggregation->groupByNode->unionNamedResults.at(i-1);
         std::vector<mlir::Attribute> unionAttributes;
         for (size_t j = 0; j<unionNamedResults.size(); j++) {
            auto left = currentAttributes[j]->createRef(builder, attrManager);
            auto right = currentLocalAttributes[j]->createRef(builder, attrManager);
            auto unionAttribute = unionNamedResults[j]->createDef(builder, attrManager, builder.getArrayAttr({left, right}));
            unionAttribute.getColumn().type = currentLocalAttributes[j]->createDef(builder, attrManager).getColumn().type;
            unionAttributes.push_back(unionAttribute);
         }
         currentAttributes = unionNamedResults;
         currTree = builder.create<relalg::UnionOp>(location, relalg::SetSemanticAttr::get(mlirContext, relalg::SetSemantic::all), currTree, parts[i].tree, builder.getArrayAttr(unionAttributes));
      }
      tree = currTree;

      for (auto& [element, namedResult] :aggregation->groupByNode->groupingFunctions) {
         auto shiftAmount = element;
         auto tree2 = mapCheckBit(builder,  shiftAmount, namedResult,  currentAttributes.back(), tree);
         tree = tree2;
      }

   } else {
      auto groupNamedResults = aggregation->groupByNode->groupByNamedResults;
      auto aggregations = aggregation->aggregations;
      std::string mapName = aggregation->mapName;

      return translateGroupByAttributesAndAggregate(builder, tree, location, groupNamedResults, aggregations, mapName);
   }

   return tree;
}


mlir::Value SQLMlirTranslator::createMap(mlir::OpBuilder& builder, mlir::Location loc, std::string mapName, std::vector<std::shared_ptr<ast::BoundExpression>> toMap, std::shared_ptr<analyzer::SQLContext> context, mlir::Value tree) {
   auto *mlirContext = builder.getContext();
   if (toMap.empty()) {
      return tree;
   }
   auto* block = new mlir::Block;
   static size_t mapId = 0;

   mlir::OpBuilder mapBuilder(mlirContext);
   block->addArgument(tuples::TupleType::get(mlirContext), loc);
   auto tupleScope = translationContext->createTupleScope();
   mlir::Value tuple = block->getArgument(0);
   translationContext->setCurrentTuple(tuple);

   mapBuilder.setInsertionPointToStart(block);
   std::vector<mlir::Value> createdValues;
   std::vector<mlir::Attribute> createdCols;

   for (auto p : toMap) {
      mlir::Value expr = translateExpression(mapBuilder, p, context);
      assert(p->namedResult.has_value());
      //TODO: Evaluate if setting the scope here is necessary. All current tests pass without it
      //p->namedResult.value()->scope = mapName;
      auto attrDef = p->namedResult.value()->createDef(builder, attrManager);

      attrDef.getColumn().type = expr.getType();
      createdCols.push_back(attrDef);
      createdValues.push_back(expr);
   }
   auto mapOp = builder.create<relalg::MapOp>(loc, tuples::TupleStreamType::get(mlirContext), tree, builder.getArrayAttr(createdCols));
   mapOp.getPredicate().push_back(block);
   mapBuilder.create<tuples::ReturnOp>(loc, createdValues);
   return mapOp.getResult();
}

/*
 * Utility
 */

mlir::Block* SQLMlirTranslator::translatePredicate(mlir::OpBuilder& builder, std::shared_ptr<ast::BoundExpression> expression, std::shared_ptr<analyzer::SQLContext> context) {
   auto *mlirContext = builder.getContext();
   auto location = getLocationFromBison(expression->loc, mlirContext);
   auto* block = new mlir::Block;
   mlir::OpBuilder predBuilder(mlirContext);
   block->addArgument(tuples::TupleType::get(mlirContext), location);
   auto tupleScope = translationContext->createTupleScope();
   translationContext->setCurrentTuple(block->getArgument(0));

   predBuilder.setInsertionPointToStart(block);
   mlir::Value expr = translateExpression(predBuilder, expression, context);
   predBuilder.create<tuples::ReturnOp>(location, expr);
   return block;
}

mlir::Location SQLMlirTranslator::getLocationFromBison(const location& loc, mlir::MLIRContext* ctx) {
   return mlir::FileLineColLoc::get(
     ctx,
     mlir::StringAttr::get(ctx, loc.begin.filename == nullptr ? "No filename" : *loc.begin.filename),
     loc.begin.line,
     loc.begin.column
 );
}

/*
 * Helper functions
 */

mlir::Type SQLMlirTranslator::createBaseTypeFromColumnType(mlir::MLIRContext* context, const catalog::Type& t) {
   return t.getMLIRTypeCreator()->createType(context);
}

} // namespace lingodb::translator