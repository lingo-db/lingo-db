#include "lingodb/compiler/frontend/sql_analyzer.h"


#include "lingodb/compiler/frontend/ast/bound/bound_aggregation.h"
#include "lingodb/compiler/frontend/ast/bound/bound_create_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_extend_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_groupby.h"
#include "lingodb/compiler/frontend/ast/bound/bound_insert_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_pipe_operator.h"
#include "lingodb/compiler/frontend/ast/bound/bound_query_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_tableref.h"
#include "lingodb/runtime/RecordBatchInfo.h"

#include <ranges>
#include <cctype>
namespace lingodb::analyzer {
using ResolverScope = llvm::ScopedHashTable<std::string, std::shared_ptr<ast::NamedResult>, StringInfo>::ScopeTy;
/*
 * SQLCanonicalizer
*/
std::shared_ptr<ast::TableProducer> SQLCanonicalizer::canonicalize(std::shared_ptr<ast::TableProducer> rootNode, std::shared_ptr<ASTTransformContext> context) {
   switch (rootNode->nodeType) {
      case ast::NodeType::QUERY_NODE: {
         auto queryNode = std::static_pointer_cast<ast::QueryNode>(rootNode);
         switch (queryNode->type) {
            case ast::QueryNodeType::SELECT_NODE: {
               auto selectNode = std::static_pointer_cast<ast::SelectNode>(queryNode);
               std::shared_ptr<ast::TableProducer> transformed = nullptr;
               //Transform from_clause
               if (selectNode->from_clause) {
                  auto transformedFrom = canonicalizeCast<ast::TableRef>(selectNode->from_clause, context);

                  selectNode->from_clause = nullptr;
                  transformed = transformedFrom;
               }

               auto extendPipeOp = drv.nf.node<ast::PipeOperator>(selectNode->select_list->loc, ast::PipeOperatorType::EXTEND, context->currentScope->extendNode);
               extendPipeOp->input = transformed;
               transformed = extendPipeOp;
               //Transform where_clause
               if (selectNode->where_clause) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(selectNode->where_clause->loc, ast::PipeOperatorType::WHERE, selectNode->where_clause);
                  auto transFormededWhereClause = canonicalizeCast<ast::PipeOperator>(pipe, context);
                  transFormededWhereClause->input = transformed;
                  selectNode->where_clause = nullptr;
                  transformed = transFormededWhereClause;
               }

               auto aggPipeNode = drv.nf.node<ast::PipeOperator>(selectNode->loc, ast::PipeOperatorType::AGGREGATE, context->currentScope->aggregationNode);
               auto transFormedAggregation = canonicalizeCast<ast::PipeOperator>(aggPipeNode, context);
               transFormedAggregation->input = transformed;
               transformed = transFormedAggregation;

               //Transform target selection
               auto select_list = selectNode->select_list;
               if (select_list) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(select_list->loc, ast::PipeOperatorType::SELECT, select_list);
                  auto transformedSelect = canonicalizeCast<ast::PipeOperator>(pipe, context);
                  transformedSelect->input = transformed;
                  transformed = transformedSelect;
                  selectNode->select_list = nullptr;
               }

               //Transform Group by
               if (selectNode->groups) {
                  auto loc = selectNode->groups->loc;
                  context->currentScope->aggregationNode->groupByNode = std::move(selectNode->groups);
               }

               if (selectNode->having) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(select_list->loc, ast::PipeOperatorType::WHERE, selectNode->having);
                  auto transformedHaving = canonicalizeCast<ast::PipeOperator>(pipe, context);
                  transformedHaving->input = transformed;
                  transformed = transformedHaving;
                  selectNode->having = nullptr;
               }

               //Transform modifiers
               for (auto modifier : selectNode->modifiers) {
                  auto transformedModifier = canonicalizeCast<ast::ResultModifier>(modifier, context);
                  transformedModifier->input = transformed;
                  transformed = transformedModifier;
               }
               selectNode->modifiers.clear();


               return transformed;
            }
            case ast::QueryNodeType::CTE_NODE: {
               auto cteNode = std::static_pointer_cast<ast::CTENode>(queryNode);
               if (cteNode->query) {

                  cteNode->query = canonicalizeCast<ast::TableProducer>(cteNode->query, context);

               }
               if (cteNode->child) {
                  context->pushNewScope();
                  cteNode->child = canonicalizeCast<ast::TableProducer>(cteNode->child, context);
                  context->popScope();
               }
               return cteNode;

            }
            case ast::QueryNodeType::SET_OPERATION_NODE: {
               std::shared_ptr<ast::TableProducer> transformed = nullptr;
               auto setOperationNode = std::static_pointer_cast<ast::SetOperationNode>(queryNode);
               context->pushNewScope();
               setOperationNode->left = canonicalize(setOperationNode->left, context);
               context->popScope();
               context->pushNewScope();
               setOperationNode->right = canonicalize(setOperationNode->right, context);
               context->popScope();
               transformed = setOperationNode;
               for (auto modifier : setOperationNode->modifiers) {
                  auto transformedModifier = canonicalizeCast<ast::ResultModifier>(modifier, context);
                  transformedModifier->input = transformed;
                  transformed = transformedModifier;
               }
               setOperationNode->modifiers.clear();
               return transformed;
            }
            default: error("Not implemented", queryNode->loc);
         }
      }
      case ast::NodeType::PIPE_OP: {
         auto pipeOp = std::static_pointer_cast<ast::PipeOperator>(rootNode);
         if (pipeOp->input) {
            pipeOp->input = canonicalize(pipeOp->input, context);
         }

         switch (pipeOp->pipeOpType) {
            case ast::PipeOperatorType::SELECT: {
               auto selectNode = std::static_pointer_cast<ast::TargetsExpression>(pipeOp->node);
               //Extract AggFunctions
               std::vector<std::shared_ptr<ast::ParsedExpression>> toRemove{};
               int i = 0;
               std::ranges::transform(selectNode->targets, selectNode->targets.begin(), [&](auto& target) {
                  return canonicalizeParsedExpression(target, context);
               });
               for (auto& target : selectNode->targets) {

                  if (target->exprClass == ast::ExpressionClass::FUNCTION) {
                     auto function = std::static_pointer_cast<ast::FunctionExpression>(target);
                     if (target->type == ast::ExpressionType::AGGREGATE) {
                        context->currentScope->aggregationNode->aggregations.push_back(function);
                     } else {
                        context->currentScope->extendNode->extensions.push_back(function);
                     }
                     //TODO better
                     if (function->alias.empty()) {
                        //TODO make unique alias
                        function->alias = function->functionName + "_" + std::to_string(i);
                     }
                     toRemove.emplace_back(target);
                     i++;
                  }
               }

               for (auto& target : toRemove) {
                  std::replace_if(selectNode->targets.begin(), selectNode->targets.end(), [&target](const auto& t) { return t == target; }, drv.nf.node<ast::ColumnRefExpression>(target->loc, target->alias));
               }

               return pipeOp;
            }
            case ast::PipeOperatorType::WHERE: {
               assert(pipeOp->node->nodeType == ast::NodeType::EXPRESSION);
               pipeOp->node = canonicalizeParsedExpression(std::static_pointer_cast<ast::ParsedExpression>(pipeOp->node), context);
               return pipeOp;
            }

            case ast::PipeOperatorType::RESULT_MODIFIER: {
               auto resultModifier = std::static_pointer_cast<ast::ResultModifier>(pipeOp->node);
               //TODO Support more complex modifiers
               resultModifier->input = pipeOp->input;
               return resultModifier;
            }
            case ast::PipeOperatorType::UNION:
            case ast::PipeOperatorType::UNION_ALL: {
               throw std::runtime_error("Not yet impleted. Transform into SetOperationNode");
            }


            default: return pipeOp;
         }

         return pipeOp;
      }
      case ast::NodeType::TABLE_REF: {
         auto tableRef = std::static_pointer_cast<ast::TableRef>(rootNode);
         switch (tableRef->type) {
            case ast::TableReferenceType::BASE_TABLE: {
               return tableRef;
            }
            case ast::TableReferenceType::JOIN: {
               auto joinRef = std::static_pointer_cast<ast::JoinRef>(tableRef);
               if (joinRef->left) {
                  joinRef->left = canonicalize(joinRef->left, context);
               }
               if (joinRef->right) {
                  joinRef->right = canonicalize(joinRef->right, context);
               }

               return tableRef;
            }
            case ast::TableReferenceType::SUBQUERY: {
               auto subquery = std::static_pointer_cast<ast::SubqueryRef>(tableRef);
               auto transformedSubSelectNode = canonicalize(subquery->subSelectNode, std::make_shared<ASTTransformContext>());

               subquery->subSelectNode = transformedSubSelectNode;
               return subquery;
            }
            case ast::TableReferenceType::EXPRESSION_LIST: {
               auto expressionListRef = std::static_pointer_cast<ast::ExpressionListRef>(tableRef);
               for (auto exprList : expressionListRef->values) {
                  std::ranges::transform(exprList, exprList.begin(), [&](auto& value) {
                     return canonicalizeParsedExpression(value, context);
                  });
               }
               return expressionListRef;
            }
            default: return tableRef;
         }
      }

      default:
         return rootNode;
   }
}

std::shared_ptr<ast::ParsedExpression> SQLCanonicalizer::canonicalizeParsedExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<ASTTransformContext> context) {
   switch (rootNode->exprClass) {
      case ast::ExpressionClass::SUBQUERY: {
         auto subqueryExpr = std::static_pointer_cast<ast::SubqueryExpression>(rootNode);
         subqueryExpr->subquery = canonicalizeCast<ast::TableProducer>(subqueryExpr->subquery, std::make_shared<ASTTransformContext>());
         return subqueryExpr;
      }
      case ast::ExpressionClass::OPERATOR: {
         auto operatorExpr = std::static_pointer_cast<ast::OperatorExpression>(rootNode);
         std::ranges::transform(operatorExpr->children, operatorExpr->children.begin(), [&](auto& child) {
            return canonicalizeParsedExpression(child, context);
         });
         return operatorExpr;
      }
      case ast::ExpressionClass::CONJUNCTION: {
         auto conjunctionExpr = std::static_pointer_cast<ast::ConjunctionExpression>(rootNode);

         std::ranges::transform(conjunctionExpr->children, conjunctionExpr->children.begin(), [&](auto& child) {
            return canonicalizeParsedExpression(child, context);
         });
         return conjunctionExpr;
      }
      case ast::ExpressionClass::COMPARISON: {
         auto comparisonExpr = std::static_pointer_cast<ast::ComparisonExpression>(rootNode);
         if (comparisonExpr->left) {
            comparisonExpr->left = canonicalizeParsedExpression(comparisonExpr->left, context);
         }
         std::ranges::transform(comparisonExpr->rightChildren, comparisonExpr->rightChildren.begin(), [&](auto& child) {
            return canonicalizeParsedExpression(child, context);
         });

         return comparisonExpr;
      }
      case ast::ExpressionClass::FUNCTION: {
         auto functionExpr = std::static_pointer_cast<ast::FunctionExpression>(rootNode);
         static int i = 0;
         if (functionExpr->type == ast::ExpressionType::AGGREGATE) {
            if (functionExpr->alias.empty()) {
               //TODO make unique alias
               functionExpr->alias = functionExpr->functionName + "_" + std::to_string(i);
               i++;
            }
            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(functionExpr->loc, functionExpr->alias);
            context->currentScope->aggregationNode->aggregations.push_back(functionExpr);

            return columnRef;

         }
         return functionExpr;
      }
      case ast::ExpressionClass::CASE: {
         auto caseExpr = std::static_pointer_cast<ast::CaseExpression>(rootNode);
         for (auto& caseCheck : caseExpr->caseChecks) {
            caseCheck.thenExpr = canonicalizeParsedExpression(caseCheck.thenExpr, context);
            caseCheck.whenExpr =canonicalizeParsedExpression(caseCheck.whenExpr, context);
         }
         caseExpr->elseExpr = canonicalizeParsedExpression(caseExpr->elseExpr, context);
         return caseExpr;

      }
      default: return rootNode;
   }
}

template <class T>
std::shared_ptr<T> SQLCanonicalizer::canonicalizeCast(std::shared_ptr<ast::TableProducer> rootNode, std::shared_ptr<ASTTransformContext> context) {
   return std::static_pointer_cast<T>(canonicalize(rootNode, context));
}



/*
 * SQLQueryAnalyzer
*/
SQLQueryAnalyzer::SQLQueryAnalyzer(std::shared_ptr<catalog::Catalog> catalog) : catalog(std::move(catalog)) {
}
std::shared_ptr<ast::AstNode> SQLQueryAnalyzer::canonicalizeAndAnalyze(std::shared_ptr<ast::AstNode> astRootNode, std::shared_ptr<SQLContext> context) {

   auto rootNode = std::dynamic_pointer_cast<ast::TableProducer>(astRootNode);
   if (!rootNode) {
      //RootNode is not a TableProducer
      switch (astRootNode->nodeType) {
         case ast::NodeType::CREATE_NODE: {
            auto createNode = std::static_pointer_cast<ast::CreateNode>(astRootNode);
            auto scope = context->createResolverScope();

            return analyzeCreateNode(createNode, context, scope);;

         }
         case ast::NodeType::INSERT_NODE: {
            auto insertNode = std::static_pointer_cast<ast::InsertNode>(astRootNode);

            context->pushNewScope();
            auto scope = context->createResolverScope();
            insertNode->producer = sqlCanonicalizer.canonicalize(insertNode->producer, std::make_shared<ASTTransformContext>());
            auto i = analyzeInsertNode(insertNode, context, scope);
            //context->popCurrentScope();
            return i;
         }
         default: throw std::runtime_error("Invalid root node type");
      }





   } else {
      //rootNode is a TableProducer
      auto transformed = sqlCanonicalizer.canonicalize(rootNode, std::make_shared<ASTTransformContext>());
      ast::NodeIdGenerator idGen{};
      std::cout << std::endl
                << std::endl;
      std::cout << "digraph ast {" << std::endl;
      std::cout << transformed->toDotGraph(1, idGen) << std::endl;
      std::cout << "}" << std::endl;
      context->pushNewScope();
      auto scope = context->createResolverScope();
      transformed = analyzeTableProducer(transformed, context, scope);
      return transformed;
   }
}
std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeTableProducer(std::shared_ptr<ast::TableProducer> rootNode, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   switch (rootNode->nodeType) {
      case ast::NodeType::PIPE_OP: {
         auto pipeOp = std::static_pointer_cast<ast::PipeOperator>(rootNode);
         if (pipeOp->input) {
            pipeOp->input = analyzeTableProducer(pipeOp->input, context, resolverScope);
         }
         return analyzePipeOperator(pipeOp, context, resolverScope);
      }
      case ast::NodeType::TABLE_REF: {
         auto tableRef = std::static_pointer_cast<ast::TableRef>(rootNode);
         return analyzeTableRef(tableRef, context, resolverScope);
      }
      case ast::NodeType::RESULT_MODIFIER: {
         auto resultModifier = std::static_pointer_cast<ast::ResultModifier>(rootNode);
         if (resultModifier->input) {
            resultModifier->input = analyzeTableProducer(resultModifier->input, context, resolverScope);
         }
         return analyzeResultModifier(resultModifier, context);
      }
      case ast::NodeType::QUERY_NODE: {
         auto queryNode = std::static_pointer_cast<ast::QueryNode>(rootNode);
         switch (queryNode->type) {
            case ast::QueryNodeType::CTE_NODE: {
               auto cteNode = std::static_pointer_cast<ast::CTENode>(queryNode);

               if (cteNode->query) {
                  ast::TargetInfo targetInfo{};
                  {
                     auto subQueryResolverScope = context->createResolverScope();
                     auto defineScope = context->createDefineScope();
                     context->pushNewScope();
                     auto subQueryScope = context->currentScope;
                     cteNode->query = analyzeTableProducer(cteNode->query, context, subQueryResolverScope);
                     targetInfo = context->currentScope->targetInfo;
                     auto evalBefore = context->currentScope->evalBeforeAggr;
                     context->popCurrentScope();

                     cteNode->subQueryScope = *subQueryScope;
                     std::vector<std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>>> renamedResults;
                     for (auto targetColumns : targetInfo.targetColumns) {
                        auto from = targetColumns;
                        auto to = std::make_shared<ast::NamedResult>(from->type, context->getUniqueScope(cteNode->alias), from->resultType, from->name);
                        to->displayName = from->displayName;
                        renamedResults.emplace_back(std::pair{from, to});
                     }
                     cteNode->renamedResults = std::move(renamedResults);

                     context->ctes.insert({cteNode->alias, {targetInfo, cteNode}});
                  }
               }
               if (cteNode->child) {
                  cteNode->child = analyzeTableProducer(cteNode->child, context, resolverScope);
               }
               return cteNode;
            }
            case ast::QueryNodeType::SET_OPERATION_NODE: {
               auto setOperationNode = std::static_pointer_cast<ast::SetOperationNode>(rootNode);
               std::shared_ptr<ast::TableProducer> boundLeft = nullptr;
               std::shared_ptr<ast::TableProducer> boundRight = nullptr;
               std::shared_ptr<SQLScope> leftScope, rightScope;
               {
                  auto subqueryResScope = context->createResolverScope();
                  auto defineScope = context->createDefineScope();
                  context->pushNewScope();
                  boundLeft = analyzeTableProducer(setOperationNode->left, context, subqueryResScope);
                  leftScope = context->currentScope;
                  context->popCurrentScope();
               }
               {
                  auto subqueryResScope = context->createResolverScope();
                  auto defineScope = context->createDefineScope();
                  context->pushNewScope();
                  boundRight = analyzeTableProducer(setOperationNode->right, context, subqueryResScope);
                  rightScope = context->currentScope;
                  context->popCurrentScope();
               }
               if (leftScope->targetInfo.targetColumns.size() != rightScope->targetInfo.targetColumns.size()) {
                  error("Left and right side must have the same number of columns" , rootNode->loc);
               }

               auto newScopeName = context->getUniqueScope("setop");
               std::vector<std::shared_ptr<ast::NamedResult>> newTargetInfos;
               for (size_t i = 0; i<leftScope->targetInfo.targetColumns.size(); i++) {
                  auto leftColumn = leftScope->targetInfo.targetColumns[i];
                  auto rightColumn = rightScope->targetInfo.targetColumns[i];
                  auto commonTypes = SQLTypeUtils::toCommonTypes(std::vector{leftColumn->resultType, rightColumn->resultType});
                  leftColumn->resultType = commonTypes[0];
                  rightColumn->resultType = commonTypes[1];
                  auto commonType = SQLTypeUtils::getCommonType(leftColumn->resultType, rightColumn->resultType);
                  auto newNamedResult = std::make_shared<ast::NamedResult>(leftColumn->type, newScopeName, commonType, leftColumn->name);


                  newNamedResult->displayName = leftColumn->displayName;
                  newTargetInfos.emplace_back(newNamedResult);

               }

               context->mapAttribute(resolverScope, setOperationNode->alias.empty() ? context->getUniqueScope("union") : setOperationNode->alias, newTargetInfos);
               context->currentScope->targetInfo.targetColumns = newTargetInfos;
               auto boundSetOperationNode = drv.nf.node<ast::BoundSetOperationNode>(setOperationNode->loc, setOperationNode->alias, setOperationNode->setType, setOperationNode->setOpAll, boundLeft, boundRight, leftScope, rightScope);
               return boundSetOperationNode;

              /* auto t = std::dynamic_pointer_cast<ast::TableProducer>(pipeOperator->node);
               if (t == nullptr) {
                  error("Pipe operator node for union is not a table producer", pipeOperator->loc);
               }
               auto unionContext = std::make_shared<SQLContext>();
               unionContext->catalog = context->catalog;
               unionContext->pushNewScope();
               auto scope = unionContext->createResolverScope();
               boundAstNode = analyzeTableProducer(t, unionContext, scope);
               auto rightTargetInfo = unionContext->currentScope->targetInfo;
               auto leftTargetInfo = context->currentScope->targetInfo;
               if (rightTargetInfo.targetColumns.size() != leftTargetInfo.targetColumns.size()) {
                  error("Right and left child of union must have same amount of columns", pipeOperator->loc);
               }
               auto newScope = context->getUniqueScope("setop");
               auto boundSetPipeOp = drv.nf.node<ast::BoundSetPipeOperator>(pipeOperator->loc, pipeOperator->pipeOpType, t, pipeOperator->input);
               boundSetPipeOp->rightScope = unionContext->currentScope;
               boundSetPipeOp->leftScope = context->currentScope;

               std::vector<std::shared_ptr<ast::NamedResult>> newTargetInfos;
               for (auto n : context->currentScope->targetInfo.targetColumns) {
                  auto newNamedResult = std::make_shared<ast::NamedResult>(n->type, newScope, n->resultType, n->name );
                  newNamedResult->displayName = n->displayName;
                  newTargetInfos.emplace_back(newNamedResult);
               }

               context->mapAttribute(resolverScope, pipeOperator->alias.empty() ? context->getUniqueScope("union") : pipeOperator->alias, newTargetInfos);
               context->currentScope->targetInfo.targetColumns = newTargetInfos;
               pipeOperator = boundSetPipeOp;*/





            }


            default: throw std::runtime_error("Not implemented");
         }
      }

      default: throw std::runtime_error("Not implemented");
   }
}

std::shared_ptr<ast::CreateNode> SQLQueryAnalyzer::analyzeCreateNode(std::shared_ptr<ast::CreateNode> createNode, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   switch (createNode->createInfo->type) {
      case ast::CatalogType::TABLE_ENTRY: {
         auto createTableInfo = std::static_pointer_cast<ast::CreateTableInfo>(createNode->createInfo);
         //TODO check
         if (catalog->getEntry(createTableInfo->tableName).has_value()) {
            error("Table " + createTableInfo->tableName + " already exists", createNode->loc);
         }

         std::vector<std::shared_ptr<ast::TableElement>> boundTableElements{};
         for (auto& tableElement : createTableInfo->tableElements) {
            switch (tableElement->type) {
               case ast::TableElementType::COLUMN: {
                  auto columnElement = std::static_pointer_cast<ast::ColumnElement>(tableElement);
                  if (columnElement->name.empty()) {
                     error("Column name cannot be empty", columnElement->loc);
                  }
                  std::vector<std::variant<size_t, std::string>> typeModifiers;
                  catalog::NullableType nullableType = SQLTypeUtils::typemodsToCatalogType(columnElement->logicalTypeWithMods.logicalType, columnElement->logicalTypeWithMods.typeModifiers);
                  nullableType.isNullable = true;
                  bool primary = false;
                  //TODO constraints
                  for (auto& constraint : columnElement->constraints) {
                     switch (constraint->type) {
                        case ast::ConstraintType::NOT_NULL: {
                           nullableType.isNullable = false;
                           break;
                        }
                        case ast::ConstraintType::UNIQUE: {
                           primary = true;
                           break;
                        }
                        default: error("Constraint type not implemented", constraint->loc);
                     }
                  }

                  auto boundColumnElement = std::make_shared<ast::BoundColumnElement>(columnElement->name, nullableType, primary);
                  boundTableElements.emplace_back(boundColumnElement);
                  break;
               }
               case ast::TableElementType::CONSTRAINT: {
                  auto tableConstraintElement = std::static_pointer_cast<ast::TableConstraintElement>(tableElement);
                  if (tableConstraintElement->constraint->type != ast::ConstraintType::UNIQUE || !std::static_pointer_cast<ast::UniqueConstraint>(tableConstraintElement->constraint)->isPrimaryKey) {
                     error("Unsupported TableConstraint constraint type", tableConstraintElement->loc);
                  }
                  boundTableElements.emplace_back(tableConstraintElement);
                  break;
               }
               default: error("TableElementType here not supported", tableElement->loc);
            }
         }

         createTableInfo->tableElements = std::move(boundTableElements);

         //TODO check for catalog and shema
         return createNode;
      }
      default: error("Not implemented", createNode->loc);
   }
}

std::shared_ptr<ast::BoundInsertNode> SQLQueryAnalyzer::analyzeInsertNode(std::shared_ptr<ast::InsertNode> insertNode, std::shared_ptr<SQLContext> context, SQLContext::ResolverScope& resolverScope) {
   auto maybeRel =context->catalog->getTypedEntry<catalog::TableCatalogEntry>(insertNode->tableName);
   if (!maybeRel.has_value()) {
      error("Table " << insertNode->tableName << " does not exist", insertNode->loc);
   }
   auto boundTableProducer = analyzeTableProducer(insertNode->producer, context, resolverScope);
   //TODO Maybe add BoundTableProducer which has a produced columns and their type
   if (boundTableProducer->nodeType != ast::NodeType::BOUND_TABLE_REF || std::static_pointer_cast<ast::BoundTableRef>(boundTableProducer)->type != ast::TableReferenceType::EXPRESSION_LIST) {
      error("Table producer type for insert node not yet supported", boundTableProducer->loc);
   }

   auto exprListTableRef = std::static_pointer_cast<ast::BoundExpressionListRef>(boundTableProducer);
   auto rel = maybeRel.value();
   std::unordered_map<std::string, catalog::NullableType>  allCollumnTypes;
   //Check for correct Type
   for (auto c: rel->getColumns()) {
     allCollumnTypes.emplace(c.getColumnName(), catalog::NullableType(c.getLogicalType(), c.getIsNullable()));
   }
   //TODO move loop up
   if (insertNode->columns.empty()) {
      for (auto c : rel->getColumns()) {
         insertNode->columns.emplace_back(c.getColumnName());
      }
   }


   return drv.nf.node<ast::BoundInsertNode>(insertNode->loc, insertNode->schema, insertNode->tableName,exprListTableRef, insertNode->columns,allCollumnTypes);

}
std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzePipeOperator(std::shared_ptr<ast::PipeOperator> pipeOperator, std::shared_ptr<SQLContext>& context, ResolverScope& resolverScope) {
   std::shared_ptr<ast::AstNode> boundAstNode = pipeOperator->node;
   switch (pipeOperator->pipeOpType) {
      case ast::PipeOperatorType::SELECT: {
         assert(pipeOperator->node->nodeType == ast::NodeType::EXPRESSION);
         auto targetSelection = std::static_pointer_cast<ast::TargetsExpression>(pipeOperator->node);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundTargetExpressions{};
         std::vector<std::shared_ptr<ast::NamedResult>> targetColumns{};
         for (auto& target : targetSelection->targets) {
            auto parsedExpression = analyzeExpression(target, context, resolverScope);
            switch (parsedExpression->exprClass) {
               case ast::ExpressionClass::BOUND_COLUMN_REF: {
                  //ADD column_ref to targetInfo for the current scope!
                  auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(parsedExpression);
                  targetColumns.emplace_back(columnRef->namedResult);
                  if (!columnRef->alias.empty()) {
                     context->mapAttribute(resolverScope, columnRef->alias, columnRef->namedResult);
                  }
                  context->currentScope->targetInfo.add(columnRef->namedResult);
                  break;
               }
               case ast::ExpressionClass::BOUND_STAR: {
                  auto star = std::static_pointer_cast<ast::BoundStarExpression>(parsedExpression);
                  std::vector<catalog::Catalog> catalogs;
                  std::string scope;
                  std::vector<catalog::Column> columns;
                  //TODO implement x.*
                  for (auto [scope, namedResult] : star->namedResults) {
                     targetColumns.emplace_back(namedResult);
                     context->currentScope->targetInfo.add(namedResult);
                  }

                  break;
               }
               case ast::ExpressionClass::BOUND_FUNCTION: {
                  error("Not implemented", target->loc);
                  auto function = std::static_pointer_cast<ast::BoundFunctionExpression>(parsedExpression);
                  auto fName = function->alias.empty() ? function->functionName : function->alias;
                  assert(function->functionInfo && function->namedResult.has_value());
                  context->currentScope->targetInfo.add(function->functionInfo);
                  break;
               }
               case ast::ExpressionClass::BOUND_CONSTANT:
               case ast::ExpressionClass::BOUND_OPERATOR:
               case ast::ExpressionClass::BOUND_CAST:
               case ast::ExpressionClass::BOUND_CASE: {
                  assert(parsedExpression->resultType.has_value());
                  auto scope = parsedExpression->alias.empty() ? parsedExpression->alias : createTmpScope();
                  auto n = std::make_shared<ast::NamedResult>(ast::NamedResultType::EXPRESSION, scope, parsedExpression->resultType.value(), createTmpScope());
                  n->displayName = parsedExpression->alias.empty() ? "" : parsedExpression->alias;
                  context->mapAttribute(resolverScope, parsedExpression->alias.empty() ? n->name : parsedExpression->alias, n);
                  targetColumns.emplace_back(n);
                  context->currentScope->targetInfo.add(n);
                  context->currentScope->evalBeforeAggr.emplace_back(parsedExpression);
                  parsedExpression->namedResult = n;
                  break;
               }
               default: error("Not implemented", target->loc);
            }
         }
         boundAstNode = drv.nf.node<ast::BoundTargetsExpression>(targetSelection->loc, targetSelection->alias,  boundTargetExpressions, targetColumns);
         break;
      }
      case ast::PipeOperatorType::WHERE: {
         //TODO check if correct
         auto whereResolverScope = context->createResolverScope();
         auto whereClause = std::static_pointer_cast<ast::ParsedExpression>(pipeOperator->node);
         boundAstNode = analyzeExpression(whereClause, context, whereResolverScope);
         if (std::static_pointer_cast<ast::BoundExpression>(boundAstNode)->resultType->type.getTypeId() != catalog::LogicalTypeId::BOOLEAN) {
            error("Where clause is not a boolean expression", whereClause->loc);
         }
         break;
      }
      case ast::PipeOperatorType::AGGREGATE: {
         auto aggregationNode = std::static_pointer_cast<ast::AggregationNode>(pipeOperator->node);

         //TODO parse aggregations sets
         std::vector<std::shared_ptr<ast::BoundExpression>> transFormedGroupExpressions{};
         if (aggregationNode->groupByNode) {
            std::ranges::transform(aggregationNode->groupByNode->group_expressions, std::back_inserter(transFormedGroupExpressions), [&](auto expr) {
               return analyzeExpression(expr, context, resolverScope);
            });
         }

         std::vector<std::shared_ptr<ast::BoundFunctionExpression>> boundAggregationExpressions{};

         std::ranges::transform(aggregationNode->aggregations, std::back_inserter(boundAggregationExpressions), [&](auto expr) {
            auto boundExpr = analyzeExpression(expr, context, resolverScope);
            assert(boundExpr->exprClass == ast::ExpressionClass::BOUND_FUNCTION);
            //boundExpr->namedResult = std::make_shared<ast::FunctionInfo>("boundExpr->scope", "boundExpr->alias", boundExpr->resultType.value());
            return std::static_pointer_cast<ast::BoundFunctionExpression>(boundExpr);
         });
         auto mapName = createMapName();
         auto boundGroupByNode = drv.nf.node<ast::BoundGroupByNode>(aggregationNode->groupByNode ? aggregationNode->groupByNode->loc : aggregationNode->loc, transFormedGroupExpressions);
         std::vector<std::shared_ptr<ast::BoundExpression>> toMap{};
         for (auto& aggr : boundAggregationExpressions) {
            if (aggr->arguments.empty() || aggr->arguments[0]->type == ast::ExpressionType::BOUND_COLUMN_REF) {
               continue;
            }
            toMap.emplace_back(aggr->arguments[0]);
            aggr->arguments[0]->alias = createTmpScope();
            aggr->arguments[0]->namedResult = std::make_shared<ast::NamedResult>(ast::NamedResultType::EXPRESSION, mapName, aggr->arguments[0]->resultType.value(), aggr->arguments[0]->alias);
         }
         //ADD to TargetInfo, see Google PIPE sql paper!
         //Maybe Not the best way!
         if (!aggregationNode->groupByNode || aggregationNode->groupByNode->group_expressions.empty()) {
            for (auto boundAggr : boundAggregationExpressions) {
               boundAggr->resultType->isNullable = true;
            }
         }


         boundAstNode = drv.nf.node<ast::BoundAggregationNode>(pipeOperator->loc,boundGroupByNode, boundAggregationExpressions, toMap, mapName );

         break;
      }
      case ast::PipeOperatorType::EXTEND: {
         assert(pipeOperator->node->nodeType == ast::NodeType::EXTEND_NODE);
         auto extendNode = std::static_pointer_cast<ast::ExtendNode>(pipeOperator->node);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundExtensions;
         std::ranges::transform(extendNode->extensions, std::back_inserter(boundExtensions), [&](auto& expr) {
            return analyzeExpression(expr, context, resolverScope);
         });
         boundAstNode =  drv.nf.node<ast::BoundExtendNode>(extendNode->loc, createMapName(), std::move(boundExtensions));
         break;
      }
      case ast::PipeOperatorType::UNION_ALL:
      case ast::PipeOperatorType::UNION: {
        error("Should not happen", pipeOperator->loc);

      }
      default: error("Not implemented", pipeOperator->loc);
   }
   pipeOperator->node = boundAstNode;
   return pipeOperator;
}

std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeTableRef(std::shared_ptr<ast::TableRef> tableRef, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   switch (tableRef->type) {
      case ast::TableReferenceType::BASE_TABLE: {
         auto baseTableRef = std::static_pointer_cast<ast::BaseTableRef>(tableRef);
         auto catalogEntry = catalog->getTypedEntry<catalog::TableCatalogEntry>(baseTableRef->tableName);
         //Add to current scope
         auto sqlScopeName = baseTableRef->alias.empty() ? baseTableRef->tableName : baseTableRef->alias;
         //Get unique mlirScope
         auto uniqueScope = context->getUniqueScope(sqlScopeName);
         if (!catalogEntry.has_value()) {

            //Check for cte
            if (context->ctes.contains(baseTableRef->tableName)) {
               auto [cteInfo, cteNode] = context->ctes.at(baseTableRef->tableName);


               std::vector<std::shared_ptr<ast::NamedResult>> namedResults{};

               std::ranges::transform(cteNode->renamedResults, std::back_inserter(namedResults), [&](auto& pair) {
                  auto namedResult = std::make_shared<ast::NamedResult>(pair.second->type, context->getUniqueScope(baseTableRef->tableName), pair.second->resultType, pair.second->name );
                  namedResult->displayName = pair.second->displayName;
                  return namedResult;
               });

               context->mapAttribute(resolverScope, sqlScopeName, namedResults);

               auto boundBaseTableRef = drv.nf.node<ast::BoundBaseTableRef>(baseTableRef->loc, namedResults, baseTableRef->alias, baseTableRef->tableName, uniqueScope);
               return boundBaseTableRef;
            } else {
               error("No Catalog found with name " + baseTableRef->tableName, baseTableRef->loc);
            }


         } else {

            auto namedResults = context->mapAttribute(resolverScope, sqlScopeName, uniqueScope, catalogEntry.value());

            auto boundBaseTableRef = drv.nf.node<ast::BoundBaseTableRef>(baseTableRef->loc, namedResults, baseTableRef->alias, catalogEntry.value()->getName(),uniqueScope);
            return boundBaseTableRef;
         }




         break;
      }
      case ast::TableReferenceType::JOIN: {

         auto join = std::static_pointer_cast<ast::JoinRef>(tableRef);
         switch (join->type) {
            case ast::JoinType::CROSS: {
               std::shared_ptr<ast::TableProducer> left, right;
               std::shared_ptr<SQLScope> leftScope, rightScope;


               std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> mapping{};
               if (join->left) {
                  context->pushNewScope();
                  auto leftResolverScope = context->createResolverScope();
                  auto defineScope = context->createDefineScope();
                  left = analyzeTableProducer(join->left, context, leftResolverScope);
                  auto localMapping = context->getTopDefinedColumns();
                  for (auto& [name, column] : localMapping) {
                     mapping.push_back({name, column});
                  }
                  leftScope = context->currentScope;
                  context->popCurrentScope();
               } else {
                  error("Left side of join is empty", join->loc);
               }
               if (join->right) {
                  context->pushNewScope();
                  auto rightResolverScope = context->createResolverScope();
                  auto defineScope = context->createDefineScope();
                  right = analyzeTableProducer(join->right, context, rightResolverScope);
                  auto localMapping = context->getTopDefinedColumns();
                  for (auto& [name, column] : localMapping) {
                     mapping.push_back({name, column});
                  }
                  rightScope = context->currentScope;
                  context->popCurrentScope();
               } else {
                  error("Right side of join is empty", join->loc);
               }

               for (auto& [name, column] : mapping) {
                  context->mapAttribute(resolverScope, name, column);
               }
               auto boundJoin = drv.nf.node<ast::BoundJoinRef>(join->loc, join->type, join->refType, left, right, nullptr);
               boundJoin->leftScope = leftScope;
               boundJoin->rightScope = rightScope;
               return boundJoin;

            }
            case ast::JoinType::RIGHT: {
               throw std::runtime_error("Should not happen");
            }
            case ast::JoinType::LEFT: {
               std::shared_ptr<ast::TableProducer> left, right;
               std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> mapping{};
               std::shared_ptr<SQLScope> leftScope, rightScope;
               {
                  context->pushNewScope();
                  left = analyzeTableProducer(join->left, context, resolverScope);
                  leftScope = context->currentScope;
                  context->popCurrentScope();
               }
               {
                  auto rightContext = std::make_shared<SQLContext>();
                  //Create new context
                  rightContext->pushNewScope();
                  rightContext->ctes = context->ctes;
                  auto rightResolverScope = rightContext->createResolverScope();
                  right = analyzeTableProducer(join->right, rightContext, rightResolverScope);
                  rightScope = rightContext->currentScope;
                  mapping = rightContext->getTopDefinedColumns();
               }

               for (auto x : mapping) {
                  context->mapAttribute(resolverScope, x.first, x.second);
               }
               std::shared_ptr<ast::BoundExpression> boundCondition;
               {
                  auto predScope = context->createResolverScope();
                  if (!std::holds_alternative<std::shared_ptr<ast::ParsedExpression>>(join->condition)) {
                     error("Not implemented", join->loc);
                  }
                  boundCondition = analyzeExpression(std::get<std::shared_ptr<ast::ParsedExpression>>(join->condition), context, resolverScope);
               }
               //TODO

               std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> outerJoinMapping;
               std::string outerjoinName;
               static size_t id = 0;
               if (!mapping.empty()) {
                  outerjoinName = "oj" + std::to_string(id++);
                  std::unordered_map<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>> remapped;
                  for (auto x : mapping) {
                     if (!remapped.contains(x.second)) {
                        auto scope = x.second->scope;
                        auto name = x.second->name;
                        auto namedResult = std::make_shared<ast::NamedResult>(x.second->type, outerjoinName, x.second->resultType, name);


                        //Make mapping output nullable
                        namedResult->resultType.isNullable = true;
                        outerJoinMapping.push_back({scope, namedResult});
                        remapped.insert({x.second, namedResult});
                        context->mapAttribute(resolverScope, x.first, namedResult);
                     }

                  }





               }


               auto boundJoin = drv.nf.node<ast::BoundJoinRef>(join->loc, join->type, join->refType, left, right, boundCondition);
               boundJoin->outerJoinMapping = outerJoinMapping;
               boundJoin->leftScope = leftScope;
               boundJoin->rightScope = rightScope;
               return boundJoin;

            }

               default: error("Join type not implemented", join->loc);

         }
         break;
      }
      case ast::TableReferenceType::SUBQUERY: {
         auto subquery = std::static_pointer_cast<ast::SubqueryRef>(tableRef);
         ast::TargetInfo targetInfo;
         std::shared_ptr<ast::TableProducer> t;
         std::shared_ptr<SQLScope> subQueryScope;
         std::vector<std::shared_ptr<ast::BoundExpression>> evalBefore;
         {
            auto subQueryResolverScope = context->createResolverScope();
            auto defineScope = context->createDefineScope();
            context->pushNewScope();
            subQueryScope = context->currentScope;
            t = analyzeTableProducer(subquery->subSelectNode, context, subQueryResolverScope);
            targetInfo = context->currentScope->targetInfo;
            evalBefore = context->currentScope->evalBeforeAggr;
            context->popCurrentScope();


         }


         for (auto target : targetInfo.targetColumns) {
            assert(!subquery->alias.empty());


            if (!target->displayName.empty()) {
               context->mapAttribute(resolverScope, subquery->alias + "." + target->displayName, target);
               context->mapAttribute(resolverScope,  target->displayName, target);
            } else {
               context->mapAttribute(resolverScope, subquery->alias  +"." + target->name, target);
               context->mapAttribute(resolverScope, target->name, target);
            }


         }

         return drv.nf.node<ast::BoundSubqueryRef>(subquery->loc, subQueryScope, t);

         break;
      }
      case ast::TableReferenceType::EXPRESSION_LIST: {
         auto expressionListRef = std::static_pointer_cast<ast::ExpressionListRef>(tableRef);
         if (expressionListRef->values.empty() || expressionListRef->values[0].empty()) {
            error("Expression list is empty", expressionListRef->loc);
         }

         std::vector<std::vector<std::shared_ptr<ast::BoundConstantExpression>>> boundValues{};
         size_t sizePerExprList = expressionListRef->values[0].size();
         std::vector<std::vector<catalog::NullableType>> types{sizePerExprList};

         for (auto exprList : expressionListRef->values) {
            if (exprList.size() != sizePerExprList) {
               error("All expression lists must have the same size", expressionListRef->loc);
            }
            std::vector<std::shared_ptr<ast::BoundConstantExpression>> boundExprList{};
            for (size_t i = 0; i< sizePerExprList; i++) {
               std::shared_ptr<ast::BoundExpression> boundExpr = analyzeExpression(exprList.at(i), context, resolverScope);
               if (boundExpr->exprClass != ast::ExpressionClass::BOUND_CONSTANT) {
                  error("Expression list must only contain constant expressions", exprList.at(i)->loc);
               }
               assert(boundExpr->resultType.has_value());
               types.at(i).push_back(boundExpr->resultType.value());
               boundExprList.emplace_back(std::static_pointer_cast<ast::BoundConstantExpression>(boundExpr));
            }
            boundValues.emplace_back(boundExprList);

         }
         std::vector<catalog::NullableType> commonTypes{};
         std::ranges::transform(types, std::back_inserter(commonTypes), [&](auto& typeList) {
            auto t =  SQLTypeUtils::getCommonBaseType(typeList);
            SQLTypeUtils::toCommonTypes(typeList);

            return t;
         });
         std::vector<std::shared_ptr<ast::NamedResult>> namedResults{};
         auto scope = createTmpScope();
         for (size_t i = 0; i < commonTypes.size(); i++) {
            auto name = "const_" + std::to_string(i);
            auto namedResult = std::make_shared<ast::NamedResult>(ast::NamedResultType::EXPRESSION, scope, commonTypes[i], name);
            namedResults.push_back(namedResult);
            context->currentScope->targetInfo.add(namedResult);

         }
         //TODO move to upper for loop
         context->mapAttribute(resolverScope, scope, namedResults);

         return drv.nf.node<ast::BoundExpressionListRef>(expressionListRef->loc, boundValues, namedResults);
      }

      default: error("Not implemented", tableRef->loc);
   }
}

std::shared_ptr<ast::BoundResultModifier> SQLQueryAnalyzer::analyzeResultModifier(std::shared_ptr<ast::ResultModifier> resultModifier, std::shared_ptr<SQLContext> context) {
   auto resolverScope = context->createResolverScope();
   switch (resultModifier->modifierType) {
      case ast::ResultModifierType::ORDER_BY: {
         auto orderByModifier = std::static_pointer_cast<ast::OrderByModifier>(resultModifier);
         std::vector<std::shared_ptr<ast::BoundOrderByElement>> boundOrderByElements{};
         for (auto orderByElement : orderByModifier->orderByElements) {
            if (orderByElement->expression) {
               auto boundExpression = analyzeExpression(orderByElement->expression, context, resolverScope);
               auto boundOrderByElement = drv.nf.node<ast::BoundOrderByElement>(orderByElement->loc, orderByElement->type, orderByElement->nullOrder, boundExpression);
               boundOrderByElements.push_back(boundOrderByElement);
            }
         }
         return drv.nf.node<ast::BoundOrderByModifier>(resultModifier->loc, boundOrderByElements, resultModifier->input);
      }
      case ast::ResultModifierType::LIMIT: {
         auto limitModifier = std::static_pointer_cast<ast::LimitModifier>(resultModifier);
         auto limitExpression = analyzeExpression(limitModifier->limitExpression, context, resolverScope);
         if (limitExpression->exprClass != ast::ExpressionClass::BOUND_CONSTANT) {
            error("Limit expression must be a constant expression", limitModifier->loc);
         }
         if (limitExpression->resultType->type.getTypeId() != catalog::LogicalTypeId::INT) {
            error("Limit expression must be of type INT", limitModifier->loc);
         }
         return drv.nf.node<ast::BoundLimitModifier>(limitModifier->loc, limitExpression, resultModifier->input);
      }
      default: error("Not implemented", resultModifier->loc);
   }
}

/*
 * Expressions
*/

std::shared_ptr<ast::BoundExpression> SQLQueryAnalyzer::analyzeExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   switch (rootNode->exprClass) {
      case ast::ExpressionClass::CONSTANT: {
         auto constExpr = std::static_pointer_cast<ast::ConstantExpression>(rootNode);
         if (!constExpr->value) {
            throw std::runtime_error("Value of constExpr is empty");
         }
         catalog::NullableType type = catalog::Type::int64();
         switch (constExpr->value->type) {
            case ast::ConstantType::INT:
               type = catalog::Type::int32();
               break;
            case ast::ConstantType::STRING: {
               auto strValue = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
               if (strValue.length() <= 8) {
                  type = catalog::Type::charType(strValue.length());
               } else {
                  type = catalog::Type::stringType();
               }
               break;
            }

            case ast::ConstantType::INTERVAL:
               //TODO hardcoded
               type = catalog::Type::intervalDaytime();
               break;
            case ast::ConstantType::FLOAT: {
               auto floatConstant = std::static_pointer_cast<ast::FloatValue>(constExpr->value);
               std::string value(floatConstant->fVal);
               auto decimalPos = value.find('.');
               if (decimalPos == std::string::npos) {
                  type = catalog::Type::int64();
               } else {
                  auto s = value.size() - decimalPos - 1;
                  auto p = value.size() - 1;
                  type = catalog::Type::decimal(p, s);
               }
               break;
            }
            case ast::ConstantType::NULL_P: {
               type = catalog::Type::noneType();
               type.isNullable = true;
               break;

            }
            default:
               error("Not implemented", constExpr->loc);
         }
         return drv.nf.node<ast::BoundConstantExpression>(constExpr->loc, type, constExpr->value, constExpr->alias);
      }
      case ast::ExpressionClass::COLUMN_REF: {
         auto columnRef = std::static_pointer_cast<ast::ColumnRefExpression>(rootNode);
         return analyzeColumnRefExpression(columnRef, context);
      }
      case ast::ExpressionClass::STAR: {
         auto star = std::static_pointer_cast<ast::StarExpression>(rootNode);
         std::vector<catalog::Catalog> catalogs;
         std::string relationName = star->relationName;
         std::vector<std::pair<std::string, catalog::Column>> columns{};
         auto topDefinedColumnsAll = context->getTopDefinedColumns();
         std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> topDefinedColumnsWithoutDuplicates;
         //TODO better solution!!!!
         for (auto [scope, namedResult] : topDefinedColumnsAll) {
            if (std::find_if(topDefinedColumnsWithoutDuplicates.begin(), topDefinedColumnsWithoutDuplicates.end(), [&](std::pair<std::string, std::shared_ptr<ast::NamedResult>> p) {
               return (p.first == scope || star->relationName.empty()) && p.second->name == namedResult->name;
            }) == topDefinedColumnsWithoutDuplicates.end()) {
               topDefinedColumnsWithoutDuplicates.emplace_back(std::pair{scope, namedResult});
            }
         }

         auto boundStar = drv.nf.node<ast::BoundStarExpression>(star->loc, relationName, topDefinedColumnsWithoutDuplicates);
         return boundStar;
      }
      case ast::ExpressionClass::COMPARISON: {
         auto comparison = std::static_pointer_cast<ast::ComparisonExpression>(rootNode);

         if (comparison->type != ast::ExpressionType::COMPARE_IN && comparison->type != ast::ExpressionType::COMPARE_NOT_IN) {
            if (comparison->rightChildren.size() != 1) {
               error("ComparisonExpression expects exactly one right child for type: " <<  std::to_string(static_cast<int>(comparison->type)), comparison->loc)
            }

         } else {


         }

         auto left = analyzeExpression(comparison->left, context, resolverScope);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundRightChildren{};
         std::ranges::transform(comparison->rightChildren, std::back_inserter(boundRightChildren), [&](auto& child) {
            return analyzeExpression(child, context, resolverScope);
         });
         if (!left->resultType.has_value()) {
            error("Left side of comparison is not a valid expression", comparison->left->loc);
         }
         std::vector<catalog::NullableType> types{};
         types.push_back(left->resultType.value());
         std::ranges::transform(boundRightChildren, std::back_inserter(types), [](auto& child) {
            return child->resultType.value();
         });

         auto commonTypes = SQLTypeUtils::toCommonTypes(types);
         left->resultType = commonTypes[0];
         size_t x = 1;
         for (auto boundChild : boundRightChildren) {
            boundChild->resultType = commonTypes[x];
            x++;
         }


         auto boundComparison = drv.nf.node<ast::BoundComparisonExpression>(comparison->loc, comparison->type, comparison->alias, left, boundRightChildren);
         return boundComparison;
      }
      case ast::ExpressionClass::CONJUNCTION: {
         auto conjunction = std::static_pointer_cast<ast::ConjunctionExpression>(rootNode);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundChildren{};
         for (auto expr : conjunction->children) {
            auto boundExpr = analyzeExpression(expr, context, resolverScope);
            boundChildren.push_back(boundExpr);
            if (!boundExpr->resultType.has_value() || boundExpr->resultType.value().type.getTypeId() != catalog::Type::boolean().getTypeId()) {
               error("Conjunction is not possible with children of type boolean", expr->loc);
            }
         }
         return drv.nf.node<ast::BoundConjunctionExpression>(conjunction->loc, conjunction->type, conjunction->alias, boundChildren);
         break;
      }
      case ast::ExpressionClass::OPERATOR: {
         auto operatorExpr = std::static_pointer_cast<ast::OperatorExpression>(rootNode);

         if (operatorExpr->children.size() == 0) {
            error("Operator expression has no children", operatorExpr->loc);
         }
         std::vector<std::shared_ptr<ast::BoundExpression>> boundChildren{};
         std::ranges::transform(operatorExpr->children, std::back_inserter(boundChildren), [&](auto c) {
            return analyzeExpression(c, context, resolverScope);
         });
         auto resultType = std::find_if(boundChildren.begin(), boundChildren.end(), [](auto c) {
            return !c->resultType.has_value();
         });
         if (resultType != boundChildren.end()) {
            error("Operator expression has children with different types", boundChildren[0]->loc);
         }
         //Get common type
         std::vector<catalog::NullableType> types{};
         std::ranges::transform(boundChildren, std::back_inserter(types), [](auto c) {
            return c->resultType.value();
         });

         auto commonNumbers = SQLTypeUtils::toCommonNumber(types);
         std::vector<catalog::NullableType> castValues{};
         std::ranges::transform(commonNumbers, std::back_inserter(castValues), [](auto c) {
            if (c.castType) {
               return *c.castType;
            }
            return c;

         });
         auto commonType = SQLTypeUtils::getCommonBaseType(castValues, operatorExpr->type);
         size_t t = 0;
         for (auto boundChild : boundChildren) {
            boundChild->resultType = commonNumbers[t];
            t++;
         }
         //TODO base
         //TODO hardcode

         return drv.nf.node<ast::BoundOperatorExpression>(operatorExpr->loc, operatorExpr->type, commonType, operatorExpr->alias, boundChildren);
      }
      case ast::ExpressionClass::FUNCTION: {
         auto function = std::static_pointer_cast<ast::FunctionExpression>(rootNode);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundArguments{};
         if (rootNode->type == ast::ExpressionType::AGGREGATE) {
            //TODO Better

            if (function->arguments.size() > 1) {
               error("Aggregation with more than one argument not supported", function->loc);
            }
            for (auto arg : function->arguments) {
               auto boundArg = analyzeExpression(arg, context, resolverScope);
               boundArguments.push_back(boundArg);

            }
            //TODO Check for correct value
            /**
             * SUM, AVG, MIN, MAX
             */
            if (function->functionName == "sum" || function->functionName == "avg" || function->functionName == "min" || function->functionName == "max") {
               if (function->arguments.size() > 1) {
                  error("Aggregation with more than one argument not supported", function->loc);
               }
               if (!boundArguments[0]->resultType.has_value() && !function->star) {
                  error("Argument of aggregation function is not a valid expression", boundArguments[0]->loc);
               }
               if (boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INT &&
                   boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::FLOAT &&
                   boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DECIMAL &&
                   boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DOUBLE) {
                  error("AVG function needs argument of type int or float", function->loc);
               }
               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               //Find correct resultType
               auto resultType = boundArguments[0]->resultType.value();
               /**
                * AVG
               */
               if (function->functionName == "avg") {
                  //TODO type
                  if (resultType.type.getTypeId() == catalog::LogicalTypeId::INT) {
                     resultType = SQLTypeUtils::getCommonTypeAfterOperation(catalog::Type::decimal(19,0), catalog::Type::decimal(19,0), ast::ExpressionType::OPERATOR_DIVIDE);
                  } else if (resultType.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
                     resultType = SQLTypeUtils::getCommonTypeAfterOperation(resultType, catalog::Type::decimal(19,0), ast::ExpressionType::OPERATOR_DIVIDE);
                  }
                  resultType.isNullable = true;
                  //else keep type

               }
               //TODO check if this line is needed
               resultType.isNullable = true;

               auto fInfo = std::make_shared<ast::FunctionInfo>(scope, fName, resultType);
               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);
               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments, fInfo);
            }
            //TODO better and cleaner!
            /*
             * COUNT
            */
            if (function->functionName == "count") {
               //TODO parse agrguments if not star!!

               if (function->arguments.size() > 1) {
                  error("Aggregation with more than one argument not supported", function->loc);
               }
               if (function->arguments.size() == 0 && !function->star) {
                  error("Argument of aggregation function is not a valid expression", boundArguments[0]->loc);
               }
               if (function->star) {
                  function->functionName = function->functionName+"*";
               }

               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               auto resultType = catalog::Type::int64();
               auto fInfo = std::make_shared<ast::FunctionInfo>(scope, fName, resultType);
               //TODO better
               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);
               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments, fInfo);
            }

            /*
             * STDDEV_SAMP
             */
            if (function->functionName == "stddev_samp") {
               catalog::NullableType resultType = catalog::Type::f64();
               if (boundArguments.size() != 1) {
                  error("Aggregation with more than one argument not supported", function->loc);
               }
               if (!boundArguments[0]->resultType.has_value() || !boundArguments[0]->resultType->isNumeric()) {
                  error("Argument of aggregation function has not a valid return type", boundArguments[0]->loc);
               }
               resultType.isNullable = boundArguments[0]->resultType->isNullable;

               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               auto fInfo = std::make_shared<ast::FunctionInfo>(scope, fName, resultType);

               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);

               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments, fInfo);



            }
            error("Not implemented", rootNode->loc);

         } else {
            //TODO hardcoded
            std::ranges::transform(function->functionName, function->functionName.begin(), ::toupper);
            if (function->functionName == "DATE") {
               if (function->arguments.size() != 1) {
                  error("Function date needs exactly one argument", function->loc);
               }
               auto arg = analyzeExpression(function->arguments[0], context, resolverScope);
               if (arg->resultType.has_value() && arg->resultType.value().type.getTypeId() != catalog::Type::stringType().getTypeId()) {
                  error("Function date needs argument of type string", function->loc);
               }
               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               auto resultType = catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY));
               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, "", function->alias, function->distinct, std::vector{arg}, nullptr);
            }
            if (function->functionName == "COUNT") {
               if (function->arguments.size() != 1 && !function->star) {
                  error("Function count needs exactly one argument", function->loc);
               }
               if (!function->star && (function->arguments[0]->type != ast::ExpressionType::COLUMN_REF && function->arguments[0]->type != ast::ExpressionType::STAR)) {
                  error("Function count needs argument of type column or star", function->loc);
               }
               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               auto fInfo =  std::make_shared<ast::FunctionInfo>(scope, fName, boundArguments[0]->resultType.value());

               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);
               auto resultType = catalog::Type::int64();
               if (function->star) {
                  return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType , function->functionName + "*", "", function->alias, function->distinct, std::vector<std::shared_ptr<ast::BoundExpression>>{}, fInfo);
               }
               auto arg = analyzeExpression(function->arguments[0], context, resolverScope);

               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, "", function->alias, function->distinct, std::vector{arg}, fInfo);
            }
            if (function->functionName == "EXTRACT") {
               if (function->arguments.size() != 2) {
                  error("Function extract needs exactly two arguments", function->loc);
               }
               auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
               auto arg2 = analyzeExpression(function->arguments[1], context, resolverScope);
               if (arg2->resultType.has_value() && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DATE && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INTERVAL) {
                  error("Function extract needs second argument of type date or interval", function->loc);
               }
               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               auto resultType = catalog::Type::int64();

               auto fInfo =  std::make_shared<ast::FunctionInfo>(scope, fName, resultType);
               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);
               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1, arg2}, fInfo);

            }
            if (function->functionName == "SUBSTRING" || function->functionName == "SUBSTR") {
               if (function->arguments.size() < 1 && function->arguments.size() >= 4) {
                  error("Function extract needs one,two or three arguments", function->loc);
               }
               auto stringArg = analyzeExpression(function->arguments[0], context, resolverScope);
               auto fromArg = function->arguments[1] ? analyzeExpression(function->arguments[1], context, resolverScope) : nullptr;
               auto forArg = function->arguments[2] ? analyzeExpression(function->arguments[2], context, resolverScope) : nullptr;

               if (!stringArg->resultType.has_value() || stringArg->resultType->type.getTypeId() != catalog::LogicalTypeId::STRING) {
                  error("The first argument of the SUBSTRING function must have a result type of STRING", stringArg->loc);
               }
               if (!fromArg->resultType.has_value() || fromArg->resultType->type.getTypeId() != catalog::LogicalTypeId::INT) {
                  error("The second argument of the SUBSTRING function must have a result type of INT", fromArg->loc);
               }
               if (!forArg->resultType.has_value() || forArg->resultType->type.getTypeId() != catalog::LogicalTypeId::INT) {
                  error("The second argument of the SUBSTRING function must have a result type of INT", forArg->loc);
               }

               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               auto resultType = catalog::Type::stringType();

               auto fInfo =  std::make_shared<ast::FunctionInfo>(scope, fName, resultType);
               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);
               auto boundArgs = std::vector{stringArg};
               if (fromArg) {
                  boundArgs.emplace_back(fromArg);
               }
               if (forArg) {
                  boundArgs.emplace_back(forArg);
               }
               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs, fInfo);

            }
            if (function->functionName == "ROUND") {
               if (function->arguments.size() != 2) {
                  error("Function extract needs two arguments", function->loc);
               }
               auto numberArg = analyzeExpression(function->arguments[0], context, resolverScope);
               auto decimalsArg = analyzeExpression(function->arguments[1], context, resolverScope);
               if (decimalsArg->exprClass != ast::ExpressionClass::BOUND_CONSTANT) {
                  error("The second argument of the round function must be a constant", decimalsArg->loc);
               }
               if (decimalsArg->resultType.has_value() && decimalsArg->resultType->type.getTypeId() != catalog::LogicalTypeId::INT) {
                  error("The second argument of the round function must have a result type of INT", decimalsArg->loc);
               }
               if (!numberArg->resultType.has_value() || !numberArg->resultType->isNumeric() ) {
                  error("The first argument of the round function must have a numeric type" , numberArg->loc);
               }
               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               //Get resultType after round
               catalog::NullableType resultType = numberArg->resultType.value();
               resultType.isNullable = true;
               auto fInfo =  std::make_shared<ast::FunctionInfo>(scope, fName, resultType);
               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);
               auto boundArgs = std::vector{numberArg, decimalsArg};
               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs, fInfo);
            }
            if (function->functionName == "UPPER") {

               if (function->arguments.size() != 1) {
                  error("Function with more than one argument not supported", function->loc);
               }
               auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
               //TODO check for string
               if (!arg1->resultType.has_value()) {
                  error("Argument of aggregation function has not a valid return type", arg1->loc);
               }
               catalog::NullableType resultType = arg1->resultType.value();

               auto scope = createTmpScope();
               auto fName = function->alias.empty() ? function->functionName : function->alias;
               auto fInfo = std::make_shared<ast::FunctionInfo>(scope, fName, resultType);

               fInfo->displayName = function->alias;
               context->mapAttribute(resolverScope, fName, fInfo);

               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1}, fInfo);

            }
            error("Function '" << function->functionName << "' not implemented", function->loc);
         }
         break;
      }
      case ast::ExpressionClass::CAST: {
         auto castExpr = std::static_pointer_cast<ast::CastExpression>(rootNode);
         auto boundChild = analyzeExpression(castExpr->child, context, resolverScope);
         if (!castExpr->logicalType.has_value()) {
            error("Cast expression must have logicalType", castExpr->loc);
         }
         switch (castExpr->logicalType.value()) {
            case ast::DATE: {
               switch (boundChild->type) {
                  case ast::ExpressionType::VALUE_CONSTANT: {
                     auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(boundChild);
                     if (constExpr->value->type != ast::ConstantType::STRING) {
                        error("Cannot cast " + constExpr->value->toString() + " to date", constExpr->loc);
                     }
                     std::string stringRep = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
                     stringRep += "days";
                     return drv.nf.node<ast::BoundCastExpression>(castExpr->loc, catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY)), castExpr->alias, boundChild, castExpr->logicalType, stringRep);
                  }
                  default: error("Cast not implemented", rootNode->loc);
               }
            }
            case ast::LogicalType::INTERVAL: {
               auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(boundChild);
               if (constExpr->value->type != ast::ConstantType::STRING) {
                  error("Cannot cast " + constExpr->value->toString() + " to date", constExpr->loc);
               }
               //TODO hardcoded
               //!Shortcutted here, implement different interval types later
               auto resultType = catalog::Type::intervalDaytime();
               std::string stringRepresentation = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
               if (castExpr->optInterval.has_value()) {
                  switch (castExpr->optInterval.value()) {
                     case ast::LogicalType::YEARS: {
                        resultType = catalog::Type::intervalMonths();
                        stringRepresentation += "years";
                        break;
                     }
                     default: stringRepresentation += "days";
                  }
               } else {
                  if (stringRepresentation.ends_with("years")) {
                     resultType = catalog::Type::intervalMonths();
                  } else if (stringRepresentation.ends_with("days")) {

                  } else {
                     error("Cast for strRep: " << stringRepresentation << " not implemented", boundChild->loc);
                  }

               }
               auto boundCast = drv.nf.node<ast::BoundCastExpression>(castExpr->loc, resultType, castExpr->alias, boundChild, castExpr->logicalType, stringRepresentation);

               return boundCast;
            }
            default: error("Cast not implemented", rootNode->loc);
         }
      }
      case ast::ExpressionClass::BETWEEN: {
         auto between = std::static_pointer_cast<ast::BetweenExpression>(rootNode);
         auto boundInput = analyzeExpression(between->input, context, resolverScope);
         auto boundLower = analyzeExpression(between->lower, context, resolverScope);
         auto boundUpper = analyzeExpression(between->upper, context, resolverScope);
         if (!boundInput->resultType.has_value() || !boundLower->resultType.has_value() || !boundUpper->resultType.has_value()) {
            error("Between expression has no valid type", rootNode->loc);
         }
         //Check for correct Types
         auto commonTypes = SQLTypeUtils::toCommonTypes({boundInput->resultType.value(), boundLower->resultType.value(), boundUpper->resultType.value()});
         boundInput->resultType = commonTypes[0];
         boundLower->resultType = commonTypes[1];
         boundUpper->resultType = commonTypes[2];
         return drv.nf.node<ast::BoundBetweenExpression>(rootNode->loc, between->type, catalog::Type::boolean(), rootNode->alias, boundInput, boundLower, boundUpper);
      }
      case ast::ExpressionClass::SUBQUERY: {
         auto subqueryExpr = std::static_pointer_cast<ast::SubqueryExpression>(rootNode);
         if (subqueryExpr->subQueryType == ast::SubqueryType::INVALID) {
            error("Should not happen, subquery type is invalid", subqueryExpr->loc);
         }
         std::shared_ptr<ast::TableProducer> boundSubquery;
         std::shared_ptr<SQLScope> subqueryScope;
         ast::TargetInfo subqueryTargetInfo;
         {
            auto subqueryResolver = context->createResolverScope();
            auto subqueryDefineScope = context->createDefineScope();
            context->pushNewScope();
            subqueryScope = context->currentScope;
            boundSubquery = analyzeTableProducer(subqueryExpr->subquery, context, subqueryResolver);
            subqueryTargetInfo = context->currentScope->targetInfo;
            context->popCurrentScope();
         }
         //TODO check if subquery is a valid expression and has valid type
         if (subqueryTargetInfo.targetColumns.size() != 1 && subqueryExpr->subQueryType != ast::SubqueryType::EXISTS && subqueryExpr->subQueryType != ast::SubqueryType::NOT_EXISTS ) {
            error("subquery expressions must produce a single value", subqueryExpr->loc);
         }

         auto namedResult = subqueryTargetInfo.targetColumns[0];
         auto resultType = namedResult->resultType;

         if (subqueryExpr->subQueryType != ast::SubqueryType::SCALAR) {
            resultType = catalog::Type::boolean();
         }
         std::shared_ptr<ast::BoundExpression> boundToTestExpr = nullptr;
         if (subqueryExpr->testExpr) {
            boundToTestExpr = analyzeExpression(subqueryExpr->testExpr, context, resolverScope);
         }

         return drv.nf.node<ast::BoundSubqueryExpression>(subqueryExpr->loc, subqueryExpr->subQueryType, resultType, subqueryExpr->alias, namedResult, subqueryScope, boundSubquery, boundToTestExpr);

      }
      case ast::ExpressionClass::CASE: {
         auto caseExpr = std::static_pointer_cast<lingodb::ast::CaseExpression>(rootNode);
         if (!caseExpr->elseExpr) {
            error("Case expr must have else clause", caseExpr->loc);
         }
         if (caseExpr->caseChecks.empty()) {
            error("Case expression does not have any case checks", caseExpr->loc);
         }
         std::vector<ast::BoundCaseExpression::BoundCaseCheck> boundCaseChecks;
         std::vector<catalog::NullableType> thenTypes{};
         std::ranges::transform(caseExpr->caseChecks, std::back_inserter(boundCaseChecks), [&](ast::CaseExpression::CaseCheck& caseCheck) {
            if (!caseCheck.thenExpr || !caseCheck.whenExpr) {
               error("Should not happen", caseExpr->loc);
            }

            ast::BoundCaseExpression::BoundCaseCheck boundCheck{analyzeExpression(caseCheck.whenExpr, context, resolverScope), analyzeExpression(caseCheck.thenExpr, context, resolverScope)};
            //TODO check correct type of when clause
            if (!boundCheck.thenExpr->resultType.has_value()) {
               error("Then expression has invalid type", boundCheck.thenExpr->loc);
            }
            thenTypes.emplace_back(boundCheck.thenExpr->resultType.value());
            return boundCheck;
         });
         std::shared_ptr<ast::BoundExpression> boundElse = analyzeExpression(caseExpr->elseExpr, context, resolverScope);
         if (!boundElse->resultType.has_value()) {
            error("Else has invalid type", boundElse->loc);
         }
         thenTypes.emplace_back(boundElse->resultType.value());
         //Find common then type
         auto commonType = SQLTypeUtils::toCommonTypes(thenTypes);
         auto resultType = SQLTypeUtils::getCommonBaseType(thenTypes);

         return drv.nf.node<ast::BoundCaseExpression>(caseExpr->loc,resultType, caseExpr->alias, boundCaseChecks, boundElse );
      }
      default: error("Expression type not implemented", rootNode->loc);
   }
}

std::shared_ptr<ast::BoundColumnRefExpression> SQLQueryAnalyzer::analyzeColumnRefExpression(std::shared_ptr<ast::ColumnRefExpression> columnRef, std::shared_ptr<SQLContext> context) {
   //new implementation which uses the new concept of TableProducers
   auto columnName = columnRef->column_names.size() == 1 ? columnRef->column_names[0] : columnRef->column_names[1];

   std::string scope;
   std::shared_ptr<ast::NamedResult> found;
   if (columnRef->column_names.size() == 2) {
      found = context->getNamedResultInfo(columnRef->loc, columnRef->column_names[0]+ "." + columnRef->column_names[1]);

   } else if (columnRef->column_names.size() == 1) {

      found = context->getNamedResultInfo(columnRef->loc, columnRef->column_names[0]);
   } else {
      throw std::runtime_error("Not implemented");
   }

   if (!found) {
      error("Column not found", columnRef->loc);
   }
   found->displayName = columnRef->alias.empty() ? found->displayName : columnRef->alias;
   return drv.nf.node<ast::BoundColumnRefExpression>(columnRef->loc,found->scope, found->resultType, found, columnRef->alias);
}















//TODO cleanup
/*
 * SQLTypeUtils
 */
catalog::NullableType SQLTypeUtils::getCommonType(catalog::NullableType nullableType1, catalog::NullableType nullableType2) {
   auto type1 = nullableType1.type;
   auto type2 = nullableType2.type;
   catalog::Type commonType = type1;
   if (type1.getTypeId() == type2.getTypeId()) {
      if (type1.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
         //Get higher decimal
         return getHigherDecimalType(nullableType1, nullableType2);
      }
   } else if (type1.getTypeId() == catalog::LogicalTypeId::DATE && type2.getTypeId() == catalog::LogicalTypeId::STRING) {
      commonType =type1;
   } else if (type1.getTypeId() == catalog::LogicalTypeId::STRING && type2.getTypeId() == catalog::LogicalTypeId::DATE) {
      commonType = type2;
   } else if (type1.getTypeId() == catalog::LogicalTypeId::DATE && type2.getTypeId() == catalog::LogicalTypeId::INTERVAL || type1.getTypeId() == catalog::LogicalTypeId::INTERVAL && type2.getTypeId() == catalog::LogicalTypeId::DATE) {
      return catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY));
   } else if (type1.getTypeId() == catalog::LogicalTypeId::INT && type2.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
      commonType = type2;
   } else if (type1.getTypeId() == catalog::LogicalTypeId::DECIMAL && type2.getTypeId() == catalog::LogicalTypeId::INT) {
      commonType = type1;
   } else if (type1.getTypeId() == catalog::LogicalTypeId::CHAR && type2.getTypeId() == catalog::LogicalTypeId::STRING) {
      commonType = type2;
   } else if (type2.getTypeId() == catalog::LogicalTypeId::CHAR && type1.getTypeId() == catalog::LogicalTypeId::STRING) {
      commonType = type1;
   } else if (type2.getTypeId() == catalog::LogicalTypeId::NONE ) {
      commonType = type1;
   } else if (type1.getTypeId() == catalog::LogicalTypeId::NONE ) {
      commonType = type2;
   } else {
      throw std::runtime_error("No common type found for " + type1.toString() + " and " + type2.toString());
   }
   //TODO is this correct
   return catalog::NullableType(commonType, nullableType1.isNullable || nullableType2.isNullable);


}

catalog::NullableType SQLTypeUtils::getHigherDecimalType(catalog::NullableType left, catalog::NullableType right) {
   assert(left.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && right.type.getTypeId() == catalog::LogicalTypeId::DECIMAL);
   auto leftInfo = left.type.getInfo<catalog::DecimalTypeInfo>();
   auto rightInfo = right.type.getInfo<catalog::DecimalTypeInfo>();
   int hidig = std::max(leftInfo->getPrecision() - leftInfo->getScale(), rightInfo->getPrecision() - rightInfo->getScale());
   int maxs = std::max(leftInfo->getScale(), rightInfo->getScale());
   return catalog::NullableType(catalog::Type::decimal(hidig+maxs, maxs), left.isNullable || right.isNullable);
}


catalog::NullableType SQLTypeUtils::getCommonBaseType(std::vector<catalog::NullableType> types) {
   auto commonType = types.front();
   for (size_t i = 1; i < types.size(); ++i) {
      commonType = getCommonType(commonType, types[i]);
   }
   return commonType;
}

catalog::NullableType SQLTypeUtils::getCommonBaseType(std::vector<catalog::NullableType> types, ast::ExpressionType operationType) {
   auto commonType = types.front();
   for (size_t i = 1; i < types.size(); ++i) {
      commonType = getCommonTypeAfterOperation(commonType, types[i], operationType);
   }
   return commonType;
}

catalog::NullableType SQLTypeUtils::getCommonTypeAfterOperation(catalog::NullableType type1, catalog::NullableType type2, ast::ExpressionType operationType) {
   auto commonType = getCommonType(type1, type2);

   //Maybe the other way arround
   switch (operationType) {
      case ast::ExpressionType::OPERATOR_DIVIDE: {
         if (type1.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && type2.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            auto type1Info = type1.type.getInfo<catalog::DecimalTypeInfo>();
            auto type2Info = type2.type.getInfo<catalog::DecimalTypeInfo>();



            auto [p , s] = getAdaptedDecimalPAndSAfterMulDiv(type1Info->getPrecision() - type1Info->getScale() + type2Info->getScale() + std::max<unsigned long>(6, type1Info->getScale() + type2Info->getPrecision()), std::max<unsigned long>(6, type1Info->getScale() + type2Info->getPrecision()));


            return catalog::NullableType(catalog::Type::decimal(p,s), commonType.isNullable );
         }

         return commonType;

      }
      case ast::ExpressionType::OPERATOR_TIMES: {
         if (type1.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && type2.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            auto type1Info = type1.type.getInfo<catalog::DecimalTypeInfo>();
            auto type2Info = type2.type.getInfo<catalog::DecimalTypeInfo>();
            auto [p,s] = getAdaptedDecimalPAndSAfterMulDiv(type1Info->getPrecision() + type2Info->getPrecision(), type1Info->getScale() + type2Info->getScale());
            return catalog::NullableType(catalog::Type::decimal(p, s), commonType.isNullable);
         }


         return commonType;




      }
      default: return commonType;
   }
}


std::vector<catalog::NullableType> SQLTypeUtils::toCommonTypes(std::vector<catalog::NullableType> types) {
   auto commonType = getCommonBaseType(types);
   std::vector<catalog::NullableType> res;
   for (auto type : types) {
      type.castType = std::make_shared<catalog::NullableType>(commonType);
      type.castType->isNullable = type.isNullable;
      res.push_back(type);
   }

   return res;
}

std::vector<catalog::NullableType> SQLTypeUtils::toCommonNumber(std::vector<catalog::NullableType> types) {
   auto anyDecimal = llvm::any_of(types, [](catalog::NullableType type) {return type.type.getTypeId() == catalog::LogicalTypeId::DECIMAL;});
   auto anyFloat = llvm::any_of(types, [](catalog::NullableType type) { return type.type.getTypeId() == catalog::LogicalTypeId::FLOAT; });
   if (anyDecimal && !anyFloat) {
      std::vector<catalog::NullableType> res;
      for (auto type: types) {
         if (type.type.getTypeId() != catalog::LogicalTypeId::DECIMAL) {
            type.castType = std::make_shared<catalog::NullableType>(catalog::Type::decimal(19,0));
            res.push_back(type);

         } else {
            res.push_back(type);
         }
      }
      return res;
   }

   return toCommonTypes(types);

}

std::pair<unsigned long, unsigned long> SQLTypeUtils::getAdaptedDecimalPAndSAfterMulDiv(unsigned long p, unsigned long s) {
   auto beforeComma = p - s;

   if (beforeComma > 32 && s > 6) {
      p = 38;
      s = 6;
   } else if (beforeComma > 32 && s <= 6) {
      p = 38;
   } else {
      p = std::min<unsigned long>(p, 38);
      s = std::min<unsigned long>(s, 38 - beforeComma);
   }
   return {p, s};
}

catalog::NullableType SQLTypeUtils::typemodsToCatalogType(ast::LogicalType logicalType, std::vector<std::shared_ptr<ast::Value>>& typeModifiers) {
   switch (logicalType) {
      case ast::LogicalType::INT: {
         return catalog::Type::int32();
      }
      case ast::LogicalType::BIGINT: {
         return catalog::Type::int64();
      }
      case ast::LogicalType::SMALLINT: {
         return catalog::Type::int8();
      }
      case ast::LogicalType::BOOLEAN: {
         return catalog::Type::boolean();
      }
      case ast::LogicalType::STRING: {
         return catalog::Type::stringType();
      }
      case ast::LogicalType::CHAR: {
         if (typeModifiers.size() == 0) {
            return catalog::Type::charType(1);
         }
         if (typeModifiers.size()> 1 || typeModifiers[0]->type != ast::ConstantType::UINT) {
            throw std::runtime_error("Invalid Typemodfiers for type: char");
         }
         size_t l = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifiers[0])->iVal;
         if (l <= 8) {
            return catalog::Type::charType(l);
         }
         return catalog::Type::stringType();
      }
      case ast::LogicalType::DECIMAL: {

         if (typeModifiers.size() != 2 || typeModifiers[0]->type != ast::ConstantType::UINT || typeModifiers[1]->type != ast::ConstantType::UINT) {
            throw std::runtime_error("Invalid Typemodfiers for type: descimal");
         }
         size_t p = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifiers[0])->iVal;
         size_t s = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifiers[1])->iVal;

         return catalog::Type::decimal(p,s);

      }
      case ast::LogicalType::DATE: {
         return catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY));
      }
      case ast::LogicalType::TIMESTAMP: {
         return catalog::Type::timestamp();
      }
      case ast::LogicalType::FLOAT4: {
         return catalog::Type::f32();
      }
      case ast::LogicalType::FLOAT8: {
         return catalog::Type::f64();
      }
      default: throw std::runtime_error("Not implemented typeMods");
   }
}

} // namespace lingodb::analyzer