#include "lingodb/compiler/frontend/sql_analyzer.h"

#include "lingodb/compiler/frontend/ast/bound/bound_aggregation.h"
#include "lingodb/compiler/frontend/ast/bound/bound_create_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_extend_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_groupby.h"
#include "lingodb/compiler/frontend/ast/bound/bound_insert_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_pipe_operator.h"
#include "lingodb/compiler/frontend/ast/bound/bound_query_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_tableref.h"
#include <boost/context/fiber_fcontext.hpp>
#include <cctype>
#include <chrono>
#include <ranges>
#include <sys/resource.h>
namespace lingodb::analyzer {
using ResolverScope = llvm::ScopedHashTable<std::string, std::shared_ptr<ast::NamedResult>, StringInfo>::ScopeTy;

StackGuardNormal::StackGuardNormal() {
   rlimit rlp{};
   auto suc = getrlimit(RLIMIT_STACK, &rlp);
   if (suc != 0) {
      limit = 0;
   }
   limit = 0.095*rlp.rlim_cur;
   startFameAddress = __builtin_frame_address(0);
}
void StackGuardNormal::reset() {
   startFameAddress = __builtin_frame_address(0);
}
bool StackGuardNormal::newStackNeeded() {
   rlimit rlp{};
   auto suc = getrlimit(RLIMIT_STACK, &rlp);
   assert(suc==0);
   void* currentFrameAddress = __builtin_frame_address(0);
   size_t size = reinterpret_cast<size_t>(startFameAddress) - reinterpret_cast<size_t>(currentFrameAddress);

   if (size > limit) {
      std::cerr << "StackLimit: " << rlp.rlim_cur << " Max: " << rlp.rlim_max << " recorded size: " << size << " Perc: " << ((size*1.0)/rlp.rlim_cur) * 100 << std::endl;
      return true;
   }
   return false;
}


StackGuardFiber::StackGuardFiber(boost::context::stack_context& stackContext) : StackGuard(), stackContext(stackContext) {
   startFameAddress = stackContext.sp;
   limit = stackContext.size*0.65;
}
void StackGuardFiber::reset() {
   startFameAddress = stackContext.sp;
}
bool StackGuardFiber::newStackNeeded() {
   void* currentFrameAddress = __builtin_frame_address(0);
   size_t size = reinterpret_cast<size_t>(startFameAddress) - reinterpret_cast<size_t>(currentFrameAddress);
   if (size > limit) {
      std::cerr << "Fiber: StackLimit: " << stackContext.size << " recorded size: " << size << " Perc: " << ((size*1.0)/stackContext.size) * 100 << std::endl;
      return true;
   }
   return false;
}

/*
* SQLCanonicalizer
*/
std::shared_ptr<ast::TableProducer> SQLCanonicalizer::canonicalize(std::shared_ptr<ast::TableProducer> rootNode, std::shared_ptr<ASTTransformContext> context) {
   switch (rootNode->nodeType) {
      case ast::NodeType::QUERY_NODE: {
         context->currentScope->classicalSelect = true;
         auto queryNode = std::static_pointer_cast<ast::QueryNode>(rootNode);
         switch (queryNode->type) {
            case ast::QueryNodeType::SELECT_NODE: {
               auto selectNode = std::static_pointer_cast<ast::SelectNode>(queryNode);
               std::shared_ptr<ast::TableProducer> transformed = nullptr;
               //Transform from_clause
               if (selectNode->from_clause) {
                  auto transformedFrom = selectNode->from_clause;

                  selectNode->from_clause = nullptr;
                  transformed = transformedFrom;
               }
               //Transform where_clause
               if (selectNode->where_clause) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(selectNode->where_clause->loc, ast::PipeOperatorType::WHERE, selectNode->where_clause);
                  pipe->input = transformed;
                  selectNode->where_clause = nullptr;
                  transformed = pipe;
               }


               //Transform Group by
               if (selectNode->groups) {
                  context->currentScope->aggregationNode->groupByNode = std::move(selectNode->groups);
               }
               auto aggPipeNode = drv.nf.node<ast::PipeOperator>(selectNode->loc, ast::PipeOperatorType::AGGREGATE, context->currentScope->aggregationNode);
               aggPipeNode->input = transformed;
               transformed = aggPipeNode;



               auto extendBeforeWindowPipeOp = drv.nf.node<ast::PipeOperator>(selectNode->select_list->loc, ast::PipeOperatorType::EXTEND, context->currentScope->extendNodeBeforeWindowFunctions);
               extendBeforeWindowPipeOp->input = transformed;
               transformed = extendBeforeWindowPipeOp;




               //Transform target selection
               if (selectNode->select_list) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(selectNode->select_list->loc, ast::PipeOperatorType::SELECT, selectNode->select_list);
                  for (auto& target : selectNode->select_list->targets) {
                     if (target->alias.empty()) {
                        continue;
                     }
                     context->currentScope->selectSymbolList.emplace(target->alias, target);
                  }
                  pipe->input = transformed;

                  transformed = pipe;
                  selectNode->select_list = nullptr;
               }

               if (selectNode->having) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(selectNode->having->loc, ast::PipeOperatorType::WHERE, selectNode->having);
                  pipe->input = transformed;

                  transformed = pipe;
                  selectNode->having = nullptr;
               }



               //Transform modifiers
               for (auto modifier : selectNode->modifiers) {
                  modifier->input = transformed;
                  auto transformedModifier = modifier;

                  transformed = transformedModifier;
               }
               selectNode->modifiers.clear();
               transformed = canonicalize(transformed, context);


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
            case ast::QueryNodeType::VALUES: {
               auto valuesNode = std::static_pointer_cast<ast::ValuesQueryNode>(queryNode);
               valuesNode->expressionListRef = canonicalizeCast<ast::ExpressionListRef>(valuesNode->expressionListRef, context);
               return valuesNode;
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
               auto extendNode = drv.nf.node<ast::ExtendNode>(selectNode->loc, true);
               auto extendPipeOp = drv.nf.node<ast::PipeOperator>(selectNode->loc, ast::PipeOperatorType::EXTEND, extendNode);
               //Extract AggFunctions
               std::vector<std::pair<std::string, std::shared_ptr<ast::ParsedExpression>>> toRemove{};
               int i = 0;
               //Canonicalize target expressions
               std::ranges::transform(selectNode->targets, selectNode->targets.begin(), [&](std::shared_ptr<ast::ParsedExpression>& target) {
                  return canonicalizeParsedExpression(target, context, true, extendNode);
               });
               //Canonicalize distinct expressions list
               if (selectNode->distinctExpressions.has_value()) {
                  std::ranges::transform(selectNode->distinctExpressions.value(), selectNode->distinctExpressions->begin(), [&](auto& target) {
                     return canonicalizeParsedExpression(target, context, true, extendNode);
                  });
               }
               extendPipeOp->input = pipeOp->input;
               pipeOp->input = extendPipeOp;


              break;
            }
            case ast::PipeOperatorType::WHERE: {
               assert(pipeOp->node->nodeType == ast::NodeType::EXPRESSION);
               pipeOp->node = canonicalizeParsedExpression(std::static_pointer_cast<ast::ParsedExpression>(pipeOp->node), context, false, nullptr);
               break;
            }
            case ast::PipeOperatorType::AGGREGATE: {
               auto aggNode = std::dynamic_pointer_cast<ast::AggregationNode>(pipeOp->node);
               auto extendNode = drv.nf.node<ast::ExtendNode>(aggNode->loc, true);
               auto extendPipeOp = drv.nf.node<ast::PipeOperator>(aggNode->loc, ast::PipeOperatorType::EXTEND, extendNode);
               assert(aggNode);
               std::vector<std::shared_ptr<ast::ParsedExpression>> newGroupByExpressions{};
               if (aggNode->groupByNode) {

                  if (aggNode->groupByNode->rollup) {
                     for (size_t i = 0; i <= aggNode->groupByNode->groupByExpressions.size(); i++) {
                        size_t n = aggNode->groupByNode->groupByExpressions.size() - i;
                        std::set<size_t> set;
                        for (size_t j = 0; j < n; j++) {
                           set.emplace(j);
                        }
                        aggNode->groupByNode->groupingSet.emplace_back(set);
                     }

                  }
                  for (auto e : aggNode->groupByNode->groupByExpressions) {

                     if (e->type == ast::ExpressionType::COLUMN_REF) {
                        auto columnRef = std::static_pointer_cast<ast::ColumnRefExpression>(e);
                        auto name = columnRef->columnNames.size() == 1 ? columnRef->columnNames[0] : columnRef->columnNames[0] + "." + columnRef->columnNames[1];
                        auto found = context->currentScope->selectSymbolList.find(name);
                        if (found != context->currentScope->selectSymbolList.end()) {
                           newGroupByExpressions.emplace_back(canonicalizeParsedExpression(found->second, context, true, extendNode));
                           context->currentScope->groupedByExpressions.emplace(found->second);
                           continue;
                        }


                     }
                     auto canonicalized = canonicalizeParsedExpression(e, context, true, extendNode);
                     newGroupByExpressions.emplace_back(canonicalized);
                     context->currentScope->groupedByExpressions.emplace(e);

                  }
                  aggNode->groupByNode->groupByExpressions = std::move(newGroupByExpressions);
               }
               extendPipeOp->input = pipeOp->input;
               pipeOp->input = extendPipeOp;

               break;
            }

            case ast::PipeOperatorType::RESULT_MODIFIER: {
               auto resultModifier = std::static_pointer_cast<ast::ResultModifier>(pipeOp->node);
               resultModifier->input = pipeOp->input;
               return resultModifier;
            }
            case ast::PipeOperatorType::EXTEND: {
               auto extendNode = std::static_pointer_cast<ast::ExtendNode>(pipeOp->node);
               auto extendNode2 = drv.nf.node<ast::ExtendNode>(extendNode->loc, true);
               auto extendPipeOp = drv.nf.node<ast::PipeOperator>(extendNode->loc, ast::PipeOperatorType::EXTEND, extendNode2);
               std::ranges::transform(extendNode->extensions, extendNode->extensions.begin(), [&](auto& expr) {
                  if (expr->exprClass == ast::ExpressionClass::WINDOW && context->currentScope->classicalSelect) {
                     return expr;
                  }
                  return canonicalizeParsedExpression(expr, context, false, extendNode2);
               });
               extendPipeOp->input = pipeOp->input;
               pipeOp->input = extendPipeOp;
               return pipeOp;


            }
            case ast::PipeOperatorType::JOIN: {
               assert(pipeOp->node->nodeType == ast::NodeType::TABLE_REF);
               auto join = std::static_pointer_cast<ast::JoinRef>(pipeOp->node);
               /**
                * Removing unnecessary PipeOperator wrapping and moving input to left side of the join
                */
               join->left = pipeOp->input;
               return join;
            }
            case ast::PipeOperatorType::SET_OPERATION: {
               auto setOperationNode = std::static_pointer_cast<ast::SetOperationNode>(pipeOp->node);
               assert(!setOperationNode->left);
               setOperationNode->right = canonicalize(setOperationNode->right, std::make_shared<ASTTransformContext>());
               setOperationNode->left = pipeOp->input;
               return setOperationNode;
            }
            case ast::PipeOperatorType::SET: {
               auto setExpression = std::static_pointer_cast<ast::SetExpression>(pipeOp->node);
               assert(setExpression->type == ast::ExpressionType::SET);
               std::ranges::transform(setExpression->sets, setExpression->sets.begin(), [&](auto& pair) {
                    return std::pair(pair.first, canonicalizeParsedExpression(pair.second, context, false, nullptr));
               });
               return pipeOp;
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
               if (joinRef->type == ast::JoinType::RIGHT) {
                  std::swap(joinRef->left, joinRef->right);
                  joinRef->type = ast::JoinType::LEFT;
               }
               //Erase last element
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
                     return canonicalizeParsedExpression(value, context, false, nullptr);
                  });
               }
               return expressionListRef;
            }
            default: return tableRef;
         }
      }
      case ast::NodeType::RESULT_MODIFIER: {
         auto resultModifier = std::static_pointer_cast<ast::ResultModifier>(rootNode);
         if (resultModifier->input)
            resultModifier->input = canonicalize(resultModifier->input, context);
         auto extendNode = drv.nf.node<ast::ExtendNode>(resultModifier->loc, true);
         auto extendPipeOp = drv.nf.node<ast::PipeOperator>(resultModifier->loc, ast::PipeOperatorType::EXTEND, extendNode);
         switch (resultModifier->modifierType) {
            case ast::ResultModifierType::ORDER_BY: {
               auto orderBy = std::static_pointer_cast<ast::OrderByModifier>(resultModifier);
               for (auto expr : orderBy->orderByElements) {
                  auto canonicalized = canonicalizeParsedExpression(expr->expression, context, expr->expression->exprClass != ast::ExpressionClass::CONSTANT, extendNode);
                  context->currentScope->groupedByExpressions.emplace(expr->expression);
                  expr->expression = canonicalized;
               }

               break;
            }
            default:;
         }
         extendPipeOp->input = resultModifier->input;
         resultModifier->input = extendPipeOp;

         return resultModifier;
      }
      default:
         return rootNode;
   }
}

std::shared_ptr<ast::ParsedExpression> SQLCanonicalizer::canonicalizeParsedExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<ASTTransformContext> context, bool extend, std::shared_ptr<ast::ExtendNode> extendNode ) {
   if (stackGuard->newStackNeeded()) {
      boost::context::fixedsize_stack salloc(1024*1024);
      boost::context::stack_context sctx = salloc.allocate();
      auto sGuard = stackGuard;
      stackGuard = std::make_shared<StackGuardFiber>(sctx);
      void * sp=static_cast<char*>(sctx.sp);
      std::size_t size=sctx.size;
      std::shared_ptr<ast::ParsedExpression> expression;

      boost::context::fiber f(std::allocator_arg, boost::context::preallocated(sp, size, sctx),salloc, [&](boost::context::fiber&& sink) {
         expression = canonicalizeParsedExpression(rootNode, context, extend, extendNode);
         return std::move(sink);
      });

      f = std::move(f).resume();
      stackGuard = sGuard;

      return expression;
   }
   static int i = 0;


   switch (rootNode->exprClass) {
      case ast::ExpressionClass::SUBQUERY: {
         auto subqueryExpr = std::static_pointer_cast<ast::SubqueryExpression>(rootNode);
         auto canacolizedSubquery = canonicalizeCast<ast::PipeOperator>(subqueryExpr->subquery, std::make_shared<ASTTransformContext>());
         subqueryExpr->subquery = canacolizedSubquery;
         assert(subqueryExpr->subquery->nodeType == ast::NodeType::PIPE_OP && "Should not happen");
         if (subqueryExpr->testExpr) {
           subqueryExpr->testExpr = canonicalizeParsedExpression(subqueryExpr->testExpr, context, true, extendNode);
         }

         std::string alias = subqueryExpr->alias.empty() ? "" : subqueryExpr->alias;


         if (extend) {
            auto find = context->currentScope->groupedByExpressions.find(subqueryExpr);
            if (find == context->currentScope->groupedByExpressions.end()) {
               if (subqueryExpr->alias.empty()) {
                  subqueryExpr->alias = "op_" + std::to_string(i);
               }
               i++;
               extendNode->extensions.push_back(subqueryExpr);
            } else {
               subqueryExpr->alias = find->get()->alias;
            }
            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(subqueryExpr->loc, subqueryExpr->alias);
            columnRef->alias = alias;
            return columnRef;
         }

         return subqueryExpr;
      }
      case ast::ExpressionClass::OPERATOR: {
         auto operatorExpr = std::static_pointer_cast<ast::OperatorExpression>(rootNode);

         std::ranges::transform(operatorExpr->children, operatorExpr->children.begin(), [&](auto& child) {
            return canonicalizeParsedExpression(child, context, false, extendNode);
         });
         std::string alias = operatorExpr->alias.empty() ? "" : operatorExpr->alias;
         if (extend) {
            auto find = context->currentScope->groupedByExpressions.find(operatorExpr);
            if (find == context->currentScope->groupedByExpressions.end()) {
               if (operatorExpr->alias.empty()) {
                  operatorExpr->alias = "op_" + std::to_string(i);
               }
               i++;
               extendNode->extensions.push_back(operatorExpr);
            } else {
               operatorExpr->alias = find->get()->alias;
            }

            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(operatorExpr->loc, operatorExpr->alias);
            columnRef->alias = alias;
            return columnRef;

         }
         return operatorExpr;

      }
      case ast::ExpressionClass::CONJUNCTION: {
         auto conjunctionExpr = std::static_pointer_cast<ast::ConjunctionExpression>(rootNode);

         std::vector<std::shared_ptr<ast::ParsedExpression>> combinedChildren;
         for (auto& child : conjunctionExpr->children) {
            child = canonicalizeParsedExpression(child, context, false, extendNode);
            combinedChildren.emplace_back(child);
         }
         conjunctionExpr->children = std::move(combinedChildren);

         if (extend) {
            error("Not implemented", conjunctionExpr->loc);
         }
         return conjunctionExpr;
      }
      case ast::ExpressionClass::COMPARISON: {
         auto comparisonExpr = std::static_pointer_cast<ast::ComparisonExpression>(rootNode);
         if (comparisonExpr->left) {
            comparisonExpr->left = canonicalizeParsedExpression(comparisonExpr->left, context, false, extendNode);
         }
         std::ranges::transform(comparisonExpr->rightChildren, comparisonExpr->rightChildren.begin(), [&](auto& child) {
            return canonicalizeParsedExpression(child, context, false, extendNode);
         });
         if (extend) {
            error("Not implemented", comparisonExpr->loc);
         }
         return comparisonExpr;
      }
      case ast::ExpressionClass::FUNCTION: {
         auto functionExpr = std::static_pointer_cast<ast::FunctionExpression>(rootNode);
         std::string columnAlias = functionExpr->alias.empty() ? functionExpr->functionName : functionExpr->alias;


         std::string alias = functionExpr->alias.empty() ? functionExpr->functionName : functionExpr->alias;
         if (functionExpr->type == ast::ExpressionType::AGGREGATE) {
            functionExpr->alias = functionExpr->functionName + "_" + std::to_string(i);
            i++;

            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(functionExpr->loc, functionExpr->alias);
            columnRef->alias = columnAlias;
            context->currentScope->aggregationNode->aggregations.push_back(functionExpr);


            return columnRef;

         } else {
            i++;
            std::string upperCaseName = functionExpr->functionName;
            std::ranges::transform(upperCaseName.begin(), upperCaseName.end(), upperCaseName.begin(), ::toupper);

            //Extract Grouping into aggregation node
            if (upperCaseName == "GROUPING") {

               auto find = context->currentScope->aggregationNode->groupByNode->groupingFunctions.find(functionExpr);

               if (find ==   context->currentScope->aggregationNode->groupByNode->groupingFunctions.end()) {
                  if (functionExpr->alias.empty()) {
                     functionExpr->alias = "grouping_" + std::to_string(i);
                  }
                  context->currentScope->aggregationNode->groupByNode->groupingFunctions.emplace(functionExpr);
               } else {
                  functionExpr->alias = find->get()->alias;
               }

               auto columnRef = drv.nf.node<ast::ColumnRefExpression>(functionExpr->loc, functionExpr->alias);
               columnRef->alias = columnAlias;

               return columnRef;
            }

            if (!extend || functionExpr->functionName == "GROUPING")
               return rootNode;
            auto find = context->currentScope->groupedByExpressions.find(functionExpr);
            if (find == context->currentScope->groupedByExpressions.end()) {
               if (functionExpr->alias.empty()) {
                  functionExpr->alias = functionExpr->functionName + "_" + std::to_string(i);
               }
               extendNode->extensions.push_back(functionExpr);
            } else {
               functionExpr->alias = find->get()->alias;
            }

            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(functionExpr->loc, functionExpr->alias);
            columnRef->alias = columnAlias;
            return columnRef;
         }

         return functionExpr;
      }
      case ast::ExpressionClass::WINDOW: {
         auto windowExpr = std::static_pointer_cast<ast::WindowExpression>(rootNode);
         auto extendNodeToUse = context->currentScope->classicalSelect ? context->currentScope->extendNodeBeforeWindowFunctions : extendNode;
         assert(windowExpr->functionExpression);

         if (windowExpr->filter) {
            windowExpr->filter = canonicalizeParsedExpression(windowExpr->filter, context, true, extendNode);
         }
         std::ranges::transform(windowExpr->functionExpression->arguments, windowExpr->functionExpression->arguments.begin(), [&](auto& arg) {
            return canonicalizeParsedExpression(arg, context, true, extendNodeToUse);
         });

         std::ranges::transform(windowExpr->partitions, windowExpr->partitions.begin(), [&](auto& partition) {
            return canonicalizeParsedExpression(partition, context, true, extendNodeToUse);
         });
         if (windowExpr->order.has_value()) {
            std::ranges::transform(windowExpr->order.value()->orderByElements, windowExpr->order.value()->orderByElements.begin(), [&](auto& orderByElement) {
               orderByElement->expression = canonicalizeParsedExpression(orderByElement->expression, context, true, extendNodeToUse);
               return orderByElement;
            });
         }
         if (windowExpr->startExpr) {
            windowExpr->startExpr = canonicalizeParsedExpression(windowExpr->startExpr, context, false, extendNode);
         }
         if (windowExpr->endExpr) {
            windowExpr->endExpr = canonicalizeParsedExpression(windowExpr->endExpr, context, false, extendNode);
         }
         if (windowExpr->offsetExpr) {
            windowExpr->offsetExpr = canonicalizeParsedExpression(windowExpr->offsetExpr, context, false, extendNode);
         }
         if (windowExpr->defaultExpr) {
            windowExpr->defaultExpr = canonicalizeParsedExpression(windowExpr->defaultExpr, context, false, extendNode);
         }

         std::string alias = windowExpr->alias.empty() ? windowExpr->functionExpression->functionName : windowExpr->alias;

         windowExpr->alias = windowExpr->alias.empty() ? windowExpr->functionExpression->functionName + "_" + std::to_string(i) : windowExpr->alias;
         i++;

         auto columnRef = drv.nf.node<ast::ColumnRefExpression>(windowExpr->loc, windowExpr->alias);
         columnRef->alias = alias;
         extendNode->extensions.emplace_back(windowExpr);
         return columnRef;





         return windowExpr;


      }
      case ast::ExpressionClass::CASE: {
         auto caseExpr = std::static_pointer_cast<ast::CaseExpression>(rootNode);

         for (auto& caseCheck : caseExpr->caseChecks) {
            caseCheck.thenExpr = canonicalizeParsedExpression(caseCheck.thenExpr, context, false, extendNode);
            caseCheck.whenExpr = canonicalizeParsedExpression(caseCheck.whenExpr, context, false, extendNode);
         }
         if (!caseExpr->elseExpr) {
            auto constExpr = drv.nf.node<ast::ConstantExpression>(caseExpr->loc);
            constExpr->value = std::make_shared<ast::NullValue>();
            caseExpr->elseExpr = constExpr;
         }
         caseExpr->elseExpr = canonicalizeParsedExpression(caseExpr->elseExpr, context, false, extendNode);
         std::string alias = caseExpr->alias.empty() ? "" : caseExpr->alias;
         if (extend) {
            auto find = context->currentScope->groupedByExpressions.find(caseExpr);
            if (find == context->currentScope->groupedByExpressions.end()) {
               if (caseExpr->alias.empty()) {
                  caseExpr->alias = "case__" + std::to_string(i);
               }
               i++;

               extendNode->extensions.push_back(caseExpr);
            } else {
               caseExpr->alias = find->get()->alias;
            }


            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(caseExpr->loc, caseExpr->alias);
            columnRef->alias = alias;
            return columnRef;

         }
         return caseExpr;

      }
      case ast::ExpressionClass::CAST: {
         auto castExpr = std::static_pointer_cast<ast::CastExpression>(rootNode);
         castExpr->child = canonicalizeParsedExpression(castExpr->child, context, false, extendNode);
         std::string alias = castExpr->alias.empty() ? "" : castExpr->alias;
         if (extend) {
            auto find = context->currentScope->groupedByExpressions.find(castExpr);
            if (find == context->currentScope->groupedByExpressions.end()) {
               if (castExpr->alias.empty()) {
                  castExpr->alias = "constant__" + std::to_string(i);
               }
               i++;
               extendNode->extensions.push_back(castExpr);
            } else {
               castExpr->alias = find->get()->alias;
            }

            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(castExpr->loc, castExpr->alias);
            columnRef->alias = alias;
            return columnRef;

         }
         return castExpr;

      }
      case ast::ExpressionClass::CONSTANT: {
         auto constantExpr = std::static_pointer_cast<ast::ConstantExpression>(rootNode);
         std::string alias = constantExpr->alias.empty() ? "" : constantExpr->alias;
         if (extend) {
            auto find = context->currentScope->groupedByExpressions.find(constantExpr);
            if (find == context->currentScope->groupedByExpressions.end()) {
               if (constantExpr->alias.empty()) {
                  constantExpr->alias = "constant__" + std::to_string(i);
               }
               i++;
               extendNode->extensions.push_back(constantExpr);
            } else {
               constantExpr->alias = find->get()->alias;
            }

            auto columnRef = drv.nf.node<ast::ColumnRefExpression>(constantExpr->loc, constantExpr->alias);
            columnRef->alias = alias;
            return columnRef;

         }

         return constantExpr;
      }
      case ast::ExpressionClass::BETWEEN: {
         auto betweenExpr = std::static_pointer_cast<ast::BetweenExpression>(rootNode);
         betweenExpr->input = canonicalizeParsedExpression(betweenExpr->input, context, false, extendNode);
         betweenExpr->upper = canonicalizeParsedExpression(betweenExpr->upper, context, false, extendNode);
         betweenExpr->lower = canonicalizeParsedExpression(betweenExpr->lower, context, false, extendNode);
         return betweenExpr;
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
SQLQueryAnalyzer::SQLQueryAnalyzer(std::shared_ptr<catalog::Catalog> catalog) : catalog(std::move(catalog)){
   stackGuard = std::make_shared<StackGuardNormal>();
}
std::shared_ptr<ast::AstNode> SQLQueryAnalyzer::canonicalizeAndAnalyze(std::shared_ptr<ast::AstNode> astRootNode, std::shared_ptr<SQLContext> context) {
   stackGuard->reset();
   auto startCanonicalizeAndAnalyze = std::chrono::high_resolution_clock::now();

   auto rootNode = std::dynamic_pointer_cast<ast::TableProducer>(astRootNode);
   if (!rootNode) {
      //RootNode is not a TableProducer
      switch (astRootNode->nodeType) {
         case ast::NodeType::CREATE_NODE: {
            auto createNode = std::static_pointer_cast<ast::CreateNode>(astRootNode);
            auto scope = context->createResolverScope();
            auto endCanonicalizeAndAnalyze = std::chrono::high_resolution_clock::now();
            this->totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endCanonicalizeAndAnalyze - startCanonicalizeAndAnalyze).count() / 1000.0;
            return analyzeCreateNode(createNode, context, scope);
            ;
         }
         case ast::NodeType::INSERT_NODE: {
            auto insertNode = std::static_pointer_cast<ast::InsertNode>(astRootNode);

            context->pushNewScope();
            auto scope = context->createResolverScope();
            insertNode->producer = sqlCanonicalizer.canonicalize(insertNode->producer, std::make_shared<ASTTransformContext>());
            auto i = analyzeInsertNode(insertNode, context, scope);
            //context->popCurrentScope();
            auto endCanonicalizeAndAnalyze = std::chrono::high_resolution_clock::now();
            this->totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endCanonicalizeAndAnalyze - startCanonicalizeAndAnalyze).count() / 1000.0;
            return i;
         }
         case ast::NodeType::SET_NODE: {
            auto setNode = std::static_pointer_cast<ast::SetNode>(astRootNode);
            auto scope = context->createResolverScope();
            setNode = analyzeSetNode(setNode, context, scope);

            return setNode;
         }
         default: error("Invalid root node type", astRootNode->loc);
      }

   } else {
      //rootNode is a TableProducer
      auto transformed = sqlCanonicalizer.canonicalize(rootNode, std::make_shared<ASTTransformContext>());
      ast::NodeIdGenerator idGen{};
      if (DEBUG) {
         std::cout << std::endl
               << std::endl;
         std::cout << "digraph ast {" << std::endl;
         std::cout << transformed->toDotGraph(1, idGen) << std::endl;
         std::cout << "}" << std::endl;
      }

      context->pushNewScope();
      auto scope = context->createResolverScope();
      transformed = analyzeTableProducer(transformed, context, scope);
      auto endCanonicalizeAndAnalyze = std::chrono::high_resolution_clock::now();
      this->totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endCanonicalizeAndAnalyze - startCanonicalizeAndAnalyze).count() / 1000.0;

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
               auto boundCteNode = drv.nf.node<ast::BoundCTENode>(cteNode->loc);
               boundCteNode->alias = cteNode->alias;

               if (cteNode->query) {
                  ast::TargetInfo targetInfo{};
                  {
                     auto subQueryResolverScope = context->createResolverScope();
                     auto defineScope = context->createDefineScope();
                     context->pushNewScope();
                     auto subQueryScope = context->currentScope;
                     boundCteNode->query = analyzeTableProducer(cteNode->query, context, subQueryResolverScope);
                     targetInfo = context->currentScope->targetInfo;
                     context->popCurrentScope();

                     boundCteNode->subQueryScope = *subQueryScope;
                     std::vector<std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>>> renamedResults;
                     size_t i = 0;
                     for (auto targetColumns : targetInfo.targetColumns) {
                        auto from = targetColumns;
                        auto to = std::make_shared<ast::NamedResult>(context->getUniqueScope(cteNode->alias), from->resultType, from->name);
                        to->displayName = from->displayName;
                        if (cteNode->columnNames.size() > i) {
                           to->displayName = cteNode->columnNames[i];
                           to->name = cteNode->columnNames[i];
                        }
                        renamedResults.emplace_back(std::pair{from, to});
                        i++;
                     }
                     boundCteNode->renamedResults = std::move(renamedResults);


                     context->ctes.insert({cteNode->alias, {targetInfo, boundCteNode}});
                  }
               }
               if (cteNode->child) {
                  boundCteNode->child = analyzeTableProducer(cteNode->child, context, resolverScope);
               }
               return boundCteNode;
            }
            case ast::QueryNodeType::SET_OPERATION_NODE: {
               auto setOperationNode = std::static_pointer_cast<ast::SetOperationNode>(rootNode);
               if (setOperationNode->input) {
                  setOperationNode->input = analyzeTableProducer(setOperationNode->input, context, resolverScope);
               }
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
                  error("Left and right side must have the same number of columns", rootNode->loc);
               }

               auto newScopeName = context->getUniqueScope("setop");
               std::vector<std::shared_ptr<ast::NamedResult>> newTargetInfos;
               for (size_t i = 0; i < leftScope->targetInfo.targetColumns.size(); i++) {
                  auto leftColumn = leftScope->targetInfo.targetColumns[i];
                  auto rightColumn = rightScope->targetInfo.targetColumns[i];
                  auto commonTypes = SQLTypeUtils::toCommonTypes(std::vector{leftColumn->resultType, rightColumn->resultType});
                  leftColumn->resultType = commonTypes[0];
                  rightColumn->resultType = commonTypes[1];
                  auto commonType = SQLTypeUtils::getCommonType(leftColumn->resultType, rightColumn->resultType);
                  auto newNamedResult = std::make_shared<ast::NamedResult>(newScopeName, commonType, leftColumn->name);

                  newNamedResult->displayName = leftColumn->displayName;
                  newTargetInfos.emplace_back(newNamedResult);
               }

               context->mapAttribute(resolverScope, setOperationNode->alias.empty() ? context->getUniqueScope("setOp") : setOperationNode->alias, newTargetInfos);
               context->currentScope->targetInfo.targetColumns = newTargetInfos;
               auto boundSetOperationNode = drv.nf.node<ast::BoundSetOperationNode>(setOperationNode->loc, setOperationNode->alias, setOperationNode->setType, setOperationNode->setOpAll, boundLeft, boundRight, leftScope, rightScope);
               return boundSetOperationNode;
            }
            case ast::QueryNodeType::VALUES: {
               auto valuesNode = std::static_pointer_cast<ast::ValuesQueryNode>(queryNode);
               assert(valuesNode->expressionListRef);
               auto boundExpressionListRef = analyzeTableRef(valuesNode->expressionListRef, context, resolverScope);
               assert(std::static_pointer_cast<ast::TableProducer>(boundExpressionListRef)->nodeType == ast::NodeType::BOUND_TABLE_REF && std::static_pointer_cast<ast::TableRef>(boundExpressionListRef)->type == ast::TableReferenceType::BOUND_EXPRESSION_LIST);
               return drv.nf.node<ast::BoundValuesQueryNode>(valuesNode->loc, valuesNode->alias, std::static_pointer_cast<ast::BoundExpressionListRef>(boundExpressionListRef));
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
         return createNode;
      }
      default: error("Not implemented", createNode->loc);
   }
}

std::shared_ptr<ast::BoundInsertNode> SQLQueryAnalyzer::analyzeInsertNode(std::shared_ptr<ast::InsertNode> insertNode, std::shared_ptr<SQLContext> context, SQLContext::ResolverScope& resolverScope) {
   auto maybeRel = context->catalog->getTypedEntry<catalog::TableCatalogEntry>(insertNode->tableName);
   if (!maybeRel.has_value()) {
      error("Table " << insertNode->tableName << " does not exist", insertNode->loc);
   }
   auto boundTableProducer = analyzeTableProducer(insertNode->producer, context, resolverScope);
   if (boundTableProducer->nodeType != ast::NodeType::QUERY_NODE || std::static_pointer_cast<ast::QueryNode>(boundTableProducer)->type != ast::QueryNodeType::BOUND_VALUES) {
      error("Table producer type for insert node not yet supported", boundTableProducer->loc);
   }
   if (!std::static_pointer_cast<ast::BoundValuesQueryNode>(boundTableProducer)->modifiers.empty()) {
      error("Modifiers for insert node not yet supported", boundTableProducer->loc);
   }

   auto exprListTableRef = std::static_pointer_cast<ast::BoundValuesQueryNode>(boundTableProducer)->expressionListRef;
   auto rel = maybeRel.value();
   std::unordered_map<std::string, catalog::NullableType> allCollumnTypes;
   //Check for correct Type
   for (auto c : rel->getColumns()) {
      allCollumnTypes.emplace(c.getColumnName(), catalog::NullableType(c.getLogicalType(), c.getIsNullable()));
   }
   if (insertNode->columns.empty()) {
      for (auto c : rel->getColumns()) {
         insertNode->columns.emplace_back(c.getColumnName());
      }
   }

   return drv.nf.node<ast::BoundInsertNode>(insertNode->loc, insertNode->schema, insertNode->tableName, exprListTableRef, insertNode->columns, allCollumnTypes);
}
std::shared_ptr<ast::SetNode> SQLQueryAnalyzer::analyzeSetNode(std::shared_ptr<ast::SetNode> setNode, std::shared_ptr<SQLContext> context, SQLContext::ResolverScope& resolverScope) {
   switch (setNode->setType) {
      case ast::SetType::SET: {
         auto setVariableOperation = std::static_pointer_cast<ast::SetVariableStatement>(setNode);
         if (setVariableOperation->values.size() != 1) {
            error("Only one value is supported for SET (for now)", setVariableOperation->loc);
         }
         if (setVariableOperation->values[0]->nodeType != ast::NodeType::EXPRESSION && std::static_pointer_cast<ast::ParsedExpression>(setVariableOperation->values[0])->exprClass != ast::ExpressionClass::CONSTANT) {
            error("Only constant expressions are supported for SET (for now)", setVariableOperation->loc);
         }
         return setVariableOperation;
      }
      case ast::SetType::RESET: {
         error("Reset not yet supported", setNode->loc);
      }
      default:error("Set type not supported", setNode->loc);
   }

}

std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzePipeOperator(std::shared_ptr<ast::PipeOperator> pipeOperator, std::shared_ptr<SQLContext>& context, ResolverScope& resolverScope) {
   std::shared_ptr<ast::AstNode> boundAstNode = pipeOperator->node;
   switch (pipeOperator->pipeOpType) {
      case ast::PipeOperatorType::SELECT: {
         assert(pipeOperator->node->nodeType == ast::NodeType::EXPRESSION);
         auto targetSelection = std::static_pointer_cast<ast::TargetsExpression>(pipeOperator->node);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundTargetExpressions{};
         std::vector<std::shared_ptr<ast::NamedResult>> targetColumns{};
         std::optional<std::vector<std::shared_ptr<ast::BoundExpression>>> boundDistinctExpressions = std::nullopt;
         if (targetSelection->distinctExpressions.has_value()) {
            //Distinct is set
            boundDistinctExpressions = std::vector<std::shared_ptr<ast::BoundExpression>>{};
            std::ranges::transform(targetSelection->distinctExpressions.value(), std::back_inserter(boundDistinctExpressions.value()), [&](auto& expr) {
               return analyzeExpression(expr, context, resolverScope);
            });
         }
         context->currentScope->targetInfo.targetColumns.clear();

         for (auto& target : targetSelection->targets) {
            auto parsedExpression = analyzeExpression(target, context, resolverScope);

            switch (parsedExpression->exprClass) {
               case ast::ExpressionClass::BOUND_COLUMN_REF: {
                  assert(parsedExpression->namedResult.has_value());
                  //ADD column_ref to targetInfo for the current scope!
                  auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(parsedExpression);
                  targetColumns.emplace_back(columnRef->namedResult.value());
                  if (!columnRef->alias.empty()) {
                     context->mapAttribute(resolverScope, columnRef->alias, columnRef->namedResult.value());
                  }
                  context->currentScope->targetInfo.add(columnRef->namedResult.value());
                  break;
               }
               case ast::ExpressionClass::BOUND_STAR: {
                  auto star = std::static_pointer_cast<ast::BoundStarExpression>(parsedExpression);
                  std::vector<catalog::Catalog> catalogs;
                  std::string scope;
                  std::vector<catalog::Column> columns;
                  for (auto& namedResult : star->namedResults) {
                     targetColumns.emplace_back(namedResult);
                     context->currentScope->targetInfo.add(namedResult);
                  }

                  break;
               }
               //NOTE: All other expressions should be moved in an ExtendNode or AggregationNode by canonicalize
               default: error("Not implemented", target->loc);
            }
         }
         boundAstNode = drv.nf.node<ast::BoundTargetsExpression>(targetSelection->loc, targetSelection->alias, boundTargetExpressions, boundDistinctExpressions, targetColumns);
         break;
      }
      case ast::PipeOperatorType::WHERE: {
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
         std::vector<std::shared_ptr<ast::NamedResult>> groupNamedResults{};
         std::vector<std::shared_ptr<ast::BoundExpression>> evalBeforeAggr;

         //Clear targetinfo (see PIPE SQL Syntax)
         context->currentScope->targetInfo.targetColumns.clear();
         if (aggregationNode->groupByNode) {

            std::ranges::transform(aggregationNode->groupByNode->groupByExpressions, std::back_inserter(groupNamedResults), [&](auto expr) {
               auto boundExpression = analyzeExpression(expr, context, resolverScope);
               assert(boundExpression->namedResult.has_value());
               context->mapAttribute(resolverScope, boundExpression->namedResult.value()->name, boundExpression->namedResult.value());
               switch (boundExpression->exprClass) {
                  case ast::ExpressionClass::BOUND_FUNCTION: {
                     assert(boundExpression->resultType.has_value());
                     auto scope = boundExpression->alias.empty() ? boundExpression->alias : context->getUniqueScope("tmp_attr");
                     auto n = std::make_shared<ast::NamedResult>( scope, boundExpression->resultType.value(),  context->getUniqueScope("tmp_attr"));
                     n->displayName = boundExpression->alias.empty() ? "" : boundExpression->alias;
                     evalBeforeAggr.emplace_back(boundExpression);
                     ast::NodeIdGenerator g{};

                     boundExpression->namedResult = n;
                     break;
                  }
                  default:;
               }
               //Add GROUP BY to TargetInfo for the current scope (see PIPE SQL Syntax)
               context->currentScope->targetInfo.add(boundExpression->namedResult.value());
               return boundExpression->namedResult.value();
            });
         }

         std::vector<std::shared_ptr<ast::BoundFunctionExpression>> boundAggregationExpressions{};

         std::ranges::transform(aggregationNode->aggregations, std::back_inserter(boundAggregationExpressions), [&](auto expr) {
            auto boundExpr = analyzeExpression(expr, context, resolverScope);
            assert(boundExpr->exprClass == ast::ExpressionClass::BOUND_FUNCTION);
            //boundExpr->namedResult = std::make_shared<ast::FunctionInfo>("boundExpr->scope", "boundExpr->alias", boundExpr->resultType.value());
            return std::static_pointer_cast<ast::BoundFunctionExpression>(boundExpr);
         });
         auto mapName = context->getUniqueScope("aggMap");
         auto boundGroupByNode = drv.nf.node<ast::BoundGroupByNode>(aggregationNode->groupByNode ? aggregationNode->groupByNode->loc : aggregationNode->loc, groupNamedResults, aggregationNode->groupByNode ? aggregationNode->groupByNode->groupingSet : std::vector<std::set<size_t>>{});
         std::vector<std::shared_ptr<ast::BoundExpression>> toMap{};

         for (auto& aggr : boundAggregationExpressions) {
            //Add Aggregations to TargetInfo for the current scope (see PIPE SQL Syntax)
            context->currentScope->targetInfo.add(aggr->namedResult.value());
            if (aggr->arguments.empty() || aggr->arguments[0]->type == ast::ExpressionType::BOUND_COLUMN_REF) {
               continue;
            }
            toMap.emplace_back(aggr->arguments[0]);
            aggr->arguments[0]->alias =  context->getUniqueScope("tmp_attr");
            aggr->arguments[0]->namedResult = std::make_shared<ast::NamedResult>(mapName, aggr->arguments[0]->resultType.value(), aggr->arguments[0]->alias);

         }
         //ADD to TargetInfo, see Google PIPE sql paper!
         //Maybe Not the best way!
         if (!aggregationNode->groupByNode || aggregationNode->groupByNode->groupByExpressions.empty()) {
            for (auto boundAggr : boundAggregationExpressions) {
               boundAggr->resultType->isNullable = boundAggr->functionName != "COUNT";
            }
         }



         auto boundAggrNode = drv.nf.node<ast::BoundAggregationNode>(pipeOperator->loc, boundGroupByNode, boundAggregationExpressions, toMap, mapName, evalBeforeAggr);
         boundAstNode = boundAggrNode;

         if (aggregationNode->groupByNode && !aggregationNode->groupByNode->groupingSet.empty()) {
            static size_t groupingSetId = 0;
            auto groupingSets = aggregationNode->groupByNode->groupingSet;
            for (size_t i = 0; i < groupingSets.size(); i++) {
               auto groupingSet = groupingSets[i];
               std::vector<std::shared_ptr<ast::NamedResult>> localGroupBy{};
               std::vector<std::shared_ptr<ast::NamedResult>> mapToNull{};
               std::vector<std::shared_ptr<ast::NamedResult>> notAvailable{};
               int present = 0;

               for (size_t j = 0; j < aggregationNode->groupByNode->groupByExpressions.size(); j++) {
                  if (groupingSet.contains(j)) {

                     localGroupBy.emplace_back(groupNamedResults[j]);
                     auto mappedNamedResult = std::make_shared<ast::NamedResult>("groupingSet_" + std::to_string(groupingSetId), groupNamedResults[j]->resultType, "tmp_" + std::to_string(j));
                     mappedNamedResult->displayName = groupNamedResults[j]->displayName;
                     mapToNull.emplace_back(mappedNamedResult);
                     mappedNamedResult->resultType.isNullable = true;
                  } else {
                     present |= (1 << j);
                     auto mappedNamedResult = std::make_shared<ast::NamedResult>( "groupingSet_" + std::to_string(groupingSetId), groupNamedResults[j]->resultType, "tmp_" + std::to_string(j));
                     notAvailable.emplace_back(mappedNamedResult);
                     mappedNamedResult->displayName = groupNamedResults[j]->displayName;
                     mappedNamedResult->resultType.isNullable = true;

                  }
               }

               std::vector<std::shared_ptr<ast::NamedResult>> aggregationNamedResults{};
               for (auto& aggr : boundAggrNode->aggregations) {
                  auto namedResultAggr = std::make_shared<ast::NamedResult>("groupingSetAgg_" + std::to_string(groupingSetId), aggr->namedResult.value()->resultType, aggr->namedResult.value()->name);
                  namedResultAggr->displayName = aggr->namedResult.value()->displayName;
                  aggregationNamedResults.emplace_back(namedResultAggr);
               }
               boundAggrNode->groupByNode->localAggregationNamedResults.emplace_back(std::move(aggregationNamedResults));



               boundAggrNode->groupByNode->localGroupByNamedResults.emplace_back(std::move(localGroupBy));
               boundAggrNode->groupByNode->localMapToNullNamedResults.emplace_back(std::move(mapToNull));
               boundAggrNode->groupByNode->localNotAvailableNamedResults.emplace_back(std::move(notAvailable));
               auto presentNamedResult = std::make_shared<ast::NamedResult>(boundAggrNode->mapName, catalog::Type::int64(), "intval" + std::to_string(present));
               boundAggrNode->groupByNode->localPresentIntval.emplace_back(std::pair{present,presentNamedResult});
               groupingSetId++;

            }




            std::vector<std::shared_ptr<ast::NamedResult>> currentAttributes(
               boundAggrNode->groupByNode->localMapToNullNamedResults.at(0).begin(),
               boundAggrNode->groupByNode->localMapToNullNamedResults.at(0).end());

            currentAttributes.insert(currentAttributes.end(),
               boundAggrNode->groupByNode->localNotAvailableNamedResults.at(0).begin(),
               boundAggrNode->groupByNode->localNotAvailableNamedResults.at(0).end());

            currentAttributes.insert(currentAttributes.end(),
               boundAggrNode->groupByNode->localAggregationNamedResults.at(0).begin(),
               boundAggrNode->groupByNode->localAggregationNamedResults.at(0).end());
            currentAttributes.emplace_back(boundAggrNode->groupByNode->localPresentIntval.at(0).second);
            for (size_t i = 1; i < boundAggrNode->groupByNode->localGroupByNamedResults.size(); i++) {
               auto rollUpUnionName = context->getUniqueScope("rollupUnion");
               std::vector<std::shared_ptr<ast::NamedResult>> currentLocalAttributes(
               boundAggrNode->groupByNode->localMapToNullNamedResults.at(i).begin(),
               boundAggrNode->groupByNode->localMapToNullNamedResults.at(i).end());
               currentLocalAttributes.insert(currentLocalAttributes.end(),
               boundAggrNode->groupByNode->localNotAvailableNamedResults.at(i).begin(),
               boundAggrNode->groupByNode->localNotAvailableNamedResults.at(i).end());
               currentLocalAttributes.insert(currentLocalAttributes.end(),
               boundAggrNode->groupByNode->localAggregationNamedResults.at(i).begin(),
               boundAggrNode->groupByNode->localAggregationNamedResults.at(i).end());
               currentLocalAttributes.emplace_back(boundAggrNode->groupByNode->localPresentIntval.at(i).second);

               size_t id = 0;
               std::vector<std::shared_ptr<ast::NamedResult>> unionNamedResults{};
               for (size_t j = 0; j < currentLocalAttributes.size(); j++) {
                  auto left = currentAttributes[j];
                  auto right = currentLocalAttributes[j];
                  auto unionNamedResult = std::make_shared<ast::NamedResult>(rollUpUnionName + std::to_string(i), right->resultType, left->name);
                  unionNamedResults.emplace_back(unionNamedResult);
                  unionNamedResult->displayName = left->displayName;

               }
               currentAttributes= unionNamedResults;

               boundAggrNode->groupByNode->unionNamedResults.emplace_back(std::move(unionNamedResults));



            }
            for (size_t i = 0; i < boundAggrNode->groupByNode->groupNamedResults.size(); i++) {
               auto old = boundAggrNode->groupByNode->groupNamedResults[i];
               auto newN = boundAggrNode->groupByNode->unionNamedResults.back().at(i);
               context->replace(resolverScope, old, newN);
            }
            for (size_t i = 0; i < boundAggrNode->aggregations.size(); i++) {
               auto old = boundAggrNode->aggregations[i]->namedResult.value();
               auto newN = boundAggrNode->groupByNode->unionNamedResults.back().at(boundAggrNode->groupByNode->groupNamedResults.size()+i);
               context->replace(resolverScope, old, newN);
            }


            std::vector<std::shared_ptr<ast::BoundFunctionExpression>> boundGroupingFunctions;
            for (size_t i = 0; i < aggregationNode->groupByNode->groupingFunctions.size(); i++) {
               auto boundGroupingFunction = analyzeExpression(*std::next(aggregationNode->groupByNode->groupingFunctions.begin(), i), context, resolverScope);
               boundGroupingFunctions.emplace_back(std::static_pointer_cast<ast::BoundFunctionExpression>(boundGroupingFunction));
               assert(std::static_pointer_cast<ast::BoundFunctionExpression>(boundGroupingFunction)->functionName == "GROUPING");
               context->mapAttribute(resolverScope, boundGroupingFunction->alias, boundGroupingFunction->namedResult.value());

               assert(std::static_pointer_cast<ast::BoundFunctionExpression>(boundGroupingFunction)->arguments[0]->namedResult.has_value());
               auto functionArgNamedResult = std::static_pointer_cast<ast::BoundFunctionExpression>(boundGroupingFunction)->arguments[0];;
               size_t j = 0;
               for (; j < boundAggrNode->groupByNode->groupNamedResults.size(); j++) {
                  auto groupNamedResult =  boundAggrNode->groupByNode->unionNamedResults.back().at(j);

                  if (groupNamedResult->name == functionArgNamedResult->namedResult.value()->name) {
                     break;
                  }

               }

               boundAggrNode->groupByNode->groupingFunctions.emplace_back(std::pair{j, boundGroupingFunction->namedResult.value()});



            }




            groupingSetId++;
         }


         break;
      }

      case ast::PipeOperatorType::EXTEND: {
         assert(pipeOperator->node->nodeType == ast::NodeType::EXTEND_NODE);
         auto extendNode = std::static_pointer_cast<ast::ExtendNode>(pipeOperator->node);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundExtensions;

         std::ranges::transform(extendNode->extensions, std::back_inserter(boundExtensions), [&](auto& expr) {
            auto boundExpression = analyzeExpression(expr, context, resolverScope);
            return boundExpression;
         });
         auto mapName = context->getUniqueScope("map");
         //TODO: Refactor to unify storage of window expressions and other extensions, avoiding separate handling.
         std::vector<std::shared_ptr<ast::BoundExpression>> boundExpressions;
         std::vector<std::shared_ptr<ast::BoundWindowExpression>> boundWindowExpressions;

         for (auto& parsedExpression: boundExtensions) {
            switch (parsedExpression->exprClass) {
               case ast::ExpressionClass::BOUND_STAR:
               case ast::ExpressionClass::BOUND_COLUMN_REF: {
                  assert(parsedExpression->namedResult.has_value());
                  context->mapAttribute(resolverScope, parsedExpression->namedResult.value()->displayName, parsedExpression->namedResult.value());
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(parsedExpression->namedResult.value());
                  }
                  break;

               }
               case ast::ExpressionClass::BOUND_FUNCTION: {
                  assert(parsedExpression->resultType.has_value() && parsedExpression->namedResult.has_value());
                  auto function = std::static_pointer_cast<ast::BoundFunctionExpression>(parsedExpression);

                  auto fName = function->alias.empty() ? function->functionName : function->alias;
                  context->mapAttribute(resolverScope, fName, function->namedResult.value());
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(parsedExpression->namedResult.value());
                  }
                  boundExpressions.emplace_back(function);
                  break;
               }

               case ast::ExpressionClass::BOUND_WINDOW: {
                  assert(parsedExpression->resultType.has_value() && parsedExpression->namedResult.has_value());
                  auto window = std::static_pointer_cast<ast::BoundWindowExpression>(parsedExpression);



                  auto fName = window->alias;
                  context->mapAttribute(resolverScope, fName, window->namedResult.value());
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(parsedExpression->namedResult.value());
                  }
                  boundWindowExpressions.emplace_back(window);
                  break;
               }
               case ast::ExpressionClass::BOUND_CONSTANT:
               case ast::ExpressionClass::BOUND_OPERATOR:
               case ast::ExpressionClass::BOUND_CAST:
               case ast::ExpressionClass::BOUND_SUBQUERY:
               case ast::ExpressionClass::BOUND_CASE: {
                  assert(parsedExpression->resultType.has_value());
                  auto scope = parsedExpression->alias.empty() ? parsedExpression->alias : mapName;
                  auto resultType = parsedExpression->resultType.value();
                  if (resultType.useZeroInsteadOfNull) {
                     resultType.isNullable = false;
                  }
                  auto n = std::make_shared<ast::NamedResult>(scope, resultType ,  context->getUniqueScope("tmp_attr"));
                  n->displayName = parsedExpression->alias.empty() ? "" : parsedExpression->alias;
                  context->mapAttribute(resolverScope, parsedExpression->alias.empty() ? n->name : parsedExpression->alias, n);
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(n);
                  }




                  parsedExpression->namedResult = n;

                  boundExpressions.emplace_back(parsedExpression);
                  break;
               }


               default: error("Extend: Not implemented", parsedExpression->loc);
            }
         }
         auto boundExtendNode = drv.nf.node<ast::BoundExtendNode>(extendNode->loc, mapName, std::move(boundExpressions));
         boundExtendNode->windowExpressions = boundWindowExpressions;
         boundAstNode = boundExtendNode;
         break;
      }
      case ast::PipeOperatorType::SET_OPERATION: {
         error("Should not happen", pipeOperator->loc);
      }
      case ast::PipeOperatorType::DROP: {
         assert(pipeOperator->node->nodeType == ast::NodeType::EXPRESSION && std::static_pointer_cast<ast::ParsedExpression>(pipeOperator->node)->exprClass == ast::ExpressionClass::TARGETS);
         auto targets = std::static_pointer_cast<ast::TargetsExpression>(pipeOperator->node);
         for (auto& target : targets->targets) {
            if (target->exprClass != ast::ExpressionClass::COLUMN_REF) {
               error("Only column references are allowed in DROP", target->loc);
            }
            auto boundExpression = analyzeExpression(target, context, resolverScope);
            assert(boundExpression->namedResult.has_value());
            std::erase_if(context->currentScope->targetInfo.targetColumns,[&](const std::shared_ptr<ast::NamedResult>& other) {
               return *other == *boundExpression->namedResult.value();
            });
         }
         return pipeOperator->input;


      }
      case ast::PipeOperatorType::SET: {
         auto setExpression = std::static_pointer_cast<ast::SetExpression>(pipeOperator->node);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundExpressions{};
         for (auto [columnRef, expr]: setExpression->sets) {
            auto boundExpression = analyzeExpression(expr, context, resolverScope);
            auto boundColumnRef = analyzeExpression(columnRef, context, resolverScope);

            assert(boundColumnRef->namedResult.has_value());
            context->mapAttribute(resolverScope, boundColumnRef->namedResult.value()->name, boundColumnRef->namedResult.value());
            boundExpressions.emplace_back(boundExpression);
            boundExpression->namedResult = boundColumnRef->namedResult.value();
            boundColumnRef->namedResult.value()->resultType = boundExpression->resultType.value();
         }
         boundAstNode = drv.nf.node<ast::BoundSetExpression>(setExpression->loc,  context->getUniqueScope("setMap"),boundExpressions);

         break;



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
                  auto namedResult = std::make_shared<ast::NamedResult>(context->getUniqueScope(baseTableRef->tableName), pair.second->resultType, pair.second->name);
                  namedResult->displayName = pair.second->displayName;

                  return namedResult;
               });

               for (auto& namedResult : namedResults) {
                  context->currentScope->targetInfo.add(namedResult);
               }

               context->mapAttribute(resolverScope, sqlScopeName, namedResults);


               auto boundBaseTableRef = drv.nf.node<ast::BoundBaseTableRef>(baseTableRef->loc, namedResults, baseTableRef->alias, baseTableRef->tableName, uniqueScope);
               return boundBaseTableRef;
            } else {
               error("No Catalog found with name " + baseTableRef->tableName, baseTableRef->loc);
            }

         } else {
            auto namedResults = context->mapAttribute(resolverScope, sqlScopeName, uniqueScope, catalogEntry.value());
            for (auto& namedResult : namedResults) {
               context->currentScope->targetInfo.add(namedResult);
            }


            auto boundBaseTableRef = drv.nf.node<ast::BoundBaseTableRef>(baseTableRef->loc, namedResults, baseTableRef->alias, catalogEntry.value()->getName(), uniqueScope);
            return boundBaseTableRef;
         }

         break;
      }
      case ast::TableReferenceType::JOIN: {
         auto join = std::static_pointer_cast<ast::JoinRef>(tableRef);
         switch (join->type) {
            //TOD check for correctness
            case ast::JoinType::INNER:
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
               std::shared_ptr<ast::BoundExpression> boundCondition;
               {
                  auto predScope = context->createResolverScope();
                  if (!std::holds_alternative<std::shared_ptr<ast::ParsedExpression>>(join->condition)) {
                     error("Not implemented", join->loc);
                  }
                  if (std::get<std::shared_ptr<ast::ParsedExpression>>(join->condition)) {
                     boundCondition = analyzeExpression(std::get<std::shared_ptr<ast::ParsedExpression>>(join->condition), context, resolverScope);
                  }
               }

               auto boundJoin = drv.nf.node<ast::BoundJoinRef>(join->loc, join->type, join->refType, left, right, boundCondition);
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
                  rightContext->scopeUnifier = context->scopeUnifier;
                  //Create new context
                  rightContext->pushNewScope();
                  rightContext->ctes = context->ctes;
                  auto rightResolverScope = rightContext->createResolverScope();
                  right = analyzeTableProducer(join->right, rightContext, rightResolverScope);
                  rightScope = rightContext->currentScope;
                  mapping = rightContext->getTopDefinedColumns();
                  context->scopeUnifier = rightContext->scopeUnifier;
               }


               std::shared_ptr<ast::BoundExpression> boundCondition;
               {
                  auto predScope = context->createResolverScope();
                  auto defineScope = context->createDefineScope();
                  for (auto x : mapping) {
                     context->mapAttribute(resolverScope, x.first, x.second);
                  }
                  if (!std::holds_alternative<std::shared_ptr<ast::ParsedExpression>>(join->condition)) {
                     error("Not implemented", join->loc);
                  }

                  boundCondition = analyzeExpression(std::get<std::shared_ptr<ast::ParsedExpression>>(join->condition), context, resolverScope);

               }

               std::vector<std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>>> outerJoinMapping;
               std::string outerjoinName;
               static size_t id = 0;
               if (!mapping.empty()) {
                  outerjoinName = "oj" + std::to_string(id++);
                  std::unordered_map<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>> remapped;
                  for (auto x : mapping) {
                     auto it = remapped.find(x.second);
                     if (it == remapped.end()) {
                        auto scope = x.second->scope;
                        auto name = x.second->name;
                        auto namedResult = std::make_shared<ast::NamedResult>(outerjoinName, x.second->resultType, name);

                        //Make mapping output nullable
                        namedResult->resultType.isNullable = true;
                        namedResult->displayName = x.second->displayName;
                        outerJoinMapping.push_back({x.second, namedResult});
                        remapped.insert({x.second, namedResult});
                        context->mapAttribute(resolverScope, x.first, namedResult);
                     } else {
                        context->mapAttribute(resolverScope, x.first, it->second);
                     }
                  }
               }

               auto boundJoin = drv.nf.node<ast::BoundJoinRef>(join->loc, join->type, join->refType, left, right, boundCondition);
               boundJoin->outerJoinMapping = outerJoinMapping;
               boundJoin->leftScope = leftScope;
               boundJoin->rightScope = rightScope;
               return boundJoin;
            }
            case ast::JoinType::FULL: {
               std::shared_ptr<ast::TableProducer> left, right;
               std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> mapping;
               std::shared_ptr<SQLScope> leftScope, rightScope;

               {
                  auto rightContext = std::make_shared<SQLContext>();
                  //TODO find better way to share scope unifier
                  rightContext->scopeUnifier = context->scopeUnifier;
                  //Create new context
                  rightContext->pushNewScope();
                  rightContext->ctes = context->ctes;
                  auto rightResolverScope = rightContext->createResolverScope();
                  right = analyzeTableProducer(join->right, rightContext, rightResolverScope);
                  rightScope = rightContext->currentScope;
                  auto localMapping = rightContext->getTopDefinedColumns();
                  mapping.insert(mapping.end(), localMapping.begin(), localMapping.end());
                  context->scopeUnifier = rightContext->scopeUnifier;
               }

               {
                  auto leftContext = std::make_shared<SQLContext>();
                  //TODO find better way to share scope unifier
                  leftContext->scopeUnifier = context->scopeUnifier;
                  //Create new context
                  leftContext->pushNewScope();
                  leftContext->ctes = context->ctes;
                  auto leftResolverScope = leftContext->createResolverScope();
                  left = analyzeTableProducer(join->left, leftContext, leftResolverScope);
                  leftScope = leftContext->currentScope;
                  auto localMapping = leftContext->getTopDefinedColumns();
                  mapping.insert(mapping.end(), localMapping.begin(), localMapping.end());
                  context->scopeUnifier = leftContext->scopeUnifier;
               }

               std::shared_ptr<ast::BoundExpression> boundCondition;
               {
                  auto predScope = context->createResolverScope();
                  auto defineScope = context->createDefineScope();
                  for (auto x : mapping) {
                     context->mapAttribute(resolverScope, x.first, x.second);
                  }

                  if (!std::holds_alternative<std::shared_ptr<ast::ParsedExpression>>(join->condition)) {
                     error("Not implemented", join->loc);
                  }
                  boundCondition = analyzeExpression(std::get<std::shared_ptr<ast::ParsedExpression>>(join->condition), context, resolverScope);
               }
               std::vector<std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>>> outerJoinMapping;
               std::string outerjoinName;
               static size_t id = 0;
               if (!mapping.empty()) {
                  outerjoinName = "foj" + std::to_string(id++);
                  //Remap all attributes to the new named result: remapped.first = original, remapped.second = new named result
                  std::unordered_map<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>> remapped;
                  for (auto x : mapping) {
                     auto it = remapped.find(x.second);
                     if (it == remapped.end()) {
                        auto scope = x.second->scope;
                        auto name = x.second->name + "_" +  std::to_string(id++);
                        auto namedResult = std::make_shared<ast::NamedResult>(outerjoinName, x.second->resultType,   name);

                        //Make mapping output nullable
                        namedResult->resultType.isNullable = true;
                        namedResult->displayName = x.second->displayName;
                        outerJoinMapping.push_back({x.second, namedResult});
                        remapped.insert({x.second, namedResult});
                        context->mapAttribute(resolverScope, x.first, namedResult);
                        id++;
                     } else {
                        context->mapAttribute(resolverScope, x.first, it->second);
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
            context->popCurrentScope();
         }
         size_t i = 0;
         for (auto target : targetInfo.targetColumns) {
            assert(!subquery->alias.empty());
            if (subquery->columnNames.size() > i) {
               target->displayName = subquery->columnNames[i];
            }
            if (!target->displayName.empty()) {
               context->mapAttribute(resolverScope, subquery->alias + "." + target->displayName, target);
               context->mapAttribute(resolverScope, target->displayName, target);
            } else {
               context->mapAttribute(resolverScope, subquery->alias + "." + target->name, target);
               context->mapAttribute(resolverScope, target->name, target);
            }
            i++;
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
            for (size_t i = 0; i < sizePerExprList; i++) {
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
            auto t = SQLTypeUtils::getCommonBaseType(typeList);
            SQLTypeUtils::toCommonTypes(typeList);

            return t;
         });
         std::vector<std::shared_ptr<ast::NamedResult>> namedResults{};
         auto scope = context->getUniqueScope("constantTable");
         for (size_t i = 0; i < commonTypes.size(); i++) {
            auto name = context->getUniqueScope("const");
            auto namedResult = std::make_shared<ast::NamedResult>(scope, commonTypes[i], name);
            namedResults.push_back(namedResult);
            context->currentScope->targetInfo.add(namedResult);
         }
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

               std::shared_ptr<ast::NamedResult> namedResult = nullptr;

               switch (boundExpression->type) {
                  case ast::ExpressionType::BOUND_COLUMN_REF: {
                     auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(boundExpression);
                     assert(columnRef->namedResult.has_value());
                     namedResult = columnRef->namedResult.value();
                     break;
                  }
                  case ast::ExpressionType::VALUE_CONSTANT: {
                     auto boundConstant = std::static_pointer_cast<ast::BoundConstantExpression>(boundExpression);
                     assert(boundConstant->resultType.has_value());
                     if (!boundConstant->resultType->isNumeric() || boundConstant->value->type != ast::ConstantType::INT) {
                        error("Order by element not supported", boundConstant->loc);
                     }
                     assert(boundConstant->value);
                     auto constantValue = std::static_pointer_cast<ast::IntValue>(boundConstant->value);
                     if (context->currentScope->targetInfo.targetColumns.size() < constantValue->iVal || constantValue->iVal <= 0) {
                        error("Invalid order by element", boundConstant->loc);
                     }
                     namedResult = context->currentScope->targetInfo.targetColumns.at(constantValue->iVal - 1);
                     break;
                  }
                  default: error("Order by element not implemented", orderByElement->expression->loc);
               }
               assert(namedResult);
               auto boundOrderByElement = drv.nf.node<ast::BoundOrderByElement>(orderByElement->loc, orderByElement->type, orderByElement->nullOrder, namedResult);
               boundOrderByElements.push_back(boundOrderByElement);
            }
         }
         return drv.nf.node<ast::BoundOrderByModifier>(resultModifier->loc, boundOrderByElements, resultModifier->input);
      }
      case ast::ResultModifierType::LIMIT: {
         auto limitModifier = std::static_pointer_cast<ast::LimitModifier>(resultModifier);
         auto limitExpression = analyzeExpression(limitModifier->limitExpression, context, resolverScope);
         auto offset = limitModifier->offset == nullptr ? nullptr : analyzeExpression(limitModifier->offset, context, resolverScope);
         if (limitExpression->exprClass != ast::ExpressionClass::BOUND_CONSTANT) {
            error("Limit expression must be a constant expression", limitModifier->loc);
         }
         if (limitExpression->resultType->type.getTypeId() != catalog::LogicalTypeId::INT) {
            error("Limit expression must be of type INT", limitModifier->loc);
         }
         if (offset != nullptr) {
            std::cerr << "Note: " << "Offset in limit is not supported yet and is ignored for now" << std::endl;
         }
         return drv.nf.node<ast::BoundLimitModifier>(limitModifier->loc, limitExpression, offset, resultModifier->input);
      }
      default: error("Not implemented", resultModifier->loc);
   }
}

/*
    * Expressions
    */
size_t get_stack_size() {
   pthread_attr_t attr;
   pthread_getattr_np(pthread_self(), &attr);

   void *stack_addr;
   size_t stack_size;
   pthread_attr_getstack(&attr, &stack_addr, &stack_size);

   pthread_attr_destroy(&attr);
   return stack_size;
}

std::shared_ptr<ast::BoundExpression> SQLQueryAnalyzer::analyzeExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   if (stackGuard->newStackNeeded()) {
      boost::context::fixedsize_stack salloc(1024*1024);
      boost::context::stack_context sctx = salloc.allocate();
      auto sGuard = stackGuard;
      stackGuard = std::make_shared<StackGuardFiber>(sctx);
      void * sp=static_cast<char*>(sctx.sp);
      std::size_t size=sctx.size;
      std::shared_ptr<ast::BoundExpression> boundExpression;

      boost::context::fiber f(std::allocator_arg, boost::context::preallocated(sp, size, sctx),salloc, [&](boost::context::fiber&& sink) {
         boundExpression = analyzeExpression(rootNode, context, resolverScope);
         return std::move(sink);
      });

      f = std::move(f).resume();
      stackGuard = sGuard;

      return boundExpression;
   }









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
         std::vector<std::shared_ptr<ast::NamedResult>> topDefinedColumnsWithoutDuplicates;

         /**
          * TODO: Find a more elegant solution to this problem
          * Why is this elimination of duplactes needed:
          * The context stores each column (of a table) in definedAttributes with two keys: columnName and tableName.columnName
          * Therefore you must elimniate these duplactes for statements like Select * ....
          */
         for (auto& [scope, namedResult] : topDefinedColumnsAll) {
            if (std::find_if(topDefinedColumnsWithoutDuplicates.begin(), topDefinedColumnsWithoutDuplicates.end(), [&](std::shared_ptr<ast::NamedResult> p) {
                   return p->name == namedResult->name && namedResult->scope == p->scope;
                }) == topDefinedColumnsWithoutDuplicates.end()) {
               topDefinedColumnsWithoutDuplicates.emplace_back(namedResult);
            }
         }

         auto boundStar = drv.nf.node<ast::BoundStarExpression>(star->loc, relationName, topDefinedColumnsWithoutDuplicates);
         return boundStar;
      }
      case ast::ExpressionClass::COMPARISON: {
         auto comparison = std::static_pointer_cast<ast::ComparisonExpression>(rootNode);

         if (comparison->type != ast::ExpressionType::COMPARE_IN && comparison->type != ast::ExpressionType::COMPARE_NOT_IN) {
            if (comparison->rightChildren.size() != 1) {
               error("ComparisonExpression expects exactly one right child for type: " << std::to_string(static_cast<int>(comparison->type)), comparison->loc)
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
         bool isNullable = commonTypes[0].castType->isNullable;
         auto boundComparison = drv.nf.node<ast::BoundComparisonExpression>(comparison->loc, comparison->type, comparison->alias, isNullable, left, boundRightChildren);
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
         if (operatorExpr->type == ast::ExpressionType::OPERATOR_UNKNOWN) {
            operatorExpr->type = stringToExpressionType(operatorExpr->opString);
            if (operatorExpr->type == ast::ExpressionType::OPERATOR_UNKNOWN) {
               error("Unknown operator: " << operatorExpr->opString, operatorExpr->loc);
            }

         }

         if (operatorExpr->children.size() == 0) {
            error("Operator expression has no children", operatorExpr->loc);
         }
         std::vector<std::shared_ptr<ast::BoundExpression>> boundChildren{};
         std::ranges::transform(operatorExpr->children, std::back_inserter(boundChildren), [&](auto c) {
            return analyzeExpression(c, context, resolverScope);
         });
         auto it = std::find_if(boundChildren.begin(), boundChildren.end(), [](auto c) {
            return !c->resultType.has_value();
         });
         if (it != boundChildren.end()) {
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
         catalog::NullableType resultType = SQLTypeUtils::getCommonBaseType(castValues, operatorExpr->type);
         switch (operatorExpr->type) {
            case ast::ExpressionType::OPERATOR_IS_NOT_NULL:
            case ast::ExpressionType::OPERATOR_IS_NULL: {
               if (!resultType.isNullable) {
                  return drv.nf.node<ast::BoundConstantExpression>(operatorExpr->loc, catalog::Type::boolean(), std::make_shared<ast::BoolValue>(operatorExpr->type == ast::ExpressionType::OPERATOR_IS_NOT_NULL), operatorExpr->alias);
               }
               resultType = catalog::Type::boolean();
               break;
            }
            case ast::ExpressionType::OPERATOR_CONCAT: {
               resultType = catalog::NullableType(catalog::Type::stringType(), resultType.isNullable);

               break;
            }
            case ast::ExpressionType::OPERATOR_NOT: {
               resultType = catalog::NullableType(catalog::Type::boolean());
               if (boundChildren.size() != 1) {
                  error("Operator NOT expects exactly one child", operatorExpr->loc);
               }
               if (boundChildren[0]->resultType->type.getTypeId() != catalog::LogicalTypeId::BOOLEAN) {
                  error("Operator NOT expects child of type BOOLEAN", boundChildren[0]->loc);
               }
               break;
            }
            default:;
         }

         size_t t = 0;
         for (auto boundChild : boundChildren) {
            boundChild->resultType = commonNumbers[t];
            t++;
         }

         return drv.nf.node<ast::BoundOperatorExpression>(operatorExpr->loc, operatorExpr->type, resultType, operatorExpr->alias, boundChildren);
      }
      case ast::ExpressionClass::FUNCTION: {
         auto function = std::static_pointer_cast<ast::FunctionExpression>(rootNode);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundArguments{};
         if (rootNode->type == ast::ExpressionType::AGGREGATE) {
            auto scope =  context->getUniqueScope("tmp_attr");
            auto fName = function->alias.empty() ? function->functionName : function->alias;
            std::ranges::transform(function->functionName, function->functionName.begin(), ::toupper);
            std::shared_ptr<ast::BoundFunctionExpression> boundFunctionExpression = nullptr;
            catalog::NullableType resultType{catalog::Type::noneType()};

            if (function->arguments.size() > 1) {
               error("Aggregation with more than one argument not supported", function->loc);
            }
            for (auto arg : function->arguments) {
               auto boundArg = analyzeExpression(arg, context, resolverScope);
               boundArguments.push_back(boundArg);
            }

            /**
                * SUM, AVG, MIN, MAX
                */
            if (function->functionName == "SUM" || function->functionName == "AVG" || function->functionName == "MIN" || function->functionName == "MAX") {
               if (function->arguments.size() > 1) {
                  error("Aggregation with more than one argument not supported", function->loc);
               }
               if (!boundArguments[0]->resultType.has_value() && !function->star) {
                  error("Argument of aggregation function is not a valid expression", boundArguments[0]->loc);
               }
               if ((function->functionName == "SUM" || function->functionName == "AVG") && boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INT &&
                   boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::FLOAT &&
                   boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DECIMAL &&
                   boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DOUBLE) {


               }

               //Find correct resultType
               resultType = boundArguments[0]->resultType.value();

               /**
                    * AVG
                */
               if (function->functionName == "AVG") {
                  if (resultType.type.getTypeId() == catalog::LogicalTypeId::INT) {
                     resultType = SQLTypeUtils::getCommonTypeAfterOperation(catalog::Type::decimal(19, 0), catalog::Type::decimal(19, 0), ast::ExpressionType::OPERATOR_DIVIDE);
                  } else if (resultType.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
                     resultType = SQLTypeUtils::getCommonTypeAfterOperation(resultType, catalog::Type::decimal(19, 0), ast::ExpressionType::OPERATOR_DIVIDE);
                  }
                  resultType.isNullable = true;
                  //else keep type
               }
               resultType.isNullable = true;

               boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments);
            }
            if (function->functionName == "RANK" || function->functionName == "ROW_NUMBER") {
               if (!function->arguments.empty()) {
                  error("RANK and ROW_NUMBER do not support any arguments", function->loc);
               }
               resultType = catalog::Type::int64();

               boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments);

            }

            /*
                * COUNT
                */
            if (function->functionName == "COUNT") {

               if (function->arguments.size() > 1) {
                  error("Aggregation with more than one argument not supported", function->loc);
               }
               if (function->arguments.size() == 0 && !function->star) {
                  error("Argument of aggregation function is not a valid expression", boundArguments[0]->loc);
               }
                 if (function->arguments.size() == 1 && !function->star && boundArguments[0]->exprClass != ast::ExpressionClass::BOUND_COLUMN_REF && boundArguments[0]->exprClass != ast::ExpressionClass::BOUND_CONSTANT) {
                    error("Argument of count must be a column reference or a constant", boundArguments[0]->loc);
                 }
               resultType = catalog::Type::int64();
               if (function->star) {
                  function->functionName = function->functionName + "*";
                  resultType.useZeroInsteadOfNull = true;
               }



               boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments);
            }

            /*
            * STDDEV_SAMP
            */
            if (function->functionName == "STDDEV_SAMP") {
               if (boundArguments.size() != 1) {
                  error("Aggregation with more than one argument not supported", function->loc);
               }
               if (!boundArguments[0]->resultType.has_value() || !boundArguments[0]->resultType->isNumeric()) {
                  error("Argument of aggregation function has not a valid return type", boundArguments[0]->loc);
               }
               resultType = catalog::Type::f64();
               resultType.isNullable = boundArguments[0]->resultType->isNullable;

               boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments);
            }
            if (boundFunctionExpression == nullptr) {
               error("Not implemented", rootNode->loc);
            }

            auto fInfo = std::make_shared<ast::NamedResult>(scope, resultType, fName);

            fInfo->displayName = function->alias;
            context->mapAttribute(resolverScope, fName, fInfo);
            boundFunctionExpression->namedResult = fInfo;

            return boundFunctionExpression;
         }
         std::ranges::transform(function->functionName, function->functionName.begin(), ::toupper);
         auto scope =  context->getUniqueScope("tmp_attr");
         auto fName = function->alias.empty() ? function->functionName : function->alias;
         std::shared_ptr<ast::BoundFunctionExpression> boundFunctionExpression = nullptr;
         catalog::NullableType resultType{catalog::Type::noneType()};

         if (function->functionName == "DATE") {
            if (function->arguments.size() != 1) {
               error("Function date needs exactly one argument", function->loc);
            }
            auto arg = analyzeExpression(function->arguments[0], context, resolverScope);
            if (arg->resultType.has_value() && arg->resultType.value().type.getTypeId() != catalog::Type::stringType().getTypeId()) {
               error("Function date needs argument of type string", function->loc);
            }
            resultType = catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY));
            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, "", function->alias, function->distinct, std::vector{arg});

         } else if (function->functionName == "COUNT") {
            if (function->arguments.size() != 1 && !function->star) {
               error("Function count needs exactly one argument", function->loc);
            }
            if (!function->star && (function->arguments[0]->type != ast::ExpressionType::COLUMN_REF && function->arguments[0]->type != ast::ExpressionType::STAR)) {
               error("Function count needs argument of type column or star", function->loc);
            }

            resultType = catalog::Type::int64();

            if (function->star) {
               return drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName + "*", "", function->alias, function->distinct, std::vector<std::shared_ptr<ast::BoundExpression>>{});
            }
            auto arg = analyzeExpression(function->arguments[0], context, resolverScope);

            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, "", function->alias, function->distinct, std::vector{arg});

         } else if (function->functionName == "EXTRACT") {
            if (function->arguments.size() != 2) {
               error("Function extract needs exactly two arguments", function->loc);
            }
            auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
            auto arg2 = analyzeExpression(function->arguments[1], context, resolverScope);
            if (arg2->resultType.has_value() && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DATE &&  arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::TIMESTAMP && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INTERVAL) {
               error("Function extract needs second argument of type date, interval or timestamp", function->loc);
            }
            resultType = catalog::NullableType{catalog::Type::int64(), arg2->resultType->isNullable};

            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1, arg2});

         } else if (function->functionName == "DATE_TRUNC") {
            if (function->arguments.size() != 2) {
               error("Function DATE_TRUNC needs exactly two arguments", function->loc);
            }
            auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
            auto arg2 = analyzeExpression(function->arguments[1], context, resolverScope);
            if (arg1->resultType.has_value() && arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING && arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::CHAR) {
               error("Function DATE_TRUNC needs first argument of type string", function->loc);
            }
            if (arg2->resultType.has_value() && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DATE &&  arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::TIMESTAMP && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INTERVAL) {
               error("Function DATE_TRUNC needs second argument of type date, interval or timestamp", function->loc);
            }
            //TODO wrong resulttype
            resultType = catalog::Type::int64();


            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1, arg2});

         } else if (function->functionName == "SUBSTRING" || function->functionName == "SUBSTR") {
            if (function->arguments.size() < 1 && function->arguments.size() >= 4) {
               error("Function extract needs one,two or three arguments", function->loc);
            }
            auto stringArg = analyzeExpression(function->arguments[0], context, resolverScope);
            auto fromArg = function->arguments[1] ? analyzeExpression(function->arguments[1], context, resolverScope) : nullptr;
            auto forArg = function->arguments[2] ? analyzeExpression(function->arguments[2], context, resolverScope) : nullptr;

            if (!stringArg->resultType.has_value() || (stringArg->resultType->type.getTypeId() != catalog::LogicalTypeId::STRING && stringArg->resultType->type.getTypeId() != catalog::LogicalTypeId::CHAR)) {
               error("The first argument of the SUBSTRING function must have a result type of STRING", stringArg->loc);
            }
            if (!fromArg->resultType.has_value() || fromArg->resultType->type.getTypeId() != catalog::LogicalTypeId::INT) {
               error("The second argument of the SUBSTRING function must have a result type of INT", fromArg->loc);
            }
            if (!forArg->resultType.has_value() || forArg->resultType->type.getTypeId() != catalog::LogicalTypeId::INT) {
               error("The second argument of the SUBSTRING function must have a result type of INT", forArg->loc);
            }

            resultType = stringArg->resultType.value();

            auto boundArgs = std::vector{stringArg};
            if (fromArg) {
               boundArgs.emplace_back(fromArg);
            }
            if (forArg) {
               boundArgs.emplace_back(forArg);
            }
            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs);

         } else if (function->functionName == "ROUND") {
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
            if (!numberArg->resultType.has_value() || !numberArg->resultType->isNumeric()) {
               error("The first argument of the round function must have a numeric type", numberArg->loc);
            }
            //Get resultType after round
            resultType = numberArg->resultType.value();
            resultType.isNullable = true;

            auto boundArgs = std::vector{numberArg, decimalsArg};
            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs);
         } else if (function->functionName == "UPPER") {
            if (function->arguments.size() != 1) {
               error("Function with more than one argument not supported", function->loc);
            }
            auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);

            if (!arg1->resultType.has_value() || ( arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING && arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::CHAR)) {
               error("Argument of aggregation function has not a valid return type", arg1->loc);
            }
            resultType = arg1->resultType.value();

            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1});
         } else if (function->functionName == "ABS") {
            if (function->arguments.size() != 1) {
               error("Function with more than one argument not supported", function->loc);
            }
            auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
            if (!arg1->resultType.has_value() || !arg1->resultType->isNumeric()) {
               error("Argument of function has not a valid return type", arg1->loc);
            }
            resultType = arg1->resultType.value();

            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1});

         } else if (function->functionName == "COALESCE") {

            if (function->arguments.size() < 2) {
               error("Function with less than two argument not supported", function->loc);
            }

            std::vector<std::shared_ptr<ast::BoundExpression>> boundArgs{};
            std::vector<catalog::NullableType> types{};
            std::ranges::transform(function->arguments, std::back_inserter(boundArgs), [&](auto c) {
               auto e = analyzeExpression(c, context, resolverScope);
               types.push_back(e->resultType.value());
               return e;
            });

            resultType = SQLTypeUtils::getCommonBaseType(types);
            resultType.isNullable = true;


            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs);

         } else if (function->functionName == "GROUPING") {
            if (function->arguments.size() != 1) {
               error("Function grouping needs exactly one argument", function->loc);
            }
            if (function->arguments[0]->type != ast::ExpressionType::COLUMN_REF) {
               error("Function grouping needs argument of type column", function->loc);
            }
            auto arg = analyzeExpression(function->arguments[0], context, resolverScope);
            if (arg->exprClass != ast::ExpressionClass::BOUND_COLUMN_REF) {
               error("Function grouping needs argument of type column", function->loc);
            }

            resultType = catalog::Type::int64();
            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg});

         } else if (function->functionName == "LENGTH") {
            if (function->arguments.size() != 1) {
               error("Function LENGTH needs exactly one argument", function->loc);
            }
            if (function->arguments[0]->type != ast::ExpressionType::COLUMN_REF) {
               error("Function LENGTH needs argument of type column", function->loc);
            }
            auto arg = analyzeExpression(function->arguments[0], context, resolverScope);
            if (arg->exprClass != ast::ExpressionClass::BOUND_COLUMN_REF) {
               error("Function grouping needs argument of type column", function->loc);
            }
            resultType = catalog::Type::int64();
            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg});
         } else if (function->functionName == "REGEXP_REPLACE") {
            if (function->arguments.size() != 3) {
               error("Function REGEXP_REPLACE needs exactly 3 arguments", function->loc);
            }
            auto text = analyzeExpression(function->arguments[0], context, resolverScope);
            auto pattern = analyzeExpression(function->arguments[1], context, resolverScope);
            auto replace = analyzeExpression(function->arguments[2], context, resolverScope);
            if (text->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING ) {
               error("Function REGEXP_REPLACE needs text of type string", text->loc);
            }
            if (pattern->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING && pattern->resultType.value().type.getTypeId() != catalog::LogicalTypeId::CHAR) {
               error("Function REGEXP_REPLACE needs pattern of type string or char", text->loc);
            }
            if (replace->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING && replace->resultType.value().type.getTypeId() != catalog::LogicalTypeId::CHAR) {
               error("Function REGEXP_REPLACE needs replacement of type string or char", text->loc);
            }
            resultType = text->resultType.value();
            boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{text, pattern, replace});
         }

         if (boundFunctionExpression == nullptr) {
            error("Function '" << function->functionName << "' not implemented", function->loc);
         }
         auto fInfo = std::make_shared<ast::NamedResult>(scope, resultType, fName);


         boundFunctionExpression->namedResult = fInfo;

         return boundFunctionExpression;
      }
      case ast::ExpressionClass::CAST: {
         auto castExpr = std::static_pointer_cast<ast::CastExpression>(rootNode);
         auto boundChild = analyzeExpression(castExpr->child, context, resolverScope);
         if (!castExpr->logicalTypeWithMods.has_value()) {
            error("Cast expression must have logicalType", castExpr->loc);
         }
         switch (castExpr->logicalTypeWithMods.value().logicalType) {
            case ast::DATE: {
               switch (boundChild->type) {
                  case ast::ExpressionType::VALUE_CONSTANT: {
                     auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(boundChild);
                     if (constExpr->value->type != ast::ConstantType::STRING) {
                        error("Cannot cast " + constExpr->value->toString() + " to date", constExpr->loc);
                     }
                     std::string stringRep = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
                     stringRep += "days";
                     return drv.nf.node<ast::BoundCastExpression>(castExpr->loc, catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY)), castExpr->alias, boundChild, castExpr->logicalTypeWithMods, stringRep);
                  }
                  case ast::ExpressionType::BOUND_COLUMN_REF: {
                     auto boundColRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(boundChild);
                     assert(boundColRef->resultType.has_value());
                     if (boundColRef->resultType.value().type.getTypeId() == catalog::LogicalTypeId::DATE) {
                        return boundColRef;
                     }
                     if (boundColRef->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING) {
                        error("Cannot cast " + boundColRef->alias + " to date", boundColRef->loc);
                     }
                  }
                  default: error("Cast not implemented", rootNode->loc);
               }
            }
            case ast::LogicalType::INTERVAL: {
               auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(boundChild);
               if (constExpr->value->type != ast::ConstantType::STRING) {
                  error("Cannot cast " + constExpr->value->toString() + " to date", constExpr->loc);
               }
               //!Shortcutted here, implement different interval types later
               auto resultType = catalog::Type::intervalDaytime();
               std::string stringRepresentation = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
               if (castExpr->optInterval.has_value()) {
                  switch (castExpr->optInterval.value()) {
                     case ast::LogicalType::YEARS: {
                        resultType = catalog::Type::intervalMonths();
                        stringRepresentation = std::to_string(std::stol(stringRepresentation) * 12);
                        break;
                     }
                     case ast::LogicalType::MONTHS: {
                        resultType = catalog::Type::intervalMonths();
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
               auto boundCast = drv.nf.node<ast::BoundCastExpression>(castExpr->loc, resultType, castExpr->alias, boundChild, castExpr->logicalTypeWithMods, stringRepresentation);

               return boundCast;
            }

            default: {
               auto castType = SQLTypeUtils::typemodsToCatalogType(castExpr->logicalTypeWithMods.value().logicalType, castExpr->logicalTypeWithMods.value().typeModifiers);
               if (castType != boundChild->resultType.value()) {
                  castType.isNullable = boundChild->resultType.value().isNullable;
                  return drv.nf.node<ast::BoundCastExpression>(castExpr->loc, castType, castExpr->alias, boundChild, castExpr->logicalTypeWithMods, "");
               } else {
                  return boundChild;
               }
            };
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
         if (subqueryTargetInfo.targetColumns.size() != 1 && subqueryExpr->subQueryType != ast::SubqueryType::EXISTS && subqueryExpr->subQueryType != ast::SubqueryType::NOT_EXISTS) {
            error("subquery expressions must produce a single value", subqueryExpr->loc);
         }

         auto namedResult = subqueryTargetInfo.targetColumns[0];
         auto x = std::make_shared<ast::NamedResult>(namedResult->scope, namedResult->resultType, namedResult->name);
         auto resultType = namedResult->resultType;
         resultType.isNullable = true;
         if (subqueryExpr->subQueryType != ast::SubqueryType::SCALAR) {
            resultType = catalog::Type::boolean();
         }
         std::shared_ptr<ast::BoundExpression> boundToTestExpr = nullptr;
         if (subqueryExpr->testExpr) {
            boundToTestExpr = analyzeExpression(subqueryExpr->testExpr, context, resolverScope);
         }

         auto boundSubqueryExpression = drv.nf.node<ast::BoundSubqueryExpression>(subqueryExpr->loc, subqueryExpr->subQueryType, resultType, subqueryExpr->alias, namedResult, subqueryScope, boundSubquery, boundToTestExpr);
         boundSubqueryExpression->namedResult = x;
         boundSubqueryExpression->comparisonType = subqueryExpr->comparisonType;
         return boundSubqueryExpression;
      }
      case ast::ExpressionClass::CASE: {
         auto caseExpr = std::static_pointer_cast<lingodb::ast::CaseExpression>(rootNode);
         if (!caseExpr->elseExpr) {
            error("Case expr must have else clause", caseExpr->loc);
         }
         if (caseExpr->caseChecks.empty()) {
            error("Case expression does not have any case checks", caseExpr->loc);
         }
         std::optional<std::shared_ptr<ast::BoundExpression>> boundCaseExpr = std::nullopt;
         std::vector<ast::BoundCaseExpression::BoundCaseCheck> boundCaseChecks;
         std::vector<catalog::NullableType> thenTypes{};
         std::vector<catalog::NullableType> whenTypes{};
         std::ranges::transform(caseExpr->caseChecks, std::back_inserter(boundCaseChecks), [&](ast::CaseExpression::CaseCheck& caseCheck) {
            if (!caseCheck.thenExpr || !caseCheck.whenExpr) {
               error("Should not happen", caseExpr->loc);
            }

            ast::BoundCaseExpression::BoundCaseCheck boundCheck{analyzeExpression(caseCheck.whenExpr, context, resolverScope), analyzeExpression(caseCheck.thenExpr, context, resolverScope)};

            if (!boundCheck.thenExpr->resultType.has_value()) {
               error("Then expression has invalid type", boundCheck.thenExpr->loc);
            }
            thenTypes.emplace_back(boundCheck.thenExpr->resultType.value());
            whenTypes.emplace_back(boundCheck.whenExpr->resultType.value());
            return boundCheck;
         });
         std::shared_ptr<ast::BoundExpression> boundElse = analyzeExpression(caseExpr->elseExpr, context, resolverScope);
         if (!boundElse->resultType.has_value()) {
            error("Else has invalid type", boundElse->loc);
         }
         if (caseExpr->caseExpr.has_value()) {
            boundCaseExpr = analyzeExpression(caseExpr->caseExpr.value(), context, resolverScope);
            whenTypes.emplace_back(boundCaseExpr.value()->resultType.value());
         }
         thenTypes.emplace_back(boundElse->resultType.value());
         //Find common then type
         auto commonThenType = SQLTypeUtils::toCommonTypes(thenTypes);
         auto commonWhenType = SQLTypeUtils::toCommonTypes(whenTypes);
         if (boundCaseExpr.has_value()) {
            boundCaseExpr.value()->resultType = commonWhenType.back();
         }
         for (size_t i = 0; i < boundCaseChecks.size(); i++) {
            boundCaseChecks[i].thenExpr->resultType = commonThenType[i];
            boundCaseChecks[i].whenExpr->resultType = commonWhenType[i];
         }
         auto resultType = SQLTypeUtils::getCommonBaseType(thenTypes);

         return drv.nf.node<ast::BoundCaseExpression>(caseExpr->loc, resultType, caseExpr->alias, boundCaseExpr, boundCaseChecks, boundElse);
      }
      case ast::ExpressionClass::WINDOW: {
         auto windowExpr = std::static_pointer_cast<ast::WindowExpression>(rootNode);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundPartitions;
         std::optional<std::shared_ptr<ast::BoundOrderByModifier>> boundOrderByModifier;
         std::shared_ptr<ast::BoundFunctionExpression> boundFunction;
         auto tmp = analyzeExpression(windowExpr->functionExpression, context, resolverScope);
         assert(tmp->exprClass == ast::ExpressionClass::BOUND_FUNCTION);
         if (tmp->type != ast::ExpressionType::AGGREGATE) {
            error("Window function must be aggregate", windowExpr->loc);
         }
         boundFunction = std::static_pointer_cast<ast::BoundFunctionExpression>(tmp);
         std::ranges::transform(windowExpr->partitions, std::back_inserter(boundPartitions), [&](auto& partition) {
            return analyzeExpression(partition, context, resolverScope);
         });


         if (windowExpr->order.has_value()) {
            auto boundRModifier = analyzeResultModifier(windowExpr->order.value(), context);
            assert(boundRModifier->modifierType == ast::ResultModifierType::BOUND_ORDER_BY);
            boundOrderByModifier = std::static_pointer_cast<ast::BoundOrderByModifier>(boundRModifier);
         }
         auto boundWindowBoundary = std::make_shared<ast::BoundWindowBoundary>();

         if (!windowExpr->windowBoundary) {
            windowExpr->windowBoundary = drv.nf.node<ast::WindowBoundary>(windowExpr->loc, ast::WindowBoundaryType::UNBOUNDED_PRECEDING);
            if (windowExpr->order.has_value()) {
               windowExpr->windowBoundary->end = ast::WindowBoundaryType::CURRENT_ROW;
            } else {
               windowExpr->windowBoundary->end = ast::WindowBoundaryType::UNBOUNDED_FOLLOWING;
            }

            windowExpr->windowBoundary->windowMode = ast::WindowMode::ROWS;

         }

         boundWindowBoundary->loc = windowExpr->windowBoundary->loc;
         boundWindowBoundary->windowMode = windowExpr->windowBoundary->windowMode;
         //Start
         switch (windowExpr->windowBoundary->start) {
            case ast::WindowBoundaryType::CURRENT_ROW: {
               boundWindowBoundary->start = 0;
               break;
            }
            case ast::WindowBoundaryType::EXPR_PRECEDING:
            case ast::WindowBoundaryType::EXPR_FOLLOWING: {
               assert(windowExpr->windowBoundary->startExpr);
               assert(windowExpr->windowBoundary->startExpr->exprClass == ast::ExpressionClass::CONSTANT);

               auto constantExpr = std::static_pointer_cast<ast::BoundConstantExpression>(analyzeExpression(windowExpr->windowBoundary->startExpr, context, resolverScope));
               if (constantExpr->value->type != ast::ConstantType::INT) {
                  error("unsupported window start specification", boundWindowBoundary->loc);
               }
               boundWindowBoundary->start = std::static_pointer_cast<ast::IntValue>(constantExpr->value)->iVal;
               if (windowExpr->windowBoundary->start == ast::WindowBoundaryType::EXPR_PRECEDING) {
                  boundWindowBoundary->start = -boundWindowBoundary->start;
               }
            }

            case ast::WindowBoundaryType::INVALID: {
               error("Invalid boundary type", boundWindowBoundary->loc)
               break;
            }
               default: ;

         }
         //End
         switch (windowExpr->windowBoundary->end) {
            case ast::WindowBoundaryType::CURRENT_ROW: {
               boundWindowBoundary->end = 0;
               break;
            }
            case ast::WindowBoundaryType::EXPR_PRECEDING:
            case ast::WindowBoundaryType::EXPR_FOLLOWING: {
               assert(windowExpr->windowBoundary->endExpr);
               assert(windowExpr->windowBoundary->endExpr->exprClass == ast::ExpressionClass::CONSTANT);

               auto constantExpr = std::static_pointer_cast<ast::BoundConstantExpression>(analyzeExpression(windowExpr->windowBoundary->endExpr, context, resolverScope));
               if (constantExpr->value->type != ast::ConstantType::INT) {
                  error("unsupported window start specification", boundWindowBoundary->loc);
               }
               boundWindowBoundary->end = std::static_pointer_cast<ast::IntValue>(constantExpr->value)->iVal;
               if (windowExpr->windowBoundary->end == ast::WindowBoundaryType::EXPR_PRECEDING) {
                  boundWindowBoundary->end = -boundWindowBoundary->end;
               }
            }

            case ast::WindowBoundaryType::INVALID: {
               error("Invalid boundary type", boundWindowBoundary->loc)
               break;
            }
            default: ;

         }


         ast::ExpressionType windowType = ast::ExpressionType::WINDOW_AGGREGATE;
         catalog::Type resultType = catalog::Type::int64();


         boundFunction->namedResult.value()->displayName = windowExpr->alias;
         context->mapAttribute(resolverScope, windowExpr->alias, boundFunction->namedResult.value());



         auto boundWindowExpression =  drv.nf.node<ast::BoundWindowExpression>(windowExpr->loc, windowExpr->type, windowExpr->alias, resultType, boundFunction, boundPartitions, boundOrderByModifier, boundWindowBoundary);
         boundWindowExpression->namedResult = boundFunction->namedResult;
         return boundWindowExpression;


      }
      default: error("Expression type not implemented", rootNode->loc);
   }
}

std::shared_ptr<ast::BoundColumnRefExpression> SQLQueryAnalyzer::analyzeColumnRefExpression(std::shared_ptr<ast::ColumnRefExpression> columnRef, std::shared_ptr<SQLContext> context) {
   //new implementation which uses the new concept of TableProducers
   auto columnName = columnRef->columnNames.size() == 1 ? columnRef->columnNames[0] : columnRef->columnNames[1];

   std::string scope;
   std::shared_ptr<ast::NamedResult> found;
   if (columnRef->columnNames.size() == 2) {
      found = context->getNamedResultInfo(columnRef->loc, columnRef->columnNames[0] + "." + columnRef->columnNames[1]);

   } else if (columnRef->columnNames.size() == 1) {
      found = context->getNamedResultInfo(columnRef->loc, columnRef->columnNames[0]);
   } else {
      throw std::runtime_error("Not implemented");
   }

   if (!found) {
      error("Column not found", columnRef->loc);
   }
   found->displayName = columnRef->alias.empty() ? found->displayName : columnRef->alias;
   return drv.nf.node<ast::BoundColumnRefExpression>(columnRef->loc, found->resultType, found, columnRef->alias);
}

/*
    * SQLTypeUtils
    */
catalog::NullableType SQLTypeUtils::getCommonType(catalog::NullableType nullableType1, catalog::NullableType nullableType2) {
    const bool isNullable = nullableType1.isNullable || nullableType2.isNullable;

    // If types are identical, handle special case for DECIMAL or return the type
    if (nullableType1.type.getTypeId() == nullableType2.type.getTypeId()) {
        if (nullableType1.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            return getHigherDecimalType(nullableType1, nullableType2);
        }

       if (nullableType1.type.getTypeId() == catalog::LogicalTypeId::CHAR) {
          auto charInfo1 = nullableType1.type.getInfo<catalog::CharTypeInfo>();
          auto charInfo2 = nullableType2.type.getInfo<catalog::CharTypeInfo>();
          //If char have different lengths, return string type
          if (charInfo1->getLength() != charInfo2->getLength()) {
             return catalog::NullableType(catalog::Type::stringType(), isNullable);
          }

       }
        return catalog::NullableType(nullableType1.type, isNullable);
    }

    for (size_t i = 0; i<2; i++) {
       const auto& type1 = nullableType1.type;
       const auto& type2 = nullableType2.type;

       // Check combinations in one direction
       if (type1.getTypeId() == catalog::LogicalTypeId::DECIMAL &&
           type2.getTypeId() == catalog::LogicalTypeId::INT) {
          return catalog::NullableType(type1, isNullable);
           }
       if ((type1.getTypeId() == catalog::LogicalTypeId::INT || type1.getTypeId() == catalog::LogicalTypeId::DECIMAL) && type2.getTypeId() == catalog::LogicalTypeId::DOUBLE) {
          return catalog::NullableType(type2, isNullable);
           }


       if (type1.getTypeId() == catalog::LogicalTypeId::STRING &&
           type2.getTypeId() == catalog::LogicalTypeId::CHAR) {
          return catalog::NullableType(type1, isNullable);
           }
       if (type1.getTypeId() == catalog::LogicalTypeId::STRING) {
         if (type2.getTypeId() == catalog::LogicalTypeId::INT || type2.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            return catalog::NullableType(type1, isNullable);
         }
       }

       if (type1.getTypeId() == catalog::LogicalTypeId::DATE) {
          if (type2.getTypeId() == catalog::LogicalTypeId::STRING) {
             return catalog::NullableType(type1, isNullable);
          }
          if (type2.getTypeId() == catalog::LogicalTypeId::INTERVAL) {
             return catalog::NullableType(
                 catalog::Type(catalog::LogicalTypeId::DATE,
                             std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY)),
                 isNullable);
          }
       }


       if (type1.getTypeId() == catalog::LogicalTypeId::INT &&
           type2.getTypeId() == catalog::LogicalTypeId::CHAR) {
          return catalog::NullableType(type1, isNullable);
           }

       // Handle NONE type
       if (type2.getTypeId() == catalog::LogicalTypeId::NONE) {
          return catalog::NullableType(type1, isNullable);
       }
       if (type1.getTypeId() == catalog::LogicalTypeId::NONE) {
          return catalog::NullableType(type2, isNullable);
       }
       std::swap(nullableType1, nullableType2);
    }

    throw std::runtime_error("No common type found for " + nullableType1.type.toString() + " and " + nullableType2.type.toString());
}

catalog::NullableType SQLTypeUtils::getHigherDecimalType(catalog::NullableType left, catalog::NullableType right) {
   assert(left.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && right.type.getTypeId() == catalog::LogicalTypeId::DECIMAL);
   auto leftInfo = left.type.getInfo<catalog::DecimalTypeInfo>();
   auto rightInfo = right.type.getInfo<catalog::DecimalTypeInfo>();
   int hidig = std::max(leftInfo->getPrecision() - leftInfo->getScale(), rightInfo->getPrecision() - rightInfo->getScale());
   int maxs = std::max(leftInfo->getScale(), rightInfo->getScale());
   return catalog::NullableType(catalog::Type::decimal(hidig + maxs, maxs), left.isNullable || right.isNullable);
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

            auto [p, s] = getAdaptedDecimalPAndSAfterMulDiv(type1Info->getPrecision() - type1Info->getScale() + type2Info->getScale() + std::max<unsigned long>(6, type1Info->getScale() + type2Info->getPrecision()), std::max<unsigned long>(6, type1Info->getScale() + type2Info->getPrecision()));

            return catalog::NullableType(catalog::Type::decimal(p, s), commonType.isNullable);
         }
         return commonType;
      }
      case ast::ExpressionType::OPERATOR_TIMES: {
         if (type1.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && type2.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            auto type1Info = type1.type.getInfo<catalog::DecimalTypeInfo>();
            auto type2Info = type2.type.getInfo<catalog::DecimalTypeInfo>();
            auto [p, s] = getAdaptedDecimalPAndSAfterMulDiv(type1Info->getPrecision() + type2Info->getPrecision(), type1Info->getScale() + type2Info->getScale());
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
   auto anyDecimal = llvm::any_of(types, [](catalog::NullableType type) { return type.type.getTypeId() == catalog::LogicalTypeId::DECIMAL; });
   auto anyFloat = llvm::any_of(types, [](catalog::NullableType type) { return type.type.getTypeId() == catalog::LogicalTypeId::FLOAT || type.type.getTypeId() == catalog::LogicalTypeId::DOUBLE; });
   if (anyDecimal && !anyFloat) {
      std::vector<catalog::NullableType> res;
      for (auto type : types) {
         if (type.type.getTypeId() != catalog::LogicalTypeId::DECIMAL) {
            type.castType = std::make_shared<catalog::NullableType>(catalog::Type::decimal(19, 0), type.isNullable);
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
         if (typeModifiers.size() > 1 || typeModifiers[0]->type != ast::ConstantType::UINT) {
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

         return catalog::Type::decimal(p, s);
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