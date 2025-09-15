#include "lingodb/compiler/frontend/sql_analyzer.h"

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/compiler/frontend/ast/bound/bound_aggregation.h"
#include "lingodb/compiler/frontend/ast/bound/bound_create_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_extend_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_groupby.h"
#include "lingodb/compiler/frontend/ast/bound/bound_insert_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_pipe_operator.h"
#include "lingodb/compiler/frontend/ast/bound/bound_query_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_tableref.h"
#include "lingodb/compiler/frontend/ast/bound/bound_target_list.h"
#include "lingodb/scheduler/Tasks.h"

#include <boost/context/fiber_fcontext.hpp>
#include <cctype>
#include <ranges>
#include <sys/resource.h>
namespace lingodb::analyzer {
using ResolverScope = llvm::ScopedHashTable<std::string, std::shared_ptr<ast::ColumnReference>, StringInfo>::ScopeTy;

StackGuardNormal::StackGuardNormal() {
#ifdef ASAN_ACTIVE
   rlimit rlp{};
   auto suc = getrlimit(RLIMIT_STACK, &rlp);
   if (suc != 0) {
      limit = 0;
   }
   limit = 0.080 * rlp.rlim_cur;
   startFameAddress = __builtin_frame_address(0);
#else
   static constexpr size_t stackSize = 1 << 20;
   limit = 0.8 * stackSize;
   startFameAddress = __builtin_frame_address(0);
#endif
}
void StackGuardNormal::reset() {
   startFameAddress = __builtin_frame_address(0);
}
bool StackGuardNormal::newStackNeeded() {
   rlimit rlp{};
   auto suc = getrlimit(RLIMIT_STACK, &rlp);
   assert(suc == 0);
   void* currentFrameAddress = __builtin_frame_address(0);
   size_t size = reinterpret_cast<size_t>(startFameAddress) - reinterpret_cast<size_t>(currentFrameAddress);

   if (size > limit) {
      std::cerr << "StackLimit: " << rlp.rlim_cur << " Max: " << rlp.rlim_max << " recorded size: " << size << " Perc: " << ((size * 1.0) / rlp.rlim_cur) * 100 << std::endl;
#ifdef ASAN_ACTIVE
      throw std::runtime_error("StackLimit reached");
#endif
      return true;
   }
   return false;
}

StackGuardFiber::StackGuardFiber(boost::context::stack_context& stackContext) : StackGuard(), stackContext(stackContext) {
   startFameAddress = stackContext.sp;
   limit = stackContext.size * 0.65;
}
void StackGuardFiber::reset() {
   startFameAddress = stackContext.sp;
}
bool StackGuardFiber::newStackNeeded() {
   void* currentFrameAddress = __builtin_frame_address(0);
   size_t size = reinterpret_cast<size_t>(startFameAddress) - reinterpret_cast<size_t>(currentFrameAddress);
   if (size > limit) {
      std::cerr << "Fiber: StackLimit: " << stackContext.size << " recorded size: " << size << " Perc: " << ((size * 1.0) / stackContext.size) * 100 << std::endl;
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
               if (selectNode->fromClause) {
                  auto transformedFrom = selectNode->fromClause;

                  selectNode->fromClause = nullptr;
                  transformed = transformedFrom;
               }
               //Transform where_clause
               if (selectNode->whereClause) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(selectNode->whereClause->loc, ast::PipeOperatorType::WHERE, selectNode->whereClause);
                  pipe->input = transformed;
                  selectNode->whereClause = nullptr;
                  transformed = pipe;
               }

               //Transform Group by
               if (selectNode->groups) {
                  context->currentScope->aggregationNode->groupByNode = std::move(selectNode->groups);
               }
               auto aggPipeNode = drv.nf.node<ast::PipeOperator>(selectNode->loc, ast::PipeOperatorType::AGGREGATE, context->currentScope->aggregationNode);
               aggPipeNode->input = transformed;
               transformed = aggPipeNode;

               auto extendBeforeWindowPipeOp = drv.nf.node<ast::PipeOperator>(selectNode->selectList->loc, ast::PipeOperatorType::EXTEND, context->currentScope->extendNodeBeforeWindowFunctions);
               extendBeforeWindowPipeOp->input = transformed;
               transformed = extendBeforeWindowPipeOp;

               //Transform target selection
               if (selectNode->selectList) {
                  auto pipe = drv.nf.node<ast::PipeOperator>(selectNode->selectList->loc, ast::PipeOperatorType::SELECT, selectNode->selectList);
                  for (auto& target : selectNode->selectList->targets) {
                     if (target->alias.empty()) {
                        continue;
                     }
                     context->currentScope->selectSymbolList.emplace(target->alias, target);
                  }
                  pipe->input = transformed;

                  transformed = pipe;
                  selectNode->selectList = nullptr;
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
            default: error("canonicalize: queryNode is not yet supported", queryNode->loc);
         }
      }
      case ast::NodeType::PIPE_OP: {
         auto pipeOp = std::static_pointer_cast<ast::PipeOperator>(rootNode);
         if (pipeOp->input) {
            pipeOp->input = canonicalize(pipeOp->input, context);
         }

         switch (pipeOp->pipeOpType) {
            case ast::PipeOperatorType::SELECT: {
               auto selectNode = std::static_pointer_cast<ast::TargetList>(pipeOp->node);
               auto extendNode = drv.nf.node<ast::ExtendNode>(selectNode->loc, true);
               auto extendPipeOp = drv.nf.node<ast::PipeOperator>(selectNode->loc, ast::PipeOperatorType::EXTEND, extendNode);
               //Extract AggFunctions
               std::vector<std::pair<std::string, std::shared_ptr<ast::ParsedExpression>>> toRemove{};
               //Canonicalize target expressions
               std::ranges::transform(selectNode->targets, selectNode->targets.begin(), [&](std::shared_ptr<ast::ParsedExpression>& target) {
                  return canonicalizeParsedExpression(target, context, true, extendNode);
               });
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
               auto join = std::static_pointer_cast<ast::JoinRef>(pipeOp->node);
               auto extendNode = drv.nf.node<ast::ExtendNode>(join->loc, true);
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
               auto setExpression = std::static_pointer_cast<ast::SetColumnExpression>(pipeOp->node);
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
            case ast::TableReferenceType::CROSS_PRODUCT: {
               auto crossProd = std::static_pointer_cast<ast::CrossProductRef>(tableRef);
               std::ranges::transform(crossProd->tables, crossProd->tables.begin(), [&](auto& table) {
                  return canonicalize(table, context);
               });
               return crossProd->tables.size() == 1 ? crossProd->tables[0] : crossProd;
            }
            case ast::TableReferenceType::JOIN: {
               auto joinRef = std::static_pointer_cast<ast::JoinRef>(tableRef);
               auto extendNode = drv.nf.node<ast::ExtendNode>(joinRef->loc, true);
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

               if (std::holds_alternative<std::shared_ptr<ast::ParsedExpression>>(joinRef->condition)) {
                  auto condition = std::get<std::shared_ptr<ast::ParsedExpression>>(joinRef->condition);
                  if (condition) {
                     joinRef->condition = canonicalizeParsedExpression(condition, context, false, extendNode);
                  }
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

std::shared_ptr<ast::ParsedExpression> SQLCanonicalizer::canonicalizeWindowExpression(std::shared_ptr<ast::WindowExpression> windowExpr, std::shared_ptr<ast::ExtendNode> extendNode, int& i, std::shared_ptr<ASTTransformContext> context) {
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
}
std::shared_ptr<ast::ParsedExpression> SQLCanonicalizer::canonicalizeFunctionExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<ast::FunctionExpression> functionExpr, bool extend, std::shared_ptr<ast::ExtendNode> extendNode, int& i, std::shared_ptr<ASTTransformContext> context) {
   std::string columnAlias = functionExpr->alias.empty() ? functionExpr->functionName : functionExpr->alias;
   std::ranges::transform(functionExpr->arguments, functionExpr->arguments.begin(), [&](auto& arg) {
      return canonicalizeParsedExpression(arg, context, false, extendNode);
   });

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

         if (find == context->currentScope->aggregationNode->groupByNode->groupingFunctions.end()) {
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
std::shared_ptr<ast::ParsedExpression> SQLCanonicalizer::canonicalizeParsedExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<ASTTransformContext> context, bool extend, std::shared_ptr<ast::ExtendNode> extendNode) {
   if (stackGuard->newStackNeeded()) {
      boost::context::fixedsize_stack salloc(1024 * 1024);
      boost::context::stack_context sctx = salloc.allocate();
      auto sGuard = stackGuard;
      stackGuard = std::make_shared<StackGuardFiber>(sctx);
      void* sp = static_cast<char*>(sctx.sp);
      std::size_t size = sctx.size;
      std::shared_ptr<ast::ParsedExpression> expression;

      boost::context::fiber f(std::allocator_arg, boost::context::preallocated(sp, size, sctx), salloc, [&](boost::context::fiber&& sink) {
         expression = canonicalizeParsedExpression(rootNode, context, extend, extendNode);
         return std::move(sink);
      });

      f = std::move(f).resume();
      stackGuard = sGuard;

      return expression;
   }
   static int i = 0;
   auto extendExpr = [this, context, extendNode](std::shared_ptr<ast::ParsedExpression> expr) {
      std::string alias = expr->alias.empty() ? "" : expr->alias;
      auto find = context->currentScope->groupedByExpressions.find(expr);
      if (find == context->currentScope->groupedByExpressions.end()) {
         if (expr->alias.empty()) {
            expr->alias = "op_" + std::to_string(i);
         }
         i++;
         extendNode->extensions.push_back(expr);
      } else {
         expr->alias = find->get()->alias;
      }
      auto columnRef = drv.nf.node<ast::ColumnRefExpression>(expr->loc, expr->alias);
      columnRef->alias = alias;
      columnRef->forceToUseAlias = true;
      return columnRef;
   };

   switch (rootNode->exprClass) {
      case ast::ExpressionClass::SUBQUERY: {
         auto subqueryExpr = std::static_pointer_cast<ast::SubqueryExpression>(rootNode);
         auto canacolizedSubquery = canonicalizeCast<ast::PipeOperator>(subqueryExpr->subquery, std::make_shared<ASTTransformContext>());
         subqueryExpr->subquery = canacolizedSubquery;
         // assert(subqueryExpr->subquery->nodeType == ast::NodeType::PIPE_OP && "Should not happen");
         if (subqueryExpr->testExpr) {
            subqueryExpr->testExpr = canonicalizeParsedExpression(subqueryExpr->testExpr, context, true, extendNode);
         }

         if (extend) {
            return extendExpr(subqueryExpr);
         }

         return subqueryExpr;
      }
      case ast::ExpressionClass::OPERATOR: {
         auto operatorExpr = std::static_pointer_cast<ast::OperatorExpression>(rootNode);

         std::ranges::transform(operatorExpr->children, operatorExpr->children.begin(), [&](auto& child) {
            return canonicalizeParsedExpression(child, context, false, extendNode);
         });
         if (extend) {
            return extendExpr(operatorExpr);
         }
         return operatorExpr;
      }
      case ast::ExpressionClass::CONJUNCTION: {
         auto conjunctionExpr = std::static_pointer_cast<ast::ConjunctionExpression>(rootNode);

         std::vector<std::shared_ptr<ast::ParsedExpression>> combinedChildren;
         for (auto& child : conjunctionExpr->children) {
            child = canonicalizeParsedExpression(child, context, false, extendNode);
            if (child->exprClass == ast::ExpressionClass::CONJUNCTION && child->type == conjunctionExpr->type) {
               combinedChildren.insert(combinedChildren.end(), std::static_pointer_cast<ast::ConjunctionExpression>(child)->children.begin(), std::static_pointer_cast<ast::ConjunctionExpression>(child)->children.end());
            } else {
               combinedChildren.emplace_back(child);
            }
         }
         conjunctionExpr->children = std::move(combinedChildren);

         if (extend) {
            return extendExpr(conjunctionExpr);
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
            return extendExpr(comparisonExpr);
         }
         return comparisonExpr;
      }
      case ast::ExpressionClass::FUNCTION: {
         auto functionExpr = std::static_pointer_cast<ast::FunctionExpression>(rootNode);
         return canonicalizeFunctionExpression(rootNode, functionExpr, extend, extendNode, i, context);
      }
      case ast::ExpressionClass::WINDOW: {
         auto windowExpr = std::static_pointer_cast<ast::WindowExpression>(rootNode);
         return canonicalizeWindowExpression(windowExpr, extendNode, i, context);
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

         if (extend) {
            return extendExpr(caseExpr);
         }
         return caseExpr;
      }
      case ast::ExpressionClass::CAST: {
         auto castExpr = std::static_pointer_cast<ast::CastExpression>(rootNode);
         castExpr->child = canonicalizeParsedExpression(castExpr->child, context, false, extendNode);
         if (extend) {
            return extendExpr(castExpr);
         }
         return castExpr;
      }
      case ast::ExpressionClass::CONSTANT: {
         auto constantExpr = std::static_pointer_cast<ast::ConstantExpression>(rootNode);
         if (extend) {
            return extendExpr(constantExpr);
         }

         return constantExpr;
      }
      case ast::ExpressionClass::BETWEEN: {
         auto betweenExpr = std::static_pointer_cast<ast::BetweenExpression>(rootNode);
         betweenExpr->input = canonicalizeParsedExpression(betweenExpr->input, context, false, extendNode);
         betweenExpr->upper = canonicalizeParsedExpression(betweenExpr->upper, context, false, extendNode);
         betweenExpr->lower = canonicalizeParsedExpression(betweenExpr->lower, context, false, extendNode);
         if (extend) {
            return extendExpr(betweenExpr);
         }
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
SQLQueryAnalyzer::SQLQueryAnalyzer(catalog::Catalog* catalog) : catalog(std::move(catalog)) {
   stackGuard = std::make_shared<StackGuardNormal>();
}
std::shared_ptr<ast::AstNode> SQLQueryAnalyzer::canonicalizeAndAnalyze(std::shared_ptr<ast::AstNode> astRootNode, std::shared_ptr<SQLContext> context) {
   stackGuard->reset();

   auto rootNode = std::dynamic_pointer_cast<ast::TableProducer>(astRootNode);
   if (!rootNode) {
      //RootNode is not a TableProducer
      switch (astRootNode->nodeType) {
         case ast::NodeType::CREATE_NODE: {
            auto createNode = std::static_pointer_cast<ast::CreateNode>(astRootNode);
            return analyzeCreateNode(createNode);
         }
         case ast::NodeType::INSERT_NODE: {
            auto insertNode = std::static_pointer_cast<ast::InsertNode>(astRootNode);

            context->pushNewScope();
            auto scope = context->createResolverScope();
            insertNode->producer = sqlCanonicalizer.canonicalize(insertNode->producer, std::make_shared<ASTTransformContext>());
            auto i = analyzeInsertNode(insertNode, context, scope);
            return i;
         }
         case ast::NodeType::SET_NODE: {
            auto setNode = std::static_pointer_cast<ast::SetNode>(astRootNode);
            setNode = analyzeSetNode(setNode);

            return setNode;
         }
         case ast::NodeType::COPY_NODE: {
            auto copyNode = std::static_pointer_cast<ast::CopyNode>(astRootNode);
            return copyNode;
         }
         default: error("Invalid root node type", astRootNode->loc);
      }

   } else {
      //rootNode is a TableProducer
      auto transformed = sqlCanonicalizer.canonicalize(rootNode, std::make_shared<ASTTransformContext>());

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
                     std::vector<std::pair<std::shared_ptr<ast::ColumnReference>, std::shared_ptr<ast::ColumnReference>>> renamedColumnReferences;
                     size_t i = 0;
                     for (auto targetColumns : targetInfo.targetColumns) {
                        auto from = targetColumns;
                        auto to = std::make_shared<ast::ColumnReference>(context->getUniqueScope(cteNode->alias), from->resultType, from->name);
                        to->displayName = from->displayName;
                        if (cteNode->columnNames.size() > i) {
                           to->displayName = cteNode->columnNames[i];
                           to->name = cteNode->columnNames[i];
                        }
                        renamedColumnReferences.emplace_back(std::pair{from, to});
                        i++;
                     }
                     boundCteNode->renamedColumnReferences = std::move(renamedColumnReferences);

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
               std::vector<std::shared_ptr<ast::ColumnReference>> newTargetInfos;
               for (size_t i = 0; i < leftScope->targetInfo.targetColumns.size(); i++) {
                  auto leftColumn = leftScope->targetInfo.targetColumns[i];
                  auto rightColumn = rightScope->targetInfo.targetColumns[i];
                  auto commonTypes = SQLTypeUtils::toCommonTypes(std::vector{leftColumn->resultType, rightColumn->resultType});
                  leftColumn->resultType = commonTypes[0];
                  rightColumn->resultType = commonTypes[1];
                  auto commonType = SQLTypeUtils::getCommonType(leftColumn->resultType, rightColumn->resultType);
                  auto newColumnReference = std::make_shared<ast::ColumnReference>(newScopeName, commonType, leftColumn->name);

                  newColumnReference->displayName = leftColumn->displayName;
                  newTargetInfos.emplace_back(newColumnReference);
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

            default: throw std::runtime_error("QueryNodeType not implemented");
         }
      }

      default: throw std::runtime_error("Node type no implemented");
   }
}

std::shared_ptr<ast::CreateNode> SQLQueryAnalyzer::analyzeCreateNode(std::shared_ptr<ast::CreateNode> createNode) {
   switch (createNode->createInfo->type) {
      case catalog::CatalogEntry::CatalogEntryType::LINGODB_TABLE_ENTRY: {
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
                  NullableType nullableType = SQLTypeUtils::typemodsToCatalogType(columnElement->logicalTypeWithMods.logicalTypeId, columnElement->logicalTypeWithMods.typeModifiers);
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
      case catalog::CatalogEntry::CatalogEntryType::C_FUNCTION_ENTRY: {
         auto createFunctionInfo = std::static_pointer_cast<ast::CreateFunctionInfo>(createNode->createInfo);
         auto resolverScope = context->createResolverScope();
         analyzeFunctionCreate(createNode, createFunctionInfo, context, resolverScope);
         return createNode;
      }
      default: error("Create info not supported", createNode->loc);
   }
}

std::shared_ptr<ast::CreateNode> SQLQueryAnalyzer::analyzeFunctionCreate(std::shared_ptr<ast::CreateNode> createNode, std::shared_ptr<ast::CreateFunctionInfo> createFunctionInfo, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   auto logicalTypeToCTypeString = [createNode](ast::LogicalTypeWithMods& type) -> std::string {
      switch (type.logicalTypeId) {
         case catalog::LogicalTypeId::INT: {
            if (type.typeModifiers.size() == 1) {
               auto typeModifier = type.typeModifiers[0];
               if (typeModifier->type != ast::ConstantType::UINT) {
                  throw std::runtime_error("Invalid Typemodfiers for type: int");
               }
               auto size = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifier)->iVal;
               if (size>=8) {
                  return "int64_t";
               }
            }
            return "int32_t";
         }
         case catalog::LogicalTypeId::CHAR: {
            if (type.typeModifiers.empty()) {
               return "char";
            }
            size_t l = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(type.typeModifiers[0])->iVal;
            if (l>1) {
               return "char*";
            } else {
               return "char";
            }
            break;
         }
         case catalog::LogicalTypeId::STRING: {
            return "char*";
         }
         case catalog::LogicalTypeId::FLOAT: {
            if (type.typeModifiers.size() >= 1) {
               auto sizeValue = type.typeModifiers.at(0);
               if (sizeValue->type == ast::ConstantType::UINT) {
                  auto size = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(sizeValue)->iVal;
                  if (size <= 2) {
                     return "float";
                  }
               }
            }
            return "double";
         }
         case catalog::LogicalTypeId::BOOLEAN: {
            return "bool";
         }

         default: error("return type for c-udf not implemented", createNode->loc);
      }
   };
   std::string language = "unknown";
   std::string code = "";
   for (auto [optionName, optionValue] : createFunctionInfo->options) {
      if (optionName == "LANGUAGE") {
         language = optionValue;
      } else if (optionName == "AS") {
         code = optionValue;
      }
   }

   if (language != "c") {
      error("Currently only c is allowed" , createNode->loc);
   }
   std::string returnTypeStringRepresentation = logicalTypeToCTypeString(createFunctionInfo->returnType);
   NullableType returnType = SQLTypeUtils::typemodsToCatalogType(createFunctionInfo->returnType.logicalTypeId, createFunctionInfo->returnType.typeModifiers);

   std::string argumentsStringRepresentation = "(";
   for (size_t i = 0; i<createFunctionInfo->argumentTypes.size(); i++) {
      auto functionArgument = createFunctionInfo->argumentTypes[i];
      argumentsStringRepresentation+=logicalTypeToCTypeString(functionArgument.type) + " " + functionArgument.name;
      if (i+1<createFunctionInfo->argumentTypes.size()) {
         argumentsStringRepresentation+=", ";
      }
   }
   argumentsStringRepresentation+=")";

   code = returnTypeStringRepresentation + " " + createFunctionInfo->functionName + argumentsStringRepresentation +  " { " + code + "}";
   auto boundCreateFunctionInfo = std::make_shared<ast::BoundCreateFunctionInfo>(createFunctionInfo->functionName, createFunctionInfo->replace, returnType);
   boundCreateFunctionInfo->language = language;
   boundCreateFunctionInfo->code = code;
   for (auto& fArgument : createFunctionInfo->argumentTypes) {
      boundCreateFunctionInfo->argumentTypes.push_back(SQLTypeUtils::typemodsToCatalogType(fArgument.type.logicalTypeId, fArgument.type.typeModifiers).type);
   }


   createNode->createInfo = boundCreateFunctionInfo;
   return createNode;
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
   for (auto c : context->currentScope->targetInfo.targetColumns) {
      if (c->resultType.type.getTypeId() == catalog::LogicalTypeId::CHAR) {
         auto charInfo = c->resultType.type.getInfo<catalog::CharTypeInfo>();
         if (charInfo->getLength() > 8) {
            c->resultType.type = catalog::Type::stringType();
         }
      }
   }

   auto exprListTableRef = std::static_pointer_cast<ast::BoundValuesQueryNode>(boundTableProducer)->expressionListRef;
   auto rel = maybeRel.value();
   std::unordered_map<std::string, NullableType> allCollumnTypes;
   //Check for correct Type
   for (auto c : rel->getColumns()) {
      allCollumnTypes.emplace(c.getColumnName(), NullableType(c.getLogicalType(), c.getIsNullable()));
   }
   if (insertNode->columns.empty()) {
      for (auto c : rel->getColumns()) {
         insertNode->columns.emplace_back(c.getColumnName());
      }
   }

   return drv.nf.node<ast::BoundInsertNode>(insertNode->loc, insertNode->schema, insertNode->tableName, exprListTableRef, insertNode->columns, allCollumnTypes);
}
std::shared_ptr<ast::SetNode> SQLQueryAnalyzer::analyzeSetNode(std::shared_ptr<ast::SetNode> setNode) {
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
      default: error("Set type not supported", setNode->loc);
   }
}

std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzePipeOperator(std::shared_ptr<ast::PipeOperator> pipeOperator, std::shared_ptr<SQLContext>& context, ResolverScope& resolverScope) {
   std::shared_ptr<ast::AstNode> boundAstNode = pipeOperator->node;
   switch (pipeOperator->pipeOpType) {
      case ast::PipeOperatorType::SELECT: {
         assert(pipeOperator->node->nodeType == ast::NodeType::TARGET_LIST);
         auto targetSelection = std::static_pointer_cast<ast::TargetList>(pipeOperator->node);
         std::vector<std::shared_ptr<ast::ColumnReference>> targetColumns{};
         context->currentScope->targetInfo.targetColumns.clear();

         for (auto& target : targetSelection->targets) {
            auto parsedExpression = analyzeExpression(target, context, resolverScope);

            switch (parsedExpression->exprClass) {
               case ast::ExpressionClass::BOUND_COLUMN_REF: {
                  assert(parsedExpression->columnReference.has_value());
                  //ADD column_ref to targetInfo for the current scope!
                  auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(parsedExpression);
                  targetColumns.emplace_back(columnRef->columnReference.value());
                  if (!columnRef->alias.empty()) {
                     context->mapAttribute(resolverScope, columnRef->alias, columnRef->columnReference.value());
                  }
                  context->currentScope->targetInfo.add(columnRef->columnReference.value());
                  break;
               }
               case ast::ExpressionClass::BOUND_STAR: {
                  auto star = std::static_pointer_cast<ast::BoundStarExpression>(parsedExpression);
                  targetColumns.resize(star->columnReferences.size());
                  context->currentScope->targetInfo.targetColumns.resize(star->columnReferences.size());
                  std::vector<catalog::Catalog> catalogs;
                  std::string scope;
                  std::vector<catalog::Column> columns;
                  for (auto& [columnReference, index] : star->columnReferences) {
                     targetColumns[index] = columnReference;
                     context->currentScope->targetInfo.targetColumns[index] = columnReference;
                  }

                  break;
               }
               //NOTE: All other expressions should be moved into an ExtendNode or AggregationNode by canonicalize
               default: error("Invalid expression inside select clause", target->loc);
            }
         }
         boundAstNode = drv.nf.node<ast::BoundTargetList>(targetSelection->loc, targetSelection->distinct, targetColumns);
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
         std::vector<std::shared_ptr<ast::ColumnReference>> groupColumnReferences{};
         std::vector<std::shared_ptr<ast::BoundExpression>> evalBeforeAggr;

         //Clear targetinfo (see PIPE SQL Syntax)
         context->currentScope->targetInfo.targetColumns.clear();

         std::vector<std::shared_ptr<ast::BoundFunctionExpression>> boundAggregationExpressions{};
         /**
          * Analyze AggregationExpressions
          */
         bool nullable = !aggregationNode->groupByNode || aggregationNode->groupByNode->groupByExpressions.empty();
         std::ranges::transform(aggregationNode->aggregations, std::back_inserter(boundAggregationExpressions), [&](auto expr) {
            auto boundExpr = analyzeExpression(expr, context, resolverScope);
            assert(boundExpr->exprClass == ast::ExpressionClass::BOUND_FUNCTION);
            auto boundFunction = std::static_pointer_cast<ast::BoundFunctionExpression>(boundExpr);
            //Check if count
            auto fName = boundFunction->functionName;
            std::ranges::transform(fName, fName.begin(), ::toupper);
            if (fName != "COUNT" && fName != "COUNT*" && nullable) {
               boundExpr->resultType->isNullable = nullable;
               boundExpr->columnReference.value()->resultType.isNullable = nullable;
            }
            return boundFunction;
         });
         /**
          * Analyze GroupByNode
          */
         if (aggregationNode->groupByNode) {
            std::ranges::transform(aggregationNode->groupByNode->groupByExpressions, std::back_inserter(groupColumnReferences), [&](auto expr) {
               auto boundExpression = analyzeExpression(expr, context, resolverScope);
               assert(boundExpression->columnReference.has_value());
               context->mapAttribute(resolverScope, boundExpression->columnReference.value()->name, boundExpression->columnReference.value());
               //Add GROUP BY to TargetInfo for the current scope (see PIPE SQL Syntax)
               context->currentScope->targetInfo.add(boundExpression->columnReference.value());
               return boundExpression->columnReference.value();
            });
         }
         auto boundGroupByNode = drv.nf.node<ast::BoundGroupByNode>(aggregationNode->groupByNode ? aggregationNode->groupByNode->loc : aggregationNode->loc, groupColumnReferences);
         std::vector<std::shared_ptr<ast::BoundExpression>> toMap{};

         /**
          * Find Arguments of aggregation functions that need to be extended/mapped first
          */
         auto mapName = context->getUniqueScope("aggMap");
         for (auto& aggr : boundAggregationExpressions) {
            //Add Aggregations to TargetInfo for the current scope (see PIPE SQL Syntax)
            context->currentScope->targetInfo.add(aggr->columnReference.value());
            if (aggr->arguments.empty() || aggr->arguments[0]->type == ast::ExpressionType::BOUND_COLUMN_REF) {
               continue;
            }
            toMap.emplace_back(aggr->arguments[0]);
            aggr->arguments[0]->alias = context->getUniqueScope("tmp_attr");
            aggr->arguments[0]->columnReference = std::make_shared<ast::ColumnReference>(mapName, aggr->arguments[0]->resultType.value(), aggr->arguments[0]->alias);
         }

         for (auto boundAggr : boundAggregationExpressions) {
            auto fName = boundAggr->functionName;
            std::ranges::transform(fName, fName.begin(), ::toupper);
            if (fName == "COUNT" || fName == "COUNT*") {
               boundAggr->columnReference.value()->resultType.useZeroInsteadOfNull = !aggregationNode->groupByNode || aggregationNode->groupByNode->groupByExpressions.empty();
               boundAggr->resultType->useZeroInsteadOfNull = !aggregationNode->groupByNode || aggregationNode->groupByNode->groupByExpressions.empty();
            }
         }

         auto boundAggrNode = drv.nf.node<ast::BoundAggregationNode>(pipeOperator->loc, boundGroupByNode, boundAggregationExpressions, toMap, mapName, evalBeforeAggr);
         boundAstNode = boundAggrNode;
         /**
          * Handle grouping sets
          */
         if (aggregationNode->groupByNode && !aggregationNode->groupByNode->groupingSet.empty()) {
            static size_t groupingSetId = 0;
            auto groupingSets = aggregationNode->groupByNode->groupingSet;
            for (size_t i = 0; i < groupingSets.size(); i++) {
               auto groupingSet = groupingSets[i];
               std::vector<std::shared_ptr<ast::ColumnReference>> localGroupBy{};
               std::vector<std::shared_ptr<ast::ColumnReference>> mapToNull{};
               std::vector<std::shared_ptr<ast::ColumnReference>> notAvailable{};
               int present = 0;

               for (size_t j = 0; j < aggregationNode->groupByNode->groupByExpressions.size(); j++) {
                  if (groupingSet.contains(j)) {
                     localGroupBy.emplace_back(groupColumnReferences[j]);
                     auto mappedColumnReference = std::make_shared<ast::ColumnReference>("groupingSetMapToNull_" + std::to_string(groupingSetId), groupColumnReferences[j]->resultType, "tmp_" + std::to_string(j));
                     mappedColumnReference->displayName = groupColumnReferences[j]->displayName;
                     mapToNull.emplace_back(mappedColumnReference);
                     mappedColumnReference->resultType.isNullable = true;
                  } else {
                     present |= (1 << j);
                     auto mappedColumnReference = std::make_shared<ast::ColumnReference>("groupingSet_" + std::to_string(groupingSetId), groupColumnReferences[j]->resultType, "tmp_" + std::to_string(j));
                     notAvailable.emplace_back(mappedColumnReference);
                     mappedColumnReference->displayName = groupColumnReferences[j]->displayName;
                     mappedColumnReference->resultType.isNullable = true;
                  }
               }

               std::vector<std::shared_ptr<ast::ColumnReference>> aggregationColumnReferences{};
               for (auto& aggr : boundAggrNode->aggregations) {
                  auto columnReferenceAggr = std::make_shared<ast::ColumnReference>("groupingSetAgg_" + std::to_string(groupingSetId), aggr->columnReference.value()->resultType, aggr->columnReference.value()->name);
                  columnReferenceAggr->displayName = aggr->columnReference.value()->displayName;
                  aggregationColumnReferences.emplace_back(columnReferenceAggr);
               }
               boundAggrNode->groupByNode->localAggregationColumnReferences.emplace_back(std::move(aggregationColumnReferences));

               boundAggrNode->groupByNode->localGroupByColumnReferences.emplace_back(std::move(localGroupBy));
               boundAggrNode->groupByNode->localMapToNullColumnReferences.emplace_back(std::move(mapToNull));
               boundAggrNode->groupByNode->localNotAvailableColumnReferences.emplace_back(std::move(notAvailable));
               auto presentColumnReference = std::make_shared<ast::ColumnReference>(boundAggrNode->mapName, catalog::Type::int64(), "intval" + std::to_string(present));
               boundAggrNode->groupByNode->localPresentIntval.emplace_back(std::pair{present, presentColumnReference});
               groupingSetId++;
            }

            std::vector<std::shared_ptr<ast::ColumnReference>> currentAttributes(
               boundAggrNode->groupByNode->localMapToNullColumnReferences.at(0).begin(),
               boundAggrNode->groupByNode->localMapToNullColumnReferences.at(0).end());

            currentAttributes.insert(currentAttributes.end(),
                                     boundAggrNode->groupByNode->localNotAvailableColumnReferences.at(0).begin(),
                                     boundAggrNode->groupByNode->localNotAvailableColumnReferences.at(0).end());

            currentAttributes.insert(currentAttributes.end(),
                                     boundAggrNode->groupByNode->localAggregationColumnReferences.at(0).begin(),
                                     boundAggrNode->groupByNode->localAggregationColumnReferences.at(0).end());
            currentAttributes.emplace_back(boundAggrNode->groupByNode->localPresentIntval.at(0).second);
            for (size_t i = 1; i < boundAggrNode->groupByNode->localGroupByColumnReferences.size(); i++) {
               auto rollUpUnionName = context->getUniqueScope("rollupUnion");
               std::vector<std::shared_ptr<ast::ColumnReference>> currentLocalAttributes(
                  boundAggrNode->groupByNode->localMapToNullColumnReferences.at(i).begin(),
                  boundAggrNode->groupByNode->localMapToNullColumnReferences.at(i).end());
               currentLocalAttributes.insert(currentLocalAttributes.end(),
                                             boundAggrNode->groupByNode->localNotAvailableColumnReferences.at(i).begin(),
                                             boundAggrNode->groupByNode->localNotAvailableColumnReferences.at(i).end());
               currentLocalAttributes.insert(currentLocalAttributes.end(),
                                             boundAggrNode->groupByNode->localAggregationColumnReferences.at(i).begin(),
                                             boundAggrNode->groupByNode->localAggregationColumnReferences.at(i).end());
               currentLocalAttributes.emplace_back(boundAggrNode->groupByNode->localPresentIntval.at(i).second);

               std::vector<std::shared_ptr<ast::ColumnReference>> unionColumnReferences{};
               for (size_t j = 0; j < currentLocalAttributes.size(); j++) {
                  auto left = currentAttributes[j];
                  auto right = currentLocalAttributes[j];
                  auto unionColumnReference = std::make_shared<ast::ColumnReference>(rollUpUnionName + std::to_string(i), right->resultType, left->name);
                  unionColumnReferences.emplace_back(unionColumnReference);
                  unionColumnReference->displayName = left->displayName;
               }
               currentAttributes = unionColumnReferences;

               boundAggrNode->groupByNode->unionColumnReferences.emplace_back(std::move(unionColumnReferences));
            }
            for (size_t i = 0; i < boundAggrNode->groupByNode->groupByColumnReferences.size(); i++) {
               auto old = boundAggrNode->groupByNode->groupByColumnReferences[i];
               auto newN = boundAggrNode->groupByNode->unionColumnReferences.back().at(i);
               context->replace(resolverScope, old, newN);
            }
            for (size_t i = 0; i < boundAggrNode->aggregations.size(); i++) {
               auto old = boundAggrNode->aggregations[i]->columnReference.value();
               auto newN = boundAggrNode->groupByNode->unionColumnReferences.back().at(boundAggrNode->groupByNode->groupByColumnReferences.size() + i);
               context->replace(resolverScope, old, newN);
            }

            std::vector<std::shared_ptr<ast::BoundFunctionExpression>> boundGroupingFunctions;
            for (size_t i = 0; i < aggregationNode->groupByNode->groupingFunctions.size(); i++) {
               auto boundGroupingFunction = analyzeExpression(*std::next(aggregationNode->groupByNode->groupingFunctions.begin(), i), context, resolverScope);
               boundGroupingFunctions.emplace_back(std::static_pointer_cast<ast::BoundFunctionExpression>(boundGroupingFunction));
               context->mapAttribute(resolverScope, boundGroupingFunction->alias, boundGroupingFunction->columnReference.value());

               assert(std::static_pointer_cast<ast::BoundFunctionExpression>(boundGroupingFunction)->arguments[0]->columnReference.has_value());
               auto functionArgColumnReference = std::static_pointer_cast<ast::BoundFunctionExpression>(boundGroupingFunction)->arguments[0];
               ;
               size_t j = 0;
               for (; j < boundAggrNode->groupByNode->groupByColumnReferences.size(); j++) {
                  auto groupColumnReference = boundAggrNode->groupByNode->unionColumnReferences.back().at(j);
                  if (groupColumnReference->name == functionArgColumnReference->columnReference.value()->name) {
                     break;
                  }
               }

               boundAggrNode->groupByNode->groupingFunctions.emplace_back(std::pair{j, boundGroupingFunction->columnReference.value()});
            }

            groupingSetId++;
         }

         break;
      }

      case ast::PipeOperatorType::EXTEND: {
         assert(pipeOperator->node->nodeType == ast::NodeType::EXTEND_NODE);
         auto extendNode = std::static_pointer_cast<ast::ExtendNode>(pipeOperator->node);
         auto mapName = context->getUniqueScope("map");
         std::vector<std::shared_ptr<ast::BoundExpression>> boundExtensions;
         std::vector<std::shared_ptr<ast::BoundExpression>> boundExpressions;
         std::vector<std::shared_ptr<ast::BoundWindowExpression>> boundWindowExpressions;

         for (auto& expr : extendNode->extensions) {
            auto boundExpression = analyzeExpression(expr, context, resolverScope);
            boundExtensions.emplace_back(boundExpression);
            switch (boundExpression->exprClass) {
               case ast::ExpressionClass::BOUND_STAR:
               case ast::ExpressionClass::BOUND_COLUMN_REF: {
                  assert(boundExpression->columnReference.has_value());
                  context->mapAttribute(resolverScope, boundExpression->columnReference.value()->displayName, boundExpression->columnReference.value());
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(boundExpression->columnReference.value());
                  }
                  break;
               }
               case ast::ExpressionClass::BOUND_FUNCTION: {
                  assert(boundExpression->resultType.has_value() && boundExpression->columnReference.has_value());
                  auto function = std::static_pointer_cast<ast::BoundFunctionExpression>(boundExpression);

                  auto fName = function->alias.empty() ? function->functionName : function->alias;
                  context->mapAttribute(resolverScope, fName, function->columnReference.value());
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(boundExpression->columnReference.value());
                  }
                  boundExpressions.emplace_back(function);
                  break;
               }

               case ast::ExpressionClass::BOUND_WINDOW: {
                  assert(boundExpression->resultType.has_value() && boundExpression->columnReference.has_value());
                  auto window = std::static_pointer_cast<ast::BoundWindowExpression>(boundExpression);

                  auto fName = window->alias;
                  context->mapAttribute(resolverScope, fName, window->columnReference.value());
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(boundExpression->columnReference.value());
                  }
                  boundWindowExpressions.emplace_back(window);
                  break;
               }
               case ast::ExpressionClass::BOUND_CONJUNCTION:
               case ast::ExpressionClass::BOUND_BETWEEN:
               case ast::ExpressionClass::BOUND_COMPARISON:
               case ast::ExpressionClass::BOUND_CONSTANT:
               case ast::ExpressionClass::BOUND_OPERATOR:
               case ast::ExpressionClass::BOUND_CAST:
               case ast::ExpressionClass::BOUND_SUBQUERY:
               case ast::ExpressionClass::BOUND_CASE: {
                  assert(boundExpression->resultType.has_value());
                  auto scope = boundExpression->alias.empty() ? boundExpression->alias : mapName;
                  auto resultType = boundExpression->resultType.value();
                  if (resultType.useZeroInsteadOfNull) {
                     resultType.isNullable = false;
                  }
                  auto n = std::make_shared<ast::ColumnReference>(scope, resultType, context->getUniqueScope("tmp_attr"));
                  n->displayName = boundExpression->alias.empty() ? "" : boundExpression->alias;
                  context->mapAttribute(resolverScope, boundExpression->alias.empty() ? n->name : boundExpression->alias, n);
                  if (extendNode->hidden) {
                     context->definedAttributes.top().pop_back();
                  } else {
                     context->currentScope->targetInfo.add(n);
                  }

                  boundExpression->columnReference = n;

                  boundExpressions.emplace_back(boundExpression);
                  break;
               }

               default: error("Invalid expression", boundExpression->loc);
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
         auto targets = std::static_pointer_cast<ast::TargetList>(pipeOperator->node);
         for (auto& target : targets->targets) {
            if (target->exprClass != ast::ExpressionClass::COLUMN_REF) {
               error("Only column references are allowed in DROP", target->loc);
            }
            auto boundExpression = analyzeExpression(target, context, resolverScope);
            assert(boundExpression->columnReference.has_value());
            std::erase_if(context->currentScope->targetInfo.targetColumns, [&](const std::shared_ptr<ast::ColumnReference>& other) {
               return *other == *boundExpression->columnReference.value();
            });
         }
         return pipeOperator->input;
      }
      case ast::PipeOperatorType::SET: {
         auto setExpression = std::static_pointer_cast<ast::SetColumnExpression>(pipeOperator->node);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundExpressions{};
         for (auto [columnRef, expr] : setExpression->sets) {
            auto boundExpression = analyzeExpression(expr, context, resolverScope);
            auto boundColumnRef = analyzeExpression(columnRef, context, resolverScope);

            assert(boundColumnRef->columnReference.has_value());
            context->mapAttribute(resolverScope, boundColumnRef->columnReference.value()->name, boundColumnRef->columnReference.value());
            boundExpressions.emplace_back(boundExpression);
            boundExpression->columnReference = boundColumnRef->columnReference.value();
            boundColumnRef->columnReference.value()->resultType = boundExpression->resultType.value();
         }
         boundAstNode = drv.nf.node<ast::BoundSetColumnExpression>(setExpression->loc, context->getUniqueScope("setMap"), boundExpressions);

         break;
      }
      default: error("pipe operator not implemented", pipeOperator->loc);
   }
   pipeOperator->node = boundAstNode;
   return pipeOperator;
}

std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeTableRef(std::shared_ptr<ast::TableRef> tableRef, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   if (stackGuard->newStackNeeded()) {
      boost::context::fixedsize_stack salloc(1024 * 1024);
      boost::context::stack_context sctx = salloc.allocate();
      auto sGuard = stackGuard;
      stackGuard = std::make_shared<StackGuardFiber>(sctx);
      void* sp = static_cast<char*>(sctx.sp);
      std::size_t size = sctx.size;
      std::shared_ptr<ast::TableProducer> boundTableRef;

      boost::context::fiber f(std::allocator_arg, boost::context::preallocated(sp, size, sctx), salloc, [&](boost::context::fiber&& sink) {
         boundTableRef = analyzeTableRef(tableRef, context, resolverScope);
         return std::move(sink);
      });

      f = std::move(f).resume();
      stackGuard = sGuard;

      return boundTableRef;
   }
   switch (tableRef->type) {
      case ast::TableReferenceType::BASE_TABLE: {
         auto baseTableRef = std::static_pointer_cast<ast::BaseTableRef>(tableRef);
         return analyzeBaseTableRef(baseTableRef, context, resolverScope);
      }
      case ast::TableReferenceType::CROSS_PRODUCT: {
         auto crossProdRef = std::static_pointer_cast<ast::CrossProductRef>(tableRef);
         std::vector<std::shared_ptr<ast::TableProducer>> boundTableRefs;
         std::ranges::transform(crossProdRef->tables, std::back_inserter(boundTableRefs), [&](auto table) {
            return analyzeTableProducer(table, context, resolverScope);
         });
         return drv.nf.node<ast::BoundCrossProductRef>(crossProdRef->loc, boundTableRefs);
      }
      case ast::TableReferenceType::JOIN: {
         auto join = std::static_pointer_cast<ast::JoinRef>(tableRef);
         return analyzeJoinRef(join, context, resolverScope);
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
         return analyzeExpressionListRef(expressionListRef, context, resolverScope);
      }

      default: error("Table reference not implemented", tableRef->loc);
   }
}

std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeBaseTableRef(std::shared_ptr<ast::BaseTableRef> baseTableRef, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   auto catalogEntry = catalog->getTypedEntry<catalog::TableCatalogEntry>(baseTableRef->tableName);
   //Add to current scope
   auto sqlScopeName = baseTableRef->alias.empty() ? baseTableRef->tableName : baseTableRef->alias;
   //Get unique mlirScope
   auto uniqueScope = context->getUniqueScope(sqlScopeName);
   if (!catalogEntry.has_value()) {
      //Check for cte
      if (context->ctes.contains(baseTableRef->tableName)) {
         auto [cteInfo, cteNode] = context->ctes.at(baseTableRef->tableName);

         std::vector<std::shared_ptr<ast::ColumnReference>> columnReferences{};

         std::ranges::transform(cteNode->renamedColumnReferences, std::back_inserter(columnReferences), [&](auto& pair) {
            auto columnReference = std::make_shared<ast::ColumnReference>(context->getUniqueScope(baseTableRef->tableName), pair.second->resultType, pair.second->name);
            columnReference->displayName = pair.second->displayName;

            return columnReference;
         });

         for (auto& columnReference : columnReferences) {
            context->currentScope->targetInfo.add(columnReference);
         }

         context->mapAttribute(resolverScope, sqlScopeName, columnReferences);

         auto boundBaseTableRef = drv.nf.node<ast::BoundBaseTableRef>(baseTableRef->loc, columnReferences, baseTableRef->alias, baseTableRef->tableName, uniqueScope);
         return boundBaseTableRef;
      } else {
         error("No Catalog found with name " + baseTableRef->tableName, baseTableRef->loc);
      }

   } else {
      auto columnReferences = context->mapAttribute(resolverScope, sqlScopeName, uniqueScope, catalogEntry.value());
      for (auto& columnReference : columnReferences) {
         context->currentScope->targetInfo.add(columnReference);
      }

      auto boundBaseTableRef = drv.nf.node<ast::BoundBaseTableRef>(baseTableRef->loc, columnReferences, baseTableRef->alias, catalogEntry.value()->getName(), uniqueScope);
      return boundBaseTableRef;
   }
}

std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeInnerJoin(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   std::shared_ptr<ast::TableProducer> left, right;
   std::shared_ptr<SQLScope> leftScope, rightScope;

   std::vector<std::pair<std::string, std::shared_ptr<ast::ColumnReference>>> mapping{};
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
         error("Invalid join condition", join->loc);
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
std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeLeftOuterJoin(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   std::shared_ptr<ast::TableProducer> left, right;
   std::vector<std::pair<std::string, std::shared_ptr<ast::ColumnReference>>> mapping{};
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
         error("Invalid join condition", join->loc);
      }

      boundCondition = analyzeExpression(std::get<std::shared_ptr<ast::ParsedExpression>>(join->condition), context, resolverScope);
   }

   std::vector<std::pair<std::shared_ptr<ast::ColumnReference>, std::shared_ptr<ast::ColumnReference>>> outerJoinMapping;
   std::string outerjoinName;
   static size_t id = 0;
   if (!mapping.empty()) {
      outerjoinName = "oj" + std::to_string(id++);
      std::unordered_map<std::shared_ptr<ast::ColumnReference>, std::shared_ptr<ast::ColumnReference>> remapped;
      for (auto x : mapping) {
         auto it = remapped.find(x.second);
         if (it == remapped.end()) {
            auto scope = x.second->scope;
            auto name = x.second->name;
            auto columnReference = std::make_shared<ast::ColumnReference>(outerjoinName, x.second->resultType, name);

            //Make mapping output nullable
            columnReference->resultType.isNullable = true;
            columnReference->displayName = x.second->displayName;
            outerJoinMapping.push_back({x.second, columnReference});
            remapped.insert({x.second, columnReference});
            context->mapAttribute(resolverScope, x.first, columnReference);
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
std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeFullOuterJoin(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   std::shared_ptr<ast::TableProducer> left, right;
   std::vector<std::pair<std::string, std::shared_ptr<ast::ColumnReference>>> mapping;
   std::shared_ptr<SQLScope> leftScope, rightScope;

   {
      auto rightContext = std::make_shared<SQLContext>();
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
   std::vector<std::pair<std::shared_ptr<ast::ColumnReference>, std::shared_ptr<ast::ColumnReference>>> outerJoinMapping;
   std::string outerjoinName;
   static size_t id = 0;
   if (!mapping.empty()) {
      outerjoinName = "foj" + std::to_string(id++);
      //Remap all attributes to the new named result: remapped.first = original, remapped.second = new named result
      std::unordered_map<std::shared_ptr<ast::ColumnReference>, std::shared_ptr<ast::ColumnReference>> remapped;
      for (auto x : mapping) {
         auto it = remapped.find(x.second);
         if (it == remapped.end()) {
            auto scope = x.second->scope;
            auto name = x.second->name + "_" + std::to_string(id++);
            auto columnReference = std::make_shared<ast::ColumnReference>(outerjoinName, x.second->resultType, name);

            //Make mapping output nullable
            columnReference->resultType.isNullable = true;
            columnReference->displayName = x.second->displayName;
            outerJoinMapping.push_back({x.second, columnReference});
            remapped.insert({x.second, columnReference});
            context->mapAttribute(resolverScope, x.first, columnReference);
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
std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeJoinRef(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   switch (join->type) {
      case ast::JoinType::INNER: {
         return analyzeInnerJoin(join, context, resolverScope);
      }
      case ast::JoinType::RIGHT: {
         throw std::runtime_error("Should not happen");
      }
      case ast::JoinType::LEFT: {
         return analyzeLeftOuterJoin(join, context, resolverScope);
      }
      case ast::JoinType::FULL: {
         return analyzeFullOuterJoin(join, context, resolverScope);
      }

      default: error("Join type not implemented", join->loc);
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

               std::shared_ptr<ast::ColumnReference> columnReference = nullptr;

               switch (boundExpression->type) {
                  case ast::ExpressionType::BOUND_COLUMN_REF: {
                     auto columnRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(boundExpression);
                     assert(columnRef->columnReference.has_value());
                     columnReference = columnRef->columnReference.value();
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
                     if (context->currentScope->targetInfo.targetColumns.size() < static_cast<size_t>(constantValue->iVal) || constantValue->iVal <= 0) {
                        error("Invalid order by element", boundConstant->loc);
                     }
                     columnReference = context->currentScope->targetInfo.targetColumns.at(constantValue->iVal - 1);
                     break;
                  }
                  default: error("Order by element not implemented", orderByElement->expression->loc);
               }
               assert(columnReference);
               auto boundOrderByElement = drv.nf.node<ast::BoundOrderByElement>(orderByElement->loc, orderByElement->type, orderByElement->nullOrder, columnReference);
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
      default: error("Result modifier not implemented", resultModifier->loc);
   }
}

std::shared_ptr<ast::TableProducer> SQLQueryAnalyzer::analyzeExpressionListRef(std::shared_ptr<ast::ExpressionListRef> expressionListRef, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   if (expressionListRef->values.empty() || expressionListRef->values[0].empty()) {
      error("Expression list is empty", expressionListRef->loc);
   }

   std::vector<std::vector<std::shared_ptr<ast::BoundConstantExpression>>> boundValues{};
   size_t sizePerExprList = expressionListRef->values[0].size();
   std::vector<std::vector<NullableType>> types{sizePerExprList};

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
   std::vector<NullableType> commonTypes{};
   std::ranges::transform(types, std::back_inserter(commonTypes), [&](auto& typeList) {
      auto t = SQLTypeUtils::getCommonBaseType(typeList);
      SQLTypeUtils::toCommonTypes(typeList);

      return t;
   });
   std::vector<std::shared_ptr<ast::ColumnReference>> columnReferences{};
   auto scope = context->getUniqueScope("constantTable");
   for (size_t i = 0; i < commonTypes.size(); i++) {
      auto name = context->getUniqueScope("const");
      auto columnReference = std::make_shared<ast::ColumnReference>(scope, commonTypes[i], name);
      columnReferences.push_back(columnReference);
      context->currentScope->targetInfo.add(columnReference);
   }
   context->mapAttribute(resolverScope, scope, columnReferences);

   return drv.nf.node<ast::BoundExpressionListRef>(expressionListRef->loc, boundValues, columnReferences);
}

/*
* Expressions
 */
std::shared_ptr<ast::BoundExpression> SQLQueryAnalyzer::analyzeExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   if (stackGuard->newStackNeeded()) {
      boost::context::fixedsize_stack salloc(1024 * 1024);
      boost::context::stack_context sctx = salloc.allocate();
      auto sGuard = stackGuard;
      stackGuard = std::make_shared<StackGuardFiber>(sctx);
      void* sp = static_cast<char*>(sctx.sp);
      std::size_t size = sctx.size;
      std::shared_ptr<ast::BoundExpression> boundExpression;

      boost::context::fiber f(std::allocator_arg, boost::context::preallocated(sp, size, sctx), salloc, [&](boost::context::fiber&& sink) {
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
         NullableType type = catalog::Type::int64();
         switch (constExpr->value->type) {
            case ast::ConstantType::INT:
               type = catalog::Type::int32();
               break;
            case ast::ConstantType::STRING: {
               auto strValue = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
               type = catalog::Type::charType(strValue.length());
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
            case ast::ConstantType::BOOLEAN: {
               type = catalog::Type::boolean();
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
         std::unordered_set<std::pair<std::shared_ptr<ast::ColumnReference>, size_t>, ast::ColumnRefHash, ast::ColumnRefEq> topDefinedColumnsWithoutDuplicates;

         /**
          * Why is this elimination of duplactes needed:
          * The context stores each column (of a table) in definedAttributes with two keys: columnName and tableName.columnName
          * Therefore you must elimniate these duplactes for statements like Select * ....
          */
         size_t i = 0;
         for (auto& [scope, columnReference] : topDefinedColumnsAll) {
            auto p = topDefinedColumnsWithoutDuplicates.insert({columnReference, i});
            if (p.second) {
               i++;
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
         }

         auto left = analyzeExpression(comparison->left, context, resolverScope);
         std::vector<std::shared_ptr<ast::BoundExpression>> boundRightChildren{};
         std::ranges::transform(comparison->rightChildren, std::back_inserter(boundRightChildren), [&](auto& child) {
            return analyzeExpression(child, context, resolverScope);
         });
         if (!left->resultType.has_value()) {
            error("Left side of comparison is not a valid expression", comparison->left->loc);
         }
         std::vector<NullableType> types{};
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
               error("Conjunction is only possible with children of type boolean", expr->loc);
            }
         }
         return drv.nf.node<ast::BoundConjunctionExpression>(conjunction->loc, conjunction->type, conjunction->alias, boundChildren);
         break;
      }
      case ast::ExpressionClass::OPERATOR: {
         auto operatorExpr = std::static_pointer_cast<ast::OperatorExpression>(rootNode);
         return analyzeOperatorExpression(operatorExpr, context, resolverScope);
      }
      case ast::ExpressionClass::FUNCTION: {
         auto function = std::static_pointer_cast<ast::FunctionExpression>(rootNode);
         return analyzeFunctionExpression(function, context, resolverScope);
      }
      case ast::ExpressionClass::CAST: {
         auto castExpr = std::static_pointer_cast<ast::CastExpression>(rootNode);
         return analyzeCastExpression(castExpr, context, resolverScope);
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

         auto columnReference = subqueryTargetInfo.targetColumns[0];
         auto x = std::make_shared<ast::ColumnReference>(columnReference->scope, columnReference->resultType, columnReference->name);
         auto resultType = columnReference->resultType;
         resultType.isNullable = true;
         if (subqueryExpr->subQueryType != ast::SubqueryType::SCALAR) {
            resultType = catalog::Type::boolean();
         }
         std::shared_ptr<ast::BoundExpression> boundToTestExpr = nullptr;
         if (subqueryExpr->testExpr) {
            boundToTestExpr = analyzeExpression(subqueryExpr->testExpr, context, resolverScope);
         }

         auto boundSubqueryExpression = drv.nf.node<ast::BoundSubqueryExpression>(subqueryExpr->loc, subqueryExpr->subQueryType, resultType, subqueryExpr->alias, columnReference, subqueryScope, boundSubquery, boundToTestExpr);
         boundSubqueryExpression->columnReference = x;
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
         std::vector<NullableType> thenTypes{};
         std::vector<NullableType> whenTypes{};
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
         return analyzeWindowExpression(windowExpr, context, resolverScope);
      }
      default: error("Expression type not implemented", rootNode->loc);
   }
}

std::shared_ptr<ast::BoundExpression> SQLQueryAnalyzer::analyzeOperatorExpression(std::shared_ptr<ast::OperatorExpression> operatorExpr, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
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
   std::vector<NullableType> types{};
   std::ranges::transform(boundChildren, std::back_inserter(types), [](auto c) {
      return c->resultType.value();
   });

   auto commonNumbers = SQLTypeUtils::toCommonNumber(types);
   std::vector<NullableType> castValues{};
   std::ranges::transform(commonNumbers, std::back_inserter(castValues), [](auto c) {
      if (c.castType) {
         return *c.castType;
      }
      return c;
   });
   NullableType resultType = SQLTypeUtils::getCommonBaseType(castValues, operatorExpr->type);
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
         resultType = NullableType(catalog::Type::stringType(), resultType.isNullable);

         break;
      }
      case ast::ExpressionType::OPERATOR_NOT: {
         resultType = NullableType(catalog::Type::boolean());
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

std::shared_ptr<ast::BoundExpression> SQLQueryAnalyzer::analyzeWindowExpression(std::shared_ptr<ast::WindowExpression> windowExpr, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
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
   auto boundWindowFrame = std::make_shared<ast::BoundWindowFrame>();

   if (!windowExpr->windowFrame) {
      windowExpr->windowFrame = drv.nf.node<ast::WindowFrame>(windowExpr->loc, ast::WindowFrameType::UNBOUNDED_PRECEDING);
      if (windowExpr->order.has_value()) {
         windowExpr->windowFrame->end = ast::WindowFrameType::CURRENT_ROW;
      } else {
         windowExpr->windowFrame->end = ast::WindowFrameType::UNBOUNDED_FOLLOWING;
      }

      windowExpr->windowFrame->windowMode = ast::WindowMode::ROWS;
   }

   boundWindowFrame->loc = windowExpr->windowFrame->loc;
   boundWindowFrame->windowMode = windowExpr->windowFrame->windowMode;
   //Start
   switch (windowExpr->windowFrame->start) {
      case ast::WindowFrameType::CURRENT_ROW: {
         boundWindowFrame->start = 0;
         break;
      }
      case ast::WindowFrameType::EXPR_PRECEDING:
      case ast::WindowFrameType::EXPR_FOLLOWING: {
         assert(windowExpr->windowFrame->startExpr);
         assert(windowExpr->windowFrame->startExpr->exprClass == ast::ExpressionClass::CONSTANT);

         auto constantExpr = std::static_pointer_cast<ast::BoundConstantExpression>(analyzeExpression(windowExpr->windowFrame->startExpr, context, resolverScope));
         if (constantExpr->value->type != ast::ConstantType::INT) {
            error("unsupported window start specification", boundWindowFrame->loc);
         }
         boundWindowFrame->start = std::static_pointer_cast<ast::IntValue>(constantExpr->value)->iVal;
         if (windowExpr->windowFrame->start == ast::WindowFrameType::EXPR_PRECEDING) {
            boundWindowFrame->start = -boundWindowFrame->start;
         }
         break;
      }

      case ast::WindowFrameType::INVALID: {
         error("Invalid boundary type", boundWindowFrame->loc) break;
      }
      default:;
   }
   //End
   switch (windowExpr->windowFrame->end) {
      case ast::WindowFrameType::CURRENT_ROW: {
         boundWindowFrame->end = 0;
         break;
      }
      case ast::WindowFrameType::EXPR_PRECEDING:
      case ast::WindowFrameType::EXPR_FOLLOWING: {
         assert(windowExpr->windowFrame->endExpr);
         assert(windowExpr->windowFrame->endExpr->exprClass == ast::ExpressionClass::CONSTANT);

         auto constantExpr = std::static_pointer_cast<ast::BoundConstantExpression>(analyzeExpression(windowExpr->windowFrame->endExpr, context, resolverScope));
         if (constantExpr->value->type != ast::ConstantType::INT) {
            error("unsupported window start specification", boundWindowFrame->loc);
         }
         boundWindowFrame->end = std::static_pointer_cast<ast::IntValue>(constantExpr->value)->iVal;
         if (windowExpr->windowFrame->end == ast::WindowFrameType::EXPR_PRECEDING) {
            boundWindowFrame->end = -boundWindowFrame->end;
         }
         break;
      }

      case ast::WindowFrameType::INVALID: {
         error("Invalid boundary type", boundWindowFrame->loc) break;
      }
      default:;
   }

   catalog::Type resultType = catalog::Type::int64();
   //Aggregation functions used with window must be nullable
   std::string fName = boundFunction->functionName;
   std::ranges::transform(fName, fName.begin(), ::toupper);
   if (fName != "RANK" && fName!= "ROW_NUMBER" && fName != "COUNT" && fName != "COUNT*") {
      boundFunction->columnReference.value()->resultType.isNullable = true;
      boundFunction->resultType->isNullable = true;
   }

   boundFunction->columnReference.value()->displayName = windowExpr->alias;
   context->mapAttribute(resolverScope, windowExpr->alias, boundFunction->columnReference.value());

   auto boundWindowExpression = drv.nf.node<ast::BoundWindowExpression>(windowExpr->loc, windowExpr->type, windowExpr->alias, resultType, boundFunction, boundPartitions, boundOrderByModifier, boundWindowFrame);
   boundWindowExpression->columnReference = boundFunction->columnReference.value();

   return boundWindowExpression;
}

std::shared_ptr<ast::BoundExpression> SQLQueryAnalyzer::analyzeCastExpression(std::shared_ptr<ast::CastExpression> castExpr, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   auto boundChild = analyzeExpression(castExpr->child, context, resolverScope);
   if (!castExpr->logicalTypeWithMods.has_value()) {
      error("Cast expression must have logicalType", castExpr->loc);
   }
   switch (castExpr->logicalTypeWithMods.value().logicalTypeId) {
      case catalog::LogicalTypeId::DATE: {
         switch (boundChild->type) {
            case ast::ExpressionType::VALUE_CONSTANT: {
               auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(boundChild);
               if (constExpr->value->type != ast::ConstantType::STRING) {
                  error("Cannot cast non to date", constExpr->loc);
               }
               std::string stringRep = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
               stringRep += "days";
               constExpr->resultType = catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY));
               constExpr->value = std::make_shared<ast::DateValue>(std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal);
               return drv.nf.node<ast::BoundCastExpression>(castExpr->loc, catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY)), castExpr->alias, boundChild, castExpr->logicalTypeWithMods, stringRep);
            }
            case ast::ExpressionType::BOUND_COLUMN_REF: {
               auto boundColRef = std::static_pointer_cast<ast::BoundColumnRefExpression>(boundChild);
               assert(boundColRef->resultType.has_value());
               if (boundColRef->resultType.value().type.getTypeId() == catalog::LogicalTypeId::DATE) {
                  return boundColRef;
               }
               error("Cannot cast " + boundColRef->alias + " to date", boundColRef->loc);
            }
            default: error("Expression cannot be casted to date (unsupported type)", castExpr->loc);
         }
      }
      case catalog::LogicalTypeId::INTERVAL: {
         auto constExpr = std::static_pointer_cast<ast::BoundConstantExpression>(boundChild);
         if (constExpr->value->type != ast::ConstantType::STRING) {
            error("Cannot cast non String to date", constExpr->loc);
         }
         //!Shortcutted here, implement different interval types later
         auto resultType = catalog::Type::intervalDaytime();
         std::string stringRepresentation = std::static_pointer_cast<ast::StringValue>(constExpr->value)->sVal;
         if (castExpr->logicalTypeWithMods->typeModifiers.size() == 1) {
            auto typeMod = castExpr->logicalTypeWithMods->typeModifiers[0];
            if (typeMod->type != ast::ConstantType::STRING) {
               error("Interval type modifier must be string", castExpr->loc);
            }
            auto typeModStr = std::static_pointer_cast<ast::StringValue>(typeMod)->sVal;
            stringRepresentation += typeModStr;
         } else if (castExpr->logicalTypeWithMods->typeModifiers.size() > 1) {
            error("Interval with more than one type modifier not supported yet", castExpr->loc);
         }
         if (stringRepresentation.ends_with("years")) {
            stringRepresentation.erase(stringRepresentation.size() - 5);
            resultType = catalog::Type::intervalMonths();
            stringRepresentation = std::to_string(std::stol(stringRepresentation) * 12);
            stringRepresentation += "years";
         } else if (stringRepresentation.ends_with("days")) {
         } else if (stringRepresentation.ends_with("months")) {
            resultType = catalog::Type::intervalMonths();
         } else {
            error("Cast for strRep: " << stringRepresentation << " not implemented", boundChild->loc);
         }
         boundChild->resultType = resultType;
         ast::Interval interval{};
         interval.stringRepresentation = stringRepresentation;
         constExpr->value = std::make_shared<ast::IntervalValue>(interval);
         auto boundCast = drv.nf.node<ast::BoundCastExpression>(castExpr->loc, resultType, castExpr->alias, boundChild, castExpr->logicalTypeWithMods, stringRepresentation);

         return boundCast;
      }

      default: {
         auto castType = SQLTypeUtils::typemodsToCatalogType(castExpr->logicalTypeWithMods.value().logicalTypeId, castExpr->logicalTypeWithMods.value().typeModifiers);
         if (castType != boundChild->resultType.value()) {
            castType.isNullable = boundChild->resultType.value().isNullable;
            if (boundChild->type == ast::ExpressionType::VALUE_CONSTANT) {
               boundChild->resultType = castType;
               boundChild->resultType->isNullable = false;
            }
            return drv.nf.node<ast::BoundCastExpression>(castExpr->loc, castType, castExpr->alias, boundChild, castExpr->logicalTypeWithMods, "");
         } else {
            return boundChild;
         }
      };
   }
}

std::shared_ptr<ast::BoundExpression> SQLQueryAnalyzer::analyzeFunctionExpression(std::shared_ptr<ast::FunctionExpression> function, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope) {
   std::vector<std::shared_ptr<ast::BoundExpression>> boundArguments{};
   std::string upperCaseFName = function->functionName;
   std::ranges::transform(upperCaseFName, upperCaseFName.begin(), ::toupper);
   if (function->type == ast::ExpressionType::AGGREGATE) {
      auto scope = context->getUniqueScope("tmp_attr");
      auto fName = function->alias.empty() ? function->functionName : function->alias;
      std::shared_ptr<ast::BoundFunctionExpression> boundFunctionExpression = nullptr;
      NullableType resultType{catalog::Type::noneType()};

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
      if (upperCaseFName== "SUM" || upperCaseFName == "AVG" || upperCaseFName== "MIN" || upperCaseFName== "MAX") {
         if (function->arguments.size() > 1) {
            error("Aggregation with more than one argument not supported", function->loc);
         }
         if (!boundArguments[0]->resultType.has_value() && !function->star) {
            error("Argument of aggregation function is not a valid expression", boundArguments[0]->loc);
         }
         if ((upperCaseFName== "SUM" || upperCaseFName== "AVG") && boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INT &&
             boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::FLOAT &&
             boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DECIMAL &&
             boundArguments[0]->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DOUBLE) {
         }

         //Find correct resultType
         resultType = boundArguments[0]->resultType.value();

         /**
                    * AVG
                */
         if (upperCaseFName== "AVG") {
            if (resultType.type.getTypeId() == catalog::LogicalTypeId::INT) {
               resultType = SQLTypeUtils::getCommonTypeAfterOperation(catalog::Type::decimal(19, 0), catalog::Type::decimal(19, 0), ast::ExpressionType::OPERATOR_DIVIDE);
            } else if (resultType.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
               resultType = SQLTypeUtils::getCommonTypeAfterOperation(resultType, catalog::Type::decimal(19, 0), ast::ExpressionType::OPERATOR_DIVIDE);
            }
         }
         resultType.isNullable = boundArguments[0]->resultType.value().isNullable;
         boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments);
      }
      if (upperCaseFName== "RANK" || upperCaseFName== "ROW_NUMBER") {
         if (!function->arguments.empty()) {
            error("RANK and ROW_NUMBER do not support any arguments", function->loc);
         }
         resultType = catalog::Type::int64();

         boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments);
      }

      /*
                * COUNT
                */
      if (upperCaseFName== "COUNT") {
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
            function->functionName= function->functionName + "*";
            resultType.useZeroInsteadOfNull = true;
         }


         boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArguments);
      }

      /*
            * STDDEV_SAMP
            */
      if (upperCaseFName== "STDDEV_SAMP") {
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
         error("aggregate function" << function->functionName << " not implemented", function->loc);
      }

      auto fInfo = std::make_shared<ast::ColumnReference>(scope, resultType, fName);

      fInfo->displayName = function->alias;
      context->mapAttribute(resolverScope, fName, fInfo);
      boundFunctionExpression->columnReference = fInfo;

      return boundFunctionExpression;
   }
   auto scope = context->getUniqueScope("tmp_attr");
   auto fName = function->alias.empty() ? function->functionName : function->alias;
   std::shared_ptr<ast::BoundFunctionExpression> boundFunctionExpression = nullptr;
   NullableType resultType{catalog::Type::noneType()};

   if (upperCaseFName== "DATE") {
      if (function->arguments.size() != 1) {
         error("Function date needs exactly one argument", function->loc);
      }
      auto arg = analyzeExpression(function->arguments[0], context, resolverScope);
      if (arg->resultType.has_value() && arg->resultType.value().type.getTypeId() != catalog::Type::stringType().getTypeId()) {
         error("Function date needs argument of type string", function->loc);
      }
      resultType = catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY));
      boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, "", function->alias, function->distinct, std::vector{arg});

   } else if (upperCaseFName== "COUNT") {
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

   } else if (upperCaseFName== "EXTRACT") {
      if (function->arguments.size() != 2) {
         error("Function extract needs exactly two arguments", function->loc);
      }
      auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
      auto arg2 = analyzeExpression(function->arguments[1], context, resolverScope);
      if (arg2->resultType.has_value() && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DATE && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::TIMESTAMP && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INTERVAL) {
         error("Function extract needs second argument of type date, interval or timestamp", function->loc);
      }
      resultType = NullableType{catalog::Type::int64(), arg2->resultType->isNullable};

      boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1, arg2});

   } else if (upperCaseFName== "DATE_TRUNC") {
      if (function->arguments.size() != 2) {
         error("Function DATE_TRUNC needs exactly two arguments", function->loc);
      }
      auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
      auto arg2 = analyzeExpression(function->arguments[1], context, resolverScope);
      if (arg1->resultType.has_value() && arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING && arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::CHAR) {
         error("Function DATE_TRUNC needs first argument of type string", function->loc);
      }
      if (arg2->resultType.has_value() && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::DATE && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::TIMESTAMP && arg2->resultType.value().type.getTypeId() != catalog::LogicalTypeId::INTERVAL) {
         error("Function DATE_TRUNC needs second argument of type date, interval or timestamp", function->loc);
      }
      //TODO wrong resulttype
      resultType = catalog::Type::int64();

      boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1, arg2});

   } else if (upperCaseFName== "SUBSTRING" || upperCaseFName== "SUBSTR") {
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

   } else if (upperCaseFName== "ROUND") {
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
   } else if (upperCaseFName== "UPPER") {
      if (function->arguments.size() != 1) {
         error("Function with more than one argument not supported", function->loc);
      }
      auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);

      if (!arg1->resultType.has_value() || (arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING && arg1->resultType.value().type.getTypeId() != catalog::LogicalTypeId::CHAR)) {
         error("Argument of aggregation function has not a valid return type", arg1->loc);
      }
      resultType = arg1->resultType.value();

      boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1});
   } else if (upperCaseFName== "ABS") {
      if (function->arguments.size() != 1) {
         error("Function with more than one argument not supported", function->loc);
      }
      auto arg1 = analyzeExpression(function->arguments[0], context, resolverScope);
      if (!arg1->resultType.has_value() || !arg1->resultType->isNumeric()) {
         error("Argument of function has not a valid return type", arg1->loc);
      }
      resultType = arg1->resultType.value();

      boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, std::vector{arg1});

   } else if (upperCaseFName== "COALESCE") {
      if (function->arguments.size() < 2) {
         error("Function with less than two argument not supported", function->loc);
      }

      std::vector<std::shared_ptr<ast::BoundExpression>> boundArgs{};
      std::vector<NullableType> types{};
      std::ranges::transform(function->arguments, std::back_inserter(boundArgs), [&](auto c) {
         auto e = analyzeExpression(c, context, resolverScope);
         types.push_back(e->resultType.value());
         return e;
      });

      resultType = SQLTypeUtils::getCommonBaseType(types);
      resultType.isNullable = true;

      boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs);

   } else if (upperCaseFName== "GROUPING") {
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

   } else if (upperCaseFName== "LENGTH") {
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
   } else if (upperCaseFName== "REGEXP_REPLACE") {
      if (function->arguments.size() != 3) {
         error("Function REGEXP_REPLACE needs exactly 3 arguments", function->loc);
      }
      auto text = analyzeExpression(function->arguments[0], context, resolverScope);
      auto pattern = analyzeExpression(function->arguments[1], context, resolverScope);
      auto replace = analyzeExpression(function->arguments[2], context, resolverScope);
      if (text->resultType.value().type.getTypeId() != catalog::LogicalTypeId::STRING) {
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
   } else if (upperCaseFName== "HASH") {
      std::vector<std::shared_ptr<ast::BoundExpression>> boundArgs{};
      std::ranges::transform(function->arguments, std::back_inserter(boundArgs), [&](auto c) {
         return analyzeExpression(c, context, resolverScope);
      });
      resultType = catalog::Type::index();
      boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs);

   } else {
      //UDF
      auto entry = catalog->getTypedEntry<lingodb::catalog::FunctionCatalogEntry>(function->functionName);
      if (entry.has_value()) {
         std::vector<std::shared_ptr<ast::BoundExpression>> boundArgs{};
         if (function->arguments.size() != entry.value()->getArgumentTypes().size()) {
            error("Function " << function->functionName << " needs " << entry.value()->getArgumentTypes().size() << " arguments but got " << function->arguments.size(), function->loc);
         }
         for (size_t i = 0; i<function->arguments.size(); i++) {
            auto bound = analyzeExpression(function->arguments.at(i), context, resolverScope);
            auto nullableUDFArgumentType = NullableType(entry.value()->getArgumentTypes()[i]);
            assert(bound->resultType.has_value());
            auto commonTypes = SQLTypeUtils::toCommonTypes(std::vector{nullableUDFArgumentType, bound->resultType.value()});
            bound->resultType = commonTypes[1];
            boundArgs.emplace_back(bound);

         }
         resultType = entry.value()->getReturnType();
         boundFunctionExpression = drv.nf.node<ast::BoundFunctionExpression>(function->loc, function->type, resultType, function->functionName, scope, fName, function->distinct, boundArgs);
         boundFunctionExpression->udfFunction = entry.value();
      }


   }

   if (boundFunctionExpression == nullptr) {
      error("Function '" << function->functionName << "' not implemented", function->loc);
   }
   auto fInfo = std::make_shared<ast::ColumnReference>(scope, resultType, fName);

   boundFunctionExpression->columnReference = fInfo;

   return boundFunctionExpression;
}

std::shared_ptr<ast::BoundColumnRefExpression> SQLQueryAnalyzer::analyzeColumnRefExpression(std::shared_ptr<ast::ColumnRefExpression> columnRef, std::shared_ptr<SQLContext> context) {
   //new implementation which uses the new concept of TableProducers
   auto columnName = columnRef->columnNames.size() == 1 ? columnRef->columnNames[0] : columnRef->columnNames[1];

   std::string scope;
   std::shared_ptr<ast::ColumnReference> found;
   if (columnRef->columnNames.size() == 2) {
      found = context->getColumnReference(columnRef->loc, columnRef->columnNames[0] + "." + columnRef->columnNames[1]);

   } else if (columnRef->columnNames.size() == 1) {
      found = context->getColumnReference(columnRef->loc, columnRef->columnNames[0]);
   } else {
      error("Invalid column reference: expected a structured reference (e.g. 'x'' or 'y.x'').", columnRef->loc);
   }
   if (!found) {
      error("Column not found", columnRef->loc);
   }
   found->displayName = !columnRef->alias.empty() || columnRef->forceToUseAlias ? columnRef->alias : found->displayName;
   return drv.nf.node<ast::BoundColumnRefExpression>(columnRef->loc, found->resultType, found, columnRef->alias);
}

/*
    * SQLTypeUtils
    */
NullableType SQLTypeUtils::getCommonType(NullableType nullableType1, NullableType nullableType2) {
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
            return NullableType(catalog::Type::stringType(), isNullable);
         }
      }
      if (nullableType1.type.getTypeId() == catalog::LogicalTypeId::INT) {
         auto intInfo1 = nullableType1.type.getInfo<catalog::IntTypeInfo>();
         auto intInfo2 = nullableType2.type.getInfo<catalog::IntTypeInfo>();
         if (intInfo1->getBitWidth() > intInfo2->getBitWidth()) {
            return NullableType(nullableType1.type, isNullable);
         } else {
            return NullableType(nullableType2.type, isNullable);
         }
      }
      return NullableType(nullableType1.type, isNullable);
   }

   for (size_t i = 0; i < 2; i++) {
      const auto& type1 = nullableType1.type;
      const auto& type2 = nullableType2.type;

      // Check combinations in one direction
      if (type1.getTypeId() == catalog::LogicalTypeId::DECIMAL &&
          type2.getTypeId() == catalog::LogicalTypeId::INT) {
         return NullableType(type1, isNullable);
      }
      if ((type1.getTypeId() == catalog::LogicalTypeId::INT || type1.getTypeId() == catalog::LogicalTypeId::DECIMAL) && type2.getTypeId() == catalog::LogicalTypeId::DOUBLE) {
         return NullableType(type2, isNullable);
      }

      if (type1.getTypeId() == catalog::LogicalTypeId::STRING &&
          type2.getTypeId() == catalog::LogicalTypeId::CHAR) {
         return NullableType(type1, isNullable);
      }
      if (type1.getTypeId() == catalog::LogicalTypeId::STRING) {
         if (type2.getTypeId() == catalog::LogicalTypeId::INT || type2.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            return NullableType(type1, isNullable);
         }
      }

      if (type1.getTypeId() == catalog::LogicalTypeId::DATE) {
         if (type2.getTypeId() == catalog::LogicalTypeId::STRING || type2.getTypeId() == catalog::LogicalTypeId::CHAR) {
            return NullableType(type1, isNullable);
         }
         if (type2.getTypeId() == catalog::LogicalTypeId::INTERVAL) {
            return NullableType(
               catalog::Type(catalog::LogicalTypeId::DATE,
                             std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY)),
               isNullable);
         }
      }

      if (type1.getTypeId() == catalog::LogicalTypeId::INT &&
          type2.getTypeId() == catalog::LogicalTypeId::CHAR) {
         return NullableType(type1, isNullable);
      }

      // Handle NONE type
      if (type2.getTypeId() == catalog::LogicalTypeId::NONE) {
         return NullableType(type1, isNullable);
      }
      if (type1.getTypeId() == catalog::LogicalTypeId::NONE) {
         return NullableType(type2, isNullable);
      }
      std::swap(nullableType1, nullableType2);
   }

   throw std::runtime_error("No common type found for " + nullableType1.type.toString() + " and " + nullableType2.type.toString());
}

NullableType SQLTypeUtils::getHigherDecimalType(NullableType left, NullableType right) {
   assert(left.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && right.type.getTypeId() == catalog::LogicalTypeId::DECIMAL);
   auto leftInfo = left.type.getInfo<catalog::DecimalTypeInfo>();
   auto rightInfo = right.type.getInfo<catalog::DecimalTypeInfo>();
   int hidig = std::max(leftInfo->getPrecision() - leftInfo->getScale(), rightInfo->getPrecision() - rightInfo->getScale());
   int maxs = std::max(leftInfo->getScale(), rightInfo->getScale());
   return NullableType(catalog::Type::decimal(hidig + maxs, maxs), left.isNullable || right.isNullable);
}

NullableType SQLTypeUtils::getCommonBaseType(std::vector<NullableType> types) {
   auto commonType = types.front();
   for (size_t i = 1; i < types.size(); ++i) {
      commonType = getCommonType(commonType, types[i]);
   }
   return commonType;
}

NullableType SQLTypeUtils::getCommonBaseType(std::vector<NullableType> types, ast::ExpressionType operationType) {
   auto commonType = types.front();
   for (size_t i = 1; i < types.size(); ++i) {
      commonType = getCommonTypeAfterOperation(commonType, types[i], operationType);
   }
   return commonType;
}

NullableType SQLTypeUtils::getCommonTypeAfterOperation(NullableType type1, NullableType type2, ast::ExpressionType operationType) {
   auto commonType = getCommonType(type1, type2);

   //Maybe the other way arround
   switch (operationType) {
      case ast::ExpressionType::OPERATOR_DIVIDE: {
         if (type1.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && type2.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            auto type1Info = type1.type.getInfo<catalog::DecimalTypeInfo>();
            auto type2Info = type2.type.getInfo<catalog::DecimalTypeInfo>();

            auto [p, s] = getAdaptedDecimalPAndSAfterMulDiv(type1Info->getPrecision() - type1Info->getScale() + type2Info->getScale() + std::max<unsigned long>(6, type1Info->getScale() + type2Info->getPrecision()), std::max<unsigned long>(6, type1Info->getScale() + type2Info->getPrecision()));

            return NullableType(catalog::Type::decimal(p, s), commonType.isNullable);
         }
         return commonType;
      }
      case ast::ExpressionType::OPERATOR_TIMES: {
         if (type1.type.getTypeId() == catalog::LogicalTypeId::DECIMAL && type2.type.getTypeId() == catalog::LogicalTypeId::DECIMAL) {
            auto type1Info = type1.type.getInfo<catalog::DecimalTypeInfo>();
            auto type2Info = type2.type.getInfo<catalog::DecimalTypeInfo>();
            auto [p, s] = getAdaptedDecimalPAndSAfterMulDiv(type1Info->getPrecision() + type2Info->getPrecision(), type1Info->getScale() + type2Info->getScale());
            return NullableType(catalog::Type::decimal(p, s), commonType.isNullable);
         }

         return commonType;
      }
      default: return commonType;
   }
}

std::vector<NullableType> SQLTypeUtils::toCommonTypes(std::vector<NullableType> types) {
   auto commonType = getCommonBaseType(types);
   std::vector<NullableType> res;
   for (auto type : types) {
      type.castType = std::make_shared<NullableType>(commonType);
      type.castType->isNullable = type.isNullable;
      res.push_back(type);
   }

   return res;
}

std::vector<NullableType> SQLTypeUtils::toCommonNumber(std::vector<NullableType> types) {
   auto anyDecimal = llvm::any_of(types, [](NullableType type) { return type.type.getTypeId() == catalog::LogicalTypeId::DECIMAL; });
   auto anyFloat = llvm::any_of(types, [](NullableType type) { return type.type.getTypeId() == catalog::LogicalTypeId::FLOAT || type.type.getTypeId() == catalog::LogicalTypeId::DOUBLE; });
   if (anyDecimal && !anyFloat) {
      std::vector<NullableType> res;
      for (auto type : types) {
         if (type.type.getTypeId() != catalog::LogicalTypeId::DECIMAL) {
            type.castType = std::make_shared<NullableType>(catalog::Type::decimal(19, 0), type.isNullable);
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

NullableType SQLTypeUtils::typemodsToCatalogType(catalog::LogicalTypeId logicalTypeId, std::vector<std::shared_ptr<ast::Value>>& typeModifiers) {
   switch (logicalTypeId) {
      case catalog::LogicalTypeId::INT: {
         if (typeModifiers.size() == 1) {
            auto typeModifier = typeModifiers[0];
            if (typeModifier->type != ast::ConstantType::UINT) {
               throw std::runtime_error("Invalid Typemodfiers for type: int");
            }
            auto size = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifier)->iVal;
            if (size >= 8) {
               return catalog::Type::int64();
            }
            if (size == 2) {
               return catalog::Type::int16();
            }
            if (size == 1) {
               return catalog::Type::int8();
            }
         }
         return catalog::Type::int32();
      }
      case catalog::LogicalTypeId::BOOLEAN: {
         return catalog::Type::boolean();
      }
      case catalog::LogicalTypeId::STRING: {
         return catalog::Type::stringType();
      }
      case catalog::LogicalTypeId::CHAR: {
         if (typeModifiers.size() == 0) {
            return catalog::Type::charType(1);
         }
         if (typeModifiers.size() > 1 || typeModifiers[0]->type != ast::ConstantType::UINT) {
            throw std::runtime_error("Invalid Typemodfiers for type: char");
         }
         size_t l = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifiers[0])->iVal;
         return catalog::Type::charType(l);
      }
      case catalog::LogicalTypeId::DECIMAL: {
         if (typeModifiers.size() != 2 || typeModifiers[0]->type != ast::ConstantType::UINT || typeModifiers[1]->type != ast::ConstantType::UINT) {
            throw std::runtime_error("Invalid Typemodfiers for type: descimal");
         }
         size_t p = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifiers[0])->iVal;
         size_t s = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(typeModifiers[1])->iVal;

         return catalog::Type::decimal(p, s);
      }
      case catalog::LogicalTypeId::DATE: {
         return catalog::Type(catalog::LogicalTypeId::DATE, std::make_shared<catalog::DateTypeInfo>(catalog::DateTypeInfo::DateUnit::DAY));
      }
      case catalog::LogicalTypeId::TIMESTAMP: {
         return catalog::Type::timestamp();
      }
      case catalog::LogicalTypeId::FLOAT: {
         if (typeModifiers.size() >= 1) {
            auto sizeValue = typeModifiers.at(0);
            if (sizeValue->type == ast::ConstantType::UINT) {
               auto size = std::reinterpret_pointer_cast<ast::UnsignedIntValue>(sizeValue)->iVal;
               if (size <= 2) {
                  return catalog::Type::f32();
               }
            }
         }
         return catalog::Type::f64();
      }
      case catalog::LogicalTypeId::INTERVAL: {
         return catalog::Type::intervalDaytime();
      }
      default: throw std::runtime_error("Typemod not implemented");
   }
}

} // namespace lingodb::analyzer