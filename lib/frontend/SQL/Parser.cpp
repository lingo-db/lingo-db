#include "frontend/SQL/Parser.h"
frontend::sql::ExpressionType frontend::sql::stringToExpressionType(const std::string& parserStr) {
   std::string str = parserStr;
   std::transform(str.begin(), str.end(), str.begin(), ::toupper);
   return llvm::StringSwitch<ExpressionType>(str)
      .Case("OPERATOR_UNARY_MINUS", ExpressionType::OPERATOR_UNARY_MINUS)
      .Case("OPERATOR_PLUS", ExpressionType::OPERATOR_PLUS)
      .Case("OPERATOR_MINUS", ExpressionType::OPERATOR_MINUS)
      .Case("OPERATOR_MULTIPLY", ExpressionType::OPERATOR_MULTIPLY)
      .Case("OPERATOR_DIVIDE", ExpressionType::OPERATOR_DIVIDE)
      .Case("OPERATOR_CONCAT", ExpressionType::OPERATOR_CONCAT)
      .Case("OPERATOR_MOD", ExpressionType::OPERATOR_MOD)
      .Case("+", ExpressionType::OPERATOR_PLUS)
      .Case("-", ExpressionType::OPERATOR_MINUS)
      .Case("*", ExpressionType::OPERATOR_MULTIPLY)
      .Case("/", ExpressionType::OPERATOR_DIVIDE)
      .Case("||", ExpressionType::OPERATOR_CONCAT)
      .Case("%", ExpressionType::OPERATOR_MOD)
      .Case("OPERATOR_NOT", ExpressionType::OPERATOR_NOT)
      .Case("OPERATOR_IS_NULL", ExpressionType::OPERATOR_IS_NULL)
      .Case("OPERATOR_EXISTS", ExpressionType::OPERATOR_EXISTS)
      .Case("COMPARE_EQUAL", ExpressionType::COMPARE_EQUAL)
      .Case("COMPARE_NOTEQUAL", ExpressionType::COMPARE_NOT_EQUAL)
      .Case("COMPARE_LESSTHAN", ExpressionType::COMPARE_LESS_THAN)
      .Case("COMPARE_GREATERTHAN", ExpressionType::COMPARE_GREATER_THAN)
      .Case("COMPARE_LESSTHANOREQUALTO", ExpressionType::COMPARE_LESS_THAN_OR_EQUAL_TO)
      .Case("COMPARE_GREATERTHANOREQUALTO", ExpressionType::COMPARE_GREATER_THAN_OR_EQUAL_TO)
      .Case("COMPARE_LIKE", ExpressionType::COMPARE_LIKE)
      .Case("COMPARE_NOTLIKE", ExpressionType::COMPARE_NOT_LIKE)
      .Case("=", ExpressionType::COMPARE_EQUAL)
      .Case("!=", ExpressionType::COMPARE_NOT_EQUAL)
      .Case("<>", ExpressionType::COMPARE_NOT_EQUAL)
      .Case("<", ExpressionType::COMPARE_LESS_THAN)
      .Case(">", ExpressionType::COMPARE_GREATER_THAN)
      .Case("<=", ExpressionType::COMPARE_LESS_THAN_OR_EQUAL_TO)
      .Case(">=", ExpressionType::COMPARE_GREATER_THAN_OR_EQUAL_TO)
      .Case("~~", ExpressionType::COMPARE_LIKE)
      .Case("!~~", ExpressionType::COMPARE_NOT_LIKE)
      .Case("COMPARE_IN", ExpressionType::COMPARE_IN)
      .Case("COMPARE_DISTINCT_FROM", ExpressionType::COMPARE_IS_DISTINCT_FROM)
      .Case("CONJUNCTION_AND", ExpressionType::CONJUNCTION_AND)
      .Case("CONJUNCTION_OR", ExpressionType::CONJUNCTION_OR)
      .Case("COLUMN_VALUE", ExpressionType::COLUMN_VALUE)
      .Case("VALUE_CONSTANT", ExpressionType::VALUE_CONSTANT)
      .Case("VALUE_PARAMETER", ExpressionType::VALUE_PARAMETER)
      .Case("VALUE_TUPLE", ExpressionType::VALUE_TUPLE)
      .Case("VALUE_TUPLE_ADDRESS", ExpressionType::VALUE_TUPLE_ADDRESS)
      .Case("VALUE_NULL", ExpressionType::VALUE_NULL)
      .Case("VALUE_VECTOR", ExpressionType::VALUE_VECTOR)
      .Case("VALUE_SCALAR", ExpressionType::VALUE_SCALAR)
      .Case("AGGREGATE_COUNT", ExpressionType::AGGREGATE_COUNT)
      .Case("AGGREGATE_SUM", ExpressionType::AGGREGATE_SUM)
      .Case("AGGREGATE_MIN", ExpressionType::AGGREGATE_MIN)
      .Case("AGGREGATE_MAX", ExpressionType::AGGREGATE_MAX)
      .Case("AGGREGATE_AVG", ExpressionType::AGGREGATE_AVG)
      .Case("AGGREGATE_TOP_K", ExpressionType::AGGREGATE_TOP_K)
      .Case("AGGREGATE_HISTOGRAM", ExpressionType::AGGREGATE_HISTOGRAM)
      .Case("FUNCTION", ExpressionType::FUNCTION)
      .Case("HASH_RANGE", ExpressionType::HASH_RANGE)
      .Case("OPERATOR_CASE_EXPR", ExpressionType::OPERATOR_CASE_EXPR)
      .Case("OPERATOR_NULLIF", ExpressionType::OPERATOR_NULL_IF)
      .Case("OPERATOR_COALESCE", ExpressionType::OPERATOR_COALESCE)
      .Case("ROW_SUBQUERY", ExpressionType::ROW_SUBQUERY)
      .Case("STAR", ExpressionType::STAR)
      .Case("TABLE_STAR", ExpressionType::TABLE_STAR)
      .Case("PLACEHOLDER", ExpressionType::PLACEHOLDER)
      .Case("COLUMN_REF", ExpressionType::COLUMN_REF)
      .Case("FUNCTION_REF", ExpressionType::FUNCTION_REF)
      .Case("TABLE_REF", ExpressionType::TABLE_REF)
      .Default(ExpressionType::INVALID);
}
mlir::FloatType frontend::sql::SQLTypeInference::getHigherFloatType(mlir::Type left, mlir::Type right) {
   mlir::FloatType leftFloat = left.dyn_cast_or_null<mlir::FloatType>();
   if (auto rightFloat = right.dyn_cast_or_null<mlir::FloatType>()) {
      if (!leftFloat || rightFloat.getWidth() > leftFloat.getWidth()) {
         return rightFloat;
      }
   }
   return leftFloat;
}
mlir::IntegerType frontend::sql::SQLTypeInference::getHigherIntType(mlir::Type left, mlir::Type right) {
   mlir::IntegerType leftInt = left.dyn_cast_or_null<mlir::IntegerType>();
   if (auto rightInt = right.dyn_cast_or_null<mlir::IntegerType>()) {
      if (!leftInt || rightInt.getWidth() > leftInt.getWidth()) {
         return rightInt;
      }
   }
   return leftInt;
}
mlir::db::DecimalType frontend::sql::SQLTypeInference::getHigherDecimalType(mlir::Type left, mlir::Type right) {
   auto a = left.dyn_cast_or_null<mlir::db::DecimalType>();
   if (auto b = right.dyn_cast_or_null<mlir::db::DecimalType>()) {
      if (!a) return b;
      int hidig = std::max(a.getP() - a.getS(), b.getP() - b.getS());
      int maxs = std::max(a.getS(), b.getS());
      return mlir::db::DecimalType::get(a.getContext(), hidig + maxs, maxs);
   }
   return a;
}
mlir::Value frontend::sql::Parser::translateWhenCase(mlir::OpBuilder& builder, TranslationContext& context, mlir::Value compareValue, ListCell* whenCell, Node* defaultNode) {
   auto loc = builder.getUnknownLoc();
   if (!whenCell) {
      return translateExpression(builder, defaultNode, context);
   }
   auto* w = reinterpret_cast<CaseWhen*>(whenCell->data.ptr_value);
   auto cond = translateExpression(builder, reinterpret_cast<Node*>(w->expr_), context);
   if (compareValue) {
      cond = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, cond, compareValue);
   }
   auto* whenBlock = new mlir::Block;
   auto* elseBlock = new mlir::Block;
   mlir::OpBuilder whenBuilder(builder.getContext());
   whenBuilder.setInsertionPointToStart(whenBlock);
   auto whenRes = translateExpression(whenBuilder, reinterpret_cast<Node*>(w->result_), context);
   mlir::OpBuilder elseBuilder(builder.getContext());
   elseBuilder.setInsertionPointToStart(elseBlock);
   auto elseRes = translateWhenCase(elseBuilder, context, compareValue, whenCell->next, defaultNode);
   auto commonType = SQLTypeInference::getCommonType(whenRes.getType(), elseRes.getType());
   whenRes = SQLTypeInference::toType(whenBuilder, whenRes, commonType);
   elseRes = SQLTypeInference::toType(elseBuilder, elseRes, commonType);
   whenBuilder.create<mlir::scf::YieldOp>(loc, whenRes);
   elseBuilder.create<mlir::scf::YieldOp>(loc, elseRes);
   auto ifOp = builder.create<mlir::scf::IfOp>(loc, commonType, cond, true);
   ifOp.getThenRegion().getBlocks().clear();
   ifOp.getElseRegion().getBlocks().clear();
   ifOp.getThenRegion().push_back(whenBlock);
   ifOp.getElseRegion().push_back(elseBlock);

   return ifOp.getResult(0);
}
mlir::ArrayAttr frontend::sql::Parser::translateSortSpec(List* sortClause, mlir::OpBuilder& builder, TranslationContext& context, TargetInfo targetInfo) {
   std::vector<mlir::Attribute> mapping;
   for (auto* cell = sortClause->head; cell != nullptr; cell = cell->next) {
      auto* temp = reinterpret_cast<Node*>(cell->data.ptr_value);
      mlir::relalg::SortSpec spec = mlir::relalg::SortSpec::asc;
      switch (temp->type) {
         case T_SortBy: {
            auto* sort = reinterpret_cast<SortBy*>(temp);

            switch (sort->sortby_dir_) {
               case SORTBY_DESC: {
                  spec = mlir::relalg::SortSpec::desc;
                  break;
               }
               case SORTBY_ASC: // fall through
               case SORTBY_DEFAULT: {
                  spec = mlir::relalg::SortSpec::asc;
                  break;
               }
               default: {
                  error("unknown sort type");
               }
            }

            auto* target = sort->node_;
            const mlir::relalg::Column* attr;
            switch (target->type) {
               case T_ColumnRef: {
                  attr = resolveColRef(target, context);
                  break;
               }
               case T_A_Const: {
                  auto* constExpr = reinterpret_cast<A_Const*>(target);
                  auto constVal = constExpr->val_;
                  switch (constVal.type_) {
                     case T_Integer: {
                        attr = targetInfo.namedResults.at(constVal.val_.ival_ - 1).second;
                        break;
                     }
                     default: error("unsupported sort specification");
                  }
                  break;
               }
               default: error("can only sort with column refs");
            }

            mapping.push_back(mlir::relalg::SortSpecificationAttr::get(builder.getContext(), attrManager.createRef(attr), spec));
            break;
         }
         default: {
            error("unknown orderby type");
         }
      }
   }
   return builder.getArrayAttr(mapping);
}
mlir::Value frontend::sql::Parser::translateFuncCall(Node* node, mlir::OpBuilder& builder, mlir::Location loc, TranslationContext& context) {
   auto* funcCall = reinterpret_cast<FuncCall*>(node);
   //expr = FuncCallTransform(parse_result,,context);
   std::string funcName = reinterpret_cast<value*>(funcCall->funcname_->head->data.ptr_value)->val_.str_;
   if (funcName == "pg_catalog") {
      funcName = reinterpret_cast<value*>(funcCall->funcname_->tail->data.ptr_value)->val_.str_;
   }
   if (funcName == "date_part") {
      auto part = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      auto arg2 = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, builder.getI64Type(), "ExtractFromDate", mlir::ValueRange({part, arg2})).res();
   }
   if (funcName == "substring") {
      auto str = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      auto from = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->next->data.ptr_value), context);
      auto to = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, str.getType(), "Substring", mlir::ValueRange({str, from, to})).res();
   }
   if (funcName == "abs") {
      auto val = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, val.getType(), "AbsInt", val).res();
   }
   error("could not translate func call");
   return mlir::Value();
}
std::pair<mlir::Value, frontend::sql::Parser::TargetInfo> frontend::sql::Parser::translateConstRelation(List* valuesLists, mlir::OpBuilder& builder) {
   size_t numColumns = 0;
   std::vector<mlir::Type> globalTypes;
   std::vector<mlir::Attribute> rows;
   bool first = true;
   for (auto* valueList = valuesLists->head; valueList != nullptr; valueList = valueList->next) {
      auto* target = reinterpret_cast<List*>(valueList->data.ptr_value);
      size_t i = 0;
      std::vector<mlir::Type> types;
      std::vector<mlir::Attribute> values;
      for (auto* cell = target->head; cell != nullptr; cell = cell->next, i++) {
         auto* expr = reinterpret_cast<Expr*>(cell->data.ptr_value);
         switch (expr->type_) {
            case T_A_Const: {
               auto* constExpr = reinterpret_cast<A_Const*>(expr);
               auto constVal = constExpr->val_;
               mlir::Attribute value;
               mlir::Type t;
               switch (constVal.type_) {
                  case T_Integer: {
                     t = builder.getI32Type();
                     value = builder.getI32IntegerAttr(constVal.val_.ival_);
                     break;
                  }
                  case T_String: {
                     std::string stringVal = constVal.val_.str_;
                     t = mlir::db::StringType::get(builder.getContext());
                     if (stringVal.size() <= 8) {
                        t = mlir::db::CharType::get(builder.getContext(), stringVal.size());
                     }
                     value = builder.getStringAttr(stringVal);
                     break;
                  }
                  case T_Float: {
                     std::string stringValue(constVal.val_.str_);
                     auto decimalPos = stringValue.find('.');
                     if (decimalPos == std::string::npos) {
                        t = builder.getI64Type();
                        value = builder.getI64IntegerAttr(std::stoll(constVal.val_.str_));
                     } else {
                        auto s = stringValue.size() - decimalPos - 1;
                        auto p = stringValue.size() - 1;
                        t = mlir::db::DecimalType::get(builder.getContext(), p, s);
                        value = builder.getStringAttr(constVal.val_.str_);
                     }
                     break;
                  }
                  case T_Null: {
                     t = mlir::db::NullableType::get(builder.getContext(), builder.getNoneType());
                     value = builder.getUnitAttr();
                     break;
                  }
                  default:
                     error("unhandled constant type");
               }
               types.push_back(t);
               values.push_back(value);
               break;
            }
            default: {
               error("could not handle values content");
            }
         }
      }
      rows.push_back(builder.getArrayAttr(values));
      if (first) {
         first = false;
         numColumns = i;
         globalTypes = types;
      } else {
         for (size_t j = 0; j < numColumns; j++) {
            globalTypes[j] = SQLTypeInference::getCommonType(globalTypes.at(j), types.at(j));
         }
      }
   }
   static size_t constRelId = 0;
   std::string symName = "constrel" + std::to_string(constRelId++);
   std::vector<mlir::Attribute> attributes;

   TargetInfo targetInfo;
   for (size_t i = 0; i < numColumns; i++) {
      std::string columnName = "const" + std::to_string(i);
      auto attrDef = attrManager.createDef(symName, columnName);
      attrDef.getColumn().type = globalTypes[i];
      attributes.push_back(attrDef);
      targetInfo.namedResults.push_back({columnName, &attrDef.getColumn()});
   }
   //::llvm::StringRef sym_name, ::mlir::ArrayAttr attributes, ::mlir::ArrayAttr values
   mlir::Value constRel = builder.create<mlir::relalg::ConstRelationOp>(builder.getUnknownLoc(), builder.getArrayAttr(attributes), builder.getArrayAttr(rows));
   return std::make_pair(constRel, targetInfo);
}
