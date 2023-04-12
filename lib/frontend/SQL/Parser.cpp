#include "frontend/SQL/Parser.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include <regex>

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
mlir::db::DateType frontend::sql::SQLTypeInference::getHigherDateType(mlir::Type left, mlir::Type right) {
   auto a = left.dyn_cast_or_null<mlir::db::DateType>();
   if (auto b = right.dyn_cast_or_null<mlir::db::DateType>()) {
      if (!a) return b;
   }
   return a;
}
mlir::Value frontend::sql::SQLTypeInference::castValueToType(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t) {
   bool isNullable = v.getType().isa<mlir::db::NullableType>();
   if (isNullable && !t.isa<mlir::db::NullableType>()) {
      t = mlir::db::NullableType::get(builder.getContext(), t);
   }
   bool onlyTargetIsNullable = !isNullable && t.isa<mlir::db::NullableType>();
   if (v.getType() == t) { return v; }
   if (auto* defOp = v.getDefiningOp()) {
      if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
         if (!t.isa<mlir::db::NullableType>()) {
            constOp.getResult().setType(t);
            return constOp;
         }
      }
      if (auto nullOp = mlir::dyn_cast_or_null<mlir::db::NullOp>(defOp)) {
         auto t2 = t.cast<mlir::db::NullableType>();
         nullOp.getResult().setType(t2);
         return nullOp;
      }
   }
   if (v.getType() == getBaseType(t)) {
      return builder.create<mlir::db::AsNullableOp>(builder.getUnknownLoc(), t, v);
   }
   if (onlyTargetIsNullable) {
      mlir::Value casted = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), getBaseType(t), v);
      return builder.create<mlir::db::AsNullableOp>(builder.getUnknownLoc(), t, casted);
   } else {
      return builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), t, v);
   }
}
mlir::Value frontend::sql::Parser::translateWhenCaseExpression(mlir::OpBuilder& builder, TranslationContext& context, mlir::Value compareValue, ListCell* whenCell, Node* defaultNode) {
   auto loc = builder.getUnknownLoc();
   if (!whenCell) {
      if (!defaultNode) {
         return builder.create<mlir::db::NullOp>(builder.getUnknownLoc(), mlir::db::NullableType::get(builder.getContext(), builder.getNoneType()));
      }
      return translateExpression(builder, defaultNode, context);
   }
   auto* w = reinterpret_cast<CaseWhen*>(whenCell->data.ptr_value);
   auto cond = translateExpression(builder, reinterpret_cast<Node*>(w->expr_), context);
   if (compareValue) {
      auto asCommon = SQLTypeInference::toCommonBaseTypes(builder, mlir::ValueRange{cond, compareValue});
      cond = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, asCommon.at(0), asCommon.at(1));
   }
   auto* whenBlock = new mlir::Block;
   auto* elseBlock = new mlir::Block;
   mlir::OpBuilder whenBuilder(builder.getContext());
   whenBuilder.setInsertionPointToStart(whenBlock);
   auto whenRes = translateExpression(whenBuilder, reinterpret_cast<Node*>(w->result_), context);
   mlir::OpBuilder elseBuilder(builder.getContext());
   elseBuilder.setInsertionPointToStart(elseBlock);
   auto elseRes = translateWhenCaseExpression(elseBuilder, context, compareValue, whenCell->next, defaultNode);
   auto commonType = SQLTypeInference::getCommonType(whenRes.getType(), elseRes.getType());
   whenRes = SQLTypeInference::castValueToType(whenBuilder, whenRes, commonType);
   elseRes = SQLTypeInference::castValueToType(elseBuilder, elseRes, commonType);
   whenBuilder.create<mlir::scf::YieldOp>(loc, whenRes);
   elseBuilder.create<mlir::scf::YieldOp>(loc, elseRes);
   cond = builder.create<mlir::db::DeriveTruth>(loc, cond);
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
            const mlir::tuples::Column* attr;
            switch (target->type) {
               case T_ColumnRef: {
                  attr = resolveColRef(target, context);
                  break;
               }
               case T_FakeNode: { //
                  attr = context.getAttribute(reinterpret_cast<FakeNode*>(target)->colId);
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
mlir::Value frontend::sql::Parser::translateFuncCallExpression(Node* node, mlir::OpBuilder& builder, mlir::Location loc, TranslationContext& context) {
   auto* funcCall = reinterpret_cast<FuncCall*>(node);
   //expr = FuncCallTransform(parse_result,,context);
   std::string funcName = reinterpret_cast<value*>(funcCall->funcname_->head->data.ptr_value)->val_.str_;
   if (funcName == "pg_catalog") {
      funcName = reinterpret_cast<value*>(funcCall->funcname_->tail->data.ptr_value)->val_.str_;
   }
   if (funcName == "date_diff") {
      auto unit = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      auto arg1 = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->next->data.ptr_value), context);
      auto arg2 = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, builder.getI64Type(), "DateDiff", mlir::ValueRange({unit, arg1, arg2})).getRes();
   }
   if (funcName == "date_part") {
      auto part = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      auto arg2 = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, builder.getI64Type(), "ExtractFromDate", mlir::ValueRange({part, arg2})).getRes();
   }
   if (funcName == "substring" || funcName == "substr") {
      auto str = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      auto from = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->next->data.ptr_value), context);
      auto to = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, str.getType(), "Substring", mlir::ValueRange({str, from, to})).getRes();
   }
   if (funcName == "abs") {
      auto val = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, val.getType(), getBaseType(val.getType()).isa<mlir::db::DecimalType>() ? "AbsDecimal" : "AbsInt", val).getRes();
   }
   if (funcName == "upper") {
      auto val = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, val.getType(), "ToUpper", val).getRes();
   }
   if (funcName == "round") {
      auto val = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
      auto scale = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value), context);
      return builder.create<mlir::db::RuntimeCall>(loc, val.getType(), getBaseType(val.getType()).isIntOrIndex() ? "RoundInt" + std::to_string(getBaseType(val.getType()).getIntOrFloatBitWidth()) : "RoundDecimal", mlir::ValueRange{val, scale}).getRes();
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
                     if (stringVal.size() <= 8 && stringVal.size() > 0) {
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
mlir::Value frontend::sql::Parser::translateBinaryExpression(mlir::OpBuilder& builder, frontend::sql::ExpressionType opType, mlir::Value left, mlir::Value right) {
   auto loc = builder.getUnknownLoc();
   switch (opType) {
      case ExpressionType::OPERATOR_PLUS:
         if (getBaseType(left.getType()).isa<mlir::db::DateType>() && getBaseType(right.getType()).isa<mlir::db::IntervalType>()) {
            return builder.create<mlir::db::RuntimeCall>(loc, left.getType(), "DateAdd", mlir::ValueRange({left, right})).getRes();
         }
         return builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypes(builder, {left, right}));
      case ExpressionType::OPERATOR_MINUS:
         if (left.getType().isa<mlir::db::DateType>() && right.getType().isa<mlir::db::IntervalType>()) {
            return builder.create<mlir::db::RuntimeCall>(loc, left.getType(), "DateSubtract", mlir::ValueRange({left, right})).getRes();
         }
         return builder.create<mlir::db::SubOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypes(builder, {left, right}));
      case ExpressionType::OPERATOR_MULTIPLY:
         return builder.create<mlir::db::MulOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonNumber(builder, {left, right}));
      case ExpressionType::OPERATOR_DIVIDE:
         return builder.create<mlir::db::DivOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonNumber(builder, {left, right}));
      case ExpressionType::OPERATOR_MOD:
         return builder.create<mlir::db::ModOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonNumber(builder, {left, right}));
      case ExpressionType::COMPARE_EQUAL:
      case ExpressionType::COMPARE_NOT_EQUAL:
      case ExpressionType::COMPARE_LESS_THAN:
      case ExpressionType::COMPARE_GREATER_THAN:
      case ExpressionType::COMPARE_LESS_THAN_OR_EQUAL_TO:
      case ExpressionType::COMPARE_GREATER_THAN_OR_EQUAL_TO: {
         mlir::db::DBCmpPredicate pred;
         switch (opType) {
            case ExpressionType::COMPARE_EQUAL: pred = mlir::db::DBCmpPredicate::eq; break;
            case ExpressionType::COMPARE_NOT_EQUAL: pred = mlir::db::DBCmpPredicate::neq; break;
            case ExpressionType::COMPARE_LESS_THAN: pred = mlir::db::DBCmpPredicate::lt; break;
            case ExpressionType::COMPARE_GREATER_THAN: pred = mlir::db::DBCmpPredicate::gt; break;
            case ExpressionType::COMPARE_LESS_THAN_OR_EQUAL_TO: pred = mlir::db::DBCmpPredicate::lte; break;
            case ExpressionType::COMPARE_GREATER_THAN_OR_EQUAL_TO: pred = mlir::db::DBCmpPredicate::gte; break;
            default: error("should not happen");
         }
         auto ct = SQLTypeInference::toCommonBaseTypes(builder, {left, right});
         return builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), pred, ct[0], ct[1]);
      }
      case ExpressionType::COMPARE_LIKE:
      case ExpressionType::COMPARE_NOT_LIKE: {
         auto ct = SQLTypeInference::toCommonBaseTypes(builder, {left, right});
         auto isNullable = left.getType().isa<mlir::db::NullableType>() || right.getType().isa<mlir::db::NullableType>();
         mlir::Type resType = isNullable ? (mlir::Type) mlir::db::NullableType::get(builder.getContext(), builder.getI1Type()) : (mlir::Type) builder.getI1Type();
         auto like = builder.create<mlir::db::RuntimeCall>(loc, resType, "Like", mlir::ValueRange({ct[0], ct[1]})).getRes();
         return opType == ExpressionType::COMPARE_NOT_LIKE ? builder.create<mlir::db::NotOp>(loc, like) : like;
      }
      case ExpressionType::OPERATOR_CONCAT: {
         auto leftString = SQLTypeInference::castValueToType(builder, left, mlir::db::StringType::get(builder.getContext()));
         auto rightString = SQLTypeInference::castValueToType(builder, right, mlir::db::StringType::get(builder.getContext()));
         mlir::Type resType = right.getType().isa<mlir::db::NullableType>() ? rightString.getType() : leftString.getType();
         return builder.create<mlir::db::RuntimeCall>(loc, resType, "Concatenate", mlir::ValueRange({leftString, rightString})).getRes();
      }
      default:
         error("unsupported expression type");
   }
   return mlir::Value();
}
mlir::Value frontend::sql::Parser::translateRangeVar(mlir::OpBuilder& builder, RangeVar* stmt, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   std::string relation = stmt->relname_;
   std::string alias = relation;
   if (stmt->alias_ && stmt->alias_->type_ == T_Alias && stmt->alias_->aliasname_) {
      alias = stmt->alias_->aliasname_;
   }
   if (!database.hasTable(relation)) {
      if (ctes.contains(relation)) {
         auto renamedScope = attrManager.getUniqueScope(relation);
         auto [tree, targetInfo] = ctes.at(relation);
         std::vector<mlir::Attribute> renamingDefsAsAttr;
         for (auto x : targetInfo.namedResults) {
            auto def = attrManager.createDef(renamedScope, x.first, builder.getArrayAttr({attrManager.createRef(x.second)}));
            auto* newRef = &def.getColumn();
            renamingDefsAsAttr.push_back(def);
            def.getColumn().type = x.second->type;
            context.mapAttribute(scope, x.first, newRef);
            context.mapAttribute(scope, alias + "." + x.first, newRef);
         }
         return builder.create<mlir::relalg::RenamingOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(renamingDefsAsAttr));
      } else {
         error("unknown relation " + relation);
      }
   }
   auto tableMetaData = database.getTableMetaData(relation);
   char lastCharacter = alias.back();
   std::string scopeName = attrManager.getUniqueScope(alias + (isdigit(lastCharacter) ? "_" : ""));

   std::vector<mlir::NamedAttribute> columns;
   for (auto c : tableMetaData->getOrderedColumns()) {
      auto attrDef = attrManager.createDef(scopeName, c);
      attrDef.getColumn().type = createTypeFromColumnType(builder.getContext(), tableMetaData->getColumnMetaData(c)->getColumnType());
      columns.push_back(builder.getNamedAttr(c, attrDef));
      context.mapAttribute(scope, c, &attrDef.getColumn()); //todo check for existing and overwrite...
      context.mapAttribute(scope, alias + "." + c, &attrDef.getColumn());
   }
   return builder.create<mlir::relalg::BaseTableOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), relation, mlir::relalg::TableMetaDataAttr::get(builder.getContext(), std::make_shared<runtime::TableMetaData>()), builder.getDictionaryAttr(columns));
}
std::pair<mlir::Value, frontend::sql::Parser::TargetInfo> frontend::sql::Parser::translateClassicSelectStmt(mlir::OpBuilder& builder, SelectStmt* stmt, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   // VALUES (...)
   if (stmt->values_lists_) {
      return translateConstRelation(stmt->values_lists_, builder);
   }
   // WITH ...
   if (stmt->with_clause_) {
      for (auto* cell = stmt->with_clause_->ctes_->head; cell != nullptr; cell = cell->next) {
         auto* cte = reinterpret_cast<CommonTableExpr*>(cell->data.ptr_value);
         assert(cte->ctequery_->type == T_SelectStmt);
         mlir::Value subQuery;
         TargetInfo targetInfo;
         {
            auto subQueryScope = context.createResolverScope();
            auto subQueryDefineScope = context.createDefineScope();
            auto [subQuery_, targetInfo_] = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(cte->ctequery_), context, subQueryScope);
            subQuery = subQuery_;
            targetInfo = targetInfo_;
            if (cte->aliascolnames_) {
               size_t i = 0;
               std::cout << pg_query_nodes_to_json(cte->aliascolnames_) << std::endl;
               for (auto* el = cte->aliascolnames_->head; el != nullptr; el = el->next) {
                  auto* val = reinterpret_cast<value*>(el->data.ptr_value);
                  targetInfo.namedResults.at(i++).first = val->val_.str_;
               }
            }
         }
         ctes.insert({cte->ctename_, {subQuery, targetInfo}});
      }
   }
   // FROM
   mlir::Value tree = translateFromClause(builder, stmt, context, scope);
   if (!tree) {
      auto dummyAttr = attrManager.createDef(attrManager.getUniqueScope("dummyScope"), "dummyName");
      dummyAttr.getColumn().type = builder.getI32Type();
      std::vector<mlir::Attribute> columns{dummyAttr};
      std::vector<mlir::Attribute> rows{builder.getArrayAttr({builder.getI64IntegerAttr(0)})};
      tree = builder.create<mlir::relalg::ConstRelationOp>(builder.getUnknownLoc(), builder.getArrayAttr(columns), builder.getArrayAttr(rows));
   }
   // WHERE
   if (stmt->where_clause_) {
      mlir::Block* pred = translatePredicate(builder, stmt->where_clause_, context);
      auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree);
      sel.getPredicate().push_back(pred);
      tree = sel.getResult();
   }
   auto [tree_, targetInfo] = translateSelectionTargetList(builder, stmt->group_clause_, stmt->having_clause_, stmt->target_list_, stmt->sort_clause_, stmt->distinct_clause_, tree, context, scope);
   tree = tree_;
   for (auto x : targetInfo.namedResults) {
      if (!x.first.empty()) {
         context.mapAttribute(scope, x.first, x.second);
      }
   }
   // LIMIT
   if (stmt->limit_count_) {
      size_t limit = reinterpret_cast<A_Const*>(stmt->limit_count_)->val_.val_.ival_;
      tree = builder.create<mlir::relalg::LimitOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), limit, tree);
   }
   return std::make_pair(tree, targetInfo);
}
std::pair<mlir::Value, frontend::sql::Parser::TargetInfo> frontend::sql::Parser::translateSetOperation(mlir::OpBuilder& builder, SelectStmt* stmt, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   auto setSemantic = stmt->all_ ? mlir::relalg::SetSemantic::all : mlir::relalg::SetSemantic::distinct;
   std::pair<mlir::Value, TargetInfo> leftSubQueryRes;
   std::pair<mlir::Value, TargetInfo> rightSubQueryRes;
   {
      auto subQueryScope = context.createResolverScope();
      auto subQueryDefineScope = context.createDefineScope();
      leftSubQueryRes = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(stmt->larg_), context, subQueryScope);
   }
   {
      auto subQueryScope = context.createResolverScope();
      auto subQueryDefineScope = context.createDefineScope();
      rightSubQueryRes = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(stmt->rarg_), context, subQueryScope);
   }
   auto [lTree, lTargetInfo] = leftSubQueryRes;
   auto [rTree, rTargetInfo] = rightSubQueryRes;
   if (lTargetInfo.namedResults.size() != rTargetInfo.namedResults.size()) {
      error("SET Operation expects both sides to have same number of columns");
   }
   std::vector<mlir::Attribute> attributes;
   auto scopeName = attrManager.getUniqueScope("setop");
   TargetInfo targetInfo;
   mlir::Block* leftMapBlock = new mlir::Block;
   mlir::Block* rightMapBlock = new mlir::Block;
   mlir::OpBuilder leftMapBuilder(builder.getContext());
   mlir::OpBuilder rightMapBuilder(builder.getContext());
   leftMapBuilder.setInsertionPointToStart(leftMapBlock);
   rightMapBuilder.setInsertionPointToStart(rightMapBlock);
   leftMapBlock->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   rightMapBlock->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto leftMapScope = attrManager.getUniqueScope("map");
   auto rightMapScope = attrManager.getUniqueScope("map");
   std::vector<mlir::Attribute> createdColsLeft;
   std::vector<mlir::Attribute> createdColsRight;
   mlir::Value leftTuple = leftMapBlock->getArgument(0);
   mlir::Value rightTuple = rightMapBlock->getArgument(0);
   std::vector<mlir::Value> leftMapResults;
   std::vector<mlir::Value> rightMapResults;

   for (size_t i = 0; i < lTargetInfo.namedResults.size(); i++) {
      auto newName = lTargetInfo.namedResults[i].first;
      const auto* leftColumn = lTargetInfo.namedResults[i].second;
      const auto* rightColumn = rTargetInfo.namedResults[i].second;
      auto leftType = leftColumn->type;
      auto rightType = rightColumn->type;
      auto commonType = SQLTypeInference::getCommonType(leftType, rightType);
      if (leftType != commonType) {
         auto attrDef = attrManager.createDef(leftMapScope, std::string("set_op") + std::to_string(i));
         attrDef.getColumn().type = commonType;
         auto attrRef = attrManager.createRef(leftColumn);
         createdColsLeft.push_back(attrDef);
         mlir::Value expr = leftMapBuilder.create<mlir::tuples::GetColumnOp>(leftMapBuilder.getUnknownLoc(), attrRef.getColumn().type, attrRef, leftTuple);
         leftColumn = &attrDef.getColumn();
         leftMapResults.push_back(SQLTypeInference::castValueToType(leftMapBuilder, expr, commonType));
      }
      if (rightType != commonType) {
         auto attrDef = attrManager.createDef(rightMapScope, std::string("set_op") + std::to_string(i));
         auto attrRef = attrManager.createRef(rightColumn);
         attrDef.getColumn().type = commonType;
         createdColsRight.push_back(attrDef);
         mlir::Value expr = rightMapBuilder.create<mlir::tuples::GetColumnOp>(rightMapBuilder.getUnknownLoc(), attrRef.getColumn().type, attrRef, rightTuple);
         rightColumn = &attrDef.getColumn();
         rightMapResults.push_back(SQLTypeInference::castValueToType(rightMapBuilder, expr, commonType));
      }
      auto newType = SQLTypeInference::getCommonType(leftType, rightType);
      auto newColName = attrManager.getName(leftColumn).second;
      auto newColDef = attrManager.createDef(scopeName, newColName, builder.getArrayAttr({attrManager.createRef(leftColumn), attrManager.createRef(rightColumn)}));
      auto* newCol = &newColDef.getColumn();
      newCol->type = newType;
      attributes.push_back(newColDef);
      targetInfo.map(newName, newCol);
   }
   if (!leftMapResults.empty()) {
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), lTree, builder.getArrayAttr(createdColsLeft));
      mapOp.getPredicate().push_back(leftMapBlock);
      leftMapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), leftMapResults);
      lTree = mapOp.getResult();
   }
   if (!rightMapResults.empty()) {
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), rTree, builder.getArrayAttr(createdColsRight));
      mapOp.getPredicate().push_back(rightMapBlock);
      rightMapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), rightMapResults);
      rTree = mapOp.getResult();
   }
   mlir::Value tree;
   switch (stmt->op_) {
      case SETOP_UNION: {
         tree = builder.create<mlir::relalg::UnionOp>(builder.getUnknownLoc(), ::mlir::relalg::SetSemanticAttr::get(builder.getContext(), setSemantic), lTree, rTree, builder.getArrayAttr(attributes));
         break;
      }
      case SETOP_INTERSECT: {
         tree = builder.create<mlir::relalg::IntersectOp>(builder.getUnknownLoc(), ::mlir::relalg::SetSemanticAttr::get(builder.getContext(), setSemantic), lTree, rTree, builder.getArrayAttr(attributes));
         break;
      }
      case SETOP_EXCEPT: {
         tree = builder.create<mlir::relalg::ExceptOp>(builder.getUnknownLoc(), ::mlir::relalg::SetSemanticAttr::get(builder.getContext(), setSemantic), lTree, rTree, builder.getArrayAttr(attributes));
         break;
      }
      default: error("unsupported SET operation");
   }
   return std::make_pair(tree, targetInfo);
}
mlir::Block* frontend::sql::Parser::translatePredicate(mlir::OpBuilder& builder, Node* node, frontend::sql::Parser::TranslationContext& context) {
   auto* block = new mlir::Block;
   mlir::OpBuilder predBuilder(builder.getContext());
   block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto tupleScope = context.createTupleScope();
   context.setCurrentTuple(block->getArgument(0));

   predBuilder.setInsertionPointToStart(block);
   mlir::Value expr = translateExpression(predBuilder, node, context);
   predBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), expr);
   return block;
}
mlir::Value frontend::sql::Parser::translateFromClause(mlir::OpBuilder& builder, SelectStmt* stmt, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   List* fromClause = stmt->from_clause_;
   if (!fromClause) { return mlir::Value(); };
   mlir::Value last;
   for (auto* cell = fromClause->head; cell != nullptr; cell = cell->next) {
      auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
      auto translated = translateFromClausePart(builder, node, context, scope);

      if (last) {
         last = builder.create<mlir::relalg::CrossProductOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), last, translated);
      } else {
         last = translated;
      }
   }
   return last;
}
mlir::Value frontend::sql::Parser::translateFromClausePart(mlir::OpBuilder& builder, Node* node, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   switch (node->type) {
      case T_RangeVar: return translateRangeVar(builder, reinterpret_cast<RangeVar*>(node), context, scope);
      case T_RangeSubselect: {
         std::string alias;
         auto* stmt = reinterpret_cast<RangeSubselect*>(node);
         if (stmt->alias_ && stmt->alias_->type_ == T_Alias && stmt->alias_->aliasname_) {
            alias = stmt->alias_->aliasname_;
         } else {
            error("no alias for subquery");
         }
         std::vector<std::string> colAlias = listToStringVec(stmt->alias_->colnames_);
         return translateSubSelect(builder, reinterpret_cast<SelectStmt*>(stmt->subquery_), alias, colAlias, context, scope);
      }

      case T_JoinExpr: {
         JoinExpr* joinExpr = reinterpret_cast<JoinExpr*>(node);
         if ((joinExpr->jointype_ > 4) || (joinExpr->is_natural_)) {
            error("invalid join expr");
         }

         mlir::Value left;
         mlir::Value right;
         std::vector<std::pair<std::string, const mlir::tuples::Column*>> mapping;
         if (joinExpr->jointype_ == JOIN_FULL) {
            {
               TranslationContext rightContext;
               auto rightResolverScope = rightContext.createResolverScope();
               right = translateFromClausePart(builder, joinExpr->rarg_, rightContext, rightResolverScope);
               auto localMapping = rightContext.getAllDefinedColumns();
               mapping.insert(mapping.end(), localMapping.begin(), localMapping.end());
            }
            {
               TranslationContext leftContext;
               auto leftResolverScope = leftContext.createResolverScope();
               left = translateFromClausePart(builder, joinExpr->larg_, leftContext, leftResolverScope);
               auto localMapping = leftContext.getAllDefinedColumns();
               mapping.insert(mapping.end(), localMapping.begin(), localMapping.end());
            }
         } else if (joinExpr->jointype_ == JOIN_LEFT) {
            left = translateFromClausePart(builder, joinExpr->larg_, context, scope);
            TranslationContext rightContext;
            auto rightResolverScope = rightContext.createResolverScope();
            right = translateFromClausePart(builder, joinExpr->rarg_, rightContext, rightResolverScope);
            mapping = rightContext.getAllDefinedColumns();
         } else if (joinExpr->jointype_ == JOIN_RIGHT) {
            right = translateFromClausePart(builder, joinExpr->rarg_, context, scope);
            TranslationContext leftContext;
            auto leftResolverScope = leftContext.createResolverScope();
            left = translateFromClausePart(builder, joinExpr->larg_, leftContext, leftResolverScope);
            mapping = leftContext.getAllDefinedColumns();
         } else {
            left = translateFromClausePart(builder, joinExpr->larg_, context, scope);
            right = translateFromClausePart(builder, joinExpr->rarg_, context, scope);
         }

         if (!joinExpr->quals_) {
            error("join must contain predicate");
         }
         if (joinExpr->jointype_ == JOIN_FULL) {
            mlir::Block* pred;
            {
               auto predScope = context.createResolverScope();
               for (auto x : mapping) {
                  context.mapAttribute(scope, x.first, x.second);
               }
               pred = translatePredicate(builder, joinExpr->quals_, context);
            }
            static size_t id = 0;
            std::vector<mlir::Attribute> outerJoinMapping;
            std::string outerjoinName;
            if (!mapping.empty()) {
               outerjoinName = "foj" + std::to_string(id++);
               std::unordered_map<const mlir::tuples::Column*, const mlir::tuples::Column*> remapped;
               for (auto x : mapping) {
                  if (!remapped.contains(x.second)) {
                     auto [scopename, name] = attrManager.getName(x.second);

                     auto attrDef = attrManager.createDef(outerjoinName, scopename + "_" + name, builder.getArrayAttr({attrManager.createRef(x.second)}));
                     attrDef.getColumn().type = x.second->type.isa<mlir::db::NullableType>() ? x.second->type : mlir::db::NullableType::get(builder.getContext(), x.second->type);
                     outerJoinMapping.push_back(attrDef);
                     remapped.insert({x.second, &attrDef.getColumn()});
                  }
                  context.mapAttribute(scope, x.first, remapped[x.second]);
                  context.removeFromDefinedColumns(x.second);
               }
            }
            mlir::ArrayAttr mapping = builder.getArrayAttr(outerJoinMapping);
            auto join = builder.create<mlir::relalg::FullOuterJoinOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), left, right, mapping);
            join.getPredicate().push_back(pred);
            return join;
         } else if (joinExpr->jointype_ == JOIN_LEFT || joinExpr->jointype_ == JOIN_RIGHT) {
            if (joinExpr->jointype_ == JOIN_RIGHT) {
               std::swap(left, right);
            }
            mlir::Block* pred;
            {
               auto predScope = context.createResolverScope();
               for (auto x : mapping) {
                  context.mapAttribute(scope, x.first, x.second);
               }
               pred = translatePredicate(builder, joinExpr->quals_, context);
            }
            static size_t id = 0;
            std::vector<mlir::Attribute> outerJoinMapping;
            std::string outerjoinName;
            if (!mapping.empty()) {
               outerjoinName = "oj" + std::to_string(id++);
               std::unordered_map<const mlir::tuples::Column*, const mlir::tuples::Column*> remapped;
               for (auto x : mapping) {
                  if (!remapped.contains(x.second)) {
                     auto [scopename, name] = attrManager.getName(x.second);

                     auto attrDef = attrManager.createDef(outerjoinName, name, builder.getArrayAttr({attrManager.createRef(x.second)}));
                     attrDef.getColumn().type = x.second->type.isa<mlir::db::NullableType>() ? x.second->type : mlir::db::NullableType::get(builder.getContext(), x.second->type);
                     outerJoinMapping.push_back(attrDef);
                     remapped.insert({x.second, &attrDef.getColumn()});
                  }
                  context.mapAttribute(scope, x.first, remapped[x.second]);
                  context.removeFromDefinedColumns(x.second);
               }
            }
            mlir::ArrayAttr mapping = builder.getArrayAttr(outerJoinMapping);
            auto join = builder.create<mlir::relalg::OuterJoinOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), left, right, mapping);
            join.getPredicate().push_back(pred);
            return join;
         } else if (joinExpr->jointype_ == JOIN_INNER) {
            mlir::Block* pred = translatePredicate(builder, joinExpr->quals_, context);

            auto join = builder.create<mlir::relalg::InnerJoinOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), left, right);
            join.getPredicate().push_back(pred);
            return join;
         }
         error("unsupported join type");
         break;
      }
      default: {
         error("unknown type in from clause");
      }
   }
   return mlir::Value();
}
mlir::Value frontend::sql::Parser::translateExpression(mlir::OpBuilder& builder, Node* node, frontend::sql::Parser::TranslationContext& context, bool ignoreNull) {
   auto loc = builder.getUnknownLoc();
   if (!node) {
      if (ignoreNull) {
         return mlir::Value();
      }
      error("empty expression");
   }
   switch (node->type) {
      case T_A_Const: {
         auto constVal = reinterpret_cast<A_Const*>(node)->val_;
         switch (constVal.type_) {
            case T_Integer: return builder.create<mlir::db::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(constVal.val_.ival_));
            case T_String: {
               std::string stringVal = constVal.val_.str_;
               mlir::Type stringType = mlir::db::StringType::get(builder.getContext());
               if (stringVal.size() <= 8 && stringVal.size() > 0) {
                  stringType = mlir::db::CharType::get(builder.getContext(), stringVal.size());
               }
               return builder.create<mlir::db::ConstantOp>(loc, stringType, builder.getStringAttr(stringVal));
            }
            case T_Float: {
               std::string value(constVal.val_.str_);
               auto decimalPos = value.find('.');
               if (decimalPos == std::string::npos) {
                  return builder.create<mlir::db::ConstantOp>(loc, builder.getI64Type(), builder.getI64IntegerAttr(std::stoll(constVal.val_.str_)));
               } else {
                  auto s = value.size() - decimalPos - 1;
                  auto p = value.size() - 1;
                  return builder.create<mlir::db::ConstantOp>(loc, mlir::db::DecimalType::get(builder.getContext(), p, s), builder.getStringAttr(constVal.val_.str_));
               }
               break;
            }
            case T_Null: return builder.create<mlir::db::NullOp>(loc, mlir::db::NullableType::get(builder.getContext(), builder.getNoneType()));
            default: error("unsupported value type");
         }
         //expr = ConstTransform(parse_result, reinterpret_cast<A_Const*>(node),context);
         break;
      }
      case T_TypeCast:
      case T_A_Expr: {
         auto* expr = reinterpret_cast<A_Expr*>(node);
         //expr = AExprTransform(parse_result, ,context);
         //todo:implement AEXPR_IN (q12)
         if (expr->kind_ == AEXPR_IN) {
            auto* list = reinterpret_cast<List*>(expr->rexpr_);
            std::vector<mlir::Value> values;
            for (auto* cell = list->head; cell != nullptr; cell = cell->next) {
               auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
               values.push_back(translateExpression(builder, node, context));
            }
            auto val = translateExpression(builder, expr->lexpr_, context);
            values.insert(values.begin(), val);
            return builder.create<mlir::db::OneOfOp>(loc, SQLTypeInference::toCommonBaseTypes(builder, values));
         }
         if (expr->kind_ == AEXPR_BETWEEN || expr->kind_ == AEXPR_NOT_BETWEEN) {
            mlir::Value val = translateExpression(builder, expr->lexpr_, context);
            auto* list = reinterpret_cast<List*>(expr->rexpr_);
            assert(list->length == 2);
            auto* lowerNode = reinterpret_cast<Node*>(list->head->data.ptr_value);
            auto* upperNode = reinterpret_cast<Node*>(list->tail->data.ptr_value);
            mlir::Value lower = translateExpression(builder, lowerNode, context);
            mlir::Value upper = translateExpression(builder, upperNode, context);
            auto ct = SQLTypeInference::toCommonBaseTypes(builder, {val, lower, upper});
            mlir::Value between = builder.create<mlir::db::BetweenOp>(loc, ct[0], ct[1], ct[2], true, true);
            if (expr->kind_ == AEXPR_NOT_BETWEEN) {
               between = builder.create<mlir::db::NotOp>(loc, between);
            }
            return between;
         }
         if (expr->kind_ == AEXPR_LIKE) {
            expr->kind_ = AEXPR_OP;
         }
         if (expr->kind_ == AEXPR_OP) {
            ExpressionType opType;
            mlir::Value left, right;
            if (node->type == T_TypeCast) {
               opType = ExpressionType::OPERATOR_CAST;
            } else {
               auto* name = (reinterpret_cast<value*>(expr->name_->head->data.ptr_value))->val_.str_;
               opType = stringToExpressionType(name);
               left = expr->lexpr_ ? translateExpression(builder, expr->lexpr_, context) : left;
               right = expr->rexpr_ ? translateExpression(builder, expr->rexpr_, context) : right;
            }
            if (opType == ExpressionType::OPERATOR_CAST) {
               auto* castNode = reinterpret_cast<TypeCast*>(node);
               auto* typeName = reinterpret_cast<value*>(castNode->type_name_->names_->tail->data.ptr_value)->val_.str_;
               auto toCast = translateExpression(builder, castNode->arg_, context);
               auto columnType = createColumnType(typeName, false, getTypeModList(castNode->type_name_->typmods_));
               auto resType = createTypeFromColumnType(builder.getContext(), columnType);
               if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(toCast.getDefiningOp())) {
                  if (auto intervalType = resType.dyn_cast<mlir::db::IntervalType>()) {
                     std::string unit = "";
                     auto stringRepresentation = constOp.getValue().cast<mlir::StringAttr>().str();
                     if (!columnType.modifiers.empty() && std::get<std::string>(columnType.modifiers[0]) == "years") {
                        stringRepresentation = std::to_string(std::stol(stringRepresentation) * 12);
                     }
                     if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::daytime && !stringRepresentation.ends_with("days")) {
                        stringRepresentation += "days";
                     }
                     constOp->setAttr("value", builder.getStringAttr(stringRepresentation));
                  }
                  constOp.getResult().setType(resType);
                  return constOp;
               } else {
                  return SQLTypeInference::castValueToType(builder, toCast, resType);
               }
               return mlir::Value();
            }
            return translateBinaryExpression(builder, opType, left, right);

         } else {
            error("unsupported op");
         }
         break;
      }
      case T_BoolExpr: {
         auto* boolExpr = reinterpret_cast<BoolExpr*>(node);
         std::vector<mlir::Value> values;
         for (auto* cell = boolExpr->args_->head; cell != nullptr; cell = cell->next) {
            auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
            values.push_back(translateExpression(builder, node, context));
         }
         switch (boolExpr->boolop_) {
            case AND_EXPR: return builder.create<mlir::db::AndOp>(builder.getUnknownLoc(), values);
            case OR_EXPR: return builder.create<mlir::db::OrOp>(builder.getUnknownLoc(), values);
            case NOT_EXPR: return builder.create<mlir::db::NotOp>(builder.getUnknownLoc(), values[0]);
            default: {
               error("unsupported boolean expression");
            }
         }
         break;
      }
      case T_ColumnRef: {
         const auto* attr = resolveColRef(node, context);
         return builder.create<mlir::tuples::GetColumnOp>(builder.getUnknownLoc(), attr->type, attrManager.createRef(attr), context.getCurrentTuple());
         break;
      }
      case T_FakeNode: { //
         const auto* attr = context.getAttribute(reinterpret_cast<FakeNode*>(node)->colId);
         return builder.create<mlir::tuples::GetColumnOp>(builder.getUnknownLoc(), attr->type, attrManager.createRef(attr), context.getCurrentTuple());
         break;
      }
      case T_FuncCall: return translateFuncCallExpression(node, builder, loc, context);
      case T_NullTest: {
         auto* nullTest = reinterpret_cast<NullTest*>(node);
         auto expr = translateExpression(builder, reinterpret_cast<Node*>(nullTest->arg_), context);
         if (expr.getType().isa<mlir::db::NullableType>()) {
            mlir::Value isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), expr);
            if (nullTest->nulltesttype_ == IS_NOT_NULL) {
               return builder.create<mlir::db::NotOp>(builder.getUnknownLoc(), isNull);
            } else {
               return isNull;
            }
         } else {
            return builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI1Type(), nullTest->nulltesttype_ == IS_NOT_NULL));
         }
         break;
      }
      case T_ParamRef: {
         //expr = ParamRefTransform(parse_result, reinterpret_cast<ParamRef*>(node),context);
         break;
      }
      case T_SubLink: {
         auto* subLink = reinterpret_cast<SubLink*>(node);
         //expr = FuncCallTransform(parse_result,,context);
         std::pair<mlir::Value, TargetInfo> subQueryRes;
         {
            auto subQueryScope = context.createResolverScope();
            auto subQueryDefineScope = context.createDefineScope();
            subQueryRes = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(subLink->subselect_), context, subQueryScope);
         }
         auto [subQueryTree, targetInfo] = subQueryRes;
         switch (subLink->sub_link_type_) {
            case EXPR_SUBLINK: {
               assert(!targetInfo.namedResults.empty());
               const auto* attr = targetInfo.namedResults[0].second;
               mlir::Type resType = attr->type;
               if (!resType.isa<mlir::db::NullableType>()) {
                  resType = mlir::db::NullableType::get(builder.getContext(), attr->type);
               }
               mlir::Value scalarValue = builder.create<mlir::relalg::GetScalarOp>(loc, resType, attrManager.createRef(attr), subQueryTree);
               if (context.useZeroInsteadNull.contains(attr)) {
                  mlir::Value isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), scalarValue);
                  mlir::Value nonNullValue = builder.create<mlir::db::NullableGetVal>(builder.getUnknownLoc(), scalarValue);
                  mlir::Value defaultValue = builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), getBaseType(scalarValue.getType()), builder.getIntegerAttr(getBaseType(scalarValue.getType()), 0));
                  return builder.create<mlir::arith::SelectOp>(builder.getUnknownLoc(), isNull, defaultValue, nonNullValue);
               } else {
                  return scalarValue;
               }
            }
            case EXISTS_SUBLINK:
               return builder.create<mlir::relalg::ExistsOp>(loc, builder.getI1Type(), subQueryTree);
            case ANY_SUBLINK: {
               assert(targetInfo.namedResults.size() == 1);
               mlir::tuples::ColumnRefAttr attribute = attrManager.createRef(targetInfo.namedResults[0].second);
               auto operatorName = subLink->oper_name_ ? listToStringVec(subLink->oper_name_).at(0) : "=";
               auto operatorType = stringToExpressionType(operatorName);
               auto* block = new mlir::Block;
               mlir::OpBuilder predBuilder(builder.getContext());
               block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
               auto tupleScope = context.createTupleScope();
               context.setCurrentTuple(block->getArgument(0));

               predBuilder.setInsertionPointToStart(block);
               mlir::Value expr = translateExpression(predBuilder, subLink->testexpr_, context);
               mlir::Value colVal = predBuilder.create<mlir::tuples::GetColumnOp>(loc, attribute.getColumn().type, attribute, block->getArgument(0));
               mlir::Value pred = translateBinaryExpression(predBuilder, operatorType, expr, colVal);
               predBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), pred);

               auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), subQueryTree);
               sel.getPredicate().push_back(block);
               subQueryTree = sel.getResult();

               return builder.create<mlir::relalg::ExistsOp>(loc, builder.getI1Type(), subQueryTree);
            }
            case ALL_SUBLINK: {
               assert(targetInfo.namedResults.size() == 1);
               mlir::tuples::ColumnRefAttr attribute = attrManager.createRef(targetInfo.namedResults[0].second);
               auto operatorName = listToStringVec(subLink->oper_name_).at(0);
               auto operatorType = stringToExpressionType(operatorName);
               auto* block = new mlir::Block;
               mlir::OpBuilder predBuilder(builder.getContext());
               block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
               auto tupleScope = context.createTupleScope();
               context.setCurrentTuple(block->getArgument(0));

               predBuilder.setInsertionPointToStart(block);
               mlir::Value expr = translateExpression(predBuilder, subLink->testexpr_, context);
               mlir::Value colVal = predBuilder.create<mlir::tuples::GetColumnOp>(loc, attribute.getColumn().type, attribute, block->getArgument(0));
               mlir::Value pred = translateBinaryExpression(predBuilder, operatorType, expr, colVal);
               pred = predBuilder.create<mlir::db::NotOp>(loc, pred);
               predBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), pred);

               auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), subQueryTree);
               sel.getPredicate().push_back(block);
               subQueryTree = sel.getResult();

               mlir::Value exists = builder.create<mlir::relalg::ExistsOp>(loc, builder.getI1Type(), subQueryTree);
               return builder.create<mlir::db::NotOp>(loc, exists);
            }
            default:
               error("unsupported sublink type");
         }
         break;
      }
      case T_Integer: {
         auto intVal = reinterpret_cast<Value*>(node)->val_.ival_;
         return builder.create<mlir::db::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(intVal));
      }
      case T_CaseExpr: {
         auto* caseExpr = reinterpret_cast<CaseExpr*>(node);
         mlir::Value arg = translateExpression(builder, reinterpret_cast<Node*>(caseExpr->arg_), context, true);
         Node* defaultNode = reinterpret_cast<Node*>(caseExpr->defresult_);
         auto* startWhen = caseExpr->args_->head;
         return translateWhenCaseExpression(builder, context, arg, startWhen, defaultNode);
      }
      case T_CoalesceExpr: {
         auto* coalesceExpr = reinterpret_cast<AExpr*>(node);
         return translateCoalesceExpression(builder, context, reinterpret_cast<List*>(coalesceExpr->lexpr_)->head);
      }
      default: {
         error("unsupported expression type");
      }
   }
   error("should never happen");
   return mlir::Value();
}
void frontend::sql::Parser::translateCreateStatement(mlir::OpBuilder& builder, CreateStmt* statement) {
   RangeVar* relation = statement->relation_;
   std::string tableName = relation->relname_ != nullptr ? relation->relname_ : "";
   auto tableMetaData = translateTableMetaData(statement->table_elts_);
   auto tableNameValue = createStringValue(builder, tableName);
   auto descriptionValue = createStringValue(builder, tableMetaData->serialize());
   auto databaseValue = getCurrentDatabaseValue(builder);
   rt::Database::createTable(builder, builder.getUnknownLoc())(mlir::ValueRange({databaseValue, tableNameValue, descriptionValue}));
}
mlir::Value frontend::sql::Parser::translateSubSelect(mlir::OpBuilder& builder, SelectStmt* stmt, std::string alias, std::vector<std::string> colAlias, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   mlir::Value subQuery;
   TargetInfo targetInfo;
   {
      auto subQueryScope = context.createResolverScope();
      auto subQueryDefineScope = context.createDefineScope();
      auto [subQuery_, targetInfo_] = translateSelectStmt(builder, stmt, context, subQueryScope);
      subQuery = subQuery_;
      targetInfo = targetInfo_;
   }
   if (colAlias.empty()) {
      for (auto x : targetInfo.namedResults) {
         context.mapAttribute(scope, x.first, x.second);
         context.mapAttribute(scope, alias + "." + x.first, x.second);
      }
   } else {
      for (size_t i = 0; i < colAlias.size(); i++) {
         context.mapAttribute(scope, colAlias[i], targetInfo.namedResults.at(i).second);
         context.mapAttribute(scope, alias + "." + colAlias[i], targetInfo.namedResults.at(i).second);
      }
   }

   return subQuery;
}
mlir::Value frontend::sql::Parser::translateCoalesceExpression(mlir::OpBuilder& builder, frontend::sql::Parser::TranslationContext& context, ListCell* expressions) {
   auto loc = builder.getUnknownLoc();
   if (!expressions) {
      //no expression left -> return null
      return builder.create<mlir::db::NullOp>(loc, mlir::db::NullableType::get(builder.getContext(), builder.getNoneType()));
   }
   mlir::Value value = translateExpression(builder, reinterpret_cast<Node*>(expressions->data.ptr_value), context);
   mlir::Value isNull = value.getType().isa<mlir::db::NullableType>() ? (mlir::Value) builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), value) : (mlir::Value) builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 0, builder.getI1Type());
   mlir::Value isNotNull = builder.create<mlir::db::NotOp>(loc, isNull);
   auto* whenBlock = new mlir::Block;
   auto* elseBlock = new mlir::Block;
   mlir::OpBuilder whenBuilder(builder.getContext());
   whenBuilder.setInsertionPointToStart(whenBlock);
   mlir::OpBuilder elseBuilder(builder.getContext());
   elseBuilder.setInsertionPointToStart(elseBlock);
   auto elseRes = translateCoalesceExpression(elseBuilder, context, expressions->next);
   auto commonType = SQLTypeInference::getCommonType(value.getType(), elseRes.getType());
   value = SQLTypeInference::castValueToType(whenBuilder, value, commonType);
   elseRes = SQLTypeInference::castValueToType(elseBuilder, elseRes, commonType);
   whenBuilder.create<mlir::scf::YieldOp>(loc, value);
   elseBuilder.create<mlir::scf::YieldOp>(loc, elseRes);
   auto ifOp = builder.create<mlir::scf::IfOp>(loc, commonType, isNotNull, true);
   ifOp.getThenRegion().getBlocks().clear();
   ifOp.getElseRegion().getBlocks().clear();
   ifOp.getThenRegion().push_back(whenBlock);
   ifOp.getElseRegion().push_back(elseBlock);
   return ifOp.getResult(0);
}
std::pair<mlir::Value, frontend::sql::Parser::TargetInfo> frontend::sql::Parser::translateSelectStmt(mlir::OpBuilder& builder, SelectStmt* stmt, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   switch (stmt->op_) {
      case SETOP_NONE: return translateClassicSelectStmt(builder, stmt, context, scope);
      case SETOP_EXCEPT:
      case SETOP_INTERSECT:
      case SETOP_UNION: return translateSetOperation(builder, stmt, context, scope);
      default: {
         error("could not translate select statement type");
      }
   }
   return std::make_pair(mlir::Value(), TargetInfo());
}
void frontend::sql::Parser::translateCopyStatement(mlir::OpBuilder& builder, CopyStmt* copyStatement) {
   std::string fileName = copyStatement->filename_;
   std::string tableName = copyStatement->relation_->relname_;
   std::string delimiter = ",";
   std::string escape = "";
   for (auto* optionCell = copyStatement->options_->head; optionCell != nullptr; optionCell = optionCell->next) {
      auto* defElem = reinterpret_cast<DefElem*>(optionCell->data.ptr_value);
      std::string optionName = defElem->defname_;
      if (optionName == "delimiter") {
         delimiter = reinterpret_cast<value*>(defElem->arg_)->val_.str_;
      } else if (optionName == "escape") {
         escape = reinterpret_cast<value*>(defElem->arg_)->val_.str_;
      } else if (optionName == "format") {
         std::string format = reinterpret_cast<value*>(defElem->arg_)->val_.str_;
         if (format != "csv") {
            error("copy only supports csv");
         }
      } else if (optionName == "null") {
      } else {
         error("unsupported copy option");
      }
   }
   auto databaseValue = getCurrentDatabaseValue(builder);
   auto tableNameValue = createStringValue(builder, tableName);
   auto fileNameValue = createStringValue(builder, fileName);
   auto delimiterValue = createStringValue(builder, delimiter);
   auto escapeValue = createStringValue(builder, escape);
   rt::Database::copyFromIntoTable(builder, builder.getUnknownLoc())(mlir::ValueRange{databaseValue, tableNameValue, fileNameValue, delimiterValue, escapeValue});

   // To create hashValues scan table again, then calculate hashvalues and combine
   auto tableMetaData = database.getTableMetaData(tableName);
   if (!tableMetaData->getPrimaryKey().empty()) {
      // TODO: function can be removed once inlining pass does not interfere with subop ordering
      // Create function to avoid reordering of subops and scanning an empty table; temporary workaround
      std::string uniqueFunctionName = "addHashForPrimaryKey" + tableName;
      mlir::func::FuncOp funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(uniqueFunctionName);
      assert(!funcOp);
      mlir::Block* functionEntryBlock;
      // Create function
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(moduleOp.getBody());

         funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), uniqueFunctionName, builder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}), builder.getStringAttr("private"), mlir::ArrayAttr{}, mlir::ArrayAttr{});
         functionEntryBlock = funcOp.addEntryBlock();
      }
      // Create call to declared function
      builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp);

      // Define function
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(functionEntryBlock);

      auto scopeName = attrManager.getUniqueScope("copy");
      std::vector<mlir::Attribute> names, colTypes, colMemberNames;
      std::unordered_map<std::string, ::mlir::tuples::ColumnRefAttr> attrs;
      std::vector<mlir::NamedAttribute> columns;
      auto& memberManager = builder.getContext()->getLoadedDialect<mlir::subop::SubOperatorDialect>()->getMemberManager();
      for (auto x : tableMetaData->getOrderedColumns()) {
         if (x == "primaryKeyHashValue") continue;
         names.push_back(builder.getStringAttr(x));
         auto colMemberName = memberManager.getUniqueMember(x);
         auto columnType = createTypeFromColumnType(builder.getContext(), tableMetaData->getColumnMetaData(x)->getColumnType());

         auto attrDef = attrManager.createDef(scopeName, x);
         attrs[x] = attrManager.createRef(&attrDef.getColumn());
         colTypes.push_back(mlir::TypeAttr::get(columnType));
         colMemberNames.push_back(builder.getStringAttr(colMemberName));

         attrDef.getColumn().type = columnType;
         columns.push_back(builder.getNamedAttr(x, attrDef));
      }

      mlir::Value importedTable = builder.create<mlir::relalg::BaseTableOp>(
         builder.getUnknownLoc(),
         mlir::tuples::TupleStreamType::get(builder.getContext()),
         copyStatement->relation_->relname_,
         mlir::relalg::TableMetaDataAttr::get(builder.getContext(), std::make_shared<runtime::TableMetaData>()),
         builder.getDictionaryAttr(columns));

      // Create map operation for calculation of hash values
      mlir::OpBuilder mapBuilder(builder.getContext());
      mlir::Block* block = new mlir::Block;
      block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      ;
      mlir::Value tuple = block->getArgument(0);
      mapBuilder.setInsertionPointToStart(block);

      // Extract values for primary keys
      std::vector<mlir::Value> primaryKeyValues;
      for (auto& keyAttribute : tableMetaData->getPrimaryKey()) {
         // Primary key values must be set for every row
         auto columnType = createTypeFromColumnType(builder.getContext(), tableMetaData->getColumnMetaData(keyAttribute)->getColumnType());
         auto localAttrRef = attrs[keyAttribute];
         mlir::Value expr = mapBuilder.create<mlir::tuples::GetColumnOp>(mapBuilder.getUnknownLoc(), columnType, localAttrRef, tuple);
         primaryKeyValues.push_back(expr);
      }
      // Pack and hash primary keys
      mlir::Value primaryKeyTuple = mapBuilder.create<mlir::util::PackOp>(builder.getUnknownLoc(), mlir::ValueRange{primaryKeyValues});
      mlir::Value hashValue = mapBuilder.create<mlir::db::Hash>(builder.getUnknownLoc(), mapBuilder.getIndexType(), primaryKeyTuple);

      // Create new column for hashValues
      auto mapName = attrManager.getUniqueScope("hashValuesMap");
      auto attrDef = attrManager.createDef(mapName, std::string("primaryKeyHashValue"));
      attrDef.getColumn().type = mapBuilder.getIndexType();
      mlir::Attribute attrRef = attrManager.createRef(&attrDef.getColumn());
      std::vector<mlir::Attribute> createdCols{attrDef};
      std::vector<mlir::Value> createdValues{hashValue};

      // Return hash value column
      mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);

      // Build map operation
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), importedTable, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);

      // Materialize calculated hash values
      auto resultTableType = mlir::subop::ResultTableType::get(builder.getContext(), mlir::subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr(builder.getStringAttr("primaryKeyHashValue")), builder.getArrayAttr({mlir::TypeAttr::get(mlir::IndexType::get(builder.getContext()))})));
      auto resultColumn = builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), resultTableType, mapOp.getResult(), builder.getArrayAttr({attrRef}), builder.getArrayAttr({builder.getStringAttr("primaryKeyHashValue")}));
      builder.create<mlir::subop::SetResultOp>(builder.getUnknownLoc(), 0, resultColumn);

      auto executionContextValue = getExecutionContextValue(builder);
      auto resultIdValue = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 0, builder.getI32Type());
      rt::Database::combineTableWithHashValues(builder, builder.getUnknownLoc())(mlir::ValueRange{getCurrentDatabaseValue(builder), createStringValue(builder, tableName), executionContextValue, resultIdValue});

      // TODO: find more elegant solution
      // disable unwanted output by overwriting the result
      auto emptyTableType = mlir::subop::ResultTableType::get(builder.getContext(), mlir::subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr({}), builder.getArrayAttr({})));
      auto emptyMaterialization = builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), emptyTableType, mapOp.getResult(), builder.getArrayAttr({}), builder.getArrayAttr({}));
      builder.create<mlir::subop::SetResultOp>(builder.getUnknownLoc(), 0, emptyMaterialization);

      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
   }
}
void frontend::sql::Parser::translateVariableSetStatement(mlir::OpBuilder& builder, VariableSetStmt* variableSetStatement) {
   std::string varName = variableSetStatement->name_;
   auto* args = variableSetStatement->args_;
   if (varName == "persist") {
      assert(args->head != nullptr);
      assert(args->head == args->tail);
      auto* paramNode = reinterpret_cast<Node*>(args->head->data.ptr_value);
      assert(paramNode->type == T_A_Const);
      auto* constNode = reinterpret_cast<A_Const*>(paramNode);
      assert(constNode->val_.type_ == T_Integer);
      auto persistValue = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), constNode->val_.val_.ival_, 1);
      auto databaseValue = getCurrentDatabaseValue(builder);
      rt::Database::setPersist(builder, builder.getUnknownLoc())({databaseValue, persistValue});
   }
}
std::optional<mlir::Value> frontend::sql::Parser::translate(mlir::OpBuilder& builder) {
   if (result.tree && result.tree->length == 1) {
      auto* statement = static_cast<Node*>(result.tree->head->data.ptr_value);
      switch (statement->type) {
         case T_VariableSetStmt: {
            auto* variableSetStatement = reinterpret_cast<VariableSetStmt*>(statement);
            translateVariableSetStatement(builder, variableSetStatement);
            break;
         }
         case T_CreateStmt: {
            translateCreateStatement(builder, reinterpret_cast<CreateStmt*>(statement));
            break;
         }
         case T_CopyStmt: {
            auto* copyStatement = reinterpret_cast<CopyStmt*>(statement);
            translateCopyStatement(builder, copyStatement);
            break;
         }
         case T_SelectStmt: {
            TranslationContext context;
            auto scope = context.createResolverScope();
            auto [tree, targetInfo] = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(statement), context, scope);
            //::mlir::Type result, ::mlir::Value rel, ::mlir::ArrayAttr attrs, ::mlir::ArrayAttr columns
            std::vector<mlir::Attribute> attrs;
            std::vector<mlir::Attribute> names;
            std::vector<mlir::Attribute> colMemberNames;
            std::vector<mlir::Attribute> colTypes;
            auto& memberManager = builder.getContext()->getLoadedDialect<mlir::subop::SubOperatorDialect>()->getMemberManager();
            for (auto x : targetInfo.namedResults) {
               if (x.first == "primaryKeyHashValue") continue;
               names.push_back(builder.getStringAttr(x.first));
               auto colMemberName = memberManager.getUniqueMember(x.first.empty()?"unnamed":x.first);
               auto columnType = x.second->type;
               attrs.push_back(attrManager.createRef(x.second));
               colTypes.push_back(mlir::TypeAttr::get(columnType));
               colMemberNames.push_back(builder.getStringAttr(colMemberName));
            }
            auto resultTableType = mlir::subop::ResultTableType::get(builder.getContext(), mlir::subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr(colMemberNames), builder.getArrayAttr(colTypes)));
            return builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), resultTableType, tree, builder.getArrayAttr(attrs), builder.getArrayAttr(names));
         }
         case T_InsertStmt: {
            translateInsertStmt(builder, reinterpret_cast<InsertStmt*>(statement));
            break;
         }
         default:
            error("unsupported statement type");
      }
   }
   return {};
}
frontend::sql::Parser::~Parser() {
   pg_query_free_parse_result(result);
}
std::shared_ptr<runtime::TableMetaData> frontend::sql::Parser::translateTableMetaData(List* metaData) {
   auto tableMetaData = std::make_shared<runtime::TableMetaData>();
   for (auto* cell = metaData->head; cell != nullptr; cell = cell->next) {
      auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
      switch (node->type) {
         case T_ColumnDef: {
            auto columnDef = translateColumnDef(reinterpret_cast<ColumnDef*>(node));
            tableMetaData->addColumn(columnDef.first, columnDef.second);
            break;
         }
         case T_Constraint: {
            auto* constraint = reinterpret_cast<Constraint*>(node);
            switch (constraint->contype_) {
               case CONSTR_PRIMARY: {
                  std::vector<std::string> primaryKey;
                  for (auto* keyCell = constraint->keys_->head; keyCell != nullptr; keyCell = keyCell->next) {
                     primaryKey.push_back(reinterpret_cast<value*>(keyCell->data.ptr_value)->val_.str_);
                  }
                  tableMetaData->setPrimaryKey(primaryKey);
                  break;
               }
               default: {
                  error("unsupported constraint type");
               }
            }
            break;
         }
         default: {
            error("unsupported construct in create statement");
         }
      }
   }
   tableMetaData->setNumRows(0);
   return tableMetaData;
}
runtime::ColumnType frontend::sql::Parser::createColumnType(std::string datatypeName, bool isNull, std::vector<std::variant<size_t, std::string>> typeModifiers) {
   datatypeName = llvm::StringSwitch<std::string>(datatypeName)
                     .Case("bpchar", "char")
                     .Case("varchar", "string")
                     .Case("numeric", "decimal")
                     .Case("text", "string")
                     .Default(datatypeName);
   if (datatypeName == "int4") {
      datatypeName = "int";
      typeModifiers.push_back(32ull);
   }
   if (datatypeName == "int8") {
      datatypeName = "int";
      typeModifiers.push_back(64ull);
   }
   if (datatypeName == "float4") {
      datatypeName = "float";
      typeModifiers.push_back(32ull);
   }
   if (datatypeName == "float8") {
      datatypeName = "float";
      typeModifiers.push_back(64ull);
   }
   if (datatypeName == "char" && std::get<size_t>(typeModifiers[0]) > 8) {
      typeModifiers.clear();
      datatypeName = "string";
   }
   if (datatypeName == "date") {
      typeModifiers.clear();
      typeModifiers.push_back("day");
   }
   if (datatypeName == "timestamp") {
      datatypeName = "date";
      typeModifiers.push_back("millisecond");
   }
   if (datatypeName == "interval") {
      if (typeModifiers.size() > 0 && std::holds_alternative<size_t>(typeModifiers[0])) {
         std::string unit = "";
         if (std::get<size_t>(typeModifiers[0]) & 2) {
            unit = "months";
         }
         if (std::get<size_t>(typeModifiers[0]) & 4) {
            unit = "years";
         }
         if (std::get<size_t>(typeModifiers[0]) & 8) {
            unit = "daytime";
         }
         assert(!unit.empty() && "should not happen");
         typeModifiers.clear();
         typeModifiers.push_back(unit);
      } else {
         typeModifiers.clear();
         typeModifiers.push_back("daytime");
      }
   }
   runtime::ColumnType columnType;
   columnType.base = datatypeName;
   columnType.nullable = isNull;
   columnType.modifiers = typeModifiers;
   return columnType;
}
std::pair<std::string, std::shared_ptr<runtime::ColumnMetaData>> frontend::sql::Parser::translateColumnDef(ColumnDef* columnDef) {
   auto* typeName = columnDef->type_name_;
   std::vector<std::variant<size_t, std::string>> typeModifiers = getTypeModList(typeName->typmods_);
   bool isNotNull = false;

   if (columnDef->constraints_ != nullptr) {
      for (auto* cell = columnDef->constraints_->head; cell != nullptr; cell = cell->next) {
         auto* constraint = reinterpret_cast<Constraint*>(cell->data.ptr_value);
         switch (constraint->contype_) {
            case CONSTR_NOTNULL: {
               isNotNull = true;
               break;
            }
            case CONSTR_UNIQUE: break; // do something useful
            case CONSTR_PRIMARY: break; // do something useful
            default: {
               error("unsupported column constraint");
            }
         }
      }
   }
   std::string name = columnDef->colname_;
   std::string datatypeName = reinterpret_cast<value*>(typeName->names_->tail->data.ptr_value)->val_.str_;
   auto columnMetaData = std::make_shared<runtime::ColumnMetaData>();
   columnMetaData->setColumnType(createColumnType(datatypeName, !isNotNull, typeModifiers));
   return {name, columnMetaData};
}
std::vector<std::variant<size_t, std::string>> frontend::sql::Parser::getTypeModList(List* typeMods) {
   std::vector<std::variant<size_t, std::string>> typeModifiers;
   if (typeMods != nullptr) {
      for (auto* cell = typeMods->head; cell != nullptr; cell = cell->next) {
         auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
         switch (node->type) {
            case T_A_Const: {
               auto nodeType = reinterpret_cast<A_Const*>(node)->val_.type_;
               switch (nodeType) {
                  case T_Integer: {
                     typeModifiers.push_back(static_cast<size_t>(reinterpret_cast<A_Const*>(node)->val_.val_.ival_));
                     break;
                  }
                  default: {
                     error("unsupported type mod");
                  }
               }
               break;
            }
            default: {
               error("unsupported type mod");
            }
         }
      }
   }
   return typeModifiers;
}
void frontend::sql::Parser::translateInsertStmt(mlir::OpBuilder& builder, InsertStmt* stmt) {
   assert(stmt->with_clause_ == nullptr);
   assert(stmt->on_conflict_clause_ == nullptr);
   RangeVar* relation = stmt->relation_;
   std::string tableName = relation->relname_ != nullptr ? relation->relname_ : "";

   TranslationContext context;
   auto scope = context.createResolverScope();
   auto [tree, targetInfo] = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(stmt->select_stmt_), context, scope);

   auto tableMetaData = database.getTableMetaData(tableName);
   std::unordered_map<std::string, mlir::Type> tableColumnTypes;
   for (auto c : tableMetaData->getOrderedColumns()) {
      auto type = createTypeFromColumnType(builder.getContext(), tableMetaData->getColumnMetaData(c)->getColumnType());
      tableColumnTypes[c] = type;
   }
   std::vector<std::string> insertColNames;
   if (stmt->cols_) {
      for (auto* cell = stmt->cols_->head; cell != nullptr; cell = cell->next) {
         auto* target = reinterpret_cast<ResTarget*>(cell->data.ptr_value);
         insertColNames.emplace_back(target->name_);
      }
   } else {
      insertColNames = tableMetaData->getOrderedColumns();
   }
   assert(insertColNames.size() + !tableMetaData->getPrimaryKey().empty() == tableColumnTypes.size());
   assert(insertColNames.size() == targetInfo.namedResults.size());
   std::vector<mlir::Attribute> attrs;

   std::vector<mlir::Value> createdValues;
   std::unordered_map<std::string, mlir::Value> columnNameToCreatedValue;

   mlir::Block* block = new mlir::Block;
   mlir::OpBuilder mapBuilder(builder.getContext());
   block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto tupleScope = context.createTupleScope();
   mlir::Value tuple = block->getArgument(0);
   context.setCurrentTuple(tuple);

   mapBuilder.setInsertionPointToStart(block);

   std::unordered_map<std::string, mlir::Attribute> insertedCols;

   std::vector<mlir::Attribute> createdCols;
   auto mapName = attrManager.getUniqueScope("map");
   for (size_t i = 0; i < insertColNames.size(); i++) {
      auto attrRef = attrManager.createRef(targetInfo.namedResults[i].second);
      auto currentType = attrRef.getColumn().type;
      auto tableType = tableColumnTypes.at(insertColNames[i]);
      mlir::Value expr = mapBuilder.create<mlir::tuples::GetColumnOp>(mapBuilder.getUnknownLoc(), attrRef.getColumn().type, attrRef, tuple);
      if (currentType != tableType) {
         auto attrDef = attrManager.createDef(mapName, std::string("inserted") + std::to_string(i));
         attrDef.getColumn().type = tableType;

         createdCols.push_back(attrDef);
         mlir::Value casted = SQLTypeInference::castValueToType(mapBuilder, expr, tableType);

         createdValues.push_back(casted);
         columnNameToCreatedValue[insertColNames[i]] = casted;
         insertedCols[insertColNames[i]] = attrManager.createRef(&attrDef.getColumn());
      } else {
         columnNameToCreatedValue[insertColNames[i]] = expr;
         insertedCols[insertColNames[i]] = attrRef;
      }
   }
   // calculate hash value, pack tuple, hash
   if (!tableMetaData->getPrimaryKey().empty()) {
      // extract values for primary keys
      std::vector<mlir::Value> primaryKeyValues;
      for (auto& keyAttribute : tableMetaData->getPrimaryKey()) {
         // primary key values must be set for every row
         assert(columnNameToCreatedValue.contains(keyAttribute));
         primaryKeyValues.push_back(columnNameToCreatedValue[keyAttribute]);
      }
      mlir::Value primaryKeyTuple = mapBuilder.create<mlir::util::PackOp>(builder.getUnknownLoc(), mlir::ValueRange{primaryKeyValues});
      mlir::Value hashValue = mapBuilder.create<mlir::db::Hash>(builder.getUnknownLoc(), mapBuilder.getIndexType(), primaryKeyTuple);
      auto attrDef = attrManager.createDef(mapName, std::string("inserted") + std::to_string(insertColNames.size()));
      attrDef.getColumn().type = mapBuilder.getIndexType();

      createdCols.push_back(attrDef);
      mlir::Value casted = SQLTypeInference::castValueToType(mapBuilder, hashValue, mapBuilder.getIndexType());

      createdValues.push_back(casted);
      insertedCols["primaryKeyHashValue"] = attrManager.createRef(&attrDef.getColumn());
   }

   auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
   mapOp.getPredicate().push_back(block);
   mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
   std::vector<mlir::Attribute> colMemberNames;
   std::vector<mlir::Attribute> orderedColNamesAttrs;
   std::vector<mlir::Attribute> orderedColAttrs;
   std::vector<mlir::Attribute> colTypes;
   auto& memberManager = builder.getContext()->getLoadedDialect<mlir::subop::SubOperatorDialect>()->getMemberManager();
   for (auto x : tableMetaData->getOrderedColumns()) {
      colMemberNames.push_back(builder.getStringAttr(memberManager.getUniqueMember(x)));
      orderedColNamesAttrs.push_back(builder.getStringAttr(x));
      orderedColAttrs.push_back(insertedCols.at(x));
      colTypes.push_back(mlir::TypeAttr::get(insertedCols.at(x).cast<mlir::tuples::ColumnRefAttr>().getColumn().type));
   }
   auto resultTableType = mlir::subop::ResultTableType::get(builder.getContext(), mlir::subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr(colMemberNames), builder.getArrayAttr(colTypes)));
   mlir::Value newRows = builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), resultTableType, mapOp.getResult(), builder.getArrayAttr(orderedColAttrs), builder.getArrayAttr(orderedColNamesAttrs));

   auto databaseValue = getCurrentDatabaseValue(builder);
   auto executionContextValue = getExecutionContextValue(builder);
   auto tableNameValue = createStringValue(builder, tableName);
   auto resultIdValue = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 0, builder.getI32Type());
   builder.create<mlir::subop::SetResultOp>(builder.getUnknownLoc(), 0, newRows);
   rt::Database::appendTableFromResult(builder, builder.getUnknownLoc())(mlir::ValueRange{databaseValue, tableNameValue, executionContextValue, resultIdValue});

   // TODO: find more elegant solution
   // disable unwanted output by overwriting the result
   auto emptyTableType = mlir::subop::ResultTableType::get(builder.getContext(), mlir::subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr({}), builder.getArrayAttr({})));
   auto emptyMaterialization = builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), emptyTableType, mapOp.getResult(), builder.getArrayAttr({}), builder.getArrayAttr({}));
   builder.create<mlir::subop::SetResultOp>(builder.getUnknownLoc(), 0, emptyMaterialization);
}
Node* frontend::sql::Parser::analyzeTargetExpression(Node* node, frontend::sql::ReplaceState& replaceState) {
   if (!node) return node;
   switch (node->type) {
      case T_FakeNode: {
         auto* fakeNode = reinterpret_cast<FakeNode*>(node);
         return analyzeTargetExpression(fakeNode->original, replaceState);
      }
      case T_GroupingFunc: {
         auto* groupingFunc = reinterpret_cast<GroupingFunc*>(node);
         assert(groupingFunc->args);
         assert(groupingFunc->args->head);
         auto* exprNode = reinterpret_cast<Node*>(groupingFunc->args->head->data.ptr_value);
         assert(exprNode->type == T_ColumnRef);
         auto* columnRef = reinterpret_cast<ColumnRef*>(exprNode);
         auto attrName = fieldsToString(columnRef->fields_);
         auto* fakeNode = createFakeNode("grouping(" + attrName + ")", node);
         replaceState.groupingFuncs.insert({fakeNode, attrName});
         return fakeNode;
      }
      case T_CaseExpr: {
         auto* caseExpr = reinterpret_cast<CaseExpr*>(node);
         //mlir::Value arg = translateExpression(builder, reinterpret_cast<Node*>(caseExpr->arg_), context, true);
         caseExpr->defresult_ = (Expr*) analyzeTargetExpression(reinterpret_cast<Node*>(caseExpr->defresult_), replaceState);
         caseExpr->arg_ = (Expr*) analyzeTargetExpression(reinterpret_cast<Node*>(caseExpr->arg_), replaceState);
         for (auto* whenCell = caseExpr->args_->head; whenCell != nullptr; whenCell = whenCell->next) {
            auto* w = reinterpret_cast<CaseWhen*>(whenCell->data.ptr_value);
            w->expr_ = (Expr*) analyzeTargetExpression((Node*) w->expr_, replaceState);
            w->result_ = (Expr*) analyzeTargetExpression((Node*) w->result_, replaceState);
         }
         return node;
      }
      case T_FuncCall: {
         auto* funcNode = reinterpret_cast<FuncCall*>(node);
         std::string funcName = reinterpret_cast<value*>(funcNode->funcname_->head->data.ptr_value)->val_.str_;
         if (auto* window = reinterpret_cast<WindowDef*>(funcNode->over_)) {
            Node* exprNode = nullptr;
            auto* fakeNode = createFakeNode(funcName, node);
            if (funcNode->args_ && funcNode->args_->head) {
               exprNode = reinterpret_cast<Node*>(funcNode->args_->head->data.ptr_value);
               exprNode = analyzeTargetExpression(exprNode, replaceState);
               if (exprNode->type != T_ColumnRef) {
                  auto* beforeFakeNode = createFakeNode("", exprNode);
                  replaceState.evalBeforeWindowFunc.insert({beforeFakeNode, exprNode});
                  exprNode = beforeFakeNode;
               }
            }
            WindowProperties properties;
            if (window->partition_clause_) {
               for (auto* cell = window->partition_clause_->head; cell != nullptr; cell = cell->next) {
                  auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
                  if (node->type == T_ColumnRef) {
                     properties.partitionBy.push_back(node);
                  } else {
                     node = analyzeTargetExpression(node, replaceState);
                     auto* beforeFakeNode = createFakeNode("", node);
                     replaceState.evalBeforeWindowFunc.insert({beforeFakeNode, node});
                     properties.partitionBy.push_back(beforeFakeNode);
                  }
               }
            }
            if (window->order_clause_) {
               for (auto* cell = window->order_clause_->head; cell != nullptr; cell = cell->next) {
                  auto* temp = reinterpret_cast<Node*>(cell->data.ptr_value);
                  switch (temp->type) {
                     case T_SortBy: {
                        auto* sort = reinterpret_cast<SortBy*>(temp);
                        auto* expr = analyzeTargetExpression(sort->node_, replaceState);
                        if (expr->type != T_ColumnRef) {
                           auto* beforeFakeNode = createFakeNode("", expr);
                           replaceState.evalBeforeWindowFunc.insert({beforeFakeNode, expr});
                           expr = beforeFakeNode;
                        }
                        properties.orderBy.push_back({sort->sortby_dir_, expr});
                        break;
                     }
                     default: {
                        error("unknown orderby type");
                     }
                  }
               }
            }
            size_t startOffset = 0;
            size_t endOffset = 0;
            if (auto* constExpr = reinterpret_cast<A_Const*>(window->start_offset_)) {
               assert(constExpr->type_ == T_A_Const);
               auto constVal = constExpr->val_;
               switch (constVal.type_) {
                  case T_Integer: {
                     startOffset = constVal.val_.ival_;
                     break;
                  }
                  default: error("unsupported window start specification");
               }
            }
            if (auto* constExpr = reinterpret_cast<A_Const*>(window->end_offset_)) {
               assert(constExpr->type_ == T_A_Const);
               auto constVal = constExpr->val_;
               switch (constVal.type_) {
                  case T_Integer: {
                     endOffset = constVal.val_.ival_;
                     break;
                  }
                  default: error("unsupported window start specification");
               }
            }
            if (window->frame_options_ & FRAMEOPTION_START_CURRENT_ROW) {
               properties.start = 0;
            }
            if ((window->frame_options_ & FRAMEOPTION_END_CURRENT_ROW) && window->order_clause_) {
               properties.end = 0;
            }
            if (window->frame_options_ & FRAMEOPTION_START_VALUE_FOLLOWING) {
               properties.start = startOffset;
            }
            if (window->frame_options_ & FRAMEOPTION_START_VALUE_PRECEDING) {
               properties.start = -startOffset;
            }
            if (window->frame_options_ & FRAMEOPTION_END_VALUE_FOLLOWING) {
               properties.end = endOffset;
            }
            if (window->frame_options_ & FRAMEOPTION_END_VALUE_PRECEDING) {
               properties.end = -endOffset;
            }
            replaceState.windowFunctions.insert({fakeNode, {funcName, exprNode, properties}});
            return fakeNode;
         } else {
            if (funcName == "sum" || funcName == "avg" || funcName == "min" || funcName == "max" || funcName == "count" || funcName == "stddev_samp") {
               Node* aggrExpr = nullptr;
               auto* fakeNode = createFakeNode(funcName, node);
               if (funcNode->agg_star_) {
                  funcName += "*";
               } else {
                  auto* exprNode = reinterpret_cast<Node*>(funcNode->args_->head->data.ptr_value);
                  if (exprNode->type != T_ColumnRef) {
                     auto* beforeFakeNode = createFakeNode("", exprNode);
                     replaceState.evalBeforeAggr.insert({beforeFakeNode, exprNode});
                     aggrExpr = beforeFakeNode;
                  } else {
                     aggrExpr = exprNode;
                  }
               }
               replaceState.aggrs.insert({fakeNode, {funcName, aggrExpr, funcNode->agg_distinct_}});
               return fakeNode;
            }
            return node;
         }
      }
      case T_TypeCast: {
         auto* castNode = reinterpret_cast<TypeCast*>(node);
         castNode->arg_ = analyzeTargetExpression(castNode->arg_, replaceState);
         return node;
      }
      case T_A_Expr: {
         auto* expr = reinterpret_cast<A_Expr*>(node);
         //expr = AExprTransform(parse_result, ,context);
         if (expr->kind_ == AEXPR_OP) {
            if (node->type == T_TypeCast) {
            } else {
               expr->lexpr_ = analyzeTargetExpression(expr->lexpr_, replaceState);
               expr->rexpr_ = analyzeTargetExpression(expr->rexpr_, replaceState);
            }
            return node;
         }
         break;
      }
      case T_BoolExpr: {
         auto* boolExpr = reinterpret_cast<BoolExpr*>(node);
         std::vector<mlir::Value> values;
         for (auto* cell = boolExpr->args_->head; cell != nullptr; cell = cell->next) {
            auto* nodePtr = reinterpret_cast<Node**>(&cell->data.ptr_value);
            *nodePtr = analyzeTargetExpression(*nodePtr, replaceState);
         }
         return node;
         break;
      }
      case T_NullTest: {
         auto* nullTest = reinterpret_cast<NullTest*>(node);
         nullTest->arg_ = (Expr*) analyzeTargetExpression((Node*) nullTest->arg_, replaceState);
         return node;
      }

      default: return node;
   }
   return node;
}
std::pair<mlir::Value, mlir::tuples::ColumnRefAttr> frontend::sql::Parser::mapExpressionToAttribute(mlir::Value tree, TranslationContext& context, mlir::OpBuilder& builder, TranslationContext::ResolverScope& scope, Node* expression) {
   auto* block = new mlir::Block;
   static size_t mapId = 0;
   std::string mapName = "map" + std::to_string(mapId++);

   mlir::OpBuilder mapBuilder(builder.getContext());
   block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto tupleScope = context.createTupleScope();
   mlir::Value tuple = block->getArgument(0);
   context.setCurrentTuple(tuple);

   mapBuilder.setInsertionPointToStart(block);
   auto attrDef = attrManager.createDef(mapName, "tmp");
   mlir::Value createdValue = translateExpression(mapBuilder, expression, context);
   attrDef.getColumn().type = createdValue.getType();

   mlir::Attribute createdCol = attrDef;
   auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr({createdCol}));
   mapOp.getPredicate().push_back(block);
   mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValue);
   return {mapOp.getResult(), attrManager.createRef(&attrDef.getColumn())};
}
std::string fingerprint(Node* n) {
   std::regex r(",\\s*\"location\":\\s*\\d+");
   std::string json = pg_query_nodes_to_json(n);
   return std::regex_replace(json, r, "");
}
std::tuple<mlir::Value, std::unordered_map<std::string, mlir::tuples::Column*>> frontend::sql::Parser::performAggregation(mlir::OpBuilder& builder, std::vector<mlir::Attribute> groupByAttrs, const ReplaceState& replaceState, TranslationContext& context, mlir::Value tree) {
   static size_t groupById = 0;
   auto tupleStreamType = mlir::tuples::TupleStreamType::get(builder.getContext());
   auto tupleType = mlir::tuples::TupleType::get(builder.getContext());

   std::string groupByName = "aggr" + std::to_string(groupById++);
   auto tupleScope = context.createTupleScope();
   auto* block = new mlir::Block;
   block->addArgument(tupleStreamType, builder.getUnknownLoc());
   block->addArgument(tupleType, builder.getUnknownLoc());
   mlir::Value relation = block->getArgument(0);
   mlir::OpBuilder aggrBuilder(builder.getContext());
   aggrBuilder.setInsertionPointToStart(block);
   std::vector<mlir::Value> createdValues;
   std::vector<mlir::Attribute> createdCols;
   std::unordered_map<std::string, mlir::tuples::Column*> mapping;
   for (auto toAggr : replaceState.aggrs) {
      mlir::Value expr; //todo
      auto aggrFuncName = std::get<0>(toAggr.second);
      if (aggrFuncName == "grouping") continue;
      auto* attrNode = std::get<1>(toAggr.second);
      auto distinct = std::get<2>(toAggr.second);
      auto attrDef = attrManager.createDef(groupByName, toAggr.first->colId);
      if (aggrFuncName == "count*") {
         expr = aggrBuilder.create<mlir::relalg::CountRowsOp>(builder.getUnknownLoc(), builder.getI64Type(), relation);
         if (groupByAttrs.empty()) {
            context.useZeroInsteadNull.insert(&attrDef.getColumn());
         }
      } else {
         auto aggrFunc = llvm::StringSwitch<mlir::relalg::AggrFunc>(aggrFuncName)
                            .Case("sum", mlir::relalg::AggrFunc::sum)
                            .Case("avg", mlir::relalg::AggrFunc::avg)
                            .Case("min", mlir::relalg::AggrFunc::min)
                            .Case("max", mlir::relalg::AggrFunc::max)
                            .Case("count", mlir::relalg::AggrFunc::count)
                            .Case("stddev_samp", mlir::relalg::AggrFunc::stddev_samp)
                            .Default(mlir::relalg::AggrFunc::count);
         if (aggrFunc == mlir::relalg::AggrFunc::count) {
            if (groupByAttrs.empty()) {
               context.useZeroInsteadNull.insert(&attrDef.getColumn());
            }
         }
         mlir::tuples::ColumnRefAttr refAttr;
         switch (attrNode->type) {
            case T_ColumnRef: refAttr = attrManager.createRef(resolveColRef(attrNode, context)); break;
            case T_FakeNode: refAttr = attrManager.createRef(context.getAttribute(reinterpret_cast<FakeNode*>(attrNode)->colId)); break;
            default: error("could not resolve aggr attribute");
         }
         mlir::Value currRel = relation;
         if (distinct) {
            currRel = aggrBuilder.create<mlir::relalg::ProjectionOp>(builder.getUnknownLoc(), mlir::relalg::SetSemantic::distinct, currRel, builder.getArrayAttr({refAttr}));
         }
         mlir::Type aggrResultType;
         if (aggrFunc == mlir::relalg::AggrFunc::count) {
            aggrResultType = builder.getI64Type();
         } else {
            aggrResultType = refAttr.getColumn().type;
            if (aggrFunc == mlir::relalg::AggrFunc::avg) {
               auto baseType = getBaseType(aggrResultType);
               if (baseType.isIntOrFloat() && !baseType.isIntOrIndex()) {
                  //keep aggrResultType
               } else if (baseType.isa<mlir::db::DecimalType>()) {
                  mlir::OpBuilder b(builder.getContext());
                  mlir::Value x = b.create<mlir::db::ConstantOp>(b.getUnknownLoc(), baseType, b.getUnitAttr());
                  mlir::Value x2 = b.create<mlir::db::ConstantOp>(b.getUnknownLoc(), mlir::db::DecimalType::get(b.getContext(), 19, 0), b.getUnitAttr());
                  mlir::Value div = b.create<mlir::db::DivOp>(b.getUnknownLoc(), x, x2);
                  aggrResultType = div.getType();
                  div.getDefiningOp()->erase();
                  x2.getDefiningOp()->erase();
                  x.getDefiningOp()->erase();
               } else {
                  mlir::OpBuilder b(builder.getContext());
                  mlir::Value x = b.create<mlir::db::ConstantOp>(b.getUnknownLoc(), mlir::db::DecimalType::get(b.getContext(), 19, 0), b.getUnitAttr());
                  mlir::Value div = b.create<mlir::db::DivOp>(b.getUnknownLoc(), x, x);
                  aggrResultType = div.getType();
                  div.getDefiningOp()->erase();
                  x.getDefiningOp()->erase();
               }
               if (refAttr.getColumn().type.isa<mlir::db::NullableType>()) {
                  aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
               }
            }
            if (aggrFunc == mlir::relalg::AggrFunc::stddev_samp || aggrFunc == mlir::relalg::AggrFunc::var_samp) {
               aggrResultType = builder.getF64Type();
               if (refAttr.getColumn().type.isa<mlir::db::NullableType>()) {
                  aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
               }
            }
            if (!aggrResultType.isa<mlir::db::NullableType>() && (groupByAttrs.empty())) {
               aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
            }
         }
         expr = aggrBuilder.create<mlir::relalg::AggrFuncOp>(builder.getUnknownLoc(), aggrResultType, aggrFunc, currRel, refAttr);
      }
      attrDef.getColumn().type = expr.getType();
      mapping.insert({toAggr.first->colId, &attrDef.getColumn()});
      createdCols.push_back(attrDef);
      createdValues.push_back(expr);
   }
   aggrBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
   auto groupByOp = builder.create<mlir::relalg::AggregationOp>(builder.getUnknownLoc(), tupleStreamType, tree, builder.getArrayAttr(groupByAttrs), builder.getArrayAttr(createdCols));
   groupByOp.getAggrFunc().push_back(block);

   return {groupByOp.getResult(), mapping};
}
std::pair<mlir::Value, frontend::sql::Parser::TargetInfo> frontend::sql::Parser::translateSelectionTargetList(mlir::OpBuilder& builder, List* groupBy, Node* having, List* targetList, List* sortClause, List* distinctClause, mlir::Value tree, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   auto createMap = [this](mlir::OpBuilder& builder, std::unordered_map<FakeNode*, Node*>& toMap, TranslationContext& context, mlir::Value tree, TranslationContext::ResolverScope& scope) -> mlir::Value {
      if (toMap.empty()) return tree;
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      auto tupleScope = context.createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      context.setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      for (auto p : toMap) {
         mlir::Value expr = translateExpression(mapBuilder, p.second, context);
         auto attrDef = attrManager.createDef(mapName, p.first->colId);
         attrDef.getColumn().type = expr.getType();
         context.mapAttribute(scope, p.first->colId, &attrDef.getColumn());
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);
      }
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
      return mapOp.getResult();
   };
   auto mapToNull = [this](mlir::OpBuilder& builder, std::vector<mlir::Attribute> toMap, TranslationContext& context, mlir::Value tree) {
      if (toMap.empty()) return tree;
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      auto tupleScope = context.createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      context.setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      for (auto p : toMap) {
         auto colRef = p.cast<mlir::tuples::ColumnDefAttr>();
         mlir::Value expr = mapBuilder.create<mlir::db::NullOp>(builder.getUnknownLoc(), colRef.getColumn().type);
         auto attrDef = attrManager.createDef(&colRef.getColumn());
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);
      }
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
      return mapOp.asRelation();
   };
   auto mapInt = [this](mlir::OpBuilder& builder, size_t intVal, TranslationContext& context, mlir::Value tree) -> std::pair<mlir::Value, mlir::Attribute> {
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      auto tupleScope = context.createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      context.setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      mlir::Value expr = mapBuilder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), intVal, mapBuilder.getI64Type());
      auto attrDef = attrManager.createDef(mapName, "intval");
      attrDef.getColumn().type = mapBuilder.getI64Type();
      createdCols.push_back(attrDef);
      createdValues.push_back(expr);

      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
      return {mapOp.getResult(), attrDef};
   };
   auto mapCheckBit = [this](mlir::OpBuilder& builder, mlir::Attribute val, size_t shift, TranslationContext& context, mlir::Value tree) -> std::pair<mlir::Value, mlir::Attribute> {
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      auto tupleScope = context.createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      context.setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      auto colDef = val.cast<mlir::tuples::ColumnDefAttr>();
      auto colRef = attrManager.createRef(&colDef.getColumn());
      mlir::Value shiftVal = mapBuilder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), shift, mapBuilder.getI64Type());
      mlir::Value colVal = mapBuilder.create<mlir::tuples::GetColumnOp>(builder.getUnknownLoc(), colRef.getColumn().type, colRef, tuple);
      mlir::Value shifted = mapBuilder.create<mlir::arith::ShRUIOp>(builder.getUnknownLoc(), colVal, shiftVal);
      mlir::Value one = mapBuilder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 1, mapBuilder.getI64Type());
      mlir::Value expr = mapBuilder.create<mlir::arith::AndIOp>(builder.getUnknownLoc(), shifted, one);
      auto attrDef = attrManager.createDef(mapName, "intval");
      attrDef.getColumn().type = mapBuilder.getI64Type();
      createdCols.push_back(attrDef);
      createdValues.push_back(expr);

      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
      return {mapOp.getResult(), attrDef};
   };
   auto mapToNullable = [this](mlir::OpBuilder& builder, std::vector<mlir::Attribute> toMap, std::vector<mlir::Attribute> mapTo, TranslationContext& context, mlir::Value tree) {
      if (toMap.empty()) return tree;
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      auto tupleScope = context.createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      context.setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      std::vector<mlir::Value> createdValues;
      std::vector<mlir::Attribute> createdCols;
      for (size_t i = 0; i < toMap.size(); i++) {
         auto colRef = toMap[i].cast<mlir::tuples::ColumnRefAttr>();
         auto newColDef = mapTo[i].cast<mlir::tuples::ColumnDefAttr>();
         mlir::Value expr = mapBuilder.create<mlir::tuples::GetColumnOp>(builder.getUnknownLoc(), colRef.getColumn().type, colRef, tuple);
         if (colRef.getColumn().type != newColDef.getColumn().type) {
            expr = mapBuilder.create<mlir::db::AsNullableOp>(builder.getUnknownLoc(), newColDef.getColumn().type, expr);
         }
         auto attrDef = attrManager.createDef(&newColDef.getColumn());
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);
      }
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
      mapOp.getPredicate().push_back(block);
      mapBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
      return mapOp.asRelation();
   };

   ReplaceState replaceState;
   for (auto* cell = targetList->head; cell != nullptr; cell = cell->next) {
      auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
      if (node->type == T_ResTarget) {
         auto* resTarget = reinterpret_cast<ResTarget*>(node);
         resTarget->val_ = analyzeTargetExpression(resTarget->val_, replaceState);
      } else {
         error("expected res target");
      }
   }
   having = analyzeTargetExpression(having, replaceState);

   if (sortClause) {
      for (auto* cell = sortClause->head; cell != nullptr; cell = cell->next) {
         auto* temp = reinterpret_cast<Node*>(cell->data.ptr_value);
         switch (temp->type) {
            case T_SortBy: {
               auto* sort = reinterpret_cast<SortBy*>(temp);
               sort->node_ = analyzeTargetExpression(sort->node_, replaceState);
               break;
            }
            default: {
               error("unknown orderby type");
            }
         }
      }
   }
   tree = createMap(builder, replaceState.evalBeforeAggr, context, tree, scope);
   std::vector<mlir::Attribute> groupByAttrs;
   std::unordered_map<std::string, mlir::Attribute> groupedExpressions;
   bool rollup = false;
   std::unordered_map<std::string, size_t> groupByAttrToPos;
   if (groupBy && groupBy->head != 0) {
      auto* attributes = groupBy->head;
      if (reinterpret_cast<Node*>(attributes->data.ptr_value)->type == T_GroupingSet) {
         auto* node = reinterpret_cast<GroupingSet*>(attributes->data.ptr_value);
         assert(node->kind == GROUPING_SET_ROLLUP);
         assert(attributes->next == nullptr);
         attributes = node->content->head;
         rollup = true;
      }
      size_t i = 0;
      for (auto* cell = attributes; cell != nullptr; cell = cell->next) {
         auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
         if (node->type == T_ColumnRef) {
            assert(node->type == T_ColumnRef);
            auto* columnRef = reinterpret_cast<ColumnRef*>(node);
            auto attrName = fieldsToString(columnRef->fields_);
            groupByAttrToPos[attrName] = i;
            const auto* attr = context.getAttribute(attrName);
            groupByAttrs.push_back(attrManager.createRef(attr));
         } else {
            auto [tree2, attr] = mapExpressionToAttribute(tree, context, builder, scope, node);
            std::string printed = fingerprint(node);
            groupedExpressions[printed] = attr;
            groupByAttrs.push_back(attr);
            tree = tree2;
         }
         i++;
      }
   }
   auto asNullable = [](mlir::Type t) { return t.isa<mlir::db::NullableType>() ? t : mlir::db::NullableType::get(t.getContext(), t); };
   if (!groupByAttrs.empty() || !replaceState.aggrs.empty()) {
      if (rollup) {
         struct Part {
            size_t n;
            mlir::Value tree;
            std::vector<mlir::Attribute> computed;
            std::vector<mlir::Attribute> groupByCols;
            std::vector<mlir::Attribute> groupByCols2;
            mlir::Attribute grouping;
         };
         std::vector<Part> parts;
         std::vector<std::string> orderedAggregates;
         auto scopeName = attrManager.getUniqueScope("rollup");
         static size_t rollupId = 0;
         for (size_t i = 0; i <= groupByAttrs.size(); i++) {
            size_t n = groupByAttrs.size() - i;
            std::vector<mlir::Attribute> localGroupByAttrs;
            std::vector<mlir::Attribute> localGroupByAttrsNullable;
            std::vector<mlir::Attribute> notAvailable;
            size_t present = 0;
            for (size_t j = 1; j <= n; j++) {
               localGroupByAttrs.push_back(groupByAttrs.at(j - 1));
               auto attrDef = attrManager.createDef(scopeName, "tmp" + std::to_string(rollupId++));
               auto currentType = groupByAttrs.at(j - 1).cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
               attrDef.getColumn().type = asNullable(currentType);

               localGroupByAttrsNullable.push_back(attrDef);
            }
            for (size_t j = n + 1; j <= groupByAttrs.size(); j++) {
               present |= (1 << (j - 1));
               auto currentAttr = groupByAttrs.at(j - 1);
               auto attrDef = attrManager.createDef(scopeName, "tmp" + std::to_string(rollupId++));
               auto currentType = currentAttr.cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
               attrDef.getColumn().type = asNullable(currentType);
               notAvailable.push_back(attrDef);
            }

            auto [tree2, mapping] = performAggregation(builder, localGroupByAttrs, replaceState, context, tree);

            tree2 = mapToNull(builder, notAvailable, context, tree2);
            tree2 = mapToNullable(builder, localGroupByAttrs, localGroupByAttrsNullable, context, tree2);
            auto [tree3, groupingAttr] = mapInt(builder, present, context, tree2);
            tree2 = tree3;
            if (orderedAggregates.empty()) {
               for (auto [s, c] : mapping)
                  orderedAggregates.push_back(s);
            }
            std::vector<mlir::Attribute> computed;
            for (auto s : orderedAggregates) {
               computed.push_back(attrManager.createDef(mapping[s]));
            }

            parts.push_back({n, tree2, computed, localGroupByAttrsNullable, notAvailable, groupingAttr});
         }

         mlir::Value currTree = parts[0].tree;
         std::vector<mlir::Attribute> currentAttributes(parts[0].groupByCols.begin(), parts[0].groupByCols.end());
         currentAttributes.insert(currentAttributes.end(), parts[0].groupByCols2.begin(), parts[0].groupByCols2.end());
         currentAttributes.insert(currentAttributes.end(), parts[0].computed.begin(), parts[0].computed.end());
         currentAttributes.push_back(parts[0].grouping);
         for (size_t i = 1; i < parts.size(); i++) {
            auto rollupUnionName = attrManager.getUniqueScope("rollupUnion");
            std::vector<mlir::Attribute> currentLocalAttributes(parts[i].groupByCols.begin(), parts[i].groupByCols.end());
            currentLocalAttributes.insert(currentLocalAttributes.end(), parts[i].groupByCols2.begin(), parts[i].groupByCols2.end());
            currentLocalAttributes.insert(currentLocalAttributes.end(), parts[i].computed.begin(), parts[i].computed.end());
            currentLocalAttributes.push_back(parts[i].grouping);
            std::vector<mlir::Attribute> unionAttributes;
            size_t id = 0;
            for (size_t j = 0; j < currentLocalAttributes.size(); j++) {
               auto left = attrManager.createRef(&currentAttributes[j].cast<mlir::tuples::ColumnDefAttr>().getColumn());
               auto right = attrManager.createRef(&currentLocalAttributes[j].cast<mlir::tuples::ColumnDefAttr>().getColumn());

               auto unionAttribute = attrManager.createDef(rollupUnionName, "tmp" + std::to_string(id++), builder.getArrayAttr({left, right}));
               unionAttribute.getColumn().type = currentLocalAttributes[j].cast<mlir::tuples::ColumnDefAttr>().getColumn().type;
               unionAttributes.push_back(unionAttribute);
            }
            currTree = builder.create<mlir::relalg::UnionOp>(builder.getUnknownLoc(), ::mlir::relalg::SetSemanticAttr::get(builder.getContext(), mlir::relalg::SetSemantic::all), currTree, parts[i].tree, builder.getArrayAttr(unionAttributes));
            currentAttributes = unionAttributes;
         }
         for (size_t i = 0; i < groupByAttrs.size(); i++) {
            context.replace(scope, &groupByAttrs[i].cast<mlir::tuples::ColumnRefAttr>().getColumn(), &currentAttributes[i].cast<mlir::tuples::ColumnDefAttr>().getColumn());
         }
         size_t i = 0;
         for (auto s : orderedAggregates) {
            context.mapAttribute(scope, s, &currentAttributes[groupByAttrs.size() + i++].cast<mlir::tuples::ColumnDefAttr>().getColumn());
         }
         tree = currTree;
         for (auto aggr : replaceState.groupingFuncs) {
            auto [fakeNode, name] = aggr;
            auto shiftAmount = groupByAttrToPos.at(name);
            auto [tree2, attr] = mapCheckBit(builder, currentAttributes.back(), shiftAmount, context, tree);
            tree = tree2;
            context.mapAttribute(scope, fakeNode->colId, &attr.cast<mlir::tuples::ColumnDefAttr>().getColumn());
         }
      } else {
         auto [tree2, mapping] = performAggregation(builder, groupByAttrs, replaceState, context, tree);
         tree = tree2;
         for (auto [s, c] : mapping) {
            context.mapAttribute(scope, s, c);
         }
      }
   }

   if (having) {
      mlir::Block* pred = translatePredicate(builder, having, context);
      auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree);
      sel.getPredicate().push_back(pred);
      tree = sel.getResult();
   }
   tree = createMap(builder, replaceState.evalBeforeWindowFunc, context, tree, scope);

   if (!replaceState.windowFunctions.empty()) {
      static size_t windowId = 0;
      for (auto [fakeNode, info] : replaceState.windowFunctions) {
         auto [funcName, node, window] = info;
         auto tupleStreamType = mlir::tuples::TupleStreamType::get(builder.getContext());
         auto tupleType = mlir::tuples::TupleType::get(builder.getContext());

         std::string groupByName = "window" + std::to_string(windowId++);
         auto tupleScope = context.createTupleScope();
         auto* block = new mlir::Block;
         block->addArgument(tupleStreamType, builder.getUnknownLoc());
         block->addArgument(tupleType, builder.getUnknownLoc());
         mlir::Value relation = block->getArgument(0);
         mlir::OpBuilder windowBuilder(builder.getContext());
         windowBuilder.setInsertionPointToStart(block);
         std::vector<mlir::Value> createdValues;
         std::vector<mlir::Attribute> createdCols;
         mlir::Value expr; //todo
         auto attrDef = attrManager.createDef(groupByName, fakeNode->colId);
         if (funcName == "rank") {
            expr = windowBuilder.create<mlir::relalg::RankOp>(builder.getUnknownLoc(), builder.getI64Type(), relation);

         } else if (funcName == "count*") {
            expr = windowBuilder.create<mlir::relalg::CountRowsOp>(builder.getUnknownLoc(), builder.getI64Type(), relation);
            if (groupByAttrs.empty()) {
               context.useZeroInsteadNull.insert(&attrDef.getColumn());
            }
         } else {
            auto aggrFunc = llvm::StringSwitch<mlir::relalg::AggrFunc>(funcName)
                               .Case("sum", mlir::relalg::AggrFunc::sum)
                               .Case("avg", mlir::relalg::AggrFunc::avg)
                               .Case("min", mlir::relalg::AggrFunc::min)
                               .Case("max", mlir::relalg::AggrFunc::max)
                               .Case("count", mlir::relalg::AggrFunc::count)
                               .Default(mlir::relalg::AggrFunc::count);
            if (aggrFunc == mlir::relalg::AggrFunc::count) {
               if (groupByAttrs.empty()) {
                  context.useZeroInsteadNull.insert(&attrDef.getColumn());
               }
            }
            mlir::tuples::ColumnRefAttr refAttr;
            switch (node->type) {
               case T_ColumnRef: refAttr = attrManager.createRef(resolveColRef(node, context)); break;
               case T_FakeNode: refAttr = attrManager.createRef(context.getAttribute(reinterpret_cast<FakeNode*>(node)->colId)); break;
               default: error("could not resolve window attribute");
            }
            mlir::Value currRel = relation;
            mlir::Type aggrResultType;
            if (aggrFunc == mlir::relalg::AggrFunc::count) {
               aggrResultType = builder.getI64Type();
            } else {
               aggrResultType = refAttr.getColumn().type;
               if (aggrFunc == mlir::relalg::AggrFunc::avg) {
                  auto baseType = getBaseType(aggrResultType);
                  if (baseType.isIntOrFloat() && !baseType.isIntOrIndex()) {
                     //keep aggrResultType
                  } else if (baseType.isa<mlir::db::DecimalType>()) {
                     mlir::OpBuilder b(builder.getContext());
                     mlir::Value x = b.create<mlir::db::ConstantOp>(b.getUnknownLoc(), baseType, b.getUnitAttr());
                     mlir::Value x2 = b.create<mlir::db::ConstantOp>(b.getUnknownLoc(), mlir::db::DecimalType::get(b.getContext(), 19, 0), b.getUnitAttr());
                     mlir::Value div = b.create<mlir::db::DivOp>(b.getUnknownLoc(), x, x2);
                     aggrResultType = div.getType();
                     div.getDefiningOp()->erase();
                     x2.getDefiningOp()->erase();
                     x.getDefiningOp()->erase();
                  } else {
                     mlir::OpBuilder b(builder.getContext());
                     mlir::Value x = b.create<mlir::db::ConstantOp>(b.getUnknownLoc(), mlir::db::DecimalType::get(b.getContext(), 19, 0), b.getUnitAttr());
                     mlir::Value div = b.create<mlir::db::DivOp>(b.getUnknownLoc(), x, x);
                     aggrResultType = div.getType();
                     div.getDefiningOp()->erase();
                     x.getDefiningOp()->erase();
                  }
                  if (refAttr.getColumn().type.isa<mlir::db::NullableType>()) {
                     aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
                  }
               }
               if (aggrFunc == mlir::relalg::AggrFunc::stddev_samp || aggrFunc == mlir::relalg::AggrFunc::var_samp) {
                  aggrResultType = builder.getF64Type();
                  if (refAttr.getColumn().type.isa<mlir::db::NullableType>()) {
                     aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
                  }
               }
               if (!aggrResultType.isa<mlir::db::NullableType>() && (groupByAttrs.empty())) {
                  aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
               }
            }
            expr = windowBuilder.create<mlir::relalg::AggrFuncOp>(builder.getUnknownLoc(), aggrResultType, aggrFunc, currRel, refAttr);
         }
         attrDef.getColumn().type = expr.getType();
         context.mapAttribute(scope, fakeNode->colId, &attrDef.getColumn());
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);

         windowBuilder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
         std::vector<mlir::Attribute> partitionByAttrs;
         std::vector<mlir::Attribute> orderBySpecs;

         for (auto* node : window.partitionBy) {
            if (node->type == T_ColumnRef) {
               partitionByAttrs.push_back(attrManager.createRef(resolveColRef(node, context)));
            } else {
               assert(node->type == T_FakeNode);
               partitionByAttrs.push_back(attrManager.createRef(context.getAttribute(reinterpret_cast<FakeNode*>(node)->colId)));
            }
         }
         for (auto [dir, node] : window.orderBy) {
            mlir::tuples::ColumnRefAttr attr;
            if (node->type == T_ColumnRef) {
               attr = attrManager.createRef(resolveColRef(node, context));
            } else {
               assert(node->type == T_FakeNode);
               attr = attrManager.createRef(context.getAttribute(reinterpret_cast<FakeNode*>(node)->colId));
            }
            orderBySpecs.push_back(mlir::relalg::SortSpecificationAttr::get(builder.getContext(), attr, dir == SORTBY_DESC ? mlir::relalg::SortSpec::desc : mlir::relalg::SortSpec::asc));
         }
         auto windowOp = builder.create<mlir::relalg::WindowOp>(builder.getUnknownLoc(), tupleStreamType, tree, builder.getArrayAttr(partitionByAttrs), builder.getArrayAttr(orderBySpecs), builder.getArrayAttr(createdCols), window.start, window.end);
         windowOp.getAggrFunc().push_back(block);

         tree = windowOp.getResult();
      }
   }

   TargetInfo targetInfo;
   std::unordered_map<FakeNode*, Node*> mapForTargetList;
   std::vector<std::pair<std::string, std::variant<const mlir::tuples::Column*, FakeNode*>>> targets;
   for (auto* cell = targetList->head; cell != nullptr; cell = cell->next) {
      auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
      if (node->type == T_ResTarget) {
         auto* resTarget = reinterpret_cast<ResTarget*>(node);
         auto* targetExpr = resTarget->val_;
         std::string name;
         const mlir::tuples::Column* attribute;
         FakeNode* fakeNode = nullptr;
         //todo: handle T_A_STAR
         switch (targetExpr->type) {
            case T_ColumnRef: {
               auto* colRef = reinterpret_cast<ColumnRef*>(targetExpr);
               auto* colRefFirst = reinterpret_cast<Node*>(colRef->fields_->head->data.ptr_value);

               switch (colRefFirst->type) {
                  case T_String: {
                     //todo: handle a.*

                     name = fieldsToString(colRef->fields_);
                     auto p = name.find(".");
                     if (p != std::string::npos) {
                        name = name.substr(p + 1, name.size() - p - 1);
                     }
                     attribute = resolveColRef(targetExpr, context);
                     break;
                  }
                  case T_A_Star: {
                     std::unordered_set<const mlir::tuples::Column*> handledAttrs;
                     for (auto p : context.getAllDefinedColumns()) {
                        if (!handledAttrs.contains(p.second)) {
                           targetInfo.namedResults.push_back({p.first, p.second});
                           handledAttrs.insert(p.second);
                        }
                     }
                     continue;
                  }
                  default: error("unexpected colref type in target list");
               }

               break;
            }
            case T_FakeNode: {
               auto* fakeNode = reinterpret_cast<FakeNode*>(targetExpr);
               name = fakeNode->name;
               attribute = context.getAttribute(fakeNode->colId);
               break;
            }
            default: {
               auto printed = std::string(fingerprint(targetExpr));
               if (groupedExpressions.contains(printed)) {
                  attribute = &groupedExpressions[printed].cast<mlir::tuples::ColumnRefAttr>().getColumn();
                  name = "";
               } else {
                  fakeNode = createFakeNode("", nullptr);
                  mapForTargetList.insert({fakeNode, targetExpr});
                  name = "";
               }
            }
         }
         if (resTarget->name_) {
            name = resTarget->name_;
         }
         if (!fakeNode) {
            assert(attribute);
            targets.push_back({name, attribute});
         } else {
            targets.push_back({name, fakeNode});
         }
      } else {
         error("expected res target");
      }
   }
   tree = createMap(builder, mapForTargetList, context, tree, scope);
   for (auto target : targets) {
      if (std::holds_alternative<const mlir::tuples::Column*>(target.second)) {
         targetInfo.namedResults.push_back({target.first, std::get<const mlir::tuples::Column*>(target.second)});
      } else {
         auto* fakeNode = std::get<FakeNode*>(target.second);
         targetInfo.namedResults.push_back({target.first, context.getAttribute(fakeNode->colId)});
      }
   }
   if (distinctClause) {
      assert(distinctClause->length == 1);
      std::vector<mlir::Attribute> columns;
      for (auto x : targetInfo.namedResults) {
         columns.push_back(attrManager.createRef(x.second));
      }
      tree = builder.create<mlir::relalg::ProjectionOp>(builder.getUnknownLoc(), mlir::relalg::SetSemantic::distinct, tree, builder.getArrayAttr(columns));
   }

   // ORDER BY
   if (sortClause) {
      auto sortScope = context.createResolverScope();
      for (auto x : targetInfo.namedResults) {
         if (!x.first.empty()) {
            context.mapAttribute(sortScope, x.first, x.second);
         }
      }
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
               const mlir::tuples::Column* attr;
               switch (target->type) {
                  case T_ColumnRef: {
                     attr = resolveColRef(target, context);
                     break;
                  }
                  case T_FakeNode: { //
                     attr = context.getAttribute(reinterpret_cast<FakeNode*>(target)->colId);
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
                  default: {
                     auto printed = std::string(fingerprint(target));
                     if (groupedExpressions.contains(printed)) {
                        attr = &groupedExpressions[printed].cast<mlir::tuples::ColumnRefAttr>().getColumn();
                     } else {
                        auto [tree2, attr2] = mapExpressionToAttribute(tree, context, builder, sortScope, target);
                        attr = &attr2.getColumn();
                        tree = tree2;
                     }
                  }
               }

               mapping.push_back(mlir::relalg::SortSpecificationAttr::get(builder.getContext(), attrManager.createRef(attr), spec));
               break;
            }
            default: {
               error("unknown orderby type");
            }
         }
      }
      tree = builder.create<mlir::relalg::SortOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(mapping));
   }
   return std::make_pair(tree, targetInfo);
}
mlir::Type frontend::sql::Parser::createBaseTypeFromColumnType(mlir::MLIRContext* context, const runtime::ColumnType& colType) {
   auto asInt = [](std::variant<size_t, std::string> intOrStr) -> size_t {
      if (std::holds_alternative<size_t>(intOrStr)) {
         return std::get<size_t>(intOrStr);
      } else {
         return std::stoll(std::get<std::string>(intOrStr));
      }
   };
   if (colType.base == "bool") return mlir::IntegerType::get(context, 1);
   if (colType.base == "int") return mlir::IntegerType::get(context, asInt(colType.modifiers.at(0)));
   if (colType.base == "index") return mlir::IndexType::get(context);
   if (colType.base == "float") return asInt(colType.modifiers.at(0)) == 32 ? mlir::FloatType::getF32(context) : mlir::FloatType::getF64(context);
   if (colType.base == "date") return mlir::db::DateType::get(context, mlir::db::symbolizeDateUnitAttr(std::get<std::string>(colType.modifiers.at(0))).value());
   if (colType.base == "string") return mlir::db::StringType::get(context);
   if (colType.base == "char") return mlir::db::CharType::get(context, asInt(colType.modifiers.at(0)));
   if (colType.base == "decimal") return mlir::db::DecimalType::get(context, asInt(colType.modifiers.at(0)), asInt(colType.modifiers.at(1)));
   if (colType.base == "interval") return mlir::db::IntervalType::get(context, std::get<std::string>(colType.modifiers.at(0)) == "daytime" ? mlir::db::IntervalUnitAttr::daytime : mlir::db::IntervalUnitAttr::months);
   if (colType.base == "timestamp") return mlir::db::TimestampType::get(context, mlir::db::TimeUnitAttr::second);
   assert(false);
   return mlir::Type();
}
mlir::Type frontend::sql::Parser::createTypeFromColumnType(mlir::MLIRContext* context, const runtime::ColumnType& colType) {
   mlir::Type baseType = createBaseTypeFromColumnType(context, colType);
   return colType.nullable ? mlir::db::NullableType::get(context, baseType) : baseType;
}
std::vector<std::string> frontend::sql::Parser::listToStringVec(List* l) {
   std::vector<std::string> res;
   if (l != nullptr) {
      for (auto* cell = l->head; cell != nullptr; cell = cell->next) {
         auto* const target = reinterpret_cast<Value*>(cell->data.ptr_value);
         auto* const column = target->val_.str_;
         res.push_back(column);
      }
   }
   return res;
}
