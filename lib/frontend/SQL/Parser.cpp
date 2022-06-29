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
         nullOp.getResult().setType(t);
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
mlir::Value frontend::sql::Parser::translateFuncCallExpression(Node* node, mlir::OpBuilder& builder, mlir::Location loc, TranslationContext& context) {
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
mlir::Value frontend::sql::Parser::translateBinaryExpression(mlir::OpBuilder& builder, frontend::sql::ExpressionType opType, mlir::Value left, mlir::Value right) {
   auto loc = builder.getUnknownLoc();
   switch (opType) {
      case ExpressionType::OPERATOR_PLUS:
         if (left.getType().isa<mlir::db::DateType>() && right.getType().isa<mlir::db::IntervalType>()) {
            return builder.create<mlir::db::RuntimeCall>(loc, left.getType(), "DateAdd", mlir::ValueRange({left, right})).res();
         }
         return builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypes(builder, {left, right}));
      case ExpressionType::OPERATOR_MINUS:
         if (left.getType().isa<mlir::db::DateType>() && right.getType().isa<mlir::db::IntervalType>()) {
            return builder.create<mlir::db::RuntimeCall>(loc, left.getType(), "DateSubtract", mlir::ValueRange({left, right})).res();
         }
         return builder.create<mlir::db::SubOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypes(builder, {left, right}));
      case ExpressionType::OPERATOR_MULTIPLY:
         return builder.create<mlir::db::MulOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypesExceptDecimals(builder, {left, right}));
      case ExpressionType::OPERATOR_DIVIDE:
         return builder.create<mlir::db::DivOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypes(builder, {left, right}));
      case ExpressionType::OPERATOR_MOD:
         return builder.create<mlir::db::ModOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypes(builder, {left, right}));
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
         auto like = builder.create<mlir::db::RuntimeCall>(loc, resType, "Like", mlir::ValueRange({ct[0], ct[1]})).res();
         return opType == ExpressionType::COMPARE_NOT_LIKE ? builder.create<mlir::db::NotOp>(loc, like) : like;
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
         auto [tree, targetInfo] = ctes.at(relation);
         for (auto x : targetInfo.namedResults) {
            context.mapAttribute(scope, x.first, x.second);
            context.mapAttribute(scope, alias + "." + x.first, x.second);
         }
         return tree;
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
   return builder.create<mlir::relalg::BaseTableOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), relation, mlir::relalg::TableMetaDataAttr::get(builder.getContext(), std::make_shared<runtime::TableMetaData>()), builder.getDictionaryAttr(columns));
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
      auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree);
      sel.predicate().push_back(pred);
      tree = sel.result();
   }
   auto [tree_, targetInfo] = translateSelectionTargetList(builder, stmt->group_clause_, stmt->having_clause_, stmt->target_list_, tree, context, scope);
   tree = tree_;
   for (auto x : targetInfo.namedResults) {
      if (!x.first.empty()) {
         context.mapAttribute(scope, x.first, x.second);
      }
   }
   // ORDER BY
   if (stmt->sort_clause_) {
      tree = builder.create<mlir::relalg::SortOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree, translateSortSpec(stmt->sort_clause_, builder, context, targetInfo));
   }
   // LIMIT
   if (stmt->limit_count_) {
      size_t limit = reinterpret_cast<A_Const*>(stmt->limit_count_)->val_.val_.ival_;
      tree = builder.create<mlir::relalg::LimitOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), limit, tree);
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
   for (size_t i = 0; i < lTargetInfo.namedResults.size(); i++) {
      auto newName = lTargetInfo.namedResults[i].first;
      const auto* leftColumn = lTargetInfo.namedResults[i].second;
      const auto* rightColumn = rTargetInfo.namedResults[i].second;
      auto leftType = leftColumn->type;
      auto rightType = rightColumn->type;
      if (getBaseType(leftType) != getBaseType(rightColumn->type)) {
         error("SET operation expacts same types (column:" + std::to_string(i) + ")");
      }
      auto newType = SQLTypeInference::getCommonType(leftType, rightType);
      auto newColName = attrManager.getName(leftColumn).second;
      auto newColDef = attrManager.createDef(scopeName, newColName, builder.getArrayAttr({attrManager.createRef(leftColumn), attrManager.createRef(rightColumn)}));
      auto* newCol = &newColDef.getColumn();
      newCol->type = newType;
      attributes.push_back(newColDef);
      targetInfo.map(newName, newCol);
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
   block->addArgument(mlir::relalg::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto tupleScope = context.createTupleScope();
   context.setCurrentTuple(block->getArgument(0));

   predBuilder.setInsertionPointToStart(block);
   mlir::Value expr = translateExpression(predBuilder, node, context);
   predBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), expr);
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
         last = builder.create<mlir::relalg::CrossProductOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), last, translated);
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
         std::vector<std::pair<std::string, const mlir::relalg::Column*>> mapping;
         if (joinExpr->jointype_ == JOIN_LEFT) {
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
         //todo: handle outerjoin

         if (joinExpr->jointype_ == JOIN_LEFT || joinExpr->jointype_ == JOIN_RIGHT) {
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
               std::unordered_map<const mlir::relalg::Column*, const mlir::relalg::Column*> remapped;
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
            auto join = builder.create<mlir::relalg::OuterJoinOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), left, right, mapping);
            join.predicate().push_back(pred);
            return join;
         } else if (joinExpr->jointype_ == JOIN_INNER) {
            mlir::Block* pred = translatePredicate(builder, joinExpr->quals_, context);

            auto join = builder.create<mlir::relalg::InnerJoinOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), left, right);
            join.predicate().push_back(pred);
            return join;
         }
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
               if (stringVal.size() <= 8) {
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
                     if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::daytime) {
                        unit = "days";
                     }
                     constOp->setAttr("value", builder.getStringAttr(constOp.getValue().cast<mlir::StringAttr>().str() + unit));
                  }
                  constOp.getResult().setType(resType);
                  return constOp;
               } else {
                  return builder.create<mlir::db::CastOp>(loc, resType, toCast);
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
         return builder.create<mlir::relalg::GetColumnOp>(builder.getUnknownLoc(), attr->type, attrManager.createRef(attr), context.getCurrentTuple());
         break;
      }
      case T_FakeNode: { //
         const auto* attr = context.getAttribute(reinterpret_cast<FakeNode*>(node)->colId);
         return builder.create<mlir::relalg::GetColumnOp>(builder.getUnknownLoc(), attr->type, attrManager.createRef(attr), context.getCurrentTuple());
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
               mlir::relalg::ColumnRefAttr attribute = attrManager.createRef(targetInfo.namedResults[0].second);
               auto operatorName = subLink->oper_name_ ? listToStringVec(subLink->oper_name_).at(0) : "=";
               auto operatorType = stringToExpressionType(operatorName);
               auto* block = new mlir::Block;
               mlir::OpBuilder predBuilder(builder.getContext());
               block->addArgument(mlir::relalg::TupleType::get(builder.getContext()), builder.getUnknownLoc());
               auto tupleScope = context.createTupleScope();
               context.setCurrentTuple(block->getArgument(0));

               predBuilder.setInsertionPointToStart(block);
               mlir::Value expr = translateExpression(predBuilder, subLink->testexpr_, context);
               mlir::Value colVal = predBuilder.create<mlir::relalg::GetColumnOp>(loc, attribute.getColumn().type, attribute, block->getArgument(0));
               mlir::Value pred = translateBinaryExpression(predBuilder, operatorType, expr, colVal);
               predBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), pred);

               auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), subQueryTree);
               sel.predicate().push_back(block);
               subQueryTree = sel.result();

               return builder.create<mlir::relalg::ExistsOp>(loc, builder.getI1Type(), subQueryTree);
            }
            case ALL_SUBLINK: {
               assert(targetInfo.namedResults.size() == 1);
               mlir::relalg::ColumnRefAttr attribute = attrManager.createRef(targetInfo.namedResults[0].second);
               auto operatorName = listToStringVec(subLink->oper_name_).at(0);
               auto operatorType = stringToExpressionType(operatorName);
               auto* block = new mlir::Block;
               mlir::OpBuilder predBuilder(builder.getContext());
               block->addArgument(mlir::relalg::TupleType::get(builder.getContext()), builder.getUnknownLoc());
               auto tupleScope = context.createTupleScope();
               context.setCurrentTuple(block->getArgument(0));

               predBuilder.setInsertionPointToStart(block);
               mlir::Value expr = translateExpression(predBuilder, subLink->testexpr_, context);
               mlir::Value colVal = predBuilder.create<mlir::relalg::GetColumnOp>(loc, attribute.getColumn().type, attribute, block->getArgument(0));
               mlir::Value pred = translateBinaryExpression(predBuilder, operatorType, expr, colVal);
               pred = predBuilder.create<mlir::db::NotOp>(loc, pred);
               predBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), pred);

               auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), subQueryTree);
               sel.predicate().push_back(block);
               subQueryTree = sel.result();

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
   mlir::Value isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), value);
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
            for (auto x : targetInfo.namedResults) {
               names.push_back(builder.getStringAttr(x.first));
               attrs.push_back(attrManager.createRef(x.second));
            }
            return builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), mlir::dsa::TableType::get(builder.getContext()), tree, builder.getArrayAttr(attrs), builder.getArrayAttr(names));
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
   if (datatypeName == "char" && std::get<size_t>(typeModifiers[0]) > 8) {
      typeModifiers.clear();
      datatypeName = "string";
   }
   if (datatypeName == "date") {
      typeModifiers.clear();
      typeModifiers.push_back("day");
   }
   if (datatypeName == "interval") {
      if (typeModifiers.size() > 0 && std::holds_alternative<size_t>(typeModifiers[0])) {
         std::string unit = (std::get<size_t>(typeModifiers[0]) & 8) ? "daytime" : "months";
         typeModifiers.clear();
         typeModifiers.push_back(unit);
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
   assert(insertColNames.size() == tableColumnTypes.size());
   assert(insertColNames.size() == targetInfo.namedResults.size());
   std::vector<mlir::Attribute> attrs;

   std::vector<mlir::Value> createdValues;

   mlir::Block* block = new mlir::Block;
   mlir::OpBuilder mapBuilder(builder.getContext());
   block->addArgument(mlir::relalg::TupleType::get(builder.getContext()), builder.getUnknownLoc());
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
      if (currentType != tableType) {
         mlir::Value expr = mapBuilder.create<mlir::relalg::GetColumnOp>(mapBuilder.getUnknownLoc(), attrRef.getColumn().type, attrRef, tuple);
         auto attrDef = attrManager.createDef(mapName, std::string("inserted") + std::to_string(i));
         attrDef.getColumn().type = tableType;

         createdCols.push_back(attrDef);
         mlir::Value casted = SQLTypeInference::castValueToType(mapBuilder, expr, tableType);

         createdValues.push_back(casted);
         insertedCols[insertColNames[i]] = attrManager.createRef(&attrDef.getColumn());
      } else {
         insertedCols[insertColNames[i]] = attrRef;
      }
   }
   auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
   mapOp.predicate().push_back(block);
   mapBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), createdValues);

   std::vector<mlir::Attribute> orderedColNamesAttrs;
   std::vector<mlir::Attribute> orderedColAttrs;
   for (auto x : tableMetaData->getOrderedColumns()) {
      orderedColNamesAttrs.push_back(builder.getStringAttr(x));
      orderedColAttrs.push_back(insertedCols.at(x));
   }
   mlir::Value newRows = builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), mlir::dsa::TableType::get(builder.getContext()), mapOp.result(), builder.getArrayAttr(orderedColAttrs), builder.getArrayAttr(orderedColNamesAttrs));
   auto databaseValue = getCurrentDatabaseValue(builder);
   auto tableNameValue = createStringValue(builder, tableName);
   rt::Database::appendTable(builder, builder.getUnknownLoc())(mlir::ValueRange{databaseValue, tableNameValue, newRows});
}
Node* frontend::sql::Parser::analyzeTargetExpression(Node* node, frontend::sql::ReplaceState& replaceState) {
   if (!node) return node;
   switch (node->type) {
      case T_FakeNode: {
         auto* fakeNode = reinterpret_cast<FakeNode*>(node);
         return analyzeTargetExpression(fakeNode->original, replaceState);
      }
      case T_FuncCall: {
         auto* funcNode = reinterpret_cast<FuncCall*>(node);

         std::string funcName = reinterpret_cast<value*>(funcNode->funcname_->head->data.ptr_value)->val_.str_;
         if (funcName == "sum" || funcName == "avg" || funcName == "min" || funcName == "max" || funcName == "count") {
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
      case T_TypeCast: {
         auto* castNode = reinterpret_cast<TypeCast*>(node);
         castNode->arg_ = analyzeTargetExpression(node, replaceState);
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
std::pair<mlir::Value, frontend::sql::Parser::TargetInfo> frontend::sql::Parser::translateSelectionTargetList(mlir::OpBuilder& builder, List* groupBy, Node* having, List* targetList, mlir::Value tree, frontend::sql::Parser::TranslationContext& context, frontend::sql::Parser::TranslationContext::ResolverScope& scope) {
   auto createMap = [this](mlir::OpBuilder& builder, std::unordered_map<FakeNode*, Node*>& toMap, TranslationContext& context, mlir::Value tree, TranslationContext::ResolverScope& scope) -> mlir::Value {
      if (toMap.empty()) return tree;
      auto* block = new mlir::Block;
      static size_t mapId = 0;
      std::string mapName = "map" + std::to_string(mapId++);

      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(mlir::relalg::TupleType::get(builder.getContext()), builder.getUnknownLoc());
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
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(createdCols));
      mapOp.predicate().push_back(block);
      mapBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), createdValues);
      return mapOp.result();
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
   tree = createMap(builder, replaceState.evalBeforeAggr, context, tree, scope);
   std::vector<mlir::Attribute> groupByAttrs;
   if (groupBy) {
      for (auto* cell = groupBy->head; cell != nullptr; cell = cell->next) {
         auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
         if (node->type == T_ColumnRef) {
            groupByAttrs.push_back(attrManager.createRef(resolveColRef(node, context)));
         } else {
            error("expected column ref");
         }
      }
   }
   static size_t groupById = 0;
   if (!groupByAttrs.empty() || !replaceState.aggrs.empty()) {
      auto tupleStreamType = mlir::relalg::TupleStreamType::get(builder.getContext());
      auto tupleType = mlir::relalg::TupleType::get(builder.getContext());

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
      for (auto toAggr : replaceState.aggrs) {
         mlir::Value expr; //todo
         auto aggrFuncName = std::get<0>(toAggr.second);
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
                               .Default(mlir::relalg::AggrFunc::count);
            if (aggrFunc == mlir::relalg::AggrFunc::count) {
               if (groupByAttrs.empty()) {
                  context.useZeroInsteadNull.insert(&attrDef.getColumn());
               }
            }
            mlir::relalg::ColumnRefAttr refAttr;
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
               if (!aggrResultType.isa<mlir::db::NullableType>() && groupByAttrs.empty()) {
                  aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
               }
            }
            expr = aggrBuilder.create<mlir::relalg::AggrFuncOp>(builder.getUnknownLoc(), aggrResultType, aggrFunc, currRel, refAttr);
         }
         attrDef.getColumn().type = expr.getType();
         context.mapAttribute(scope, toAggr.first->colId, &attrDef.getColumn());
         createdCols.push_back(attrDef);
         createdValues.push_back(expr);
      }
      aggrBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), createdValues);
      auto groupByOp = builder.create<mlir::relalg::AggregationOp>(builder.getUnknownLoc(), tupleStreamType, tree, builder.getArrayAttr(groupByAttrs), builder.getArrayAttr(createdCols));
      groupByOp.aggr_func().push_back(block);

      tree = groupByOp.result();
   }

   if (having) {
      mlir::Block* pred = translatePredicate(builder, having, context);
      auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree);
      sel.predicate().push_back(pred);
      tree = sel.result();
   }

   TargetInfo targetInfo;
   std::unordered_map<FakeNode*, Node*> mapForTargetList;
   std::vector<std::pair<std::string, std::variant<const mlir::relalg::Column*, FakeNode*>>> targets;
   for (auto* cell = targetList->head; cell != nullptr; cell = cell->next) {
      auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
      if (node->type == T_ResTarget) {
         auto* resTarget = reinterpret_cast<ResTarget*>(node);
         auto* targetExpr = resTarget->val_;
         std::string name;
         const mlir::relalg::Column* attribute;
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
                     attribute = resolveColRef(targetExpr, context);
                     break;
                  }
                  case T_A_Star: {
                     std::unordered_set<const mlir::relalg::Column*> handledAttrs;
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
               fakeNode = createFakeNode("", nullptr);
               mapForTargetList.insert({fakeNode, targetExpr});
               name = "";
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
      if (std::holds_alternative<const mlir::relalg::Column*>(target.second)) {
         targetInfo.namedResults.push_back({target.first, std::get<const mlir::relalg::Column*>(target.second)});
      } else {
         auto* fakeNode = std::get<FakeNode*>(target.second);
         targetInfo.namedResults.push_back({target.first, context.getAttribute(fakeNode->colId)});
      }
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
   if (colType.base == "float") return asInt(colType.modifiers.at(0)) == 32 ? mlir::FloatType::getF32(context) : mlir::FloatType::getF64(context);
   if (colType.base == "date") return mlir::db::DateType::get(context, mlir::db::symbolizeDateUnitAttr(std::get<std::string>(colType.modifiers.at(0))).getValue());
   if (colType.base == "string") return mlir::db::StringType::get(context);
   if (colType.base == "char") return mlir::db::CharType::get(context, asInt(colType.modifiers.at(0)));
   if (colType.base == "decimal") return mlir::db::DecimalType::get(context, asInt(colType.modifiers.at(0)), asInt(colType.modifiers.at(1)));
   if (colType.base == "interval") return mlir::db::IntervalType::get(context, std::get<std::string>(colType.modifiers.at(0)) == "daytime" ? mlir::db::IntervalUnitAttr::daytime : mlir::db::IntervalUnitAttr::months);
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
