#ifndef FRONTEND_SQL_PARSER_H
#define FRONTEND_SQL_PARSER_H
#include "libpg_query/pg_list.h"
#include "libpg_query/pg_query.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallString.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"

#include "parsenodes.h"
#include "runtime/Database.h"

#include "runtime-defs/Database.h"
#include "runtime-defs/ExecutionContext.h"

#include <llvm/ADT/StringSwitch.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <stack>
#include <unordered_set>
namespace frontend::sql {
enum class ExpressionType {
   INVALID

   ,
   OPERATOR_UNARY_MINUS,
   OPERATOR_PLUS,
   OPERATOR_MINUS,
   OPERATOR_MULTIPLY,
   OPERATOR_DIVIDE,
   OPERATOR_CONCAT,
   OPERATOR_MOD,
   OPERATOR_CAST,
   OPERATOR_NOT,
   OPERATOR_IS_NULL,
   OPERATOR_IS_NOT_NULL,
   OPERATOR_EXISTS

   ,
   COMPARE_EQUAL,
   COMPARE_NOT_EQUAL,
   COMPARE_LESS_THAN,
   COMPARE_GREATER_THAN,
   COMPARE_LESS_THAN_OR_EQUAL_TO,
   COMPARE_GREATER_THAN_OR_EQUAL_TO,
   COMPARE_LIKE,
   COMPARE_NOT_LIKE,
   COMPARE_IN,
   COMPARE_IS_DISTINCT_FROM

   ,
   CONJUNCTION_AND,
   CONJUNCTION_OR

   ,
   COLUMN_VALUE

   ,
   VALUE_CONSTANT,
   VALUE_PARAMETER,
   VALUE_TUPLE,
   VALUE_TUPLE_ADDRESS,
   VALUE_NULL,
   VALUE_VECTOR,
   VALUE_SCALAR,
   VALUE_DEFAULT

   ,
   AGGREGATE_COUNT,
   AGGREGATE_SUM,
   AGGREGATE_MIN,
   AGGREGATE_MAX,
   AGGREGATE_AVG,
   AGGREGATE_TOP_K,
   AGGREGATE_HISTOGRAM

   ,
   FUNCTION

   ,
   HASH_RANGE

   ,
   OPERATOR_CASE_EXPR,
   OPERATOR_NULL_IF,
   OPERATOR_COALESCE

   ,
   ROW_SUBQUERY

   ,
   STAR,
   TABLE_STAR,
   PLACEHOLDER,
   COLUMN_REF,
   FUNCTION_REF,
   TABLE_REF
};
struct SQLTypeInference {
   static mlir::FloatType getHigherFloatType(mlir::Type left, mlir::Type right);
   static mlir::IntegerType getHigherIntType(mlir::Type left, mlir::Type right);
   static mlir::db::DecimalType getHigherDecimalType(mlir::Type left, mlir::Type right);
   static mlir::Type getCommonBaseType(mlir::Type left, mlir::Type right) {
      left = getBaseType(left);
      right = getBaseType(right);
      bool stringPresent = left.isa<mlir::db::StringType>() || right.isa<mlir::db::StringType>();
      bool intPresent = left.isa<mlir::IntegerType>() || right.isa<mlir::IntegerType>();
      bool floatPresent = left.isa<mlir::FloatType>() || right.isa<mlir::FloatType>();
      bool decimalPresent = left.isa<mlir::db::DecimalType>() || right.isa<mlir::db::DecimalType>();
      if (stringPresent) return mlir::db::StringType::get(left.getContext());
      if (decimalPresent) return getHigherDecimalType(left, right);
      if (floatPresent) return getHigherFloatType(left, right);
      if (intPresent) return getHigherIntType(left, right);
      return left;
   }
   static mlir::Type getCommonType(mlir::Type left, mlir::Type right) {
      bool isNullable = left.isa<mlir::db::NullableType>() || right.isa<mlir::db::NullableType>();
      auto commonBaseType = getCommonBaseType(left, right);
      if (isNullable) {
         return mlir::db::NullableType::get(left.getContext(), commonBaseType);
      } else {
         return commonBaseType;
      }
   }
   static mlir::Type getCommonBaseType(mlir::TypeRange types) {
      mlir::Type commonType = types.front();
      for (auto t : types) {
         commonType = getCommonBaseType(commonType, t);
      }
      return commonType;
   }
   static mlir::Type getCommonType(mlir::TypeRange types) {
      mlir::Type commonType = types.front();
      for (auto t : types) {
         commonType = getCommonType(commonType, t);
      }
      return commonType;
   }
   static mlir::Value toType(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t) {
      bool isNullable = v.getType().isa<mlir::db::NullableType>();
      if (isNullable && !t.isa<mlir::db::NullableType>()) {
         t = mlir::db::NullableType::get(builder.getContext(), t);
      }
      if (v.getType() == t) { return v; }
      if (auto* defOp = v.getDefiningOp()) {
         if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
            assert(!t.isa<mlir::db::NullableType>());
            constOp.getResult().setType(t);
            return constOp;
         }
         if (auto nullOp = mlir::dyn_cast_or_null<mlir::db::NullOp>(defOp)) {
            nullOp.getResult().setType(t);
            return nullOp;
         }
      }
      return builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), t, v);
   }
   static std::vector<mlir::Value> toCommonBaseTypes(mlir::OpBuilder& builder, mlir::ValueRange values) {
      auto commonType = getCommonBaseType(values.getTypes());
      std::vector<mlir::Value> res;
      for (auto val : values) {
         res.push_back(toType(builder, val, commonType));
      }
      return res;
   }
   static std::vector<mlir::Value> toCommonBaseTypes2(mlir::OpBuilder& builder, mlir::ValueRange values) {
      std::vector<mlir::Value> res;
      for (auto val : values) {
         if (!getBaseType(val.getType()).isa<mlir::db::DecimalType>()) {
            return toCommonBaseTypes(builder, values);
         }
         res.push_back(val);
      }
      return res;
   }
};
#define T_FakeNode T_TidScan
struct FakeNode : Node {
   std::string colId;
   std::string name;
   Node* original;
   FakeNode(std::string colId, std::string name, Node* original) : colId(colId), name(name), original(original) {
      type = T_FakeNode;
   }
};
ExpressionType stringToExpressionType(const std::string& parserStr);
struct Parser {
   mlir::relalg::ColumnManager& attrManager;
   struct StringInfo {
      static bool isEqual(std::string a, std::string b) { return a == b; }
      static std::string getEmptyKey() { return ""; }
      static std::string getTombstoneKey() { return "-"; }
      static size_t getHashValue(std::string str) { return std::hash<std::string>{}(str); }
   };
   struct TranslationContext {
      std::stack<mlir::Value> currTuple;
      std::vector<std::pair<std::string, const mlir::relalg::Column*>> allAttributes;
      llvm::ScopedHashTable<std::string, const mlir::relalg::Column*, StringInfo> resolver;
      using ResolverScope = llvm::ScopedHashTable<std::string, const mlir::relalg::Column*, StringInfo>::ScopeTy;
      struct TupleScope {
         TranslationContext* context;
         bool active;
         TupleScope(TranslationContext* context) : context(context) {
            context->currTuple.push(context->currTuple.top());
         }

         ~TupleScope() {
            context->currTuple.pop();
         }
      };

      TranslationContext() : currTuple(), resolver() {
         currTuple.push(mlir::Value());
      }
      mlir::Value getCurrentTuple() {
         return currTuple.top();
      }
      void setCurrentTuple(mlir::Value v) {
         currTuple.top() = v;
      }
      void mapAttribute(ResolverScope& scope, std::string name, const mlir::relalg::Column* attr) {
         allAttributes.push_back({name, attr});
         resolver.insertIntoScope(&scope, name, attr);
      }
      const mlir::relalg::Column* getAttribute(std::string name) {
         assert(resolver.lookup(name));
         return resolver.lookup(name);
      }
      TupleScope createTupleScope() {
         return TupleScope(this);
      }
      ResolverScope createResolverScope() {
         return ResolverScope(resolver);
      }
   };
   std::string sql;
   PgQueryInternalParsetreeAndError result;
   runtime::Database& database;
   std::vector<std::unique_ptr<FakeNode>> fakeNodes;
   struct TargetInfo {
      std::vector<std::pair<std::string, const mlir::relalg::Column*>> namedResults;
      void map(std::string name, const mlir::relalg::Column* attr) {
         namedResults.push_back({name, attr});
      }
   };
   std::unordered_map<std::string, std::pair<mlir::Value, TargetInfo>> ctes;
   FakeNode* createFakeNode(std::string name, Node* original) {
      static size_t fakeNodeId = 0;
      std::string colId = "tmp_attr" + std::to_string(fakeNodeId++);
      auto node = std::make_unique<FakeNode>(colId, name, original);
      auto* ptr = node.get();
      fakeNodes.emplace_back(std::move(node));
      return ptr;
   }
   mlir::ModuleOp moduleOp;
   Parser(std::string sql, runtime::Database& database, mlir::ModuleOp moduleOp) : attrManager(moduleOp->getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager()), sql(sql), database(database), moduleOp(moduleOp) {
      moduleOp.getContext()->getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper().setParentModule(moduleOp);
      pg_query_parse_init();
      result = pg_query_parse(sql.c_str());
   }
   void error(std::string str) {
      std::cerr << str << std::endl;
      abort();
   }
   std::pair<std::string, std::shared_ptr<runtime::ColumnMetaData>> translateColumnDef(ColumnDef* columnDef) {
      auto* typeName = columnDef->type_name_;

      std::vector<std::variant<size_t, std::string>> typeModifiers;
      if (typeName->typmods_ != nullptr) {
         for (auto* cell = typeName->typmods_->head; cell != nullptr; cell = cell->next) {
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

      std::string datatypeName = reinterpret_cast<value*>(typeName->names_->tail->data.ptr_value)->val_.str_;

      //bool isPrimary = false;
      bool isNotNull = false;
      //bool isUnique = false;

      if (columnDef->constraints_ != nullptr) {
         for (auto* cell = columnDef->constraints_->head; cell != nullptr; cell = cell->next) {
            auto* constraint = reinterpret_cast<Constraint*>(cell->data.ptr_value);
            switch (constraint->contype_) {
               case CONSTR_PRIMARY: {
                  //isPrimary = true;
                  break;
               }
               case CONSTR_NOTNULL: {
                  isNotNull = true;
                  break;
               }
               case CONSTR_UNIQUE: {
                  //isUnique = true;
                  break;
               }
               default: {
                  error("unsupported column constraint");
               }
            }
         }
      }
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
      std::string name = columnDef->colname_;
      runtime::ColumnType columnType;
      columnType.base = datatypeName;
      columnType.nullable = !isNotNull;
      columnType.modifiers = typeModifiers;
      auto columnMetaData = std::make_shared<runtime::ColumnMetaData>();
      columnMetaData->setColumnType(columnType);
      return {name, columnMetaData};
   }
   mlir::Value getExecutionContext(mlir::OpBuilder& builder) {
      mlir::FuncOp funcOp = moduleOp.lookupSymbol<mlir::FuncOp>("rt_get_execution_context");
      if (!funcOp) {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(moduleOp.getBody());
         funcOp = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), "rt_get_execution_context", builder.getFunctionType({}, {mlir::util::RefType::get(builder.getContext(), builder.getI8Type())}), builder.getStringAttr("private"));
      }

      return builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange{}).getResult(0);
   }
   void translateCreate(mlir::OpBuilder& builder, CreateStmt* statement) {
      RangeVar* relation = statement->relation_;
      std::string tableName = relation->relname_ != nullptr ? relation->relname_ : "";
      auto tableMetaData = std::make_shared<runtime::TableMetaData>();
      for (auto* cell = statement->table_elts_->head; cell != nullptr; cell = cell->next) {
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
      auto executionContext = getExecutionContext(builder);
      auto tableNameValue = builder.create<mlir::util::CreateConstVarLen>(builder.getUnknownLoc(), mlir::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(tableName));
      auto descrValue = builder.create<mlir::util::CreateConstVarLen>(builder.getUnknownLoc(), mlir::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(tableMetaData->serialize()));

      auto database = rt::ExecutionContext::getDatabase(builder, builder.getUnknownLoc())(executionContext)[0];
      rt::Database::createTable(builder, builder.getUnknownLoc())(mlir::ValueRange({database, tableNameValue, descrValue}));
   }

   mlir::Value translate(mlir::OpBuilder& builder) {
      if (result.tree && result.tree->length == 1) {
         auto* statement = static_cast<Node*>(result.tree->head->data.ptr_value);
         switch (statement->type) {
            case T_VariableSetStmt: {
               auto* variableSetStatement = reinterpret_cast<VariableSetStmt*>(statement);
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
                  auto executionContext = getExecutionContext(builder);
                  auto database = rt::ExecutionContext::getDatabase(builder, builder.getUnknownLoc())(executionContext)[0];
                  rt::Database::setPersist(builder, builder.getUnknownLoc())({database, persistValue});
               }
               break;
            }
            case T_CreateStmt: {
               translateCreate(builder, reinterpret_cast<CreateStmt*>(statement));
               break;
            }
            case T_CopyStmt: {
               auto* copyStatement = reinterpret_cast<CopyStmt*>(statement);
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
               auto executionContext = getExecutionContext(builder);
               auto database = rt::ExecutionContext::getDatabase(builder, builder.getUnknownLoc())(executionContext)[0];
               auto tableNameValue = builder.create<mlir::util::CreateConstVarLen>(builder.getUnknownLoc(), mlir::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(tableName));
               auto fileNameValue = builder.create<mlir::util::CreateConstVarLen>(builder.getUnknownLoc(), mlir::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(fileName));
               auto delimiterValue = builder.create<mlir::util::CreateConstVarLen>(builder.getUnknownLoc(), mlir::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(delimiter));
               auto escapeValue = builder.create<mlir::util::CreateConstVarLen>(builder.getUnknownLoc(), mlir::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(escape));

               rt::Database::copyFromIntoTable(builder, builder.getUnknownLoc())(mlir::ValueRange{database, tableNameValue, fileNameValue, delimiterValue, escapeValue});
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
            default:
               error("unsupported statement type");
         }
      }
      return mlir::Value();
   }
   std::string fieldsToString(List* fields) {
      auto* node = reinterpret_cast<Node*>(fields->head->data.ptr_value);
      std::string colName;
      std::string tableName;
      if (fields->length == 1) {
         colName = reinterpret_cast<value*>(node)->val_.str_;
         tableName = "";
      } else {
         auto* nextNode = reinterpret_cast<Node*>(fields->head->next->data.ptr_value);
         if (nextNode->type == T_A_Star) {
            //all_columns = true;
            error("unexpected *");
         } else {
            colName = reinterpret_cast<value*>(nextNode)->val_.str_;
         }

         tableName = reinterpret_cast<value*>(node)->val_.str_;
      }
      return tableName.empty() ? colName : tableName + "." + colName;
   }

   auto resolveColRef(Node* node, TranslationContext& context) {
      assert(node->type == T_ColumnRef);
      auto* columnRef = reinterpret_cast<ColumnRef*>(node);
      auto attrName = fieldsToString(columnRef->fields_);
      const auto* attr = context.getAttribute(attrName);
      assert(attr);
      return attr;
   }
   mlir::Type strToType(std::string str, int typeMod, mlir::MLIRContext* context) {
      mlir::Type res = llvm::StringSwitch<mlir::Type>(str)
                          .Case("date", mlir::db::DateType::get(context, mlir::db::DateUnitAttr::day))
                          .Case("interval", mlir::db::IntervalType::get(context, (typeMod & 8) ? mlir::db::IntervalUnitAttr::daytime : mlir::db::IntervalUnitAttr::months))
                          .Default(mlir::Type());
      assert(res);
      return res;
   }
   int typemod(List* typemods) {
      int res = 0;
      if (typemods != nullptr) {
         auto* node = reinterpret_cast<Node*>(typemods->head->data.ptr_value);
         switch (node->type) {
            case T_A_Const: {
               auto nodeType = reinterpret_cast<A_Const*>(node)->val_.type_;
               switch (nodeType) {
                  case T_Integer: {
                     res = static_cast<int32_t>(reinterpret_cast<A_Const*>(node)->val_.val_.ival_);
                     break;
                  }
                  default: {
                     error("unsupported typemod");
                  }
               }
               break;
            }
            default: {
               error("unsupported typemod");
            }
         }
      }
      return res;
   }
   struct ReplaceState {
      std::unordered_map<FakeNode*, Node*> evalBeforeAggr;
      std::unordered_map<FakeNode*, std::tuple<std::string, Node*, bool>> aggrs;
   };

   Node* replaceWithFakeNodes(Node* node, ReplaceState& replaceState) {
      if (!node) return node;
      switch (node->type) {
         case T_FakeNode: {
            auto* fakeNode = reinterpret_cast<FakeNode*>(node);
            return replaceWithFakeNodes(fakeNode->original, replaceState);
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
            castNode->arg_ = replaceWithFakeNodes(node, replaceState);
            return node;
         }
         case T_A_Expr: {
            auto* expr = reinterpret_cast<A_Expr*>(node);
            //expr = AExprTransform(parse_result, ,context);
            if (expr->kind_ == AEXPR_OP) {
               if (node->type == T_TypeCast) {
               } else {
                  expr->lexpr_ = replaceWithFakeNodes(expr->lexpr_, replaceState);
                  expr->rexpr_ = replaceWithFakeNodes(expr->rexpr_, replaceState);
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
               *nodePtr = replaceWithFakeNodes(*nodePtr, replaceState);
            }
            return node;
            break;
         }

         default: return node;
      }
      return node;
   }
   mlir::Value translateCoalesce(mlir::OpBuilder& builder, TranslationContext& context, ListCell* values) {
      auto loc = builder.getUnknownLoc();
      if (!values) {
         return builder.create<mlir::db::NullOp>(loc, mlir::db::NullableType::get(builder.getContext(), builder.getNoneType()));
      }
      mlir::Value value = translateExpression(builder, reinterpret_cast<Node*>(values->data.ptr_value), context);
      mlir::Value isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), value);
      mlir::Value isNotNull = builder.create<mlir::db::NotOp>(loc, isNull);
      auto* whenBlock = new mlir::Block;
      auto* elseBlock = new mlir::Block;
      mlir::OpBuilder whenBuilder(builder.getContext());
      whenBuilder.setInsertionPointToStart(whenBlock);
      mlir::OpBuilder elseBuilder(builder.getContext());
      elseBuilder.setInsertionPointToStart(elseBlock);
      auto elseRes = translateCoalesce(elseBuilder, context, values->next);
      auto commonType = SQLTypeInference::getCommonType(value.getType(), elseRes.getType());
      value = SQLTypeInference::toType(whenBuilder, value, commonType);
      elseRes = SQLTypeInference::toType(elseBuilder, elseRes, commonType);
      whenBuilder.create<mlir::scf::YieldOp>(loc, value);
      elseBuilder.create<mlir::scf::YieldOp>(loc, elseRes);
      auto ifOp = builder.create<mlir::scf::IfOp>(loc, commonType, isNotNull, true);
      ifOp.getThenRegion().getBlocks().clear();
      ifOp.getElseRegion().getBlocks().clear();
      ifOp.getThenRegion().push_back(whenBlock);
      ifOp.getElseRegion().push_back(elseBlock);

      return ifOp.getResult(0);
   }

   /*
    * translate case when SQL expression
    */
   mlir::Value translateWhenCase(mlir::OpBuilder& builder, TranslationContext& context, mlir::Value compareValue, ListCell* whenCell, Node* defaultNode);
   /*
    * translate function call
    */
   mlir::Value translateFuncCall(Node* node, mlir::OpBuilder& builder, mlir::Location loc, TranslationContext& context);
   mlir::Value translateExpression(mlir::OpBuilder& builder, Node* node, TranslationContext& context, bool ignoreNull = false) {
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
            if (expr->kind_ == AEXPR_BETWEEN) {
               mlir::Value val = translateExpression(builder, expr->lexpr_, context);
               auto* list = reinterpret_cast<List*>(expr->rexpr_);
               assert(list->length == 2);
               auto* lowerNode = reinterpret_cast<Node*>(list->head->data.ptr_value);
               auto* upperNode = reinterpret_cast<Node*>(list->tail->data.ptr_value);
               mlir::Value lower = translateExpression(builder, lowerNode, context);
               mlir::Value upper = translateExpression(builder, upperNode, context);
               auto ct = SQLTypeInference::toCommonBaseTypes(builder, {val, lower, upper});
               return builder.create<mlir::db::BetweenOp>(loc, ct[0], ct[1], ct[2], true, true);
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
                     return builder.create<mlir::db::MulOp>(builder.getUnknownLoc(), SQLTypeInference::toCommonBaseTypes2(builder, {left, right}));
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
                  case ExpressionType::OPERATOR_CAST: {
                     auto* castNode = reinterpret_cast<TypeCast*>(node);
                     auto* typeName = reinterpret_cast<value*>(castNode->type_name_->names_->tail->data.ptr_value)->val_.str_;
                     int typeMods = typemod(castNode->type_name_->typmods_);

                     auto toCast = translateExpression(builder, castNode->arg_, context);
                     auto resType = strToType(typeName, typeMods, builder.getContext());
                     if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(toCast.getDefiningOp())) {
                        if (resType.isa<mlir::db::IntervalType>()) {
                           std::string unit = "";
                           if (typeMods & 8) {
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
                  default:
                     error("unsupported expression type");
               }
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
         case T_FuncCall: return translateFuncCall(node, builder, loc, context);
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
            auto subQueryScope = context.createResolverScope();
            auto [subQueryTree, targetInfo] = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(subLink->subselect_), context, subQueryScope);
            switch (subLink->sub_link_type_) {
               case EXPR_SUBLINK: {
                  assert(!targetInfo.namedResults.empty());
                  const auto* attr = targetInfo.namedResults[0].second;
                  //todo: make nullable
                  return builder.create<mlir::relalg::GetScalarOp>(loc, attr->type, attrManager.createRef(attr), subQueryTree);
               }
               case EXISTS_SUBLINK:
                  return builder.create<mlir::relalg::ExistsOp>(loc, builder.getI1Type(), subQueryTree);
               case ANY_SUBLINK: {
                  assert(targetInfo.namedResults.size() == 1);
                  //:mlir::relalg::SetSemanticAttr set_semantic, ::mlir::Value rel, ::mlir::ArrayAttr attrs
                  mlir::Attribute attribute = attrManager.createRef(targetInfo.namedResults[0].second);
                  subQueryTree = builder.create<mlir::relalg::ProjectionOp>(loc, mlir::relalg::SetSemantic::all, subQueryTree, builder.getArrayAttr({attribute}));
                  mlir::Value val = translateExpression(builder, subLink->testexpr_, context);
                  return builder.create<mlir::relalg::InOp>(loc, builder.getI1Type(), val, subQueryTree);
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
            return translateWhenCase(builder, context, arg, startWhen, defaultNode);
         }
         case T_CoalesceExpr: {
            auto* coalesceExpr = reinterpret_cast<AExpr*>(node);
            return translateCoalesce(builder, context, reinterpret_cast<List*>(coalesceExpr->lexpr_)->head);
         }
         default: {
            error("unsupported expression type");
         }
      }
      error("should never happen");
      return mlir::Value();
   }
   size_t asInt(std::variant<size_t, std::string> intOrStr) {
      if (std::holds_alternative<size_t>(intOrStr)) {
         return std::get<size_t>(intOrStr);
      } else {
         return std::stoll(std::get<std::string>(intOrStr));
      }
   }
   mlir::Type translateColBaseType(mlir::MLIRContext* context, const runtime::ColumnType& colType) {
      if (colType.base == "bool") return mlir::IntegerType::get(context, 1);
      if (colType.base == "int") return mlir::IntegerType::get(context, asInt(colType.modifiers.at(0)));
      if (colType.base == "float") return asInt(colType.modifiers.at(0)) == 32 ? mlir::FloatType::getF32(context) : mlir::FloatType::getF64(context);
      if (colType.base == "date") return mlir::db::DateType::get(context, mlir::db::symbolizeDateUnitAttr(std::get<std::string>(colType.modifiers.at(0))).getValue());
      if (colType.base == "string") return mlir::db::StringType::get(context);
      if (colType.base == "char") return mlir::db::CharType::get(context, asInt(colType.modifiers.at(0)));
      if (colType.base == "decimal") return mlir::db::DecimalType::get(context, asInt(colType.modifiers.at(0)), asInt(colType.modifiers.at(1)));
      assert(false);
      return mlir::Type();
   }
   mlir::Type translateColType(mlir::MLIRContext* context, const runtime::ColumnType& colType) {
      mlir::Type baseType = translateColBaseType(context, colType);
      return colType.nullable ? mlir::db::NullableType::get(context, baseType) : baseType;
   }
   mlir::Value translateRangeVar(mlir::OpBuilder& builder, RangeVar* stmt, TranslationContext& context, TranslationContext::ResolverScope& scope) {
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
      static size_t id = 0;
      std::string scopeName = alias + (id != 0 ? std::to_string(id) : "");
      std::vector<mlir::NamedAttribute> columns;
      for (auto c : tableMetaData->getOrderedColumns()) {
         auto attrDef = attrManager.createDef(scopeName, c);
         attrDef.getColumn().type = translateColType(builder.getContext(), tableMetaData->getColumnMetaData(c)->getColumnType());
         columns.push_back(builder.getNamedAttr(c, attrDef));
         context.mapAttribute(scope, c, &attrDef.getColumn()); //todo check for existing and overwrite...
         context.mapAttribute(scope, alias + "." + c, &attrDef.getColumn());
      }
      return builder.create<mlir::relalg::BaseTableOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), relation, mlir::relalg::TableMetaDataAttr::get(builder.getContext(), std::make_shared<runtime::TableMetaData>()), builder.getDictionaryAttr(columns));
   }
   mlir::Value translateSubSelect(mlir::OpBuilder& builder, SelectStmt* stmt, std::string alias, std::vector<std::string> colAlias, TranslationContext& context, TranslationContext::ResolverScope& scope) {
      mlir::Value subQuery;
      TargetInfo targetInfo;
      {
         auto subQueryScope = context.createResolverScope();
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
   mlir::Block* translatePredicate(mlir::OpBuilder& builder, Node* node, TranslationContext& context) {
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
   std::vector<std::string> listToStringVec(List* l) {
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
   mlir::Value translateFromPart(mlir::OpBuilder& builder, Node* node, TranslationContext& context, TranslationContext::ResolverScope& scope) {
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

            mlir::Value left = translateFromPart(builder, joinExpr->larg_, context, scope);
            mlir::Value right;
            std::vector<std::pair<std::string, const mlir::relalg::Column*>> mapping;
            if (joinExpr->jointype_ == JOIN_LEFT) {
               TranslationContext rightContext;
               auto rightResolverScope = rightContext.createResolverScope();
               right = translateFromPart(builder, joinExpr->rarg_, rightContext, rightResolverScope);
               mapping = rightContext.allAttributes;
            } else {
               right = translateFromPart(builder, joinExpr->rarg_, context, scope);
            }

            if (!joinExpr->quals_) {
               error("join must contain predicate");
            }
            //todo: handle outerjoin

            if (joinExpr->jointype_ == JOIN_LEFT) {
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
                        attrDef.getColumn().type = mlir::db::NullableType::get(builder.getContext(), x.second->type);
                        outerJoinMapping.push_back(attrDef);
                        attrDef.dump();
                        remapped.insert({x.second, &attrDef.getColumn()});
                     }
                     context.mapAttribute(scope, x.first, remapped[x.second]);
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
   mlir::Value createMap(mlir::OpBuilder& builder, std::unordered_map<FakeNode*, Node*>& toMap, TranslationContext& context, mlir::Value tree, TranslationContext::ResolverScope& scope) {
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
   }
   mlir::Value translateFrom(mlir::OpBuilder& builder, SelectStmt* stmt, TranslationContext& context, TranslationContext::ResolverScope& scope) {
      List* fromClause = stmt->from_clause_;
      if (!fromClause) { return mlir::Value(); };
      mlir::Value last;
      for (auto* cell = fromClause->head; cell != nullptr; cell = cell->next) {
         auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
         auto translated = translateFromPart(builder, node, context, scope);

         if (last) {
            last = builder.create<mlir::relalg::CrossProductOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), last, translated);
         } else {
            last = translated;
         }
      }
      return last;
   }

   std::pair<mlir::Value, TargetInfo> translateSelGroupHaving(mlir::OpBuilder& builder, List* groupBy, Node* having, List* targetList, mlir::Value tree, TranslationContext& context, TranslationContext::ResolverScope& scope) {
      static size_t id = 0;
      id++;
      ReplaceState replaceState;
      for (auto* cell = targetList->head; cell != nullptr; cell = cell->next) {
         auto* node = reinterpret_cast<Node*>(cell->data.ptr_value);
         if (node->type == T_ResTarget) {
            auto* resTarget = reinterpret_cast<ResTarget*>(node);
            resTarget->val_ = replaceWithFakeNodes(resTarget->val_, replaceState);
         } else {
            error("expected res target");
         }
      }
      having = replaceWithFakeNodes(having, replaceState);
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

            if (aggrFuncName == "count*") {
               expr = aggrBuilder.create<mlir::relalg::CountRowsOp>(builder.getUnknownLoc(), builder.getI64Type(), relation);
            } else {
               auto aggrFunc = llvm::StringSwitch<mlir::relalg::AggrFunc>(aggrFuncName)
                                  .Case("sum", mlir::relalg::AggrFunc::sum)
                                  .Case("avg", mlir::relalg::AggrFunc::avg)
                                  .Case("min", mlir::relalg::AggrFunc::min)
                                  .Case("max", mlir::relalg::AggrFunc::max)
                                  .Case("count", mlir::relalg::AggrFunc::count)
                                  .Default(mlir::relalg::AggrFunc::count);
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
            auto attrDef = attrManager.createDef(groupByName, toAggr.first->colId);
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
      std::vector<std::pair<std::string, FakeNode*>> computedTargets;
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
                        for (auto p : context.allAttributes) {
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
               targetInfo.namedResults.push_back({name, attribute});
            } else {
               computedTargets.push_back({name, fakeNode});
            }
         } else {
            error("expected res target");
         }
      }
      tree = createMap(builder, mapForTargetList, context, tree, scope);
      for (auto computedTarget : computedTargets) {
         auto* fakeNode = computedTarget.second;
         targetInfo.namedResults.push_back({computedTarget.first, context.getAttribute(fakeNode->colId)});
      }
      return std::make_pair(tree, targetInfo);
   }
   /*
    * translate sort specification into array attribute for relalg::SortOp
    */
   mlir::ArrayAttr translateSortSpec(List* sortClause, mlir::OpBuilder& builder, TranslationContext& context, TargetInfo);
   /*
    * translate values -> const relation
    */
   std::pair<mlir::Value, TargetInfo> translateConstRelation(List* valuesLists, mlir::OpBuilder& builder);
   std::pair<mlir::Value, TargetInfo> translateSelectStmt(mlir::OpBuilder& builder, SelectStmt* stmt, TranslationContext& context, TranslationContext::ResolverScope& scope) {
      switch (stmt->op_) {
         case SETOP_NONE: {
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
                     auto [subQuery_, targetInfo_] = translateSelectStmt(builder, reinterpret_cast<SelectStmt*>(cte->ctequery_), context, subQueryScope);
                     subQuery = subQuery_;
                     targetInfo = targetInfo_;
                  }
                  ctes.insert({cte->ctename_, {subQuery, targetInfo}});
               }
            }
            // FROM
            mlir::Value tree = translateFrom(builder, stmt, context, scope);
            // WHERE
            if (stmt->where_clause_) {
               mlir::Block* pred = translatePredicate(builder, stmt->where_clause_, context);
               auto sel = builder.create<mlir::relalg::SelectionOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree);
               sel.predicate().push_back(pred);
               tree = sel.result();
            }
            auto [tree_, targetInfo] = translateSelGroupHaving(builder, stmt->group_clause_, stmt->having_clause_, stmt->target_list_, tree, context, scope);
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
         default: {
            error("could not translate select statement type");
         }
      }
      return std::make_pair(mlir::Value(), TargetInfo());
   }
   ~Parser() {
      pg_query_free_parse_result(result);
   }
};
} // end namespace frontend::sql
#endif // FRONTEND_SQL_PARSER_H
