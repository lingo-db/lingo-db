#include "libpg_query/pg_list.h"
#include "libpg_query/pg_query.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallString.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"

#include "parsenodes.h"

#include <llvm/ADT/StringSwitch.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <stack>
#include <unordered_set>
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
#define T_FakeNode T_TidScan
struct FakeNode : Node {
   std::string colId;
   std::string name;
   Node* original;
   FakeNode(std::string colId, std::string name, Node* original) : colId(colId), name(name), original(original) {
      type = T_FakeNode;
   }
};
ExpressionType stringToExpressionType(const std::string& parserStr) {
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
struct SQLTranslator {
   mlir::relalg::RelationalAttributeManager attrManager;
   struct StringInfo {
      static bool isEqual(std::string a, std::string b) { return a == b; }
      static std::string getEmptyKey() { return ""; }
      static std::string getTombstoneKey() { return "-"; }
      static size_t getHashValue(std::string str) { return std::hash<std::string>{}(str); }
   };
   struct Schema {
      struct Table {
         std::vector<std::pair<std::string, mlir::Type>> columns;
         //std::unordered_map<;
         Table(std::initializer_list<std::pair<std::string, mlir::Type>> list) : columns() {
            for (auto x : list) {
               columns.push_back(x);
            }
         }
      };
      std::unordered_map<std::string, Table> tables;
      Schema(std::initializer_list<std::pair<std::string, Table>> list) : tables() {
         for (auto x : list) {
            tables.insert(x);
         }
      }
   };
   struct TranslationContext {
      std::stack<mlir::Value> currTuple;
      std::vector<std::pair<std::string, const mlir::relalg::RelationalAttribute*>> allAttributes;
      llvm::ScopedHashTable<std::string, const mlir::relalg::RelationalAttribute*, StringInfo> resolver;
      using ResolverScope = llvm::ScopedHashTable<std::string, const mlir::relalg::RelationalAttribute*, StringInfo>::ScopeTy;
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
      void mapAttribute(ResolverScope& scope, std::string name, const mlir::relalg::RelationalAttribute* attr) {
         allAttributes.push_back({name, attr});
         resolver.insertIntoScope(&scope, name, attr);
      }
      const mlir::relalg::RelationalAttribute* getAttribute(std::string name) {
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
   Schema& schema;
   std::vector<std::unique_ptr<FakeNode>> fakeNodes;
   struct TargetInfo {
      std::vector<std::pair<std::string, const mlir::relalg::RelationalAttribute*>> namedResults;
      void map(std::string name, const mlir::relalg::RelationalAttribute* attr) {
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
   SQLTranslator(std::string sql, Schema& schema, mlir::MLIRContext* context) : sql(sql), schema(schema) {
      pg_query_parse_init();
      result = pg_query_parse(sql.c_str());
      attrManager.setContext(context);
   }
   void error(std::string str) {
      std::cerr << str << std::endl;
      abort();
   }
   mlir::Value translate(mlir::OpBuilder& builder) {
      if (result.tree && result.tree->length == 1) {
         auto* statement = static_cast<Node*>(result.tree->head->data.ptr_value);
         if (statement->type == T_SelectStmt) {
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
            return builder.create<mlir::relalg::MaterializeOp>(builder.getUnknownLoc(), mlir::db::TableType::get(builder.getContext()), tree, builder.getArrayAttr(attrs), builder.getArrayAttr(names));
         } else {
            error("only select statement supported");
         }
      } else {
         error("expect query with exactly one statement");
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

   mlir::FloatType getHigherFloatType(mlir::Type left, mlir::Type right) {
      mlir::FloatType leftFloat = left.dyn_cast_or_null<mlir::FloatType>();
      if (auto rightFloat = right.dyn_cast_or_null<mlir::FloatType>()) {
         if (!leftFloat || rightFloat.getWidth() > leftFloat.getWidth()) {
            return rightFloat;
         }
      }
      return leftFloat;
   }
   mlir::IntegerType getHigherIntType(mlir::Type left, mlir::Type right) {
      mlir::IntegerType leftInt = left.dyn_cast_or_null<mlir::IntegerType>();
      if (auto rightInt = right.dyn_cast_or_null<mlir::IntegerType>()) {
         if (!leftInt || rightInt.getWidth() > leftInt.getWidth()) {
            return rightInt;
         }
      }
      return leftInt;
   }
   mlir::db::DecimalType getHigherDecimalType(mlir::Type left, mlir::Type right) {
      unsigned p = 0, s = 0;

      if (auto leftDec = left.dyn_cast_or_null<mlir::db::DecimalType>()) {
         p = std::max(p, leftDec.getP());
         s = std::max(s, leftDec.getS());
      }
      if (auto rightDec = right.dyn_cast_or_null<mlir::db::DecimalType>()) {
         p = std::max(p, rightDec.getP());
         s = std::max(s, rightDec.getS());
      }
      return mlir::db::DecimalType::get(left.getContext(), p, s);
   }
   mlir::Type getCommonType(mlir::Type left, mlir::Type right) {
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
   mlir::Type getCommonType(mlir::TypeRange types) {
      mlir::Type commonType = types.front();
      for (auto t : types) {
         commonType = getCommonType(commonType, t);
      }
      return commonType;
   }
   mlir::Value toType(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t) {
      bool isNullable = v.getType().isa<mlir::db::NullableType>();
      if (isNullable) {
         t = mlir::db::NullableType::get(builder.getContext(), t);
      }
      if (v.getType() == t) { return v; }
      if (auto* defOp = v.getDefiningOp()) {
         if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
            constOp.getResult().setType(t);
            return constOp;
         }
      }
      return builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), t, v);
   }
   std::vector<mlir::Value> toCommonTypes(mlir::OpBuilder& builder, mlir::ValueRange values) {
      auto commonType = getCommonType(values.getTypes());
      std::vector<mlir::Value> res;
      for (auto val : values) {
         res.push_back(toType(builder, val, commonType));
      }
      return res;
   }
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
   std::string getStringFromConst(Node* node) {
      assert(node->type == T_A_Const);
      auto constVal = reinterpret_cast<A_Const*>(node)->val_;
      if (constVal.type_ != T_String) error("expected string");
      return constVal.val_.str_;
   }
   int getIntFromConst(Node* node) {
      assert(node->type == T_A_Const);
      auto constVal = reinterpret_cast<A_Const*>(node)->val_;
      if (constVal.type_ != T_Integer) error("expected integer");
      return constVal.val_.ival_;
   }
   mlir::Value translateWhenCase(mlir::OpBuilder& builder, TranslationContext& context, mlir::Value compareValue, ListCell* whenCell, Node* defaultNode) {
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
      auto commonType = getCommonType(whenRes.getType(), elseRes.getType());
      whenRes = toType(whenBuilder, whenRes, commonType);
      elseRes = toType(elseBuilder, elseRes, commonType);
      whenBuilder.create<mlir::scf::YieldOp>(loc, whenRes);
      elseBuilder.create<mlir::scf::YieldOp>(loc, elseRes);
      auto ifOp = builder.create<mlir::scf::IfOp>(loc, commonType, cond, true);
      ifOp.getThenRegion().getBlocks().clear();
      ifOp.getElseRegion().getBlocks().clear();
      ifOp.getThenRegion().push_back(whenBlock);
      ifOp.getElseRegion().push_back(elseBlock);

      return ifOp.getResult(0);
   }
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
               return builder.create<mlir::db::OneOfOp>(loc, toCommonTypes(builder, values));
            }
            if (expr->kind_ == AEXPR_BETWEEN) {
               mlir::Value val = translateExpression(builder, expr->lexpr_, context);
               auto* list = reinterpret_cast<List*>(expr->rexpr_);
               assert(list->length == 2);
               auto* lowerNode = reinterpret_cast<Node*>(list->head->data.ptr_value);
               auto* upperNode = reinterpret_cast<Node*>(list->tail->data.ptr_value);
               mlir::Value lower = translateExpression(builder, lowerNode, context);
               mlir::Value upper = translateExpression(builder, upperNode, context);
               auto ct = toCommonTypes(builder, {val, lower, upper});
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
                     if (left.getType().isa<mlir::db::DateType>()) {
                        return builder.create<mlir::db::DateAddOp>(builder.getUnknownLoc(), left.getType(), left, right);
                     }
                     return builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), toCommonTypes(builder, {left, right}));
                  case ExpressionType::OPERATOR_MINUS:
                     if (left.getType().isa<mlir::db::DateType>()) {
                        if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(right.getDefiningOp())) {
                           if (auto intAttr = constOp.getValue().dyn_cast_or_null<mlir::IntegerAttr>()) {
                              constOp->setAttr("value", builder.getIntegerAttr(intAttr.getType(), -intAttr.getInt()));
                           } else if (auto strAttr = constOp.getValue().dyn_cast_or_null<mlir::StringAttr>()) {
                              constOp->setAttr("value", builder.getStringAttr("-" + strAttr.str()));
                           }
                           return builder.create<mlir::db::DateAddOp>(builder.getUnknownLoc(), left.getType(), left, right);
                        } else {
                           error("expected constant");
                        }
                     }
                     return builder.create<mlir::db::SubOp>(builder.getUnknownLoc(), toCommonTypes(builder, {left, right}));
                  case ExpressionType::OPERATOR_MULTIPLY:
                     return builder.create<mlir::db::MulOp>(builder.getUnknownLoc(), toCommonTypes(builder, {left, right}));
                  case ExpressionType::OPERATOR_DIVIDE:
                     return builder.create<mlir::db::DivOp>(builder.getUnknownLoc(), toCommonTypes(builder, {left, right}));
                  case ExpressionType::OPERATOR_MOD:
                     return builder.create<mlir::db::ModOp>(builder.getUnknownLoc(), toCommonTypes(builder, {left, right}));
                  case ExpressionType::COMPARE_EQUAL:
                  case ExpressionType::COMPARE_NOT_EQUAL:
                  case ExpressionType::COMPARE_LESS_THAN:
                  case ExpressionType::COMPARE_GREATER_THAN:
                  case ExpressionType::COMPARE_LESS_THAN_OR_EQUAL_TO:
                  case ExpressionType::COMPARE_GREATER_THAN_OR_EQUAL_TO:
                  case ExpressionType::COMPARE_LIKE: {
                     mlir::db::DBCmpPredicate pred;
                     switch (opType) {
                        case ExpressionType::COMPARE_EQUAL: pred = mlir::db::DBCmpPredicate::eq; break;
                        case ExpressionType::COMPARE_NOT_EQUAL: pred = mlir::db::DBCmpPredicate::neq; break;
                        case ExpressionType::COMPARE_LESS_THAN: pred = mlir::db::DBCmpPredicate::lt; break;
                        case ExpressionType::COMPARE_GREATER_THAN: pred = mlir::db::DBCmpPredicate::gt; break;
                        case ExpressionType::COMPARE_LESS_THAN_OR_EQUAL_TO: pred = mlir::db::DBCmpPredicate::lte; break;
                        case ExpressionType::COMPARE_GREATER_THAN_OR_EQUAL_TO: pred = mlir::db::DBCmpPredicate::gte; break;
                        case ExpressionType::COMPARE_LIKE: pred = mlir::db::DBCmpPredicate::like; break;
                        default: error("should not happen");
                     }
                     auto ct = toCommonTypes(builder, {left, right});
                     return builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), pred, ct[0], ct[1]);
                  }
                  case ExpressionType::COMPARE_NOT_LIKE: {
                     auto ct = toCommonTypes(builder, {left, right});
                     auto like = builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::DBCmpPredicate::like, ct[0], ct[1]);
                     return builder.create<mlir::db::NotOp>(loc, like);
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
            return builder.create<mlir::relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attrManager.createRef(attr), context.getCurrentTuple());
            break;
         }
         case T_FakeNode: { //
            const auto* attr = context.getAttribute(reinterpret_cast<FakeNode*>(node)->colId);
            return builder.create<mlir::relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attrManager.createRef(attr), context.getCurrentTuple());
            break;
         }
         case T_FuncCall: {
            auto* funcCall = reinterpret_cast<FuncCall*>(node);
            //expr = FuncCallTransform(parse_result,,context);
            std::string funcName = reinterpret_cast<value*>(funcCall->funcname_->head->data.ptr_value)->val_.str_;
            if (funcName == "pg_catalog") {
               funcName = reinterpret_cast<value*>(funcCall->funcname_->tail->data.ptr_value)->val_.str_;
            }
            if (funcName == "date_part") {
               auto part = getStringFromConst(reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value));
               auto arg2 = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value), context);
               auto unit = mlir::db::symbolizeExtractableTimeUnitAttr(part);
               if (!unit) {
                  error("expected valid unit");
               }
               return builder.create<mlir::db::DateExtractOp>(loc, builder.getI64Type(), unit.getValue(), arg2);
            }
            if (funcName == "substring") {
               auto str = translateExpression(builder, reinterpret_cast<Node*>(funcCall->args_->head->data.ptr_value), context);
               auto* fromNode = reinterpret_cast<Node*>(funcCall->args_->head->next->data.ptr_value);
               auto* toNode = reinterpret_cast<Node*>(funcCall->args_->tail->data.ptr_value);
               return builder.create<mlir::db::SubStrOp>(loc, str.getType(), str, getIntFromConst(fromNode), getIntFromConst(toNode));
            }

            break;
         }
         case T_NullTest: {
            //expr = NullTestTransform(parse_result, reinterpret_cast<NullTest*>(node),context);
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
         default: {
            error("unsupported expression type");
         }
      }
      error("should never happen");
      return mlir::Value();
   }
   mlir::Value translateRangeVar(mlir::OpBuilder& builder, RangeVar* stmt, TranslationContext& context, TranslationContext::ResolverScope& scope) {
      std::string relation = stmt->relname_;
      std::string alias = relation;
      if (stmt->alias_ && stmt->alias_->type_ == T_Alias && stmt->alias_->aliasname_) {
         alias = stmt->alias_->aliasname_;
      }
      if (!schema.tables.contains(relation)) {
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
      auto& table = schema.tables.at(relation);
      static size_t id = 0;
      std::string scopeName = alias + (id != 0 ? std::to_string(id) : "");
      std::vector<mlir::NamedAttribute> columns;
      attrManager.setCurrentScope(scopeName);
      for (auto c : table.columns) {
         auto attrDef = attrManager.createDef(c.first);
         attrDef.getRelationalAttribute().type = c.second;
         columns.push_back(builder.getNamedAttr(c.first, attrDef));
         context.mapAttribute(scope, c.first, &attrDef.getRelationalAttribute()); //todo check for existing and overwrite...
         context.mapAttribute(scope, alias + "." + c.first, &attrDef.getRelationalAttribute());
      }

      //::mlir::Type result, ::llvm::StringRef sym_name, ::llvm::StringRef table_identifier, ::mlir::relalg::TableMetaDataAttr meta, ::mlir::DictionaryAttr columns
      return builder.create<mlir::relalg::BaseTableOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), scopeName, relation, mlir::relalg::TableMetaDataAttr::get(builder.getContext(), std::make_shared<runtime::TableMetaData>()), builder.getDictionaryAttr(columns));
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
            std::vector<std::pair<std::string, const mlir::relalg::RelationalAttribute*>> mapping;
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
                  attrManager.setCurrentScope(outerjoinName);
                  std::unordered_map<const mlir::relalg::RelationalAttribute*, const mlir::relalg::RelationalAttribute*> remapped;
                  for (auto x : mapping) {
                     if (!remapped.contains(x.second)) {
                        auto [scopename, name] = attrManager.getName(x.second);

                        auto attrDef = attrManager.createDef(name, builder.getArrayAttr({attrManager.createRef(x.second)}));
                        attrDef.getRelationalAttribute().type = mlir::db::NullableType::get(builder.getContext(), x.second->type);
                        outerJoinMapping.push_back(attrDef);
                        attrDef.dump();
                        remapped.insert({x.second, &attrDef.getRelationalAttribute()});
                     }
                     context.mapAttribute(scope, x.first, remapped[x.second]);
                  }
               }
               mlir::ArrayAttr mapping = builder.getArrayAttr(outerJoinMapping);
               auto join = builder.create<mlir::relalg::OuterJoinOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), outerjoinName, left, right, mapping);
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
      auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), mapName, tree);
      mapOp.predicate().push_back(block);
      mlir::OpBuilder mapBuilder(builder.getContext());
      block->addArgument(mlir::relalg::TupleType::get(builder.getContext()), builder.getUnknownLoc());
      auto tupleScope = context.createTupleScope();
      mlir::Value tuple = block->getArgument(0);
      context.setCurrentTuple(tuple);

      mapBuilder.setInsertionPointToStart(block);
      for (auto p : toMap) {
         mlir::Value expr = translateExpression(mapBuilder, p.second, context);
         attrManager.setCurrentScope(mapName);
         auto attrDef = attrManager.createDef(p.first->colId);
         attrDef.getRelationalAttribute().type = expr.getType();
         context.mapAttribute(scope, p.first->colId, &attrDef.getRelationalAttribute());
         tuple = mapBuilder.create<mlir::relalg::AddAttrOp>(builder.getUnknownLoc(), mlir::relalg::TupleType::get(builder.getContext()), tuple, attrDef, expr);
         context.setCurrentTuple(tuple);
      }
      mapBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), tuple);
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
         //llvm::StringRef sym_name, ::mlir::Value rel, ::mlir::ArrayAttr group_by_attrs
         auto tupleScope = context.createTupleScope();
         auto groupByOp = builder.create<mlir::relalg::AggregationOp>(builder.getUnknownLoc(), tupleStreamType, groupByName, tree, builder.getArrayAttr(groupByAttrs));
         auto* block = new mlir::Block;
         groupByOp.aggr_func().push_back(block);
         block->addArgument(tupleStreamType, builder.getUnknownLoc());
         block->addArgument(tupleType, builder.getUnknownLoc());
         mlir::Value tuple = block->getArgument(1);
         mlir::Value relation = block->getArgument(0);
         mlir::OpBuilder aggrBuilder(builder.getContext());
         aggrBuilder.setInsertionPointToStart(block);
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
               mlir::relalg::RelationalAttributeRefAttr refAttr;
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
                  aggrResultType = refAttr.getRelationalAttribute().type;
                  if (!aggrResultType.isa<mlir::db::NullableType>() && groupByAttrs.empty()) {
                     aggrResultType = mlir::db::NullableType::get(builder.getContext(), aggrResultType);
                  }
               }
               expr = aggrBuilder.create<mlir::relalg::AggrFuncOp>(builder.getUnknownLoc(), aggrResultType, aggrFunc, currRel, refAttr);
            }
            attrManager.setCurrentScope(groupByName);
            auto attrDef = attrManager.createDef(toAggr.first->colId);
            attrDef.getRelationalAttribute().type = expr.getType();
            context.mapAttribute(scope, toAggr.first->colId, &attrDef.getRelationalAttribute());
            tuple = aggrBuilder.create<mlir::relalg::AddAttrOp>(builder.getUnknownLoc(), mlir::relalg::TupleType::get(builder.getContext()), tuple, attrDef, expr);
            context.setCurrentTuple(tuple);
         }
         aggrBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), tuple);

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
            const mlir::relalg::RelationalAttribute* attribute;
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
                        std::unordered_set<const mlir::relalg::RelationalAttribute*> handledAttrs;
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

   std::pair<mlir::Value, TargetInfo> translateSelectStmt(mlir::OpBuilder& builder, SelectStmt* stmt, TranslationContext& context, TranslationContext::ResolverScope& scope) {
      switch (stmt->op_) {
         case SETOP_NONE: {
            if (stmt->values_lists_) {
               size_t numColumns = 0;
               std::vector<mlir::Type> globalTypes;
               std::vector<mlir::Attribute> rows;
               bool first = true;
               for (auto* valueList = stmt->values_lists_->head; valueList != nullptr; valueList = valueList->next) {
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
                              //case T_Null: return builder.create<mlir::db::NullOp>(loc, mlir::db::NullableType::get(builder.getContext(), builder.getNoneType()));
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
                        globalTypes[j] = getCommonType(globalTypes.at(j), types.at(j));
                     }
                  }
               }
               static size_t constRelId = 0;
               std::string symName = "constrel" + std::to_string(constRelId++);
               attrManager.setCurrentScope(symName);
               std::vector<mlir::Attribute> attributes;

               TargetInfo targetInfo;
               for (size_t i = 0; i < numColumns; i++) {
                  std::string columnName = "const" + std::to_string(i);
                  auto attrDef = attrManager.createDef(columnName);
                  attrDef.getRelationalAttribute().type = globalTypes[i];
                  attributes.push_back(attrDef);
                  targetInfo.namedResults.push_back({columnName, &attrDef.getRelationalAttribute()});
               }
               //::llvm::StringRef sym_name, ::mlir::ArrayAttr attributes, ::mlir::ArrayAttr values
               mlir::Value constRel = builder.create<mlir::relalg::ConstRelationOp>(builder.getUnknownLoc(), symName, builder.getArrayAttr(attributes), builder.getArrayAttr(rows));
               return std::make_pair(constRel, targetInfo);
            }

            //todo: auto with = WithTransform(builder, stmt->with_clause_);
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
            mlir::Value tree = translateFrom(builder, stmt, context, scope);
            //auto selectDistinct = stmt->distinct_clause_ != nullptr;
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
            if (stmt->sort_clause_) {
               std::vector<mlir::Attribute> mapping;
               for (auto* cell = stmt->sort_clause_->head; cell != nullptr; cell = cell->next) {
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
                        if (target->type != T_ColumnRef) {
                           error("can only sort with column refs");
                        }
                        auto* attr = resolveColRef(target, context);
                        mapping.push_back(mlir::relalg::SortSpecificationAttr::get(builder.getContext(), attrManager.createRef(attr), spec));
                        break;
                     }
                     default: {
                        error("unknown orderby type");
                     }
                  }
               }
               tree = builder.create<mlir::relalg::SortOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr(mapping));
            }
            if (stmt->limit_count_) {
               size_t limit = reinterpret_cast<A_Const*>(stmt->limit_count_)->val_.val_.ival_;
               tree = builder.create<mlir::relalg::LimitOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), limit, tree);
               //todo: limit offset
            }

            return std::make_pair(tree, targetInfo);
         }
         default: {
            error("could not translate select statement type");
         }
      }
      return std::make_pair(mlir::Value(), TargetInfo());
   }
   ~SQLTranslator() {
      pg_query_free_parse_result(result);
   }
};

int main(int argc, char** argv) {
   mlir::MLIRContext context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<mlir::relalg::RelAlgDialect>();
   mlir::OpBuilder builder(&context);
   SQLTranslator::Schema schema = {{"nation", {{"n_nationkey", builder.getI32Type()}, {"n_name", mlir::db::StringType::get(builder.getContext())}, {"n_regionkey", builder.getI32Type()}, {"n_comment", mlir::db::NullableType::get(builder.getContext(), mlir::db::StringType::get(builder.getContext()))}}},

                                   {"region", {{"r_regionkey", builder.getI32Type()}, {"r_name", mlir::db::StringType::get(builder.getContext())}, {"r_comment", mlir::db::NullableType::get(builder.getContext(), mlir::db::StringType::get(builder.getContext()))}}},

                                   {"part", {{"p_partkey", builder.getI32Type()}, {"p_name", mlir::db::StringType::get(builder.getContext())}, {"p_mfgr", mlir::db::StringType::get(builder.getContext())}, {"p_brand", mlir::db::StringType::get(builder.getContext())}, {"p_type", mlir::db::StringType::get(builder.getContext())}, {"p_size", builder.getI32Type()}, {"p_container", mlir::db::StringType::get(builder.getContext())}, {"p_retailprice", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"p_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"supplier", {{"s_suppkey", builder.getI32Type()}, {"s_name", mlir::db::StringType::get(builder.getContext())}, {"s_address", mlir::db::StringType::get(builder.getContext())}, {"s_nationkey", builder.getI32Type()}, {"s_phone", mlir::db::StringType::get(builder.getContext())}, {"s_acctbal", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"s_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"partsupp", {{"ps_partkey", builder.getI32Type()}, {"ps_suppkey", builder.getI32Type()}, {"ps_availqty", builder.getI32Type()}, {"ps_supplycost", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"ps_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"customer", {{"c_custkey", builder.getI32Type()}, {"c_name", mlir::db::StringType::get(builder.getContext())}, {"c_address", mlir::db::StringType::get(builder.getContext())}, {"c_nationkey", builder.getI32Type()}, {"c_phone", mlir::db::StringType::get(builder.getContext())}, {"c_acctbal", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"c_mktsegment", mlir::db::StringType::get(builder.getContext())}, {"c_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"orders", {{"o_orderkey", builder.getI32Type()}, {"o_custkey", builder.getI32Type()}, {"o_orderstatus", mlir::db::CharType::get(builder.getContext(), 1)}, {"o_totalprice", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"o_orderdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"o_orderpriority", mlir::db::StringType::get(builder.getContext())}, {"o_clerk", mlir::db::StringType::get(builder.getContext())}, {"o_shippriority", builder.getI32Type()}, {"o_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"lineitem", {{"l_orderkey", builder.getI32Type()}, {"l_partkey", builder.getI32Type()}, {"l_suppkey", builder.getI32Type()}, {"l_linenumber", builder.getI32Type()}, {"l_quantity", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_extendedprice", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_discount", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_tax", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_returnflag", mlir::db::CharType::get(builder.getContext(), 1)}, {"l_linestatus", mlir::db::CharType::get(builder.getContext(), 1)}, {"l_shipdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"l_commitdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"l_receiptdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"l_shipinstruct", mlir::db::StringType::get(builder.getContext())}, {"l_shipmode", mlir::db::StringType::get(builder.getContext())}, {"l_comment", mlir::db::StringType::get(builder.getContext())}}},
                                   {"assistenten", {{"persnr", builder.getI64Type()}, {"name", mlir::db::StringType::get(builder.getContext())}, {"fachgebiet", mlir::db::StringType::get(builder.getContext())}, {"boss", builder.getI64Type()}}},
                                   {"hoeren", {{"matrnr", builder.getI64Type()}, {"vorlnr", builder.getI64Type()}}},
                                   {"studenten", {{"matrnr", builder.getI64Type()}, {"name", mlir::db::StringType::get(builder.getContext())}, {"semester", builder.getI64Type()}}},
                                   {"professoren", {{"persnr", builder.getI64Type()}, {"name", mlir::db::StringType::get(builder.getContext())}, {"rang", mlir::db::StringType::get(builder.getContext())}, {"raum", builder.getI64Type()}}},
                                   {"vorlesungen", {{"vorlnr", builder.getI64Type()}, {"titel", mlir::db::StringType::get(builder.getContext())}, {"sws", builder.getI64Type()}, {"gelesenvon", builder.getI64Type()}}},
                                   {"voraussetzen", {{"vorgaenger", builder.getI64Type()}, {"nachfolger", builder.getI64Type()}}},
                                   {"pruefen", {{"matrnr", builder.getI64Type()}, {"vorlnr", builder.getI64Type()}, {"persnr", builder.getI64Type()}, {"note", mlir::db::DecimalType::get(builder.getContext(), 2, 1)}}}};

   std::string filename = std::string(argv[1]);
   std::ifstream istream{filename};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   auto start = std::chrono::high_resolution_clock::now();
   SQLTranslator translator(buffer.str(), schema, &context);
   mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

   builder.setInsertionPointToStart(&moduleOp.body().front());
   mlir::FuncOp funcOp = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {mlir::db::TableType::get(builder.getContext())}));
   funcOp.body().push_back(new mlir::Block);
   builder.setInsertionPointToStart(&funcOp.body().front());
   mlir::Value val = translator.translate(builder);

   builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), val);
   auto end = std::chrono::high_resolution_clock::now();
   std::cerr << "time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

   moduleOp->print(llvm::outs());
   return 0;
}