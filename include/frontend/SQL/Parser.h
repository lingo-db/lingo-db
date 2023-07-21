#ifndef FRONTEND_SQL_PARSER_H
#define FRONTEND_SQL_PARSER_H
#include "libpg_query/pg_list.h"
#include "libpg_query/pg_query.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallString.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "parsenodes.h"
#include "runtime/Catalog.h"

#include "runtime-defs/ExecutionContext.h"
#include "runtime-defs/RelationHelper.h"

#include <llvm/ADT/StringSwitch.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stack>
#include <unordered_set>
namespace{
struct TranslationContext;
struct StringInfo {
   static bool isEqual(std::string a, std::string b) { return a == b; }
   static std::string getEmptyKey() { return ""; }
   static std::string getTombstoneKey() { return "-"; }
   static size_t getHashValue(std::string str) { return std::hash<std::string>{}(str); }
};
using ResolverScope = llvm::ScopedHashTable<std::string, const mlir::tuples::Column*, StringInfo>::ScopeTy;
} //end namespace

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
   static mlir::db::DateType getHigherDateType(mlir::Type left, mlir::Type right);
   static mlir::Type getCommonBaseType(mlir::Type left, mlir::Type right) {
      left = getBaseType(left);
      right = getBaseType(right);
      bool stringPresent = left.isa<mlir::db::StringType>() || right.isa<mlir::db::StringType>();
      bool intPresent = left.isa<mlir::IntegerType>() || right.isa<mlir::IntegerType>();
      bool charPresent = left.isa<mlir::db::CharType>() || right.isa<mlir::db::CharType>();
      bool floatPresent = left.isa<mlir::FloatType>() || right.isa<mlir::FloatType>();
      bool decimalPresent = left.isa<mlir::db::DecimalType>() || right.isa<mlir::db::DecimalType>();
      bool datePresent = left.isa<mlir::db::DateType>() || right.isa<mlir::db::DateType>();
      if (datePresent) return getHigherDateType(left, right);
      if (stringPresent) return mlir::db::StringType::get(left.getContext());
      if (charPresent) return left == right ? left : mlir::db::StringType::get(left.getContext());
      if (floatPresent) return getHigherFloatType(left, right);
      if (decimalPresent) return getHigherDecimalType(left, right);
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
   static mlir::Value castValueToType(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t);
   static std::vector<mlir::Value> toCommonBaseTypes(mlir::OpBuilder& builder, mlir::ValueRange values) {
      auto commonType = getCommonBaseType(values.getTypes());
      std::vector<mlir::Value> res;
      for (auto val : values) {
         res.push_back(castValueToType(builder, val, commonType));
      }
      return res;
   }
   static std::vector<mlir::Value> toCommonNumber(mlir::OpBuilder& builder, mlir::ValueRange values) {
      auto anyDecimal = llvm::any_of(values, [](mlir::Value v) { return getBaseType(v.getType()).isa<mlir::db::DecimalType>(); });
      auto anyFloat = llvm::any_of(values, [](mlir::Value v) { return getBaseType(v.getType()).isIntOrFloat() && !getBaseType(v.getType()).isIntOrIndex(); });
      if (anyDecimal && !anyFloat) {
         std::vector<mlir::Value> res;
         for (auto val : values) {
            if (!getBaseType(val.getType()).isa<mlir::db::DecimalType>()) {
               res.push_back(castValueToType(builder, val, mlir::db::DecimalType::get(builder.getContext(), 19, 0)));
            } else {
               res.push_back(val);
            }
         }
         return res;
      } else {
         return toCommonBaseTypes(builder, values);
      }
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
struct WindowProperties {
   std::vector<Node*> partitionBy;
   std::vector<std::pair<SortByDir, Node*>> orderBy;
   int64_t start = std::numeric_limits<int64_t>::min();
   int64_t end = std::numeric_limits<int64_t>::max();
};
struct ReplaceState {
   std::unordered_map<FakeNode*, std::string> groupingFuncs;
   std::unordered_map<FakeNode*, Node*> evalBeforeAggr;
   std::unordered_map<FakeNode*, std::tuple<std::string, Node*, bool>> aggrs;
   std::unordered_map<FakeNode*, Node*> evalBeforeWindowFunc;
   std::unordered_map<FakeNode*, std::tuple<std::string, Node*, WindowProperties>> windowFunctions;
};
ExpressionType stringToExpressionType(const std::string& parserStr);
struct Parser {
   mlir::tuples::ColumnManager& attrManager;

   std::string sql;
   PgQueryInternalParsetreeAndError result;
   runtime::Catalog& catalog;
   std::vector<std::unique_ptr<FakeNode>> fakeNodes;

   bool isParallelismAllowed() const;

   struct TargetInfo {
      std::vector<std::pair<std::string, const mlir::tuples::Column*>> namedResults;
      void map(std::string name, const mlir::tuples::Column* attr) {
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
   bool parallelismAllowed;
   Parser(std::string sql, runtime::Catalog& catalog, mlir::ModuleOp moduleOp);

   mlir::Value getExecutionContextValue(mlir::OpBuilder& builder) {
      mlir::func::FuncOp funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>("rt_get_execution_context");
      if (!funcOp) {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(moduleOp.getBody());
         funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "rt_get_execution_context", builder.getFunctionType({}, {mlir::util::RefType::get(builder.getContext(), builder.getI8Type())}), builder.getStringAttr("private"), mlir::ArrayAttr{}, mlir::ArrayAttr{});
      }

      return builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange{}).getResult(0);
   }
   mlir::Value createStringValue(mlir::OpBuilder& builder, std::string str) {
      return builder.create<mlir::util::CreateConstVarLen>(builder.getUnknownLoc(), mlir::util::VarLen32Type::get(builder.getContext()), builder.getStringAttr(str));
   }

   const mlir::tuples::Column* resolveColRef(Node* node, TranslationContext& context);

   //helper function: convert pg list to vector of int/string for type modifiers
   std::vector<std::variant<size_t, std::string>> getTypeModList(List* typeMods);

   //analyze expression from target list to e.g. first compute expressions required for computing aggregates
   Node* analyzeTargetExpression(Node* node, ReplaceState& replaceState);

   //create mlir base type from runtime column type (w.o. nullability)
   mlir::Type createBaseTypeFromColumnType(mlir::MLIRContext* context, const runtime::ColumnType& colType);

   //create mlir type from runtime column type
   mlir::Type createTypeFromColumnType(mlir::MLIRContext* context, const runtime::ColumnType& colType);

   //helper function: convert pg linked list into vector of strings (if possible)
   std::vector<std::string> listToStringVec(List* l);

   //translate target list in selection and also consider aggregation and groupby
   std::pair<mlir::Value, TargetInfo> translateSelectionTargetList(mlir::OpBuilder& builder, List* groupBy, Node* having, List* targetList, List* sortClause, List* distinctClause, mlir::Value tree, TranslationContext& context, ResolverScope& scope);

   //translate insert statement
   void translateInsertStmt(mlir::OpBuilder& builder, InsertStmt* stmt);

   //creates a column type from the given information
   runtime::ColumnType createColumnType(std::string datatypeName, bool isNull, std::vector<std::variant<size_t, std::string>> typeModifiers);

   //translate a column definition in a create statment
   std::pair<std::string, std::shared_ptr<runtime::ColumnMetaData>> translateColumnDef(ColumnDef* columnDef);

   //translates table metadata (column definitions + primary keys)
   std::shared_ptr<runtime::TableMetaData> translateTableMetaData(List* metaData);

   //translate a CREATE statement
   void translateCreateStatement(mlir::OpBuilder& builder, CreateStmt* statement);

   //translate the provided SQL statement
   std::optional<mlir::Value> translate(mlir::OpBuilder& builder);

   //translate a variable set statement (e.g. SET var = 1; )
   void translateVariableSetStatement(mlir::OpBuilder& builder, VariableSetStmt* variableSetStatement);

   //translate a COPY FROM statement
   void translateCopyStatement(mlir::OpBuilder& builder, CopyStmt* copyStatement);

   //translate a coalesce expression recursively into (nested) scf.if operations
   mlir::Value translateCoalesceExpression(mlir::OpBuilder& builder, TranslationContext& context, ListCell* expressions);

   //translate a when-case expression into (nested) scf.if operations
   mlir::Value translateWhenCaseExpression(mlir::OpBuilder& builder, TranslationContext& context, mlir::Value compareValue, ListCell* whenCell, Node* defaultNode);

   //translate a function call
   mlir::Value translateFuncCallExpression(Node* node, mlir::OpBuilder& builder, mlir::Location loc, TranslationContext& context);

   //translate binary expression
   mlir::Value translateBinaryExpression(mlir::OpBuilder& builder, ExpressionType opType, mlir::Value left, mlir::Value right);

   //translate expression into mlir operations that yield a single mlir::Value
   mlir::Value translateExpression(mlir::OpBuilder& builder, Node* node, TranslationContext& context, bool ignoreNull = false);

   //translates a rangevar expression inside a from clause, i.e. a table scan
   mlir::Value translateRangeVar(mlir::OpBuilder& builder, RangeVar* stmt, TranslationContext& context, ResolverScope& scope);

   //translate sub-query in from clause
   mlir::Value translateSubSelect(mlir::OpBuilder& builder, SelectStmt* stmt, std::string alias, std::vector<std::string> colAlias, TranslationContext& context, ResolverScope& scope);

   //translate boolean expression into an mlir block (tuple) -> bool
   mlir::Block* translatePredicate(mlir::OpBuilder& builder, Node* node, TranslationContext& context);

   //translate a single item of a from clause (e.g. RangeVar, SubSelect, explicit Joins,...)
   mlir::Value translateFromClausePart(mlir::OpBuilder& builder, Node* node, TranslationContext& context, ResolverScope& scope);

   //translate a complete from clause into a single value of type tuple stream (connect single items with cross-products)
   mlir::Value translateFromClause(mlir::OpBuilder& builder, SelectStmt* stmt, TranslationContext& context, ResolverScope& scope);

   //translate list of constant values into relalg::ConstRelationOp
   std::pair<mlir::Value, TargetInfo> translateConstRelation(List* valuesLists, mlir::OpBuilder& builder);

   //translate set operations (union, intersect, except)
   std::pair<mlir::Value, TargetInfo> translateSetOperation(mlir::OpBuilder& builder, SelectStmt* stmt, TranslationContext& context, ResolverScope& scope);

   //translate a 'classic' select statement e.g. SELECT ...
   std::pair<mlir::Value, TargetInfo> translateClassicSelectStmt(mlir::OpBuilder& builder, SelectStmt* stmt, TranslationContext& context, ResolverScope& scope);

   //translate a select statement which is either a 'classic select statement or a set operation
   std::pair<mlir::Value, TargetInfo> translateSelectStmt(mlir::OpBuilder& builder, SelectStmt* stmt, TranslationContext& context, ResolverScope& scope);

   std::pair<mlir::Value, mlir::tuples::ColumnRefAttr> mapExpressionToAttribute(mlir::Value tree, TranslationContext& context, mlir::OpBuilder& builder, ResolverScope& scope, Node* expression);
   std::tuple<mlir::Value, std::unordered_map<std::string, mlir::tuples::Column*>> performAggregation(mlir::OpBuilder& builder, std::vector<mlir::Attribute> groupByAttrs, const ReplaceState& replaceState, TranslationContext& context, mlir::Value tree);
   ~Parser();
};
} // end namespace frontend::sql
#endif // FRONTEND_SQL_PARSER_H
