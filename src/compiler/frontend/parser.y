%skeleton "lalr1.cc" // -*- C++ -*-
%require "3.0.4"

%define api.token.raw
%define api.namespace    { lingodb }
%define api.token.constructor
%define api.value.type variant
%define parse.assert



%code requires {
  # include <string>
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <utility>

  #include "lingodb/compiler/frontend/ast/table_producer.h"
  #include "lingodb/compiler/frontend/ast/create_node.h"
  #include "lingodb/compiler/frontend/ast/insert_node.h"
  #include "lingodb/compiler/frontend/ast/tableref.h"
  #include "lingodb/compiler/frontend/ast/aggregation_node.h"
  #include "lingodb/compiler/frontend/ast/result_modifier.h"
  #include "lingodb/compiler/frontend/ast/parsed_expression.h"
  #include "lingodb/compiler/frontend/ast/query_node.h"
  #include "lingodb/compiler/frontend/ast/constant_value.h"
  #include "lingodb/compiler/frontend/ast/constraint.h"
  #include "lingodb/compiler/frontend/ast/extend_node.h"
  #include "lingodb/compiler/frontend/ast/set_node.h"
  #include "lingodb/compiler/frontend/ast/copy_node.h"
  #include "lingodb/compiler/frontend/frontend_error.h"
  class driver;
}

// The parsing context.
%param { driver& drv }



%define parse.trace
%define parse.error detailed
%define parse.lac full
%locations
%code {
  #include <iostream>
  #include <memory>
  #include "lingodb/compiler/frontend/driver.h"
  #include "lingodb/compiler/frontend/node_factory.h"

  #define mkNode drv.nf.node
  #define mkListShared drv.nf.listShared
  #define mkList drv.nf.list
  #define mkNotNode drv.nf.notNode


}

%define api.token.prefix {TOK_}
//TODO check if int?
%token <int> ICONST
%token <uint64_t>	    INTEGER_VALUE	"integer_value"
%token <std::string>	FCONST
%token <std::string>     IDENT
%token <std::string>	IDENTIFIER	"identifier"
%token <std::string>	STRING_VALUE
%token <std::string>	BIT_VALUE	"bit_string"
%token <std::string>	HEX_VALUE	"hex_string"
%token <std::string>	NATIONAL_VALUE	"nat_string"
%token 			LP		"("
%token 			RP		")"
%token 			LB		"["
%token 			RB		"]"
%token 			DOT		"."
%token          PERCENT "%"
%token 			COMMA		","
%token 			SEMICOLON	";"
%token 			PLUS		"+"
%token 			SLASH		"/"
%token 			STAR		"*"
%token 			MINUS		"-"
%token 			EQUAL		"="
%token 			NOT_EQUAL	"<>"
%token 			LESS_EQUAL	"<="
%token 			LESS		"<"
%token 			GREATER_EQUAL	">="
%token 			GREATER		">"
%token          HAT         "^"
%token 			QUOTE		"'"
%token          PIPE        "|>"




%token <std::string>	        UIDENT SCONST USCONST BCONST XCONST Op
%token 	        PARAM
%token			TYPECAST DOT_DOT COLON_EQUALS

/* 
 * Taken directly from postgres grammatic 
 * TODO LINK
**/
%token <std::string> ABORT_P ABSENT ABSOLUTE_P ACCESS ACTION ADD_P ADMIN AFTER
	AGGREGATE ALL ALSO ALTER ALWAYS ANALYSE ANALYZE AND ANY ARRAY AS ASC
	ASENSITIVE ASSERTION ASSIGNMENT ASYMMETRIC ATOMIC AT ATTACH ATTRIBUTE AUTHORIZATION

	BACKWARD BEFORE BEGIN_P BETWEEN BIGINT BINARY BIT
	BOOLEAN_P BOTH BREADTH BY

	CACHE CALL CALLED CASCADE CASCADED CASE CAST CATALOG_P CHAIN CHAR_P
	CHARACTER CHARACTERISTICS CHECK CHECKPOINT CLASS CLOSE
	CLUSTER COALESCE COLLATE COLLATION COLUMN COLUMNS COMMENT COMMENTS COMMIT
	COMMITTED COMPRESSION CONCURRENTLY CONDITIONAL CONFIGURATION CONFLICT
	CONNECTION CONSTRAINT CONSTRAINTS CONTENT_P CONTINUE_P CONVERSION_P COPY
	COST CREATE CROSS CSV CUBE CURRENT_P
	CURRENT_CATALOG CURRENT_DATE CURRENT_ROLE CURRENT_SCHEMA
	CURRENT_TIME CURRENT_TIMESTAMP CURRENT_USER CURSOR CYCLE

	DATA_P DATABASE DAY_P DEALLOCATE DEC DECIMAL_P DECLARE DEFAULT DEFAULTS DATE_P
	DEFERRABLE DEFERRED DEFINER DELETE_P DELIMITER DELIMITERS DEPENDS DEPTH DESC
	DETACH DICTIONARY DISABLE_P DISCARD DISTINCT DO DOCUMENT_P DOMAIN_P
	DOUBLE_P DROP

	EACH ELSE EMPTY_P ENABLE_P ENCODING ENCRYPTED END_P ENFORCED ENUM_P ERROR_P
	ESCAPE EVENT EXCEPT EXCLUDE EXCLUDING EXCLUSIVE EXECUTE EXISTS EXPLAIN
	EXPRESSION EXTENSION EXTERNAL EXTRACT

	FALSE_P FAMILY FETCH FILTER FINALIZE FIRST_P FLOAT_P FOLLOWING FOR
	FORCE FOREIGN FORMAT FORWARD FREEZE FROM FULL FUNCTION FUNCTIONS

	GENERATED GLOBAL GRANT GRANTED GREATEST GROUP_P GROUPING GROUPS

	HANDLER HAVING HEADER_P HOLD HOUR_P

	IDENTITY_P IF_P ILIKE IMMEDIATE IMMUTABLE IMPLICIT_P IMPORT_P IN_P INCLUDE
	INCLUDING INCREMENT INDENT INDEX INDEXES INHERIT INHERITS INITIALLY INLINE_P
	INNER_P INOUT INPUT_P INSENSITIVE INSERT INSTEAD INT_P INTEGER
	INTERSECT INTERVAL INTO INVOKER IS ISNULL ISOLATION

	JOIN JSON JSON_ARRAY JSON_ARRAYAGG JSON_EXISTS JSON_OBJECT JSON_OBJECTAGG
	JSON_QUERY JSON_SCALAR JSON_SERIALIZE JSON_TABLE JSON_VALUE

	KEEP KEY KEYS

	LABEL LANGUAGE LARGE_P LAST_P LATERAL_P
	LEADING LEAKPROOF LEAST LEFT LEVEL LIKE LIMIT LISTEN LOAD LOCAL
	LOCALTIME LOCALTIMESTAMP LOCATION LOCK_P LOCKED LOGGED

	MAPPING MATCH MATCHED MATERIALIZED MAXVALUE MERGE MERGE_ACTION METHOD
	MINUTE_P MINVALUE MODE MONTH_P MOVE

	NAME_P NAMES NATIONAL NATURAL NCHAR NESTED NEW NEXT NFC NFD NFKC NFKD NO
	NONE NORMALIZE NORMALIZED
	NOT NOTHING NOTIFY NOTNULL NOWAIT NULL_P NULLIF
	NULLS_P NUMERIC

	OBJECT_P OBJECTS_P OF OFF OFFSET OIDS OLD OMIT ON ONLY OPERATOR OPTION OPTIONS OR
	ORDER ORDINALITY OTHERS OUT_P OUTER_P
	OVER OVERLAPS OVERLAY OVERRIDING OWNED OWNER

	PARALLEL PARAMETER PARSER PARTIAL PARTITION PASSING PASSWORD PATH
	PERIOD PLACING PLAN PLANS POLICY
	POSITION PRECEDING PRECISION PRESERVE PREPARE PREPARED PRIMARY
	PRIOR PRIVILEGES PROCEDURAL PROCEDURE PROCEDURES PROGRAM PUBLICATION

	//!QUOTE 
    QUOTES

    EXTEND

	RANGE READ REAL REASSIGN RECURSIVE REF_P REFERENCES REFERENCING
	REFRESH REINDEX RELATIVE_P RELEASE RENAME REPEATABLE REPLACE REPLICA
	RESET RESTART RESTRICT RETURN RETURNING RETURNS REVOKE RIGHT ROLE ROLLBACK ROLLUP
	ROUTINE ROUTINES ROW ROWS RULE

	SAVEPOINT SCALAR SCHEMA SCHEMAS SCROLL SEARCH SECOND_P SECURITY SELECT
	SEQUENCE SEQUENCES
	SERIALIZABLE SERVER SESSION SESSION_USER SET SETS SETOF SHARE SHOW
	SIMILAR SIMPLE SKIP SMALLINT SNAPSHOT SOME SOURCE SQL_P STABLE STANDALONE_P
	START STATEMENT STATISTICS STDIN STDOUT STORAGE STORED STRICT_P STRING_P STRIP_P
	SUBSCRIPTION SUBSTRING SUPPORT SYMMETRIC SYSID SYSTEM_P SYSTEM_USER

	TABLE TABLES TABLESAMPLE TABLESPACE TARGET TEMP TEMPLATE TEMPORARY TEXT_P THEN
	TIES TIME TIMESTAMP TO TRAILING TRANSACTION TRANSFORM
	TREAT TRIGGER TRIM TRUE_P
	TRUNCATE TRUSTED TYPE_P TYPES_P

	UESCAPE UNBOUNDED UNCONDITIONAL UNCOMMITTED UNENCRYPTED UNION UNIQUE UNKNOWN
	UNLISTEN UNLOGGED UNTIL UPDATE USER USING

	VACUUM VALID VALIDATE VALIDATOR VALUE_P VALUES VARCHAR VARIADIC VARYING
	VERBOSE VERSION_P VIEW VIEWS VIRTUAL VOLATILE

	WHEN WHERE WHITESPACE_P WINDOW WITH WITHIN WITHOUT WORK WRAPPER WRITE

	XML_P XMLATTRIBUTES XMLCONCAT XMLELEMENT XMLEXISTS XMLFOREST XMLNAMESPACES
	XMLPARSE XMLPI XMLROOT XMLSERIALIZE XMLTABLE

	YEAR_P YES_P

	ZONE

    EQUAL NOT_EQUAL PLUS MINUS STAR SLASH LESS LESS_EQUAL GREATER_EQUAL GREATER TYPECAST PERCENT



%token AGGREGATE
/*
 * The grammar thinks these are keywords, but they are not in the kwlist.h
 * list and so can never be entered directly.  The filter in parser.c
 * creates these tokens when required (based on looking one token ahead).
 *
 * NOT_LA exists so that productions such as NOT LIKE can be given the same
 * precedence as LIKE; otherwise they'd effectively have the same precedence
 * as NOT, at least with respect to their left-hand subexpression.
 * FORMAT_LA, NULLS_LA, WITH_LA, and WITHOUT_LA are needed to make the grammar
 * LALR(1).
 */
%token		FORMAT_LA NOT_LA NULLS_LA WITH_LA WITHOUT_LA
%type <std::vector<std::shared_ptr<lingodb::ast::AstNode>>> stmtmulti
%type <std::shared_ptr<lingodb::ast::AstNode>> toplevel_stmt stmt
%type <std::shared_ptr<lingodb::ast::TableProducer>>  SelectStmt  PreparableStmt PipeSQLStmt classic_select_and_pipe_sql_with_parens pipe_sql_with_parens pipe_sql_no_parens pipe_or_select_clause
%type <std::shared_ptr<lingodb::ast::QueryNode>> simple_select select_clause select_no_parens with_clause select_with_parens cte_list common_table_expr

%type<std::vector<std::shared_ptr<lingodb::ast::ParsedExpression>>> target_list group_by_list func_arg_list func_arg_list_opt
                                                                    extract_list expr_list expr_list_with_alias substr_list distinct_clause
                                                                    opt_partition_clause rollup_clause group_by_list_with_alias


/*
* in_expr either returns a SubQuery or a list of expressions
*/
%type<std::variant<std::vector<std::shared_ptr<lingodb::ast::ParsedExpression>>, std::shared_ptr<lingodb::ast::SubqueryExpression>>> in_expr

%type<std::shared_ptr<lingodb::ast::TargetsExpression>> opt_target_list

%type<std::shared_ptr<lingodb::ast::ParsedExpression>>  having_clause target_el a_expr c_expr b_expr  where_clause group_by_item group_by_item_with_alias
                                                        func_arg_expr select_limit_value case_expr case_default cast_expr offset_clause 
                                                        select_offset_value

%type<std::shared_ptr<lingodb::ast::ConjunctionExpression>> and_a_expr or_a_expr

%type<lingodb::ast::ExpressionType> basicComparisonType

%type<std::shared_ptr<lingodb::ast::WindowExpression>> over_clause window_specification

%type<std::vector<lingodb::ast::CaseExpression::CaseCheck>> when_clause_list
%type<lingodb::ast::CaseExpression::CaseCheck> when_clause

%type<std::shared_ptr<lingodb::ast::FunctionExpression>>  func_application func_expr func_expr_common_subexpr
%type<std::shared_ptr<lingodb::ast::WindowExpression>>  window_func_expr;

%type<std::vector<std::shared_ptr<lingodb::ast::FunctionExpression>>> func_expr_list
%type<std::shared_ptr<ast::ConstantExpression>> extract_arg

%type<lingodb::ast::jointCondOrUsingCols> join_qual

%type<std::shared_ptr<lingodb::ast::WindowBoundary>> opt_frame_clause frame_extent frame_bound

/*
* Columnref or Starexpression for instance
*/
%type<std::shared_ptr<lingodb::ast::ParsedExpression>>  columnref indirection indirection_el
%type<std::vector<std::shared_ptr<lingodb::ast::ParsedExpression>>> columnref_list

%type<std::shared_ptr<lingodb::ast::TableRef>> from_clause opt_from_clause table_ref from_list joined_table
%type<std::shared_ptr<lingodb::ast::ExpressionListRef>>  values_clause

%type<std::string>  ColId ColLabel BareColLabel attr_name
                    qualified_name relation_expr
                    name type_function_name func_name col_name_keyword unreserved_keyword reserved_keyword all_Op qual_Op MathOp subquery_op any_operator NonReservedWord type_func_name_keyword
%type<std::pair<std::string, std::vector<std::string>>> alias_clause opt_alias_clause

%type<std::vector<std::string>> name_list opt_name_list qualified_name_list

%type<std::shared_ptr<lingodb::ast::ParsedExpression>> Iconst SignedIconst AexprConst Sconst Bconst Fconst NumericOnly NonReservedWord_or_Sconst opt_boolean_or_string var_value
%type<std::vector<std::shared_ptr<lingodb::ast::ParsedExpression>>> var_list

%type<std::shared_ptr<lingodb::ast::GroupByNode>> group_clause group_clause_with_alias

%type<std::shared_ptr<lingodb::ast::PipeOperator>> pipe_operator pipe_start

%type<lingodb::ast::JoinType> join_type


%type<lingodb::ast::SubqueryType> sub_type

%type<std::shared_ptr<lingodb::ast::AggregationNode>> agg_expr

%type<std::shared_ptr<lingodb::ast::OrderByModifier>> sort_clause

%type<std::vector<std::shared_ptr<lingodb::ast::OrderByElement>>> sortby_list

%type<std::shared_ptr<lingodb::ast::OrderByElement>> sortby

%type<lingodb::ast::OrderType> opt_asc_desc

%type<lingodb::ast::OrderByNullType> opt_nulls_order

%type<std::optional<std::shared_ptr<lingodb::ast::ResultModifier>>>  opt_select_limit
%type<std::optional<std::shared_ptr<lingodb::ast::OrderByModifier>>> opt_sort_clause

%type<std::shared_ptr<lingodb::ast::LimitModifier>> select_limit limit_clause

%type<std::optional<lingodb::ast::LogicalType>> opt_interval
%type<bool> opt_asymmetric set_quantifier

%type<std::optional<std::shared_ptr<lingodb::ast::ParsedExpression>>> case_arg

%type<std::shared_ptr<lingodb::ast::CreateNode>> CreateStmt
%type<bool> OptTemp opt_varying
%type<lingodb::ast::LogicalTypeWithMods> Numeric SimpleType Type CharacterWithoutLength character Bit ConstCharacter Character CharacterWithLength ConstDatetime Typename ConstTypename Numeric_with_opt_lenghth
%type<std::shared_ptr<lingodb::ast::TableElement>> TableElement columnElement TableConstraint
%type<std::vector<std::shared_ptr<lingodb::ast::TableElement>>> TableElementList OptTableElementList
%type<std::shared_ptr<lingodb::ast::Constraint>> ColConstraint ColConstraintElem ConstraintElem
%type<std::vector<std::shared_ptr<lingodb::ast::Constraint>>> ColQualList
%type<std::vector<std::shared_ptr<lingodb::ast::Value>>> opt_type_modifiers type_modifiers
%type<std::shared_ptr<lingodb::ast::Value>> type_modifier

%type<std::shared_ptr<lingodb::ast::InsertNode>> InsertStmt insert_rest
%type<std::string> insert_target insert_column_item
%type<std::vector<std::string>> insert_column_list
%type<std::shared_ptr<lingodb::ast::SetExpression>> set_list

/*%type <nodes::RelExpression>		simple_select
%type <std::shared_ptr<nodes::Query>> select_no_parens
%type <std::shared_ptr<nodes::SelectStatement>> SelectStatement
%type <std::vector<nodes::RelExpression>>	opt_from_clause
%type <std::vector<nodes::RelExpression>>	from_list
%type <nodes::RelExpression>						table_ref
%type <std::shared_ptr<nodes::QualifiedName>>					qualified_name
%type <std::vector<std::string>>					qualifier_list
%type <std::shared_ptr<nodes::RowExpr>>					target_list
%type <std::shared_ptr<nodes::QualifiedName>>					target_element*/
//%token <int> NUMBER "number"
//%nterm <int> exp

%type<std::shared_ptr<ast::SetVariableStatement>> VariableSetStmt set_rest set_rest_more generic_set
%type<std::string> var_name


%type<std::shared_ptr<ast::CopyNode>> CopyStmt
%type<std::string> copy_file_name copy_delimiter
%type<std::vector<std::pair<std::string, std::string>>> copy_options copy_opt_list
%type<std::pair<std::string, std::string>> copy_opt_item
/* Precedence: lowest to highest */
%left		UNION EXCEPT
%left		INTERSECT
%left AND_PRIORITY
%left OR_PRIORITY
%left		OR
%left		AND
%right		NOT
%nonassoc	IS ISNULL NOTNULL	/* IS sets precedence for IS NULL, etc */
%nonassoc	GREATER LESS EQUAL   NOT_EQUAL
%nonassoc	BETWEEN IN_P LIKE ILIKE SIMILAR NOT_LA
%nonassoc	ESCAPE			/* ESCAPE must be just above LIKE/ILIKE/SIMILAR */

%left QUAL_OP        // lowest
%left  LESS_EQUAL GREATER_EQUAL



%nonassoc	UNBOUNDED NESTED /* ideally would have same precedence as IDENT */
%nonassoc	IDENT PARTITION RANGE ROWS GROUPS PRECEDING FOLLOWING CUBE ROLLUP
			SET KEYS OBJECT_P SCALAR VALUE_P WITH WITHOUT PATH
%left		Op OPERATOR		/* multi-character ops and user-defined operators */
%left		PLUS MINUS
%left		STAR SLASH PERCENT
%left		HAT
/* Unary Operators */
%left		AT				/* sets precedence for AT TIME ZONE, AT LOCAL */
%left		COLLATE
%right		UMINUS
%left		LB RB
%left		LP RP
%left		TYPECAST
%left		DOT
/*
 * These might seem to be low-precedence, but actually they are not part
 * of the arithmetic hierarchy at all in their use as JOIN operators.
 * We make them high-precedence to support their use as function names.
 * They wouldn't be given a precedence at all, were it not that we need
 * left-associativity among the JOIN rules themselves.
 */
%left		JOIN CROSS LEFT FULL RIGHT INNER_P NATURAL


%printer {  } <*>;

%%
%start parse_toplevel;

/*
 * We parse a list of statements, but if there are any special modes first we can add them here
 */
parse_toplevel:
    stmtmulti {drv.result = $stmtmulti;}
    ;
/*
 * Allows
 */
 //TODO Allow multiple
stmtmulti:
    toplevel_stmt
    {
        auto list = mkListShared<lingodb::ast::AstNode>();
        list.emplace_back($1);
        $$ = list;
    }

    | stmtmulti[list] SEMICOLON toplevel_stmt
    {
        if($toplevel_stmt != nullptr) {
            $list.emplace_back($toplevel_stmt);
        }
        $$ = $list;

    }


    ;

toplevel_stmt:
    stmt {$$=$1;}
    | %empty
  //TODO Add Later  | TransactionStmtLegacy
  ;
/*
 * TODO Add the different Statement Types, like Create, Copy etc
*/
stmt:
 SelectStmt {$$=$1;}
 | CreateStmt {$$=$1;}
 | InsertStmt {$$=$1;}
 | PipeSQLStmt {$$=$1;}
 | VariableSetStmt {$$=$1;}
 | CopyStmt {$$=$1;}
 ;

 SelectStmt:
    select_no_parens {$$=$select_no_parens;}
    | select_with_parens {$$=$select_with_parens;}
    ;
PipeSQLStmt:
    pipe_sql_no_parens {$$=$1;}
    | pipe_sql_with_parens {$$=$1;}
    ;

classic_select_and_pipe_sql_with_parens:
    select_with_parens {$$=$1;}
    | pipe_sql_with_parens {$$=$1;}   
    ; 

select_with_parens:
    LP select_no_parens RP {$$=$select_no_parens;}
    | LP select_with_parens RP {$$=$2;}
    ;
pipe_sql_with_parens:
    LP pipe_sql_no_parens RP {$$=$2;}
    | LP pipe_sql_with_parens RP {$$=$2;}
    ;
/*
* Difference to postgres the values clause is located in the select_no_parens rule so that it can be used as a standalone table producer.
*/
//TODO add missing rules
select_no_parens:
    simple_select {$$=$1;}
    | select_clause sort_clause
    {
        $select_clause->modifiers.emplace_back($sort_clause);
        $$ = $select_clause;
    }
    | select_clause opt_sort_clause  opt_select_limit
    {
        if ($opt_sort_clause.has_value()) {
            $select_clause->modifiers.emplace_back($opt_sort_clause.value());
        }
        if($opt_select_limit.has_value()) {
            $select_clause->modifiers.emplace_back($opt_select_limit.value());
        }
        $$ = $select_clause;
    }
    //TODO | select_clause opt_sort_clause select_limit opt_for_locking_clause
    | with_clause select_clause opt_sort_clause
    {
        if ($opt_sort_clause.has_value()) {
            $select_clause->modifiers.emplace_back($opt_sort_clause.value());
        }
        auto current = std::static_pointer_cast<lingodb::ast::CTENode>($with_clause);
        while(current->child != nullptr) {
            current = std::dynamic_pointer_cast<lingodb::ast::CTENode>(current->child);
            if(!current) {
                error(@$, "Should not happen");
            }
        }
        current->child = $select_clause;
        $$ = $with_clause;
    }
    | with_clause select_clause opt_sort_clause  opt_select_limit
    {
        if ($opt_sort_clause.has_value()) {
            $select_clause->modifiers.emplace_back($opt_sort_clause.value());
        }
        if($opt_select_limit.has_value()) {
            $select_clause->modifiers.emplace_back($opt_select_limit.value());
        }

        auto current = std::static_pointer_cast<lingodb::ast::CTENode>($with_clause);
        while(current->child != nullptr) {
            current = std::dynamic_pointer_cast<lingodb::ast::CTENode>(current->child);
            if(!current) {
                error(@$, "Should not happen");
            }
        }
        current->child = $select_clause;
        $$ = $with_clause;
    }
    //TODO | with_clause select_clause opt_sort_clause select_limit opt_for_locking_clause
  ;
pipe_sql_no_parens:
    | from_clause
    {

        $$ = $from_clause;

    }
    | PipeSQLStmt[parens] PIPE pipe_operator
    {

        auto pipeOp = std::static_pointer_cast<lingodb::ast::PipeOperator>($pipe_operator);
        $pipe_operator->input = $parens;

        $$ = $pipe_operator;

    }
    ;

pipe_start:
    | from_clause
     {


     }
     //TODO DOC
    | select_no_parens[parens] PIPE pipe_operator
    {

    }
    ;
pipe_or_select_clause:
    pipe_start {$$ = $1;}
    | select_clause {$$ = $1;}
    ;
select_clause:
    simple_select {$$ = $1;}
    | select_with_parens {$$=$1;}
    ;



PreparableStmt:
    SelectStmt
    {
        $$ = $1;
    }
    ;
//TODO add missing rules    
CopyStmt: 
    COPY qualified_name copy_from copy_file_name copy_options copy_delimiter
    {
        auto node = mkNode<lingodb::ast::CopyNode>(@$);
        node->copyInfo->table = $qualified_name;
        node->copyInfo->fromFileName = $copy_file_name;
        node->copyInfo->options = $copy_options;
        $$ = node;
    }
    ;
//TODO add missing rules    
copy_from:
    FROM
    ;
//TODO Add missing rules
copy_file_name:
    STRING_VALUE
    {
        $$ = $1;
    }
    ;
//TODO Add missing rules
copy_options:
    copy_opt_list
    {
        $$ = $1;
    }
    ;
copy_opt_list:
    copy_opt_list[list] copy_opt_item
    {
        $list.emplace_back($copy_opt_item);
        $$ = $list;
    }
    | %empty
    {
        $$ = mkList<std::pair<std::string,std::string>>();
    }
    ;    
copy_opt_item: 
    DELIMITER  STRING_VALUE
    {
        std::string str = $STRING_VALUE;
        $$ = std::pair<std::string, std::string>("DELIMITER", str);
    }
    | NULL_P  STRING_VALUE
    {
        std::string str = $STRING_VALUE;
        $$ = std::pair<std::string, std::string>("NULL", str);
    }
    | CSV 
    {
        $$ = std::pair<std::string, std::string>("FORMAT", "csv");
    }
    | ESCAPE STRING_VALUE
    {
        std::string str = $STRING_VALUE;
        $$ = std::pair<std::string, std::string>("ESCAPE", str);
    }
    ;
copy_delimiter:
    DELIMITERS Sconst
    ;
    | %empty
    ;

VariableSetStmt: 
    SET set_rest
    {
        $$ = $set_rest;
    }
    | SET LOCAL set_rest
    | SET SESSION set_rest
    ;
//TODO missing rules
set_rest:
    set_rest_more
    {
        $$ = $1;
    }
    ;
generic_set:
    var_name TO var_list
    {
       $$ = mkNode<lingodb::ast::SetVariableStatement>(@$, $var_name, $var_list);
    }
    | var_name EQUAL var_list
    {
         $$ = mkNode<lingodb::ast::SetVariableStatement>(@$, $var_name, $var_list);
    }
    ;
//TODO add missing rules
set_rest_more:
    generic_set 
    {
        $$ =$1;
    }
    ;

var_name:
    ColId
    {
        $$ = $1;
    }
    | var_name[name] DOT ColId
    {
        $$ = $name + "." + $ColId;
    }
    ;
var_list:
    var_value
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($var_value);
        $$ = list;
    }
    | var_list[list] COMMA var_value
    {
        $list.emplace_back($var_value);
        $$ = $list;
    }
    ;
var_value:
    opt_boolean_or_string
    {
        $$ = $1;
    }
    | NumericOnly 
    {
        $$ = $1;
    }
    ;
opt_boolean_or_string:
    TRUE_P
    {
        auto t = mkNode<lingodb::ast::ConstantExpression>(@$);
        t->value=std::make_shared<lingodb::ast::BoolValue>(true);
        $$=t;
    }
    | FALSE_P
    {
        auto t = mkNode<lingodb::ast::ConstantExpression>(@$);
        t->value=std::make_shared<lingodb::ast::BoolValue>(false);
        $$=t;
    }
    | ON 
    {
        auto t = mkNode<lingodb::ast::ConstantExpression>(@$);
        t->value=std::make_shared<lingodb::ast::StringValue>("on");
        $$=t;
    }
    | NonReservedWord_or_Sconst
    {
        $$ = $1;
    }
    ;
NonReservedWord_or_Sconst:
	NonReservedWord
    { 
        auto t = mkNode<lingodb::ast::ConstantExpression>(@$);
        t->value=std::make_shared<lingodb::ast::StringValue>($1);
        $$=t;
    }
	| Sconst    
    { 
        $$ = $1; 
    }
    ;
NonReservedWord:	
IDENT							{ $$ = $1; }
			| unreserved_keyword					{ $$ = $1; }
			| col_name_keyword						{ $$ = $1; }
			| type_func_name_keyword				{ $$ = $1; }

/*
 * This rule parses SELECT statements that can appear within set operations,
 * including UNION, INTERSECT and EXCEPT.  '(' and ')' can be used to specify
 * the ordering of the set operations.	Without '(' and ')' we want the
 * operations to be ordered per the precedence specs at the head of this file.
 *
 * As with select_no_parens, simple_select cannot have outer parentheses,
 * but can have parenthesized subclauses.
 *
 * It might appear that we could fold the first two alternatives into one
 * by using opt_distinct_clause.  However, that causes a shift/reduce conflict
 * against INSERT ... SELECT ... ON CONFLICT.  We avoid the ambiguity by
 * requiring SELECT DISTINCT [ON] to be followed by a non-empty target_list.
 *
 * Note that sort clauses cannot be included at this level --- SQL requires
 *		SELECT foo UNION SELECT bar ORDER BY baz
 * to be parsed as
 *		(SELECT foo UNION SELECT bar) ORDER BY baz
 * not
 *		SELECT foo UNION (SELECT bar ORDER BY baz)
 * !Likewise for WITH, FOR UPDATE and LIMIT.  Therefore, those clauses are
 * described as part of the select_no_parens production, not simple_select.
 * This does not limit functionality, because you can reintroduce these
 * clauses inside parentheses.
 *
 *NOTE: only the leftmost component SelectStmt should have INTO.
 * !However, this is not checked by the grammar; parse analysis must check it.
 * !Difference to postgres the values clause is located in the select_no_parens rule so that it can be used as a standalone table producer.
 */
simple_select:
    SELECT opt_all_clause opt_target_list
    //TODO into_clause
    opt_from_clause where_clause
    group_clause having_clause //TODO window_clause
    {
        auto t = mkNode<lingodb::ast::SelectNode>(@$);
        t->select_list = $opt_target_list;
        t->where_clause = $where_clause;
        t->from_clause = $opt_from_clause;
        t->groups = $group_clause;
        t->having = $having_clause;
        $$ = t;
    }
    | SELECT distinct_clause target_list into_clause from_clause where_clause group_clause having_clause window_clause
    {
        auto t = mkNode<lingodb::ast::SelectNode>(@$);
        auto target_list = mkNode<lingodb::ast::TargetsExpression>(@$);
        target_list->targets = std::move($target_list);

        target_list->distinctExpressions = $distinct_clause;
        t->select_list = target_list;
        t->where_clause = $where_clause;
        t->from_clause = $from_clause;
        t->groups = $group_clause;
        t->having = $having_clause;
        $$ = t;
    }
    //TODO | TABLE relation_expr
    | select_clause[clause1] UNION set_quantifier select_clause[clause2]
    {
        auto setOpNode = mkNode<lingodb::ast::SetOperationNode>(@$, lingodb::ast::SetOperationType::UNION, $clause1, $clause2);
        setOpNode->setOpAll = $set_quantifier;
        $$ = setOpNode;

    }
    | select_clause[clause1] INTERSECT set_quantifier select_clause[clause2]
    {
        auto setOpNode = mkNode<lingodb::ast::SetOperationNode>(@$, lingodb::ast::SetOperationType::INTERSECT, $clause1, $clause2);
        setOpNode->setOpAll = $set_quantifier;
        $$ = setOpNode;

    }
    | select_clause[clause1] EXCEPT set_quantifier select_clause[clause2]
    {
        auto setOpNode = mkNode<lingodb::ast::SetOperationNode>(@$, lingodb::ast::SetOperationType::EXCEPT, $clause1, $clause2);
        setOpNode->setOpAll = $set_quantifier;
        $$ = setOpNode;
    }
    | values_clause
    {
        $$ = mkNode<lingodb::ast::ValuesQueryNode>(@$, $1);
    }
    ;
//TODO Add missing rules
with_clause:
    WITH cte_list
    {
        $$ = $cte_list;

    }
    | WITH_LA cte_list
    {

    }
    ;
distinct_clause:
    DISTINCT {$$ = mkListShared<lingodb::ast::ParsedExpression>();}
    | DISTINCT ON LP expr_list RP
    {
        $$ = $expr_list;
    }
    ;
cte_list:
    common_table_expr
    {
        $$ = $common_table_expr;
    }
    | cte_list[list] COMMA common_table_expr
    {
        auto current = std::static_pointer_cast<lingodb::ast::CTENode>($list);
        while(current->child != nullptr) {
            current = std::dynamic_pointer_cast<lingodb::ast::CTENode>(current->child);
            if(!current) {
                error(@$, "should not happen");
            }
        }
        current->child = $common_table_expr;
        $$ = $list;

    }
    ;

//TODO missing rules
into_clause:
    %empty
    ;
//TODO missing rules
window_clause:
    %empty
    ;

//TODO more complex rules
common_table_expr:
    name opt_name_list AS opt_materialized LP PreparableStmt RP
    {
        auto cteNode = mkNode<lingodb::ast::CTENode>(@$);
        cteNode->alias = $name;
        cteNode->query = $PreparableStmt;
        cteNode->columnNames = $opt_name_list;
        $$ = cteNode;

    }
;

//TODO Add missing rules
opt_materialized:
    | %empty							{}
    ;
/*****************************************************************************
 *
 *	clauses common to all Optimizable Stmts:
 *		from_clause		- allow list of both JOIN expressions and table names
 *		where_clause	- qualifications for joins or restrictions
 *
 *****************************************************************************/
 opt_from_clause:
			FROM from_list							{ $$=$from_list; }
			| %empty								{  }
            ;


 from_clause:
			FROM from_list							{ $$=$from_list; }
            ;
from_list:
    table_ref { $$=$1;}
    | from_list[list] COMMA table_ref
    {
        auto join = mkNode<lingodb::ast::JoinRef>(@$, lingodb::ast::JoinType::CROSS, lingodb::ast::JoinCondType::CROSS);
        join->left = $list;
        join->right = $table_ref;
        $$ = join;
    }
    ;


/*
 * table_ref is where an alias clause can be attached.
 */
 //TODO add missing rules
table_ref:
    relation_expr opt_alias_clause
    {
        //TODO Alias clause
        //TODO schema
        //TODO for now it is very simplyfied
        lingodb::ast::TableDescription desc{"", "", $relation_expr };
        auto tableref = mkNode<lingodb::ast::BaseTableRef>(@$, desc);
        tableref->alias = $opt_alias_clause.first;
        $$ = tableref;

    }
    | joined_table { $$ = $1;}
    | LP joined_table RP alias_clause
    | joined_table opt_alias_clause
    | select_with_parens opt_alias_clause
    {
        //TODO
        auto subquery = mkNode<lingodb::ast::SubqueryRef>(@$, std::static_pointer_cast<lingodb::ast::SelectNode>($select_with_parens));
        subquery->alias = $opt_alias_clause.first;
        subquery->columnNames = $opt_alias_clause.second;
        $$ = subquery;
    }


    ;


set_quantifier:
    ALL
    {
        $$ = true;
    }
    | DISTINCT
    | %empty
    {
        $$ = false;
    }
    ;

/* Postgres
 * This syntax for group_clause tries to follow the spec quite closely.
 * However, the spec allows only column references, not expressions,
 * which introduces an ambiguity between implicit row constructors
 * (a,b) and lists of column references.
 *
 * We handle this by using the a_expr production for what the spec calls
 * <ordinary grouping set>, which in the spec represents either one column
 * reference or a parenthesized list of column references. Then, we check the
 * top node of the a_expr to see if it's an implicit RowExpr, and if so, just
 * grab and use the list, discarding the node. (this is done in parse analysis,
 * not here)
 *
 * (we abuse the row_format field of RowExpr to distinguish implicit and
 * explicit row constructors; it's debatable if anyone sanely wants to use them
 * in a group clause, but if they have a reason to, we make it possible.)
 *
 * Each item in the group_clause list is either an expression tree or a
 * GroupingSet node of some type.
 */
group_clause:
    GROUP_P BY set_quantifier group_by_list
    {
        auto node = mkNode<lingodb::ast::GroupByNode>(@$);
        node->groupByExpressions = $group_by_list;
        $$ = node;
        //TODO Support set_quantifier
    }
    //TODO find a better solution
    | GROUP_P BY set_quantifier rollup_clause %prec ROLLUP_PRIORITY
    {
        auto node = mkNode<lingodb::ast::GroupByNode>(@$);
        node->groupByExpressions = $rollup_clause;
        node->rollup = true;
        $$ = node;
    }
    | %empty
    ;
group_clause_with_alias:
    GROUP_P BY set_quantifier group_by_list_with_alias
    {
        auto node = mkNode<lingodb::ast::GroupByNode>(@$);
        node->groupByExpressions = $group_by_list_with_alias;
        $$ = node;
        //TODO Support set_quantifier
    }
    //TODO find a better solution
    | GROUP_P BY set_quantifier rollup_clause %prec ROLLUP_PRIORITY
    {
        auto node = mkNode<lingodb::ast::GroupByNode>(@$);
        node->groupByExpressions = $rollup_clause;
        node->rollup = true;
        $$ = node;
    }
    | %empty
    ;


group_by_list:
    group_by_item
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($group_by_item);
        $$ = list;
    }
    | group_by_list[list] COMMA group_by_item
    {
        $list.emplace_back($group_by_item);
        $$ = $list;
    }
    ;
group_by_list_with_alias:
    group_by_item_with_alias
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($group_by_item_with_alias);
        $$ = list;
    }
    | group_by_list[list] COMMA group_by_item_with_alias
    {
        $list.emplace_back($group_by_item_with_alias);
        $$ = $list;
    }
    ;
rollup_clause:
    ROLLUP LP expr_list RP
    {
        $$ = $expr_list;
    }
//TODO Add missing rules
group_by_item:
    a_expr {$$ = $1;}
    | empty_grouping_set {}
    /*| cube_clause */
    /*| grouping_sets_clause*/
    ;
group_by_item_with_alias:
    a_expr {$$ = $1;}
    | a_expr AS ColId {$1->alias = $ColId; $$ = $1;}
    | empty_grouping_set {}
    /*| cube_clause */
    /*| grouping_sets_clause*/
    ;
empty_grouping_set:
    LP RP
    ;

having_clause:
    HAVING a_expr {$$=$a_expr;}
    | %empty
    ;


for_locking_clause:
    for_locking_items
    | FOR READ ONLY
    ;
for_locking_items:
    for_locking_item
    | for_locking_items for_locking_item
    ;
for_locking_item:
        for_locking_strength locked_rels_list opt_nowait_or_skip
    ;
for_locking_strength:
    FOR UPDATE
    | FOR NO KEY UPDATE
    | FOR SHARE
    | FOR KEY SHARE
    ;
locked_rels_list:
    OF qualified_name_list
    | %empty
    ;
/* Postgres
 * It may seem silly to separate joined_table from table_ref, but there is
 * method in SQL's madness: if you don't do it this way you get reduce-
 * reduce conflicts, because it's not clear to the parser generator whether
 * to expect alias_clause after ')' or not.  For the same reason we must
 * treat 'JOIN' and 'join_type JOIN' separately, rather than allowing
 * join_type to expand to empty; if we try it, the parser generator can't
 * figure out when to reduce an empty join_type right after table_ref.
 *
 * Note that a CROSS JOIN is the same as an unqualified
 * INNER JOIN, and an INNER JOIN/ON has the same shape
 * but a qualification expression to limit membership.
 * A NATURAL JOIN implicitly matches column names between
 * tables and the shape is determined by which columns are
 * in common. We'll collect columns during the later transformations.
 */
joined_table:
    LP joined_table RP {$$=$2;}
    | table_ref CROSS JOIN table_ref
    | table_ref[left] join_type JOIN table_ref[right] join_qual
    {
        auto join = mkNode<lingodb::ast::JoinRef>(@$, $join_type, lingodb::ast::JoinCondType::REGULAR);
        join->left = $left;
        join->right = $right;
        join->condition = $join_qual;
        $$ = join;

    }
    | table_ref[left] JOIN table_ref[right] join_qual
    {
        //TODO find out correct JoinCondType
        auto join = mkNode<lingodb::ast::JoinRef>(@$, lingodb::ast::JoinType::INNER, lingodb::ast::JoinCondType::REGULAR);
        join->left = $left;
        join->right = $right;
        join->condition = $join_qual;
        $$ = join;
    }
    | table_ref NATURAL join_type JOIN table_ref
    | table_ref NATURAL JOIN table_ref
    ;

alias_clause:
    AS ColId LP name_list RP
    {
        $$ = std::make_pair($ColId, $name_list);
    }
    | AS ColId
    {
        $$ = std::make_pair($ColId, std::vector<std::string>());
    }
    | ColId LP name_list RP
    {
        $$ = std::make_pair($ColId, $name_list);
    }
    | ColId
    {
        $$ = std::make_pair($ColId, std::vector<std::string>());
    } //TODO Check if correct
    ;


//TODO AST
opt_nulls_order:
    NULLS_LA FIRST_P
    | NULLS_LA LAST_P
    | %empty
    {
        $$ = lingodb::ast::OrderByNullType::ORDER_DEFAULT;
    }

opt_alias_clause:
    alias_clause
    {
        $$ = $alias_clause;
    }
    | %empty 
    {
        $$ = std::make_pair<std::string, std::vector<std::string>>("", std::vector<std::string>());
    }
    ;

opt_alias_clause_for_join_using:
    AS ColId
    | %empty
    ;

opt_asc_desc:
    ASC
    {
        $$ = lingodb::ast::OrderType::ASCENDING;
    }
    | DESC
    {
        $$ = lingodb::ast::OrderType::DESCENDING;
    }
    | %empty {
        $$ = lingodb::ast::OrderType::ASCENDING;
    }
    ;
opt_nowait_or_skip:
    NOWAIT
    | SKIP LOCKED
    | %empty
    ;

join_type:
    FULL
    | FULL OUTER_P
    {
        $$ = lingodb::ast::JoinType::FULL;
    }
    | LEFT OUTER_P
    {
        $$ = lingodb::ast::JoinType::LEFT;
    }
    | LEFT
    {
        $$ = lingodb::ast::JoinType::LEFT;
    }
    | RIGHT
    {
        $$ = lingodb::ast::JoinType::RIGHT;
    }
    | RIGHT OUTER_P
    {
        $$ = lingodb::ast::JoinType::RIGHT;
    }
    | INNER_P
    {
        $$ = lingodb::ast::JoinType::INNER;
    }
    ;

/* JOIN qualification clauses
 * Possibilities are:
 *	USING ( column list ) [ AS alias ]
 *						  allows only unqualified column names,
 *						  which must match between tables.
 *	ON expr allows more general qualifications.
 *
 * We return USING as a two-element List (the first item being a sub-List
 * of the common column names, and the second either an Alias item or NULL).
 * An ON-expr will not be a List, so it can be told apart that way.
 */

join_qual:
    USING LP name_list RP //TODO not allowing alias after USING for now opt_alias_clause_for_join_using
    {
        auto name_list = $name_list;
        auto list = mkListShared<lingodb::ast::ColumnRefExpression>();
        for(auto& name : name_list) {
            list.emplace_back(mkNode<lingodb::ast::ColumnRefExpression>(@$,name));
        }
        $$ = list;

    }
    | ON a_expr {$$=$a_expr;}
    ;

relation_expr:
    qualified_name {$$ = $qualified_name;}
    | extended_relation_expr
    ;

extended_relation_expr:
    qualified_name STAR
    | ONLY qualified_name
    | ONLY LP qualified_name RP
    ;
opt_asymmetric:
    ASYMMETRIC {$$=true;}
    | %empty {$$=false;}
    ;
opt_all_clause:
    ALL
    | %empty
    ;
opt_sort_clause:
    sort_clause {$$ = $1;}
    | %empty {$$ = std::nullopt;}
    ;
sort_clause:
    ORDER BY sortby_list
    {
        auto n = mkNode<lingodb::ast::OrderByModifier>(@$);
        n->orderByElements = std::move($sortby_list);
        $$ = n;
    }
    ;
sortby_list:
    sortby
    {
        auto list = mkListShared<lingodb::ast::OrderByElement>();
        list.emplace_back($sortby);
        $$ = list;
    }
    | sortby_list[list] COMMA sortby
    {
        $list.emplace_back($sortby);
        $$ = $list;
    }
    ;
sortby:
    //TODO
    a_expr USING qual_all_Op opt_nulls_order
    | a_expr opt_asc_desc opt_nulls_order
    {
        auto orderByElement = mkNode<lingodb::ast::OrderByElement>(@$, $opt_asc_desc, $opt_nulls_order);
        orderByElement->expression = $a_expr;
        $$ = orderByElement;
    }
    ;



select_limit:
    limit_clause offset_clause
    {
        $limit_clause->offset = $offset_clause;
        $$ = $limit_clause;
    }
    | offset_clause limit_clause
    {
        $limit_clause->offset = $offset_clause;
        $$ = $limit_clause;
    }
    | limit_clause {$$ = $1;}
    | offset_clause
    ;

opt_select_limit:
    select_limit {$$=$1;}
    | %empty {$$=std::nullopt;}
    ;
//TODO missing rules
limit_clause:
    LIMIT select_limit_value {$$ = mkNode<lingodb::ast::LimitModifier>(@$, $select_limit_value);}
    | LIMIT select_limit_value COMMA select_offset_value
    ;
//TODO missing rules
offset_clause:
    OFFSET select_offset_value
    {
        $$ = $2;
    }
    ;

select_limit_value:
    a_expr {$$=$1;}
    | ALL
    ;

select_offset_value:
    a_expr {$$=$1;}
    ;

/*
 * We should allow ROW '(' expr_list ')' too, but that seems to require
 * making VALUES a fully reserved word, which will probably break more apps
 * than allowing the noise-word is worth.
 */
 values_clause:
    VALUES LP expr_list RP
    {
        //ExpressionListRef is a TableRef and therefore a TableProducer and therefore can be used stand alone
        auto exprListList = mkList<std::vector<std::shared_ptr<lingodb::ast::ParsedExpression>>>();
        exprListList.emplace_back($expr_list);
        auto exprListRef = mkNode<lingodb::ast::ExpressionListRef>(@$, exprListList);
        $$ = exprListRef;

    }
    | values_clause[clause] COMMA LP expr_list RP
    {
        std::static_pointer_cast<lingodb::ast::ExpressionListRef>($clause)->values.emplace_back($expr_list);
        $$ = $clause;
    }
    ;

    ;


/*****************************************************************************
 *
 *	expression grammar
 *
 *****************************************************************************/
 //TODO Add missing expressions, for instance func_expr
/*
 * POSGRES
 * General expressions
 * This is the heart of the expression syntax.
 *
 * We have two expression types: a_expr is the unrestricted kind, and
 * b_expr is a subset that must be used in some places to avoid shift/reduce
 * conflicts.  For example, we can't do BETWEEN as "BETWEEN a_expr AND a_expr"
 * because that use of AND conflicts with AND as a boolean operator.  So,
 * b_expr is used in BETWEEN and we remove boolean keywords from b_expr.
 *
 * Note that '(' a_expr ')' is a b_expr, so an unrestricted expression can
 * always be used by surrounding it with parens.
 *
 * c_expr is all the productions that are common to a_expr and b_expr;
 * it's factored out just to eliminate redundant coding.
 *
 * Be careful of productions involving more than one terminal token.
 * By default, bison will assign such productions the precedence of their
 * last terminal, but in nearly all cases you want it to be the precedence
 * of the first terminal instead; otherwise you will not get the behavior
 * you expect!  So we use %prec annotations freely to set precedences.
 */

where_clause:
    WHERE a_expr {$$=$a_expr;}
    | %empty
    ;
/*
TODO
 * Add missing rules
*/
a_expr:
    c_expr { $$ = $c_expr;}

    //TODO | a-expr COLLATE any_name
    //TODO | a_expr AT TIME ZONE a_expr
    //TODO | a_expr AT LOCAL
    | a_expr PLUS a_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_PLUS, $1,$3);
    }
    | a_expr[e] TYPECAST Type
    {
     $$ = mkNode<lingodb::ast::CastExpression>(@$, $Type, $e);
    }
    | a_expr MINUS a_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_MINUS, $1,$3);
    }
    | a_expr STAR a_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_TIMES, $1,$3);
    }
    | a_expr SLASH a_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_DIVIDE, $1,$3);
    }
    | a_expr PERCENT a_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_MOD, $1,$3);
    }
    | a_expr HAT a_expr
    | a_expr LESS a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_LESSTHAN, $1, $3 );
    }
    | a_expr GREATER a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_GREATERTHAN, $1, $3 );
    }
    | a_expr EQUAL a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_EQUAL, $1, $3 );
    }
    | a_expr LESS_EQUAL a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_LESSTHANOREQUALTO, $1, $3 );
    }
    | a_expr GREATER_EQUAL a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_GREATERTHANOREQUALTO, $1, $3 );
    }
    | a_expr NOT_EQUAL a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_NOTEQUAL, $1, $3 );
    }
    | and_a_expr
    {
       $$ = $1;
    }
    | or_a_expr
    {
        $$ = $1;
    }
    | NOT a_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_NOT, $2);
    }
    | a_expr LIKE a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_LIKE, $1, $3);
    }
    | a_expr NOT LIKE a_expr
    {
        $$ = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_NOT_LIKE, $1, $4);
    }
    | a_expr IN_P in_expr
    {
        /*
        * in_expr either returns a SubQuery or a list of expressions
        */
        if(std::holds_alternative<std::shared_ptr<lingodb::ast::SubqueryExpression>>($in_expr)) {
            auto subQuery = std::get<std::shared_ptr<lingodb::ast::SubqueryExpression>>($in_expr);
            subQuery->testExpr = $1;
            $$ = subQuery;

        } else {
            auto exprList = std::get<std::vector<std::shared_ptr<lingodb::ast::ParsedExpression>>>($in_expr);
            auto node = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_IN, $1, exprList);
            $$ = node;
        }
    }
    | a_expr NOT IN_P in_expr
    {
        /*
        * in_expr either returns a SubQuery or a list of expressions
        */
        if(std::holds_alternative<std::shared_ptr<lingodb::ast::SubqueryExpression>>($in_expr)) {
            auto subQuery = std::get<std::shared_ptr<lingodb::ast::SubqueryExpression>>($in_expr);
            subQuery->subQueryType = lingodb::ast::SubqueryType::NOT_ANY;
            subQuery->testExpr = $1;
            $$ = subQuery;

        } else {
            auto exprList = std::get<std::vector<std::shared_ptr<lingodb::ast::ParsedExpression>>>($in_expr);
            auto node = mkNode<lingodb::ast::ComparisonExpression>(@$, lingodb::ast::ExpressionType::COMPARE_NOT_IN, $1, exprList);
            $$ = node;
        }
    }
    | a_expr[input] BETWEEN opt_asymmetric b_expr[lower] AND a_expr[upper] %prec BETWEEN
    {
        auto node = mkNode<lingodb::ast::BetweenExpression>(@$, lingodb::ast::ExpressionType::COMPARE_BETWEEN, $input, $lower, $upper);
        node->asymmetric = $opt_asymmetric;
        $$ = node;
    }
    | a_expr[input] NOT BETWEEN opt_asymmetric b_expr[lower] AND a_expr[upper] %prec BETWEEN
    {
        auto node = mkNode<lingodb::ast::BetweenExpression>(@$, lingodb::ast::ExpressionType::COMPARE_NOT_BETWEEN, $input, $lower, $upper);
        node->asymmetric = $opt_asymmetric;
        $$ = node;
    }
    | a_expr IS NULL_P
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_IS_NULL, $1);
    }
    | a_expr IS NOT NULL_P
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_IS_NOT_NULL, $1);
    }
    | a_expr[left] qual_Op a_expr[right] %prec QUAL_OP
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, $2, $left, $right);
    }
    | qual_Op a_expr[right]  %prec QUAL_OP
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, $qual_Op, nullptr, $right);
    }
    | a_expr[left] basicComparisonType sub_type select_with_parens  %prec Op
    {
       auto subQueryExpression = mkNode<lingodb::ast::SubqueryExpression>(@$, $sub_type, std::static_pointer_cast<lingodb::ast::SelectNode>($select_with_parens));
       subQueryExpression->testExpr = $left;
       subQueryExpression->comparisonType = $basicComparisonType;
       $$ = subQueryExpression;
    }
    ;
and_a_expr:
    a_expr AND a_expr
    {
       $$ = mkNode<lingodb::ast::ConjunctionExpression>(@$, lingodb::ast::ExpressionType::CONJUNCTION_AND , $1, $3);
    }
    ;
or_a_expr:
    a_expr OR a_expr
    {
       $$ = mkNode<lingodb::ast::ConjunctionExpression>(@$, lingodb::ast::ExpressionType::CONJUNCTION_OR , $1, $3);
    }
    ;

b_expr:
    c_expr { $$ = $c_expr;}
    | b_expr PLUS b_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_PLUS, $1,$3);
    }
    | b_expr MINUS b_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_MINUS, $1,$3);
    }
    | b_expr STAR b_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_TIMES, $1,$3);
    }
    | b_expr SLASH b_expr
    {
        $$ = mkNode<lingodb::ast::OperatorExpression>(@$, lingodb::ast::ExpressionType::OPERATOR_DIVIDE, $1,$3);
    }
    ;
/*
 * Productions that can be used in both a_expr and b_expr.
 * Note: productions that refer recursively to a_expr or b_expr mostly cannot appear here.
 */
c_expr:
    columnref {$$ = $columnref;}
    | AexprConst {$$=$1;}
    //TODO | PARAM opt_indirection
    //TODO
    | LP a_expr RP {$$=$2;}//opt_indirection
    | case_expr
    {
        $$ = $1;
    }
    | func_expr {$$=$1;}
    | window_func_expr {$$=$1;}
    | cast_expr {$$=$1;}
    | classic_select_and_pipe_sql_with_parens %prec UMINUS
    {
        auto subquery = mkNode<lingodb::ast::SubqueryExpression>(@$, lingodb::ast::SubqueryType::SCALAR, $classic_select_and_pipe_sql_with_parens);
        $$ = subquery;
    }
    //TODO | select_with_parens indirection
    | EXISTS classic_select_and_pipe_sql_with_parens
    {
        auto subquery = mkNode<lingodb::ast::SubqueryExpression>(@$, lingodb::ast::SubqueryType::EXISTS, $classic_select_and_pipe_sql_with_parens);
        $$ = subquery;
    }
    | NOT EXISTS classic_select_and_pipe_sql_with_parens
    {
        auto subquery = mkNode<lingodb::ast::SubqueryExpression>(@$, lingodb::ast::SubqueryType::NOT_EXISTS, $classic_select_and_pipe_sql_with_parens);
        $$ = subquery;
    }
    //TODO | ARRAY select_with_parens
    //TODO | ARRAY array_expr
    //TODO | explicit_row
    //TODO | implicit_row
    //TODO | GROUPING LP expr_list RP

    ;
basicComparisonType:
    EQUAL
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_EQUAL;
    }
    | NOT_EQUAL
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_NOTEQUAL;
    }
    | LESS
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_LESSTHAN;
    }
    | GREATER
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_GREATERTHAN;
    }
    | LESS_EQUAL
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_LESSTHANOREQUALTO;
    }
    | GREATER_EQUAL
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
    }
    | LIKE
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_LIKE;
    }
    | NOT LIKE
    {
        $$ = lingodb::ast::ExpressionType::COMPARE_NOT_LIKE;
    }
    ;
//TODO Alias -> opt_alias_clause
in_expr:
    select_with_parens
    {
        auto subquery = mkNode<lingodb::ast::SubqueryExpression>(@$, lingodb::ast::SubqueryType::ANY, $select_with_parens);
        $$ = subquery;

    }
    | LP expr_list RP
    {
        $$ = $expr_list;
    }
    ;
case_expr:
    CASE case_arg when_clause_list case_default END_P
    {
        auto caseExpr = mkNode<lingodb::ast::CaseExpression>(@$, $case_arg, $when_clause_list, $case_default);
        $$ = caseExpr;
    }
    ;
when_clause_list:
    when_clause
    {
        auto list = mkList<lingodb::ast::CaseExpression::CaseCheck>();
        list.emplace_back($when_clause);
        $$ = list;

    }
    | when_clause_list[list] when_clause
    {
        $list.emplace_back($when_clause);
        $$ = $list;
    }
    ;

when_clause:
    WHEN a_expr THEN a_expr
    {
        auto whenCheck = lingodb::ast::CaseExpression::CaseCheck($2,$4);
        $$ = whenCheck;
    }
    ;

case_default:
    ELSE a_expr
    {
        $$ = $2;
    }
    | %empty
    ;

case_arg:
    a_expr
    {
        $$ = $1;
    }
    | %empty {$$=std::nullopt;}
    ;

columnref_list:
    columnref
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($columnref);
        $$ = list;
    }
    | columnref_list[list] COMMA columnref
    {
        $list.emplace_back($columnref);
        $$ = $list;
    }
    ;

columnref:
    ColId {$$ = mkNode<lingodb::ast::ColumnRefExpression>(@$, $ColId);}
    | ColId indirection {
        auto in = $indirection;
        if(in->exprClass == lingodb::ast::ExpressionClass::COLUMN_REF) {
            auto columnref = std::static_pointer_cast<lingodb::ast::ColumnRefExpression>(in);
            auto newColumnRef = mkNode<lingodb::ast::ColumnRefExpression>(@$, $ColId);
            newColumnRef->columnNames.insert(newColumnRef->columnNames.end(), columnref->columnNames.begin(), columnref->columnNames.end() );
            $$ = newColumnRef;


        } else if(in->exprClass == lingodb::ast::ExpressionClass::STAR) {
            auto star = std::static_pointer_cast<lingodb::ast::StarExpression>(in);
            star->relationName = $ColId;
            $$ = star;
        }

    } //TODO Add table name
    ;
func_application:
    func_name LP RP
    {
        std::string catalog("");
        std::string schema("");
        std::string functionName = $func_name;
        bool isOperator = false;
        bool distinct = false;
        bool exportState = false;
        auto funcExpr = mkNode<lingodb::ast::FunctionExpression>(@$, catalog, schema, functionName, isOperator, distinct, exportState);
        $$ = funcExpr;
    }
    | func_name LP func_arg_list opt_sort_clause RP
    {
        std::string catalog("");
        std::string schema("");
        std::string functionName = $func_name;
        bool isOperator = false;
        bool distinct = false;
        bool exportState = false;
        auto funcExpr = mkNode<lingodb::ast::FunctionExpression>(@$, catalog, schema, functionName, isOperator, distinct, exportState);

        funcExpr->arguments = $func_arg_list;

        $$ = funcExpr;
    }
    | func_name LP VARIADIC func_arg_expr opt_sort_clause RP
    | func_name LP func_arg_list COMMA VARIADIC func_arg_expr opt_sort_clause RP
    | func_name LP ALL func_arg_list opt_sort_clause RP
    | func_name LP DISTINCT func_arg_list opt_alias_clause RP
    {
        std::string catalog("");
        std::string schema("");
        std::string functionName = $func_name;
        bool isOperator = false;
        bool distinct = true;
        bool exportState = false;
        auto funcExpr = mkNode<lingodb::ast::FunctionExpression>(@$, catalog, schema, functionName, isOperator, distinct, exportState);

        funcExpr->arguments = $func_arg_list;

        $$ = funcExpr;
    }
    | func_name LP STAR RP
    {
        std::string catalog("");
        std::string schema("");
        std::string functionName = $func_name;
        bool isOperator = false;
        bool distinct = false;
        bool exportState = false;
        auto funcExpr = mkNode<lingodb::ast::FunctionExpression>(@$, catalog, schema, functionName, isOperator, distinct, exportState);


        auto l = mkListShared<lingodb::ast::ParsedExpression>();
        funcExpr->star = true;
        funcExpr->arguments = l;
        $$ = funcExpr;
    }
;

 //TODO add missing rules
 func_expr:
    func_application //within_group_clause filter_clause
    {
        //TODO within_group_clause filter_clause over_clause
        $$ = $func_application;
    }


    | func_expr_common_subexpr
    {
        $$ = $1;
    }
    ;
window_func_expr:
    func_application over_clause
    {
        $over_clause->functionExpression = $func_application;
        $$=$over_clause;

    }
    ;

func_arg_list_opt:
    func_arg_list
    {
        $$ = $1;
    }
    | %empty
    {
        //TODO
    }
    ;
/* function arguments can have names */
func_arg_list:
    func_arg_expr
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($func_arg_expr);
        $$ = list;
    }
    | func_arg_list[list] COMMA func_arg_expr
    {
        $list.emplace_back($func_arg_expr);
        $$= $list;
    }
    ;
expr_list:
    a_expr
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($a_expr);
        $$ = list;
    }
    | expr_list[list] COMMA a_expr
    {
        $list.emplace_back($a_expr);
        $$ = $list;
    }
    ;
expr_list_with_alias:
    a_expr opt_alias_clause
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        $a_expr->alias = $opt_alias_clause.first;
        list.emplace_back($a_expr);
        $$ = list;
    }
    | expr_list[list] COMMA a_expr opt_alias_clause
    {
        $a_expr->alias = $opt_alias_clause.first;
        $list.emplace_back($a_expr);
        $$ = $list;
    }
    ;
//TODO Allow for param_name
func_arg_expr:
    a_expr {$$=$1;}
    | param_name COLON_EQUALS a_expr
    | param_name GREATER_EQUAL a_expr
    ;

//TODO missing rules
/*
 * Special expressions that are considered to be functions.
 */
func_expr_common_subexpr:
    EXTRACT LP extract_list RP
    {
        auto function  = mkNode<lingodb::ast::FunctionExpression>(@$, "", "", "EXTRACT", false, false, false);
        function->arguments = $extract_list;

        $$ = function;
    }
    | SUBSTRING LP substr_list RP
    {
        auto function = mkNode<lingodb::ast::FunctionExpression>(@$, "", "", "SUBSTRING", false, false, false);
        function->arguments = $substr_list;
        $$ = function;
    }
    | SUBSTRING LP func_arg_list_opt RP
    {
        auto function = mkNode<lingodb::ast::FunctionExpression>(@$, "", "", "SUBSTRING", false, false, false);
        function->arguments = $func_arg_list_opt;
        $$ = function;
    }
;

cast_expr:
    CAST LP a_expr AS Typename RP
    {
        $$ = mkNode<lingodb::ast::CastExpression>(@$, $Typename, $a_expr);
    }
    ;

extract_list:
    extract_arg FROM a_expr
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($extract_arg);
        list.emplace_back($a_expr);
        $$ = list;
    }
    ;
extract_arg:
    YEAR_P
    {
        auto constant = mkNode<lingodb::ast::ConstantExpression>(@$);
        constant->value = std::make_shared<lingodb::ast::StringValue>("year");
        $$ = constant;
    }
    | MONTH_P
    {
        auto constant = mkNode<lingodb::ast::ConstantExpression>(@$);
        constant->value = std::make_shared<lingodb::ast::StringValue>("month");
        $$ = constant;
    }
    | DAY_P
    {
        auto constant = mkNode<lingodb::ast::ConstantExpression>(@$);
        constant->value = std::make_shared<lingodb::ast::StringValue>("day");
        $$ = constant;
    }
    | MINUTE_P
    {
        auto constant = mkNode<lingodb::ast::ConstantExpression>(@$);
        constant->value = std::make_shared<lingodb::ast::StringValue>("minute");
        $$ = constant;
    }
    ;
//TODO missing rules
substr_list:
    a_expr FROM a_expr FOR a_expr
    {
        auto list = mkListShared<lingodb::ast::ParsedExpression>();
        list.emplace_back($1);
        list.emplace_back($3);
        list.emplace_back($5);
        $$ = list;

    }
    | a_expr FOR a_expr FROM a_expr
    {

    }
    | a_expr FROM a_expr
    {

    }
    | a_expr FOR a_expr
    {

    }
    ;
//TODO missing rules
over_clause:
    OVER window_specification
    {
        $$ = $window_specification;
    }
    | OVER ColId
    | %empty
    ;
//TODO missing
window_specification:
    LP opt_existing_window_name opt_partition_clause opt_sort_clause opt_frame_clause RP
    {
        auto windowExpression = mkNode<lingodb::ast::WindowExpression>(@$);
        windowExpression->partitions = $opt_partition_clause;
        windowExpression->order = $opt_sort_clause;
        windowExpression->windowBoundary = $opt_frame_clause;
        $$ = windowExpression;


    }
    ;


//! TODO For what exactly is this here
indirection:
    indirection_el { $$=$1;}
    | indirection indirection_el {$$=$1;}
    ;
indirection_el:
    DOT attr_name {$$=mkNode<lingodb::ast::ColumnRefExpression>(@$, $attr_name);}
    | DOT STAR { $$=mkNode<lingodb::ast::StarExpression>(@$, "");}
    | LB a_expr RB
   //TODO | LB opt_slice_bound ':' opt_slice_bound RB


/*****************************************************************************
 *
 *	target list for SELECT
 *
 *****************************************************************************/
opt_target_list:
    target_list
    {
        auto node = mkNode<lingodb::ast::TargetsExpression>(@$);
        node->targets = std::move($target_list);
        $$ = node;
    }
    | %empty
    ;
target_list:
    target_el { auto list = mkListShared<lingodb::ast::ParsedExpression>(); list.emplace_back($target_el); $$=list;}
    | target_list[list] COMMA target_el { $list.emplace_back($target_el); $$=$list;}
    ;
target_el:
    a_expr AS ColLabel {  $a_expr->alias = $ColLabel; $$ = $a_expr;}
    | a_expr BareColLabel {$a_expr->alias = $BareColLabel; $$ = $a_expr;}
    | a_expr { $$=$a_expr;}
    | STAR {  $$ =mkNode<lingodb::ast::StarExpression>(@$,"");  }
    ;


/*
 * If we see PARTITION, RANGE, ROWS or GROUPS as the first token after the '('
 * of a window_specification, we want the assumption to be that there is
 * no existing_window_name; but those keywords are unreserved and so could
 * be ColIds.  We fix this by making them have the same precedence as IDENT
 * and giving the empty production here a slightly higher precedence, so
 * that the shift/reduce conflict is resolved in favor of reducing the rule.
 * These keywords are thus precluded from being an existing_window_name but
 * are not reserved for any other purpose.
 */
opt_existing_window_name:
    ColId						{  }
	| %empty				%prec Op		{  }
	;

opt_partition_clause:
    PARTITION BY expr_list		{ $$=$expr_list; }
	| %empty				    { $$ = mkListShared<lingodb::ast::ParsedExpression>(); }
		;
/*
 * For frame clauses, we return a WindowDef, but only some fields are used:
 * frameOptions, startOffset, and endOffset.
 */
 //TODO missing rules
opt_frame_clause:
    | RANGE frame_extent opt_window_exclusion_clause
    {
        $frame_extent->windowMode = lingodb::ast::WindowMode::RANGE;
        $$ = $frame_extent;
    }
    | ROWS frame_extent opt_window_exclusion_clause
    {
        $frame_extent->windowMode = lingodb::ast::WindowMode::ROWS;
        $$ = $frame_extent;
    }
    | GROUPS frame_extent opt_window_exclusion_clause
    {
        $frame_extent->windowMode = lingodb::ast::WindowMode::GROUPS;
        $$ = $frame_extent;
    }
    | %empty
    {

    }
    ;

frame_extent:
    frame_bound
    {
        $$ = $1;
    }
    | BETWEEN frame_bound[start] AND frame_bound[end]
    {
        $start->end = $end->start;
        $start->endExpr = $end->startExpr;
        $$ = $start;

    }
    ;

frame_bound:
    UNBOUNDED PRECEDING
    {
        $$ = mkNode<lingodb::ast::WindowBoundary>(@$, lingodb::ast::WindowBoundaryType::UNBOUNDED_PRECEDING );
    }
    | UNBOUNDED FOLLOWING
    {
        $$ = mkNode<lingodb::ast::WindowBoundary>(@$, lingodb::ast::WindowBoundaryType::UNBOUNDED_FOLLOWING );
    }
    | CURRENT_P ROW
    {
        $$ = mkNode<lingodb::ast::WindowBoundary>(@$, lingodb::ast::WindowBoundaryType::CURRENT_ROW );
    }
    | a_expr PRECEDING
    {
        $$ = mkNode<lingodb::ast::WindowBoundary>(@$, lingodb::ast::WindowBoundaryType::EXPR_PRECEDING, $a_expr );
    }
    | a_expr FOLLOWING
    {
        $$ = mkNode<lingodb::ast::WindowBoundary>(@$, lingodb::ast::WindowBoundaryType::EXPR_FOLLOWING, $a_expr);
    }
    ;
//TODO misisng rules
opt_window_exclusion_clause:
    | %empty
    ;
//TODO add missing rules
any_operator:
    all_Op
    {
        $$ = $1;
    }
   // | ColId DOT any_operator
    ;

qual_Op:
    all_Op {$$=$1;}
    //TODO | OPERATOR LP any_operator RP
    ;

qual_all_Op:
    all_Op
    | OPERATOR LP any_operator RP
    ;
//TODO missing rules
subquery_op:
    all_Op
    {
        $$ = $all_Op;
    }
    | OPERATOR LP any_operator RP
    {
        $$ = $any_operator;
    }
    ;

sub_type:
    ANY
    {
        $$ = lingodb::ast::SubqueryType::ANY;
    }
    | ALL
    {
        $$ = lingodb::ast::SubqueryType::ALL;
    }
    | SOME
    {
        $$ = lingodb::ast::SubqueryType::ANY;
    }
    ;

all_Op:
    Op {$$=$1;}

    ;
MathOp:
    PLUS
    | MINUS
    | STAR
    | SLASH
    | PERCENT
    | HAT
    | LESS
    | GREATER
    | LESS_EQUAL
    | GREATER_EQUAL
    | NOT_EQUAL
    | EQUAL
    ;

/*
 * Name classification hierarchy.
 *
 * IDENT is the lexeme returned by the lexer for identifiers that match
 * no known keyword.  In most cases, we can accept certain keywords as
 * names, not only IDENTs.	We prefer to accept as many such keywords
 * as possible to minimize the impact of "reserved words" on programmers.
 * So, we divide names into several possible classes.  The classification
 * is chosen in part to make keywords acceptable as names wherever possible.
 */

/* Column identifier --- names that can be column, table, etc names.
 */
 
ColId:
    IDENTIFIER {$$=$1;}
    | unreserved_keyword {$$=$1;}
    | col_name_keyword {$$=$1;}
   ;

/* Type/function identifier --- names that can be type or function names.*/
type_function_name: 
    IDENTIFIER {$$=$1;}
    | unreserved_keyword {$$=$1;}
    | type_func_name_keyword 
    | DATE_P {$$="date";}

type_func_name_keyword:
			  AUTHORIZATION
			| BINARY
			| COLLATION
			| CONCURRENTLY
			| CROSS
			| CURRENT_SCHEMA
			| FREEZE
			| FULL
			| ILIKE
			| INNER_P
			| IS
			| ISNULL
			| JOIN
			| LEFT
			| LIKE
			| NATURAL
			| NOTNULL
			| OUTER_P
			| OVERLAPS
			| RIGHT
			| SIMILAR
			| TABLESAMPLE
			| VERBOSE
            ;


/* Column label --- allowed labels in "AS" clauses.
 * This presently includes *all* Postgres keywords.
 */
ColLabel:
    IDENTIFIER									{ $$=$1; }
	| unreserved_keyword					{ $$ = $1;}
	| col_name_keyword						{ $$=$1; }
	//TODO | type_func_name_keyword				{ }
	| reserved_keyword						{ $$ = $1; }
	;
/* Bare column label --- names that can be column labels without writing "AS".
 * This classification is orthogonal to the other keyword categories.
 */
BareColLabel:
    IDENTIFIER								{ $$=$1; }
	//TODO | bare_label_keyword					{  }
	;






/* Reserved keyword --- these keywords are usable only as a ColLabel.
 *
 * Keywords appear here if they could not be distinguished from variable,
 * type, or function names in some contexts.  Don't put things here unless
 * forced to.
 */
reserved_keyword:
			  ALL
			| ANALYSE
			| ANALYZE
			| AND
			| ANY
			| ARRAY
			| AS
			| ASC
			| ASYMMETRIC
			| BOTH
			| CASE
			| CAST
			| CHECK
			| COLLATE
			| COLUMN
			| CONSTRAINT
			| CREATE
			| CURRENT_CATALOG
			| CURRENT_DATE
			| CURRENT_ROLE
			| CURRENT_TIME
			| CURRENT_TIMESTAMP
			| CURRENT_USER
			| DEFAULT
			| DEFERRABLE
			| DESC
			| DISTINCT
			| DO
			| ELSE
			| END_P
			| EXCEPT
			| FALSE_P
			| FETCH
			| FOR
			| FOREIGN
			| FROM
			| GRANT
			| GROUP_P
			| HAVING
			| IN_P
			| INITIALLY
			| INTERSECT
			| INTO
			| LATERAL_P
			| LEADING
			| LIMIT
			| LOCALTIME
			| LOCALTIMESTAMP
			| NOT
			| NULL_P
			| OFFSET
			| ON
			| ONLY
			| OR
			| ORDER
			| PLACING
			| PRIMARY
			| REFERENCES
			| RETURNING
			| SELECT
			| SESSION_USER
			| SOME
			| SYMMETRIC
			| SYSTEM_USER
			| TABLE
			| THEN
			| TO
			| TRAILING
			| TRUE_P
			| UNION
			| UNIQUE
			| USER
			| USING
			| VARIADIC
			| WHEN
			| WHERE
			| WINDOW
			| WITH
		;
unreserved_keyword:
			  ABORT_P
			| ABSENT
			| ABSOLUTE_P
			| ACCESS
			| ACTION
			| ADD_P
			| ADMIN
			| AFTER
			| AGGREGATE
			| ALSO
			| ALTER
			| ALWAYS
			| ASENSITIVE
			| ASSERTION
			| ASSIGNMENT
			| AT
			| ATOMIC
			| ATTACH
			| ATTRIBUTE
			| BACKWARD
			| BEFORE
			| BEGIN_P
			| BREADTH
			| BY
			| CACHE
			| CALL
			| CALLED
			| CASCADE
			| CASCADED
			| CATALOG_P
			| CHAIN
			| CHARACTERISTICS
			| CHECKPOINT
			| CLASS
			| CLOSE
			| CLUSTER
			| COLUMNS
			| COMMENT
			| COMMENTS
			| COMMIT
			| COMMITTED
			| COMPRESSION
			| CONDITIONAL
			| CONFIGURATION
			| CONFLICT
			| CONNECTION
			| CONSTRAINTS
			| CONTENT_P
			| CONTINUE_P
			| CONVERSION_P
			| COPY
			| COST
			| CSV
			| CUBE
			| CURRENT_P
			| CURSOR
			| CYCLE
			| DATA_P
			| DATABASE
			| DAY_P
			| DEALLOCATE
			| DECLARE
			| DEFAULTS
			| DEFERRED
			| DEFINER
			| DELETE_P
			| DELIMITER
			| DELIMITERS
			| DEPENDS
			| DEPTH
			| DETACH
			| DICTIONARY
			| DISABLE_P
			| DISCARD
			| DOCUMENT_P
			| DOMAIN_P
			| DOUBLE_P
			| DROP
			| EACH
			| EMPTY_P
			| ENABLE_P
			| ENCODING
			| ENCRYPTED
			| ENFORCED
			| ENUM_P
			| ERROR_P
			| ESCAPE
			| EVENT
			| EXCLUDE
			| EXCLUDING
			| EXCLUSIVE
			| EXECUTE
			| EXPLAIN
			| EXPRESSION
			| EXTENSION
			| EXTERNAL
			| FAMILY
			| FILTER
			| FINALIZE
			| FIRST_P
			| FOLLOWING
			| FORCE
			| FORMAT
			| FORWARD
			| FUNCTION
			| FUNCTIONS
			| GENERATED
			| GLOBAL
			| GRANTED
			| GROUPS
			| HANDLER
			| HEADER_P
			| HOLD
			| HOUR_P
			| IDENTITY_P
			| IF_P
			| IMMEDIATE
			| IMMUTABLE
			| IMPLICIT_P
			| IMPORT_P
			| INCLUDE
			| INCLUDING
			| INCREMENT
			| INDENT
			| INDEX
			| INDEXES
			| INHERIT
			| INHERITS
			| INLINE_P
			| INPUT_P
			| INSENSITIVE
			| INSERT
			| INSTEAD
			| INVOKER
			| ISOLATION
			| KEEP
			| KEY
			| KEYS
			| LABEL
			| LANGUAGE
			| LARGE_P
			| LAST_P
			| LEAKPROOF
			| LEVEL
			| LISTEN
			| LOAD
			| LOCAL
			| LOCATION
			| LOCK_P
			| LOCKED
			| LOGGED
			| MAPPING
			| MATCH
			| MATCHED
			| MATERIALIZED
			| MAXVALUE
			| MERGE
			| METHOD
			| MINUTE_P
			| MINVALUE
			| MODE
			| MONTH_P
			| MOVE
			| NAME_P
			| NAMES
			| NESTED
			| NEW
			| NEXT
			| NFC
			| NFD
			| NFKC
			| NFKD
			| NO
			| NORMALIZED
			| NOTHING
			| NOTIFY
			| NOWAIT
			| NULLS_P
			| OBJECT_P
			| OBJECTS_P
			| OF
			| OFF
			| OIDS
			| OLD
			| OMIT
			| OPERATOR
			| OPTION
			| OPTIONS
			| ORDINALITY
			| OTHERS
			| OVER
			| OVERRIDING
			| OWNED
			| OWNER
			| PARALLEL
			| PARAMETER
			| PARSER
			| PARTIAL
			| PARTITION
			| PASSING
			| PASSWORD
			| PATH
			| PERIOD
			| PLAN
			| PLANS
			| POLICY
			| PRECEDING
			| PREPARE
			| PREPARED
			| PRESERVE
			| PRIOR
			| PRIVILEGES
			| PROCEDURAL
			| PROCEDURE
			| PROCEDURES
			| PROGRAM
			| PUBLICATION
			| QUOTE
			| QUOTES
			| RANGE
			| READ
			| REASSIGN
			| RECURSIVE
			| REF_P
			| REFERENCING
			| REFRESH
			| REINDEX
			| RELATIVE_P
			| RELEASE
			| RENAME
			| REPEATABLE
			| REPLACE
			| REPLICA
			| RESET
			| RESTART
			| RESTRICT
			| RETURN
			| RETURNS
			| REVOKE
			| ROLE
			| ROLLBACK
			| ROLLUP
			| ROUTINE
			| ROUTINES
			| ROWS
			| RULE
			| SAVEPOINT
			| SCALAR
			| SCHEMA
			| SCHEMAS
			| SCROLL
			| SEARCH
			| SECOND_P
			| SECURITY
			| SEQUENCE
			| SEQUENCES
			| SERIALIZABLE
			| SERVER
			| SESSION
			| SET
			| SETS
			| SHARE
			| SHOW
			| SIMPLE
			| SKIP
			| SNAPSHOT
			| SOURCE
			| SQL_P
			| STABLE
			| STANDALONE_P
			| START
			| STATEMENT
			| STATISTICS
			| STDIN
			| STDOUT
			| STORAGE
			| STORED
			| STRICT_P
			| STRING_P
			| STRIP_P
			| SUBSCRIPTION
			| SUPPORT
			| SYSID
			| SYSTEM_P
			| TABLES
			| TABLESPACE
			| TARGET
			| TEMP
			| TEMPLATE
			| TEMPORARY
			| TEXT_P
			| TIES
			| TRANSACTION
			| TRANSFORM
			| TRIGGER
			| TRUNCATE
			| TRUSTED
			| TYPE_P
			| TYPES_P
			| UESCAPE
			| UNBOUNDED
			| UNCOMMITTED
			| UNCONDITIONAL
			| UNENCRYPTED
			| UNKNOWN
			| UNLISTEN
			| UNLOGGED
			| UNTIL
			| UPDATE
			| VACUUM
			| VALID
			| VALIDATE
			| VALIDATOR
			| VALUE_P
			| VARYING
			| VERSION_P
			| VIEW
			| VIEWS
			| VIRTUAL
			| VOLATILE
			| WHITESPACE_P
			| WITHIN
			| WITHOUT
			| WORK
			| WRAPPER
			| WRITE
			| XML_P
			| YEAR_P
			| YES_P
			| ZONE
		;
/* Column identifier --- keywords that can be column, table, etc names.
 *
 * Many of these keywords will in fact be recognized as type or function
 * names too; but they have special productions for the purpose, and so
 * can't be treated as "generic" type or function names.
 *
 * The type names appearing here are not usable as function names
 * because they can be followed by '(' in typename productions, which
 * looks too much like a function call for an LR(1) parser.
 */
col_name_keyword:
			  BETWEEN
			| BIGINT
			| BIT
			| BOOLEAN_P
			| CHAR_P
			| CHARACTER
			| COALESCE
			| DEC
			| DECIMAL_P
			| EXISTS
			| EXTRACT
			| FLOAT_P
			| GREATEST
			| GROUPING
			| INOUT
			| INT_P
			| INTEGER
			| INTERVAL
			| JSON
			| JSON_ARRAY
			| JSON_ARRAYAGG
			| JSON_EXISTS
			| JSON_OBJECT
			| JSON_OBJECTAGG
			| JSON_QUERY
			| JSON_SCALAR
			| JSON_SERIALIZE
			| JSON_TABLE
			| JSON_VALUE
			| LEAST
			| MERGE_ACTION
			| NATIONAL
			| NCHAR
			| NONE
			| NORMALIZE
			| NULLIF
			| NUMERIC
			| OUT_P
			| OVERLAY
			| POSITION
			| PRECISION
			| REAL
			| ROW
			| SETOF
			| SMALLINT
			| SUBSTRING
			| TIME
			| TIMESTAMP
			| TREAT
			| TRIM
			| VALUES
			| VARCHAR
			| XMLATTRIBUTES
			| XMLCONCAT
			| XMLELEMENT
			| XMLEXISTS
			| XMLFOREST
			| XMLNAMESPACES
			| XMLPARSE
			| XMLPI
			| XMLROOT
			| XMLSERIALIZE
			| XMLTABLE
		;

/*****************************************************************************
 *
 *	Names and constants
 *
 *****************************************************************************/
 //TODO Add missinge names and constants for instance qualified_name
qualified_name_list: 
    qualified_name 
    {
        auto list = mkListShared<lingodb::ast::QualifiedName>();
        list.emplace_back($qualified_name);
        $$ = list;
    }
    | qualified_name_list[list] COMMA qualified_name
    {
        $list.emplace_back($qualified_name);
        $$ = $list;
    }
    ;





    

/*
 * Postgres
 * The production for a qualified relation name has to exactly match the
 * production for a qualified func_name, because in a FROM clause we cannot
 * tell which we are parsing until we see what comes after it ('(' for a
 * func_name, something else for a relation). Therefore we allow 'indirection'
 * which may contain subscripts, and reject that case in the C code.
 */
qualified_name:
    ColId { $$=$1;}
    | ColId indirection
    ;
opt_name_list: 
    LP name_list RP
    {
        $$ = $name_list;
    }
    | %empty
    {
        $$ = std::vector<std::string>{};
    }
    ;

name_list:
    name { auto t = mkList<std::string>(); t.emplace_back($name); $$=t;}
    | name_list[list] COMMA name {$list.emplace_back($name); $$=$list;}
    ;
name: ColId {$$=$1;};
attr_name: ColLabel {$$=$1;};

/*
 * The production for a qualified func_name has to exactly match the
 * production for a qualified columnref, because we cannot tell which we
 * are parsing until we see what comes after it ('(' or Sconst for a func_name,
 * anything else for a columnref).  Therefore we allow 'indirection' which
 * may contain subscripts, and reject that case in the C code.  (If we
 * ever implement SQL99-like methods, such syntax may actually become legal!)
 */
func_name: 
    type_function_name
    | ColId indirection
    ;

param_name: type_function_name
    ;




/*****************************************************************************
 *
 *		QUERY :
 *				CREATE TABLE relname
 *
 *****************************************************************************/
 //TODO Add missing rules
 CreateStmt: 
    CREATE OptTemp TABLE qualified_name LP OptTableElementList RP
    {   
        auto createTableInfo = std::make_shared<lingodb::ast::CreateTableInfo>("", "", $OptTemp);
        createTableInfo->tableName = $qualified_name;
        createTableInfo->tableElements = $OptTableElementList;



        auto createTable = mkNode<lingodb::ast::CreateNode>(@$, createTableInfo);

        $$ = createTable;
    }
    ;


//TODO Add missing rules
OptTemp: 
    TEMP {$$=true;}
    | TEMPORARY {$$=true;}
    | %empty {$$=false;}
    ;

OptTableElementList:
    TableElementList
    {
        $$ = $1;
    }
    | %empty 
    {
        $$ = std::vector<std::shared_ptr<lingodb::ast::TableElement>>();
    }
    ;

TableElementList: 
    TableElement 
    {
        auto list = mkListShared<lingodb::ast::TableElement>();
        list.emplace_back($TableElement);
        $$ = list;
    }
    | TableElementList[list] COMMA TableElement
    {
        $list.emplace_back($TableElement);
        $$ = $list;
    }
    ;

//TODO Add missing rules
TableElement:
    columnElement {$$=$columnElement;}
    | TableConstraint {$$=$1;}
    ;

//TODO Add missing rules
TableConstraint: 
    ConstraintElem
    {
        $$ = mkNode<lingodb::ast::TableConstraintElement>(@$, $ConstraintElem);
    }
    ;

columnElement:
    ColId Type opt_column_storage opt_column_compression create_generic_options ColQualList
    {
        auto columnDef = mkNode<lingodb::ast::ColumnElement>(@$, $ColId, $Type);
        columnDef->constraints = $ColQualList;
        $$ = columnDef;
    }
    ;

create_generic_options:
    //TODO Add missing rules
     %empty
    ;
ColQualList:
    ColQualList[list] ColConstraint 
    {
        $list.emplace_back($ColConstraint);
        $$ = $list;
    }
    | %empty {$$ = std::vector<std::shared_ptr<lingodb::ast::Constraint>>();}
    ;
//TODO add missing rules
ColConstraint: 
     ColConstraintElem {$$=$1;}
    ;



/* DEFAULT NULL is already the default for Postgres.
 * But define it here and carry it forward into the system
 * to make it explicit.
 * - thomas 1998-09-13
 *
 * WITH NULL and NULL are not SQL-standard syntax elements,
 * so leave them out. Use DEFAULT NULL to explicitly indicate
 * that a column may have that value. WITH NULL leads to
 * shift/reduce conflicts with WITH TIME ZONE anyway.
 * - thomas 1999-01-08
 *
 * DEFAULT expression must be b_expr not a_expr to prevent shift/reduce
 * conflict on NOT (since NOT might start a subsequent NOT NULL constraint,
 * or be part of a_expr NOT LIKE or similar constructs).
 */
//TODO Add missing rules
ColConstraintElem:
    NOT NULL_P //opt_no_inherit 
    {
        $$ = mkNode<lingodb::ast::Constraint>(@$, lingodb::ast::ConstraintType::NOT_NULL);
    }
    | NULL_P
    {
        $$ = mkNode<lingodb::ast::Constraint>(@$, lingodb::ast::ConstraintType::NULLABLE);
    }
    | PRIMARY KEY
    {
        $$ = mkNode<lingodb::ast::UniqueConstraint>(@$, std::vector<std::string>(), true);
    }
    ;
//TODO Add missing rules
ConstraintElem: 
    PRIMARY KEY LP name_list opt_without_overlaps RP //TODO opt_c_include opt_definition OptConsTableSpace
    {
        $$ = mkNode<lingodb::ast::UniqueConstraint>(@$, $name_list, true);
    }
;

opt_without_overlaps:
    WITHOUT OVERLAPS
    {
        //TODO
        error(@$, "WITHOUT OVERLAPS is not supported yet!");
    }
    | %empty
    ;

//TODO Add missing rules
opt_column_compression:
    //column_compression
     %empty
    ;





//TODO Add missing rules
opt_column_storage:
    
     %empty
    ;

//TODO add missing rules
Type:
    SimpleType //opt_array_bounds
    {
        $$ = $SimpleType;
    }
    ;
SimpleType: 
     //GenericType
    Numeric_with_opt_lenghth {$$ = $Numeric_with_opt_lenghth;}
   //TODO | Bit
    | Character
    {
        $$ = $Character;
    }
    | ConstDatetime 
    {
        $$ = $ConstDatetime;
    }
    //TODO | ConstInterval
    //TODO | JsonType
    ;
opt_type_modifiers: 
    LP type_modifiers RP
    {
        $$ = $type_modifiers;
    }
    | %empty
    {
        $$ = std::vector<std::shared_ptr<lingodb::ast::Value>>();
    }
    ;
//TODO Postgres here uses aexprs, but lingodbs parser currently does not support aexprs as type modifiers. So I changed it to a list of ConstValues
type_modifiers: 
    type_modifier
    {
        auto list = mkListShared<lingodb::ast::Value>();
        list.emplace_back($type_modifier);
        $$ = list;
    }
    | type_modifiers[list] COMMA type_modifier
    {
        $list.emplace_back($type_modifier);
        $$ = $list;
    }
    ;
type_modifier:
    ICONST
    {
        auto value = std::make_shared<lingodb::ast::UnsignedIntValue>($ICONST);

        $$ = value;
    }
    ;
/* We have a separate ConstTypename to allow defaulting fixed-length
 * types such as CHAR() and BIT() to an unspecified length.
 * SQL9x requires that these default to a length of one, but this
 * makes no sense for constructs like CHAR 'hi' and BIT '0101',
 * where there is an obvious better choice to make.
 * Note that ConstInterval is not included here since it must
 * be pushed up higher in the rules to accommodate the postfix
 * options (e.g. INTERVAL '1' YEAR). Likewise, we have to handle
 * the generic-type-name case in AexprConst to avoid premature
 * reduce/reduce conflicts against function names.
 */
//TODO add missing rules
ConstTypename:
    Numeric {$$ = $1;}
    | ConstCharacter {$$ = $1;}
    | ConstDatetime {$$ = $1;}
    ;

NumericOnly:
    Fconst {$$ = $1;}
    | SignedIconst {$$ = $SignedIconst;}
    ;
Numeric_with_opt_lenghth:
    Numeric 
    {
        $$ = $1;
    }
    | Numeric LP type_modifier RP
    {
        $Numeric.typeModifiers.emplace_back($type_modifier);
        $$ = $Numeric;
    }
    ;

Numeric:
    INT_P
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::INT);
    } 
    | INTEGER
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::INT);
    }
    | SMALLINT
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::SMALLINT);
    }
    | BIGINT
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::BIGINT);
    }
    | REAL
    | FLOAT_P 
    {
        auto type = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::FLOAT);
        $$ = type;
    }
    | DOUBLE_P PRECISION
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::FLOAT);
    }
    | DECIMAL_P opt_type_modifiers
    {
        lingodb::ast::LogicalTypeWithMods type{lingodb::ast::LogicalType::DECIMAL};
        type.typeModifiers = $opt_type_modifiers;
        $$ = type;
        
    }
    | DEC //TODO opt_type_modifiers
    | NUMERIC  opt_type_modifiers 
    {
        lingodb::ast::LogicalTypeWithMods type{lingodb::ast::LogicalType::DECIMAL};
        type.typeModifiers = $opt_type_modifiers;
        $$ = type;
    }
    | BOOLEAN_P
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::BOOLEAN);
    }
    ;

Character:
    CharacterWithLength
    {
        $$ =$1;
    }
    | CharacterWithoutLength
    {
        $$ =$1;
    }
    ;
ConstCharacter: 
    CharacterWithLength
    {
        /* Length was not specified so allow to be unrestricted.
        * This handles problems with fixed-length (bpchar) strings
        * which in column definitions must default to a length
        * of one, but should not be constrained if the length
        * was not specified.
        */
        $CharacterWithLength.typeModifiers.clear();
        $$ =$1;
    }
    | CharacterWithoutLength
    {
        $$ =$1;
    }
    ;    
CharacterWithLength:
    character LP type_modifier RP
    {
        //Change Iconst rule to unsigned long and use it here
        $character.typeModifiers.emplace_back($type_modifier);
        $$ = $character;
    }
    ;
CharacterWithoutLength:
    character
    {
        $$ = $1;
    }
    ;
//TODO Add missing rules
character:
    CHARACTER opt_varying 
    {
        if($opt_varying) {
            $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::STRING);
        } else {
            $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::CHAR);
        }
    }
    | VARCHAR 
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::STRING);
    }
    | CHAR_P 
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::CHAR);
    }
    | TEXT_P
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::STRING);
    }
    | STRING_P
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::STRING);
    }
    ;
opt_varying:
    VARYING
    {
        $$ = true;
    }
    | %empty
    {
        $$ = false;
    }
    ;
//TODO add missing rules
ConstDatetime: 
    TIMESTAMP //TODO opt_timezone
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::TIMESTAMP);
    }

    | DATE_P
    {
        $$ = lingodb::ast::LogicalTypeWithMods(lingodb::ast::LogicalType::DATE);
    }
    ;



/*****************************************************************************
 *
 *		QUERY:
 *				INSERT STATEMENTS
 *
 *****************************************************************************/
//TODO add and complete missing rules
InsertStmt:
    INSERT INTO insert_target insert_rest
    {
        $insert_rest->tableName = $insert_target;
        $$ = $insert_rest;
    
    }
    ;
    

/*
 * Can't easily make AS optional here, because VALUES in insert_rest would
 * have a shift/reduce conflict with VALUES as an optional alias.  We could
 * easily allow unreserved_keywords as optional aliases, but that'd be an odd
 * divergence from other places.  So just require AS for now.
 */
insert_target:
    qualified_name
    {
        $$ = $qualified_name;
    }
       
   // | qualified_name AS ColId
       
;
//TODO Add missing rules
insert_rest:
    SelectStmt
    {
        $$ = mkNode<lingodb::ast::InsertNode>(@$, "", "", $SelectStmt);
    }
    | LP insert_column_list RP SelectStmt
    {
        auto insertNode = mkNode<lingodb::ast::InsertNode>(@$, "", "", $SelectStmt);
        insertNode->columns = $insert_column_list;
        $$ = insertNode;
    }
    ;


insert_column_list:
    insert_column_item 
    {
        auto list = mkList<std::string>();
        list.emplace_back($insert_column_item);
        $$ = list;
    }
    | insert_column_list[list] COMMA insert_column_item 
    {
        $list.emplace_back($insert_column_item);
        $$ = $list;
    }
    ;
//TODO add missing rules
insert_column_item:
    ColId //TODO opt_indirection
    {
        $$ = $ColId;
    }
    ;

/*
 * Constants
 */
 //TODO Add missing AexprConst rules
AexprConst: 

    SignedIconst {$$=$1;}
    | Fconst { $$=$1;}
    | Sconst {$$=$1;}
    | Bconst {$$=$1;}
    | func_name Sconst {
        //TODO move logic to analyzer?
        if($func_name == "date") {
            auto dateExpr = mkNode<lingodb::ast::CastExpression>(@$, lingodb::ast::LogicalType::DATE, $Sconst);
            
            $$ = dateExpr;
        } else {
            error(@$, "Unknown function for constant: " + $func_name);
        }
    }
    | ConstInterval Sconst opt_interval
    {
        //TODO
        auto interval = mkNode<lingodb::ast::CastExpression>(@$, lingodb::ast::LogicalType::INTERVAL, $Sconst);
        interval->optInterval = $opt_interval;
        $$ = interval;
    }
    | ConstTypename Sconst 
    {
        auto dateExpr = mkNode<lingodb::ast::CastExpression>(@$, $ConstTypename, $Sconst);
        $$ = dateExpr;
    }
    | NULL_P {
        auto t = mkNode<lingodb::ast::ConstantExpression>(@$); t->value=std::make_shared<lingodb::ast::NullValue>(); $$=t; 

    }
;
//TODO Set Iconst to unsigned long
//TODO create rule SignedIconst to handle signed integers!
Iconst:	
    ICONST	{ auto t = mkNode<lingodb::ast::ConstantExpression>(@$); t->value=std::make_shared<lingodb::ast::IntValue>($1); $$=t;  };

SignedIconst:
    PLUS Iconst
    {
    
        $$=$Iconst;
    }
    | MINUS Iconst %prec UMINUS
    {
        std::static_pointer_cast<lingodb::ast::IntValue>(std::static_pointer_cast<lingodb::ast::ConstantExpression>($Iconst)->value)->iVal = -std::static_pointer_cast<lingodb::ast::IntValue>(std::static_pointer_cast<lingodb::ast::ConstantExpression>($Iconst)->value)->iVal;
        $$=$Iconst;
       
    }
    | Iconst 
    {
        $$=$1;
    }

Fconst:
    FCONST  {auto t = mkNode<lingodb::ast::ConstantExpression>(@$); t->value=std::make_shared<lingodb::ast::FloatValue>($1); $$=t; }
    | PLUS FCONST
    | MINUS FCONST {error(@$, "Negative float constants are not supported yet!");}
Sconst: 
    STRING_VALUE { auto t = mkNode<lingodb::ast::ConstantExpression>(@$); t->value=std::make_shared<lingodb::ast::StringValue>($1); $$=t; };

Bconst: 
    TRUE_P 
    {
        auto t = mkNode<lingodb::ast::ConstantExpression>(@$);
        t->value=std::make_shared<lingodb::ast::BoolValue>(true);
        $$=t;
     };
    | FALSE_P 
    {
        auto t = mkNode<lingodb::ast::ConstantExpression>(@$);
        t->value=std::make_shared<lingodb::ast::BoolValue>(false);
        $$=t;
    }

ConstInterval:
    | INTERVAL
    ;

//TODO missing
opt_interval: 
    DAY_P 
     {
        $$ = lingodb::ast::LogicalType::DAYS;
     }
     | YEAR_P 
     {
        $$ = lingodb::ast::LogicalType::YEARS;
     }
     | MONTH_P
     {
        $$ = lingodb::ast::LogicalType::MONTHS;
     }
     | %empty
    ;
//TODO missing rules
Typename: 
    SimpleType {$$=$1;}
    ;

/*
* GOOLE PIPE syntax
*/
//TODO Add more operators
pipe_operator: 
    where_clause 
    {
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::WHERE, $where_clause);
        
    }
    | SELECT opt_target_list 
    {
       $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::SELECT, $opt_target_list);
    }
    | sort_clause 
    {
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::RESULT_MODIFIER, $sort_clause);
    }
    | limit_clause 
    {
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::RESULT_MODIFIER, $limit_clause);   
    }
    | join_type JOIN table_ref join_qual
    {
        auto joinType = $join_type;
        auto joinRef = mkNode<lingodb::ast::JoinRef>(@$, joinType, lingodb::ast::JoinCondType::REGULAR );
        joinRef->right = $table_ref;
        joinRef->condition = $join_qual;
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::JOIN, joinRef);
    }
    | JOIN table_ref join_qual
    {
        auto joinRef = mkNode<lingodb::ast::JoinRef>(@$, lingodb::ast::JoinType::INNER, lingodb::ast::JoinCondType::REGULAR );
        joinRef->right = $table_ref;
        joinRef->condition = $join_qual;
        $$ = mkNode<lingodb::ast::PipeOperator>(@$,lingodb::ast::PipeOperatorType::JOIN, joinRef);
    }
    //TODO check if this does not allow to much!
    | AGGREGATE agg_expr
    {
         $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::AGGREGATE, $agg_expr);
    }
    | EXTEND expr_list_with_alias
    {
        auto extendNode = mkNode<lingodb::ast::ExtendNode>(@$, false);
        extendNode->extensions = $expr_list_with_alias;
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::EXTEND, extendNode);
    }
    | UNION set_quantifier pipe_or_select_clause
    {
        auto setOpNode = mkNode<lingodb::ast::SetOperationNode>(@$, lingodb::ast::SetOperationType::UNION, nullptr, $pipe_or_select_clause);
        setOpNode->setOpAll = $set_quantifier;
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::SET_OPERATION, setOpNode);
    }
    | INTERSECT set_quantifier pipe_or_select_clause
    {
        auto setOpNode = mkNode<lingodb::ast::SetOperationNode>(@$, lingodb::ast::SetOperationType::INTERSECT, nullptr, $pipe_or_select_clause);
        setOpNode->setOpAll = $set_quantifier;
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::SET_OPERATION, setOpNode);
    }
    | EXCEPT set_quantifier pipe_or_select_clause
    {
        auto setOpNode = mkNode<lingodb::ast::SetOperationNode>(@$, lingodb::ast::SetOperationType::EXCEPT, nullptr, $pipe_or_select_clause);
        setOpNode->setOpAll = $set_quantifier;
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::SET_OPERATION, setOpNode);
    }
    | DROP columnref_list
    {
        auto node = mkNode<lingodb::ast::TargetsExpression>(@$);
        node->targets = std::move($columnref_list);
        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::DROP, node);
    }
    | SET set_list
    {

        $$ = mkNode<lingodb::ast::PipeOperator>(@$, lingodb::ast::PipeOperatorType::SET, $set_list);
    }

    //...

    ;

agg_expr: 
    func_expr_list group_clause_with_alias 
    {
        auto aggNode = mkNode<lingodb::ast::AggregationNode>(@$);
        aggNode->groupByNode = $group_clause_with_alias;
        aggNode->aggregations = $func_expr_list;
        $$ = aggNode;
      
        
    }
    | group_clause 
    {
        auto aggNode = mkNode<lingodb::ast::AggregationNode>(@$);
        aggNode->groupByNode = $group_clause;
        $$ = aggNode;
      
        
    }
    ;
   
func_expr_list: 
    func_expr opt_alias_clause 
    {
        auto list = mkListShared<lingodb::ast::FunctionExpression>();
        $func_expr->alias=$opt_alias_clause.first;
        list.emplace_back($func_expr);
        
        $$ = list;
    }
    | func_expr_list[list] COMMA func_expr opt_alias_clause
    {
        $list.emplace_back($func_expr);
        $$ = $list;
    }
    ;

set_list: 
    ColId EQUAL a_expr
    {
        auto columnRef = mkNode<lingodb::ast::ColumnRefExpression>(@$, $ColId);
        auto setNode = mkNode<lingodb::ast::SetExpression>(@$, std::vector<std::pair<std::shared_ptr<lingodb::ast::ColumnRefExpression>, std::shared_ptr<lingodb::ast::ParsedExpression>>>{std::pair(columnRef, $a_expr)});
        $$ = setNode;
    }
    | set_list[list] COMMA ColId EQUAL a_expr
    {
        auto columnRef = mkNode<lingodb::ast::ColumnRefExpression>(@$, $ColId);
        $list->sets.emplace_back(std::pair(columnRef, $a_expr));
        $$ = $list;
    }
%%
void
lingodb::parser::error (const location_type& l, const std::string& m)
{
  throw lingodb::syntax_error(m, l);
}
