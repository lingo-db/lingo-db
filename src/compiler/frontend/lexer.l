%{ /* -*- C++ -*- */
# include <cerrno>
# include <climits>
# include <cstdlib>
# include <cstring> // strerror
# include <string>
# include "lingodb/compiler/frontend/driver.h"
# include "parser.hpp"
# include <algorithm>
#include <lingodb/compiler/frontend/sql-parser/gen/location.hh>

%}

%{
#if defined __clang__
# define CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
#endif

// Clang and ICC like to pretend they are GCC.
#if defined __GNUC__ && !defined __clang__ && !defined __ICC
# define GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

// Pacify warnings in yy_init_buffer (observed with Flex 2.6.4)
// and GCC 6.4.0, 7.3.0 with -O3.
#if defined GCC_VERSION && 600 <= GCC_VERSION
# pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

// This example uses Flex's C back end, yet compiles it as C++.
// So expect warnings about C style casts and NULL.
#if defined CLANG_VERSION && 500 <= CLANG_VERSION
# pragma clang diagnostic ignored "-Wold-style-cast"
# pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#elif defined GCC_VERSION && 407 <= GCC_VERSION
# pragma GCC diagnostic ignored "-Wold-style-cast"
# pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif

#define FLEX_VERSION (YY_FLEX_MAJOR_VERSION * 100 + YY_FLEX_MINOR_VERSION)

// Old versions of Flex (2.5.35) generate an incomplete documentation comment.
//
//  In file included from src/scan-code-c.c:3:
//  src/scan-code.c:2198:21: error: empty paragraph passed to '@param' command
//        [-Werror,-Wdocumentation]
//   * @param line_number
//     ~~~~~~~~~~~~~~~~~^
//  1 error generated.
#if FLEX_VERSION < 206 && defined CLANG_VERSION
# pragma clang diagnostic ignored "-Wdocumentation"
#endif

// Old versions of Flex (2.5.35) use 'register'.  Warnings introduced in
// GCC 7 and Clang 6.
#if FLEX_VERSION < 206
# if defined CLANG_VERSION && 600 <= CLANG_VERSION
#  pragma clang diagnostic ignored "-Wdeprecated-register"
# elif defined GCC_VERSION && 700 <= GCC_VERSION
#  pragma GCC diagnostic ignored "-Wregister"
# endif
#endif

#if FLEX_VERSION < 206
# if defined CLANG_VERSION
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wdocumentation"
#  pragma clang diagnostic ignored "-Wshorten-64-to-32"
#  pragma clang diagnostic ignored "-Wsign-conversion"
# elif defined GCC_VERSION
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-conversion"
# endif
#endif

// Flex 2.6.4, GCC 9
// warning: useless cast to type 'int' [-Wuseless-cast]
// 1361 |   YY_CURRENT_BUFFER_LVALUE->yy_buf_size = (int) (new_size - 2);
//      |                                                 ^
#if defined GCC_VERSION && 900 <= GCC_VERSION
# pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

#define MAKE_KEYWORD(token_name) \
    return lingodb::parser::make_##token_name(std::string(yytext),loc);


%}

%option noyywrap nounput noinput batch debug

%{
  // A number symbol corresponding to the value in S.

  
%}


space				[ \t\f\v]+
newline				[\n\r]
non_newline			[^\n\r]

comment			("--"{non_newline}*)

whitespace		({comment}|{space})+

decdigit        [0-9]
decinteger		{decdigit}+

numeric         ({decinteger}\.{decinteger}?)|(\.{decinteger})
op_chars		    [\~\!\@\#\^\&\|\`\?\+\-\*\/\%\<\>\=]
ident_start     [a-zA-Z]
ident_cont      [a-zA-Z0-9_%]
identifier      {ident_start}{ident_cont}*
quoted_identifier   \"([^\"]|\"\")*\"
squote			\'
dquote			\"
strChars ([^'])
str       {squote}({strChars}|{op_chars}|[ ])*{squote}

operator		{op_chars}+


boolean [fF][aA][lL][sS][eE]

%{
  // Code run each time a pattern is matched.
  # define YY_USER_ACTION  loc.columns (yyleng);
%}

%option caseless
%%
%{
  // A handy shortcut to the location held by the driver.
  lingodb::location& loc = drv.location;
  // Code run each time yylex is called.
  loc.step ();
%}
"("             { return lingodb::parser::make_LP(loc); }
")"             { return lingodb::parser::make_RP(loc); }
"["             { return lingodb::parser::make_LB(loc); }
"]"             { return lingodb::parser::make_RB(loc); }
"."             { return lingodb::parser::make_DOT(loc); }
","             { return lingodb::parser::make_COMMA(loc); }
";"             { return lingodb::parser::make_SEMICOLON(loc); }
"+"             { MAKE_KEYWORD(PLUS); }
"-"             { MAKE_KEYWORD(MINUS); }
"/"             { MAKE_KEYWORD(SLASH); }
"*"             { MAKE_KEYWORD(STAR); }
"="             { MAKE_KEYWORD(EQUAL); }
"%"             { MAKE_KEYWORD(PERCENT); }
"<>"            { MAKE_KEYWORD(NOT_EQUAL); }
"!="            { MAKE_KEYWORD(NOT_EQUAL); }
"<="            { MAKE_KEYWORD(LESS_EQUAL); }
"<"             { MAKE_KEYWORD(LESS); }
">="            { MAKE_KEYWORD(GREATER_EQUAL); }
">"             { MAKE_KEYWORD(GREATER); }
"|>"            { return lingodb::parser::make_PIPE(loc); }
"::"            { MAKE_KEYWORD(TYPECAST); }
{newline}              { loc.lines();}

AGGREGATE   { MAKE_KEYWORD(AGGREGATE); }
ALL         { MAKE_KEYWORD(ALL); }
AND         { MAKE_KEYWORD(AND); }
AS          { MAKE_KEYWORD(AS); }
ASC         { MAKE_KEYWORD(ASC); }
ANY         { MAKE_KEYWORD(ANY); }

BETWEEN     { MAKE_KEYWORD(BETWEEN); }
BIGINT      { MAKE_KEYWORD(BIGINT); }
BOOLEAN     { MAKE_KEYWORD(BOOLEAN_P); }
BOOL        { MAKE_KEYWORD(BOOLEAN_P); }
BY          { MAKE_KEYWORD(BY); }

CASE        { MAKE_KEYWORD(CASE); }
CAST        { MAKE_KEYWORD(CAST); }
CHAR        { MAKE_KEYWORD(CHAR_P); }
CHARACTER   { MAKE_KEYWORD(CHARACTER); }
CREATE      { MAKE_KEYWORD(CREATE); }
CURRENT     { MAKE_KEYWORD(CURRENT_P); }
COPY        { MAKE_KEYWORD(COPY); }
CSV         { MAKE_KEYWORD(CSV); }

DATE        { MAKE_KEYWORD(DATE_P); }
DAY         { MAKE_KEYWORD(DAY_P); }
DECIMAL     { MAKE_KEYWORD(DECIMAL_P); }
DESC        { MAKE_KEYWORD(DESC); }
DISTINCT    { MAKE_KEYWORD(DISTINCT); }
DROP        { MAKE_KEYWORD(DROP); }
DELIMITER   { MAKE_KEYWORD(DELIMITER); }

ELSE        { MAKE_KEYWORD(ELSE); }
END         { MAKE_KEYWORD(END_P); }
EXCEPT      { MAKE_KEYWORD(EXCEPT); }
EXISTS      { MAKE_KEYWORD(EXISTS); }
EXTEND      { MAKE_KEYWORD(EXTEND); }
EXTRACT     { MAKE_KEYWORD(EXTRACT); }
ESCAPE      { MAKE_KEYWORD(ESCAPE); }

FALSE       { MAKE_KEYWORD(FALSE_P); }
FLOAT       { MAKE_KEYWORD(FLOAT_P); }
FOR         { MAKE_KEYWORD(FOR); }
FROM        { MAKE_KEYWORD(FROM); }
FULL        { MAKE_KEYWORD(FULL); }
FOLLOWING   { MAKE_KEYWORD(FOLLOWING); }

GROUP       { MAKE_KEYWORD(GROUP_P); }

HAVING      { MAKE_KEYWORD(HAVING); }

IN          { MAKE_KEYWORD(IN_P); }
INNER       { MAKE_KEYWORD(INNER_P); }
INSERT      { MAKE_KEYWORD(INSERT); }
INT         { MAKE_KEYWORD(INT_P); }
INTEGER     { MAKE_KEYWORD(INTEGER); }
INTERSECT   { MAKE_KEYWORD(INTERSECT); }
INTERVAL    { MAKE_KEYWORD(INTERVAL); }
INTO        { MAKE_KEYWORD(INTO); }
IS          { MAKE_KEYWORD(IS); }

JOIN        { MAKE_KEYWORD(JOIN); }

KEY         { MAKE_KEYWORD(KEY); }

LEFT        { MAKE_KEYWORD(LEFT); }
LIKE        { MAKE_KEYWORD(LIKE); }
LIMIT       { MAKE_KEYWORD(LIMIT); }

MINUTE      { MAKE_KEYWORD(MINUTE_P); }
MONTH       { MAKE_KEYWORD(MONTH_P); }

NATURAL     { MAKE_KEYWORD(NATURAL); }
NOT         { MAKE_KEYWORD(NOT); }
NULL        { MAKE_KEYWORD(NULL_P); }
NUMERIC     { MAKE_KEYWORD(NUMERIC); }

OFFSET      { MAKE_KEYWORD(OFFSET); }
ON          { MAKE_KEYWORD(ON); }
OR          { MAKE_KEYWORD(OR); }
ORDER       { MAKE_KEYWORD(ORDER); }
OUTER       { MAKE_KEYWORD(OUTER_P); }
OVER        { MAKE_KEYWORD(OVER); }

PARTITION   { MAKE_KEYWORD(PARTITION); }
PRECEDING   { MAKE_KEYWORD(PRECEDING); }
PRIMARY     { MAKE_KEYWORD(PRIMARY); }

RIGHT       { MAKE_KEYWORD(RIGHT); }
ROLLUP      { MAKE_KEYWORD(ROLLUP); }
ROW         { MAKE_KEYWORD(ROW); }
ROWS        { MAKE_KEYWORD(ROWS); }

SELECT      { MAKE_KEYWORD(SELECT); }
SET         { MAKE_KEYWORD(SET); }
SOME        { MAKE_KEYWORD(SOME); }
STRING      { MAKE_KEYWORD(STRING_P); }
SUBSTRING   { MAKE_KEYWORD(SUBSTRING); }

TABLE       { MAKE_KEYWORD(TABLE); }
TEXT        { MAKE_KEYWORD(TEXT_P); }
THEN        { MAKE_KEYWORD(THEN); }
TRUE        { MAKE_KEYWORD(TRUE_P); }
TIMESTAMP   { MAKE_KEYWORD(TIMESTAMP); }

UNBOUNDED   { MAKE_KEYWORD(UNBOUNDED); }
UNION       { MAKE_KEYWORD(UNION); }

VALUES      { MAKE_KEYWORD(VALUES); }
VARCHAR     { MAKE_KEYWORD(VARCHAR); }
VARYING     { MAKE_KEYWORD(VARYING); }

WHEN        { MAKE_KEYWORD(WHEN); }
WHERE       { MAKE_KEYWORD(WHERE); }
WITH        { MAKE_KEYWORD(WITH); }

YEAR        { MAKE_KEYWORD(YEAR_P); }


<<EOF>>    return lingodb::parser::make_YYEOF (loc);

{whitespace}    {
                  /*Ignore*/
                }
{identifier}    {
                    std::string ident(yytext);
                    std::transform(ident.begin(), ident.end(), ident.begin(),
                        [](unsigned char c){ return std::tolower(c); });
                    return lingodb::parser::make_IDENTIFIER(std::move(ident), loc);
                }
{quoted_identifier} {
                      std::string text(yytext);
                      std::string trimmed(text.begin() + 1, text.end()-1);
                      return lingodb::parser::make_IDENTIFIER(std::move(trimmed), loc);

                    }
{decinteger}	{
				    return lingodb::parser::make_ICONST(std::stoul(yytext), loc);
			    }
{numeric}	{
				    return lingodb::parser::make_FCONST(yytext, loc);
			    }
{str} {
  std::string ident(yytext);
  ident.erase(0,1);
  ident.erase(ident.size()-1,1);
  return lingodb::parser::make_STRING_VALUE(std::move(ident), loc);
}
{operator} {
  std::string ident(yytext);

  /*
   * For SQL compatibility multi char operators cannot end with '+' and '-'.
   * They should be parsed as two tokens
  */
  //Remove '+' and '-' at the end if necessary
  while(ident[ident.length() - 1] == '-' || ident[ident.length() - 1] == '+') {
    ident.erase(ident.length() - 1);
  }
  yyless(ident.length());

  if(ident == "<=") {
    return lingodb::parser::make_LESS_EQUAL("<=", loc);
  } else if(ident == "<>" || ident == "!=") {
    return lingodb::parser::make_NOT_EQUAL("<>", loc);
  } else if(ident == ">=") {
    return lingodb::parser::make_GREATER_EQUAL(">=", loc);
  }

  return lingodb::parser::make_Op(std::move(ident), loc);
}
%%



void
driver::scan_begin ()
{
  yy_flex_debug = trace_scanning;
  if (file.empty () || file == "-")
    yyin = stdin;
  if(file.starts_with(":")) {
     const char* sql = file.c_str() + 1; // Skip the ":" prefix
     YY_BUFFER_STATE buffer = yy_scan_string(sql);
     yy_switch_to_buffer(buffer);
  }
  else if (!(yyin = fopen (file.c_str (), "r")))
    {
      std::cerr << "cannot open " << file << ": " << strerror (errno) << '\n';
      exit (EXIT_FAILURE);
    }
}

void
driver::scan_end ()
{
  if (YY_CURRENT_BUFFER) {
      yy_delete_buffer(YY_CURRENT_BUFFER);
   }
   if (yyin && yyin != stdin) {
      fclose (yyin);
   }
}