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


%}

%option noyywrap nounput noinput batch debug

%{
  // A number symbol corresponding to the value in S.

  
%}


space				[ \t\n\r\f\v]
non_newline_space	[ \t\f\v]
newline				[\n\r]
non_newline			[^\n\r]

comment			("--"{non_newline}*)

whitespace		({comment})

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
str       {squote}({ident_cont}|{op_chars}|[ ])*{squote}

boolean [fF][aA][lL][sS][eE]

%{
  // Code run each time a pattern is matched.
  # define YY_USER_ACTION  loc.columns (yyleng);
%}
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
"+"             { return lingodb::parser::make_PLUS(loc); }
"-"             { return lingodb::parser::make_MINUS(loc); }
"/"             { return lingodb::parser::make_SLASH(loc); }
"*"             { return lingodb::parser::make_STAR(loc); }
"="             { return lingodb::parser::make_EQUAL(loc); }
"<>"            { return lingodb::parser::make_NOT_EQUAL(loc); }
"<="            { return lingodb::parser::make_LESS_EQUAL(loc); }
"<"             { return lingodb::parser::make_LESS(loc); }
">="            { return lingodb::parser::make_GREATER_EQUAL(loc); }
">"             { return lingodb::parser::make_GREATER(loc); }
"||"            { return lingodb::parser::make_CONCAT(loc); }
"#"             { return lingodb::parser::make_YYEOF(loc);}
"|>"            { return lingodb::parser::make_PIPE(loc); }
{newline}              { loc.lines();}

[aA][lL][lL]                          { return lingodb::parser::make_ALL(loc); }
[aA][gG][gG][rR][eE][gG][aA][tT][eE]  { return lingodb::parser::make_AGGREGATE(loc); }
[aA][nN][dD]                          { return lingodb::parser::make_AND(loc);}
[aA][sS]                              { return lingodb::parser::make_AS(loc);}
[aA][sS][cC]                          { return lingodb::parser::make_ASC(loc); }

[bB][eE][tT][wW][eE][eE][nN]          { return lingodb::parser::make_BETWEEN(loc);}
[bB][iI][gG][iI][nN][tT]              { return lingodb::parser::make_BIGINT(loc); }
[bB][yY]                              { return lingodb::parser::make_BY(loc);}
[cC][aA][sS][tT]                      { return lingodb::parser::make_CAST(loc);}
[cC][aA][sS][eE]                      { return lingodb::parser::make_CASE(loc); }
[cC][hH][aA][rR]                      { return lingodb::parser::make_CHAR_P(loc); }
[cC][hH][aA][rR][aA][cC][tT][eE][rR]  { return lingodb::parser::make_CHARACTER(loc); }
[cC][rR][eE][aA][tT][eE]              { return lingodb::parser::make_CREATE(loc); }

[dD][aA][tT][eE]                      { return lingodb::parser::make_DATE_P(loc); }
[dD][aA][yY]                          { return lingodb::parser::make_DAY_P(loc); }
[dD][eE][cC][iI][mM][aA][lL]          { return lingodb::parser::make_DECIMAL_P(loc); }
[dD][eE][sS][cC]                      { return lingodb::parser::make_DESC(loc); }
[dD][iI][sS][tT][iI][nN][cC][tT]      { return lingodb::parser::make_DISTINCT(loc); }

[eE][lL][sS][eE]                      { return lingodb::parser::make_ELSE(loc); }
[eE][nN][dD]                          { return lingodb::parser::make_END_P(loc); }
[eE][xX][iI][sS][tT][sS]              { return lingodb::parser::make_EXISTS(loc); }
[eE][xX][tT][rR][aA][cC][tT]          { return lingodb::parser::make_EXTRACT(loc); }

[fF][lL][oO][aA][tT]                  { return lingodb::parser::make_FLOAT_P(loc); }
[fF][oO][rR]                          { return lingodb::parser::make_FOR(loc); }
[fF][rR][oO][mM]                      { return lingodb::parser::make_FROM(loc); }
[fF][uU][lL][lL]                      { return lingodb::parser::make_FULL(loc); }

[gG][rR][oO][uU][pP]                  { return lingodb::parser::make_GROUP_P(loc);}
[hH][aA][vV][iI][nN][gG]              { return lingodb::parser::make_HAVING(loc);}

[iI][nN]                              { return lingodb::parser::make_IN_P(loc); }
[iI][nN][nN][eE][rR]                  { return lingodb::parser::make_INNER_P(loc); }
[iI][nN][sS][eE][rR][tT]              { return lingodb::parser::make_INSERT(loc); }
[iI][nN][tT]                          { return lingodb::parser::make_INT_P(loc); }
[iI][nN][tT][eE][gG][eE][rR]          { return lingodb::parser::make_INTEGER(loc); }
[iI][nN][tT][eE][rR][vV][aA][lL]      { return lingodb::parser::make_INTERVAL(loc); }
[iI][nN][tT][oO]                      { return lingodb::parser::make_INTO(loc); }
[nN][uU][mM][eE][rR][iI][cC]          { return lingodb::parser::make_NUMERIC(loc); }

[jJ][oO][iI][nN]                      { return lingodb::parser::make_JOIN(loc); }

[kK][eE][yY]                          { return lingodb::parser::make_KEY(loc); }

[lL][eE][fF][tT]                      { return lingodb::parser::make_LEFT(loc); }
[lL][iI][kK][eE]                      { return lingodb::parser::make_LIKE(loc); }
[lL][iI][mM][iI][tT]                  { return lingodb::parser::make_LIMIT(loc); }

[nN][aA][tT][uU][rR][aA][lL]          { return lingodb::parser::make_NATURAL(loc); }
[nN][oO][tT]                          { return lingodb::parser::make_NOT(loc); }
[nN][uU][lL][lL]                      { return lingodb::parser::make_NULL_P(loc); }

[oO][nN]                              { return lingodb::parser::make_ON(loc); }
[oO][rR]                              { return lingodb::parser::make_OR(loc); }
[oO][rR][dD][eE][rR]                  { return lingodb::parser::make_ORDER(loc); }
[oO][uU][tT][eE][rR]                  { return lingodb::parser::make_OUTER_P(loc); }
[oO][vV][eE][rR]                      { return lingodb::parser::make_OVER(loc); }

[pP][rR][iI][mM][aA][rR][yY]          { return lingodb::parser::make_PRIMARY(loc); }

[rR][iI][gG][hH][tT]                  { return lingodb::parser::make_RIGHT(loc); }

[sS][eE][lL][eE][cC][tT]              { return lingodb::parser::make_SELECT(loc); }
[sS][uU][bB][sS][tT][rR][iI][nN][gG]  { return lingodb::parser::make_SUBSTRING(loc); }

[tT][aA][bB][lL][eE]                  { return lingodb::parser::make_TABLE(loc); }
[tT][eE][xX][tT]                      { return lingodb::parser::make_TEXT_P(loc); }
[tT][hH][eE][nN]                      { return lingodb::parser::make_THEN(loc); }
[tT][iI][mM][eE][sS][tT][aA][mM][pP]  { return lingodb::parser::make_TIMESTAMP(loc); }

[uU][nN][iI][oO][nN]                  { return lingodb::parser::make_UNION(loc); }

[vV][aA][lL][uU][eE][sS]              { return lingodb::parser::make_VALUES(loc); }
[vV][aA][rR][cC][hH][aA][rR]          { return lingodb::parser::make_VARCHAR(loc); }
[vV][aA][rR][yY][iI][nN][gG]          { return lingodb::parser::make_VARYING(loc); }

[wW][hH][eE][nN]                      { return lingodb::parser::make_WHEN(loc); }
[wW][hH][eE][rR][eE]                  { return lingodb::parser::make_WHERE(loc); }
[wW][iI][tT][hH]                      { return lingodb::parser::make_WITH(loc); }

[yY][eE][aA][rR]                      { return lingodb::parser::make_YEAR_P(loc); }

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
