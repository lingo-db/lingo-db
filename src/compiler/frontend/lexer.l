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
"<>"            { MAKE_KEYWORD(NOT_EQUAL); }
"!="            { MAKE_KEYWORD(NOT_EQUAL); }
"<="            { MAKE_KEYWORD(LESS_EQUAL); }
"<"             { MAKE_KEYWORD(LESS); }
">="            { MAKE_KEYWORD(GREATER_EQUAL); }
">"             { MAKE_KEYWORD(GREATER); }
"|>"            { return lingodb::parser::make_PIPE(loc); }
"::"            { MAKE_KEYWORD(TYPECAST); }
{newline}              { loc.lines();}

[aA][gG][gG][rR][eE][gG][aA][tT][eE]  { MAKE_KEYWORD(AGGREGATE); }
[aA][lL][lL]                          { MAKE_KEYWORD(ALL); }
[aA][nN][dD]                          { MAKE_KEYWORD(AND); }
[aA][sS]                              { MAKE_KEYWORD(AS); }
[aA][sS][cC]                          { MAKE_KEYWORD(ASC); }
[aA][nN][yY]                          { MAKE_KEYWORD(ANY); }
[bB][eE][tT][wW][eE][eE][nN]          { MAKE_KEYWORD(BETWEEN); }
[bB][iI][gG][iI][nN][tT]              { MAKE_KEYWORD(BIGINT); }
[bB][yY]                              { MAKE_KEYWORD(BY); }

[cC][aA][sS][eE]                      { MAKE_KEYWORD(CASE); }
[cC][aA][sS][tT]                      { MAKE_KEYWORD(CAST); }
[cC][hH][aA][rR]                      { MAKE_KEYWORD(CHAR_P); }
[cC][hH][aA][rR][aA][cC][tT][eE][rR]  { MAKE_KEYWORD(CHARACTER); }
[cC][rR][eE][aA][tT][eE]              { MAKE_KEYWORD(CREATE); }
[cC][uU][rR][rR][eE][nN][tT]          { MAKE_KEYWORD(CURRENT_P); }

[dD][aA][tT][eE]                      { MAKE_KEYWORD(DATE_P); }
[dD][aA][yY]                          { MAKE_KEYWORD(DAY_P); }
[dD][eE][cC][iI][mM][aA][lL]          { MAKE_KEYWORD(DECIMAL_P); }
[dD][eE][sS][cC]                      { MAKE_KEYWORD(DESC); }
[dD][iI][sS][tT][iI][nN][cC][tT]      { MAKE_KEYWORD(DISTINCT); }
[dD][rR][oO][pP]                      { MAKE_KEYWORD(DROP); }

[eE][lL][sS][eE]                      { MAKE_KEYWORD(ELSE); }
[eE][nN][dD]                          { MAKE_KEYWORD(END_P); }
[eE][xX][cC][eE][pP][tT]              { MAKE_KEYWORD(EXCEPT); }
[eE][xX][iI][sS][tT][sS]              { MAKE_KEYWORD(EXISTS); }
[eE][xX][tT][eE][nN][dD]              { MAKE_KEYWORD(EXTEND); }
[eE][xX][tT][rR][aA][cC][tT]          { MAKE_KEYWORD(EXTRACT); }

[fF][aA][lL][sS][eE]                  { MAKE_KEYWORD(FALSE_P); }
[fF][lL][oO][aA][tT]                  { MAKE_KEYWORD(FLOAT_P); }
[fF][oO][rR]                          { MAKE_KEYWORD(FOR); }
[fF][rR][oO][mM]                      { MAKE_KEYWORD(FROM); }
[fF][uU][lL][lL]                      { MAKE_KEYWORD(FULL); }

[gG][rR][oO][uU][pP]                  { MAKE_KEYWORD(GROUP_P); }

[hH][aA][vV][iI][nN][gG]              { MAKE_KEYWORD(HAVING); }

[iI][nN]                              { MAKE_KEYWORD(IN_P); }
[iI][nN][nN][eE][rR]                  { MAKE_KEYWORD(INNER_P); }
[iI][nN][sS][eE][rR][tT]              { MAKE_KEYWORD(INSERT); }
[iI][nN][tT]                          { MAKE_KEYWORD(INT_P); }
[iI][nN][tT][eE][gG][eE][rR]          { MAKE_KEYWORD(INTEGER); }
[iI][nN][tT][eE][rR][sS][eE][cC][tT]  { MAKE_KEYWORD(INTERSECT); }
[iI][nN][tT][eE][rR][vV][aA][lL]      { MAKE_KEYWORD(INTERVAL); }
[iI][nN][tT][oO]                      { MAKE_KEYWORD(INTO); }
[iI][sS]                              { MAKE_KEYWORD(IS); }

[jJ][oO][iI][nN]                      { MAKE_KEYWORD(JOIN); }

[kK][eE][yY]                          { MAKE_KEYWORD(KEY); }

[lL][eE][fF][tT]                      { MAKE_KEYWORD(LEFT); }
[lL][iI][kK][eE]                      { MAKE_KEYWORD(LIKE); }
[lL][iI][mM][iI][tT]                  { MAKE_KEYWORD(LIMIT); }

[mM][iI][nN][uU][tT][eE]              { MAKE_KEYWORD(MINUTE_P); }
[mM][oO][nN][tT][hH]                  { MAKE_KEYWORD(MONTH_P); }

[nN][aA][tT][uU][rR][aA][lL]          { MAKE_KEYWORD(NATURAL); }
[nN][oO][tT]                          { MAKE_KEYWORD(NOT); }
[nN][uU][lL][lL]                      { MAKE_KEYWORD(NULL_P); }
[nN][uU][mM][eE][rR][iI][cC]          { MAKE_KEYWORD(NUMERIC); }

[oO][fF][fF][sS][eE][tT]              { MAKE_KEYWORD(OFFSET); }
[oO][nN]                              { MAKE_KEYWORD(ON); }
[oO][rR]                              { MAKE_KEYWORD(OR); }
[oO][rR][dD][eE][rR]                  { MAKE_KEYWORD(ORDER); }
[oO][uU][tT][eE][rR]                  { MAKE_KEYWORD(OUTER_P); }
[oO][vV][eE][rR]                      { MAKE_KEYWORD(OVER); }

[pP][aA][rR][tT][iI][tT][iI][oO][nN]  { MAKE_KEYWORD(PARTITION); }
[pP][rR][eE][cC][eE][dD][iI][nN][gG]  { MAKE_KEYWORD(PRECEDING); }
[pP][rR][iI][mM][aA][rR][yY]          { MAKE_KEYWORD(PRIMARY); }

[rR][iI][gG][hH][tT]                  { MAKE_KEYWORD(RIGHT); }
[rR][oO][lL][lL][uU][pP]              { MAKE_KEYWORD(ROLLUP); }
[rR][oO][wW]                          { MAKE_KEYWORD(ROW); }
[rR][oO][wW][sS]                      { MAKE_KEYWORD(ROWS); }

[sS][eE][lL][eE][cC][tT]              { MAKE_KEYWORD(SELECT); }
[sS][eE][tT]                          { MAKE_KEYWORD(SET); }
[sS][oO][mM][eE]                      { MAKE_KEYWORD(SOME); }
[sS][tT][rR][iI][nN][gG]              { MAKE_KEYWORD(STRING_P); }
[sS][uU][bB][sS][tT][rR][iI][nN][gG]  { MAKE_KEYWORD(SUBSTRING); }

[tT][aA][bB][lL][eE]                  { MAKE_KEYWORD(TABLE); }
[tT][eE][xX][tT]                      { MAKE_KEYWORD(TEXT_P); }
[tT][hH][eE][nN]                      { MAKE_KEYWORD(THEN); }
[tT][rR][uU][eE]                      { MAKE_KEYWORD(TRUE_P); }
[tT][iI][mM][eE][sS][tT][aA][mM][pP]  { MAKE_KEYWORD(TIMESTAMP); }

[uU][nN][bB][oO][uU][nN][dD][eE][dD]  { MAKE_KEYWORD(UNBOUNDED); }
[uU][nN][iI][oO][nN]                  { MAKE_KEYWORD(UNION); }

[vV][aA][lL][uU][eE][sS]              { MAKE_KEYWORD(VALUES); }
[vV][aA][rR][cC][hH][aA][rR]          { MAKE_KEYWORD(VARCHAR); }
[vV][aA][rR][yY][iI][nN][gG]          { MAKE_KEYWORD(VARYING); }

[wW][hH][eE][nN]                      { MAKE_KEYWORD(WHEN); }
[wW][hH][eE][rR][eE]                  { MAKE_KEYWORD(WHERE); }
[wW][iI][tT][hH]                      { MAKE_KEYWORD(WITH); }

[yY][eE][aA][rR]                      { MAKE_KEYWORD(YEAR_P); }


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