#pragma once

#include <cassert>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

namespace graphalg {

#define GRAPHALG_ENUM_TOKEN_KIND(XX)                                  \
   XX(INVALID) /* SPECIAL: For inputs that could not be tokenized. */ \
   XX(END_OF_FILE) /* SPECIAL: End of file marker. */                 \
   XX(IDENT)                                                          \
   XX(INT)                                                            \
   XX(FLOAT)                                                          \
   /* Keywords */                                                     \
   XX(FUNC)                                                           \
   XX(RETURN)                                                         \
   XX(FOR)                                                            \
   XX(IN)                                                             \
   XX(UNTIL)                                                          \
   /* Punctuation */                                                  \
   XX(LPAREN) /* ( */                                                 \
   XX(RPAREN) /* ) */                                                 \
   XX(LBRACE) /* { */                                                 \
   XX(RBRACE) /* } */                                                 \
   XX(LSBRACKET) /* [ */                                              \
   XX(RSBRACKET) /* ] */                                              \
   XX(LANGLE) /* < */                                                 \
   XX(RANGLE) /* > */                                                 \
   XX(COLON) /* : */                                                  \
   XX(COMMA) /* , */                                                  \
   XX(DOT) /* . */                                                    \
   XX(SEMI) /* ; */                                                   \
   /* Operators */                                                    \
   XX(PLUS) /* + */                                                   \
   XX(MINUS) /* - */                                                  \
   XX(TIMES) /* * */                                                  \
   XX(DIVIDE) /* / */                                                 \
   XX(ASSIGN) /* = */                                                 \
   XX(NOT) /* ! */                                                    \
   XX(ARROW) /* -> */                                                 \
   XX(ACCUM) /* += */                                                 \
   XX(EQUAL) /* == */                                                 \
   XX(NOT_EQUAL) /* != */                                             \
   XX(LEQ) /* <= */                                                   \
   XX(GEQ) /* >= */                                                   \
   /* Literals */                                                     \
   XX(TRUE)                                                           \
   XX(FALSE)

struct Token {
   enum Kind {
#define GA_CASE(X) X,
      GRAPHALG_ENUM_TOKEN_KIND(GA_CASE)
#undef GA_CASE
   };

   static llvm::StringLiteral kindName(Kind k);

   Kind type;
   mlir::Location loc;
   llvm::StringRef body = "";
};

mlir::LogicalResult lex(mlir::MLIRContext* ctx, llvm::StringRef buffer,
                        llvm::StringRef filename, int startLine, int startCol,
                        std::vector<Token>& tokens);

} // namespace graphalg
