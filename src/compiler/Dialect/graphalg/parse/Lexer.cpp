#include <optional>
#include <vector>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include "graphalg/parse/Lexer.h"

namespace graphalg {

namespace {

class Lexer {
   private:
   mlir::StringAttr _filename;
   llvm::StringRef _buffer;
   std::size_t _offset = 0;
   std::size_t _line;
   std::size_t _col;

   std::optional<char> cur() {
      if (_offset < _buffer.size()) {
         return _buffer[_offset];
      } else {
         return std::nullopt;
      }
   }

   std::optional<llvm::StringRef> peek(std::size_t n) {
      if (_offset + n < _buffer.size()) {
         return _buffer.substr(_offset, n);
      } else {
         return std::nullopt;
      }
   }

   struct LineCol {
      std::size_t line;
      std::size_t column;
   };
   LineCol currentPos() { return {_line, _col}; }

   mlir::FileLineColRange locFromTo(LineCol start, LineCol end) {
      return mlir::FileLineColRange::get(_filename, start.line, start.column,
                                         end.line, end.column);
   }

   mlir::FileLineColRange locFrom(LineCol from, std::size_t length) {
      return mlir::FileLineColRange::get(_filename, from.line, from.column,
                                         from.column + length);
   }

   void eat() {
      if (cur() == '\n') {
         _col = 1;
         _line += 1;
      } else {
         _col += 1;
      }

      _offset++;
   }

   bool tryEat(char c) {
      if (cur() == c) {
         eat();
         return true;
      } else {
         return false;
      }
   }

   void eatWhitespace();

   Token nextToken();

   public:
   Lexer(mlir::MLIRContext* ctx, llvm::StringRef buffer,
         llvm::StringRef filename, int startLine, int startCol)
      : _filename(mlir::StringAttr::get(ctx, filename)), _buffer(buffer),
        _line(startLine), _col(startCol) {}

   mlir::LogicalResult lex(std::vector<Token>& tokens);
};

} // namespace

llvm::StringLiteral Token::kindName(Kind k) {
   switch (k) {
#define GA_CASE(X) \
   case X:         \
      return #X;

      GRAPHALG_ENUM_TOKEN_KIND(GA_CASE)
#undef GA_CASE
   }
}

static std::optional<Token::Kind> tokenForKeyword(llvm::StringRef s) {
   auto token = llvm::StringSwitch<Token::Kind>(s)
                   .Case("func", Token::FUNC)
                   .Case("return", Token::RETURN)
                   .Case("for", Token::FOR)
                   .Case("in", Token::IN)
                   .Case("until", Token::UNTIL)
                   .Case("true", Token::TRUE)
                   .Case("false", Token::FALSE)
                   .Default(Token::END_OF_FILE);
   if (token == Token::END_OF_FILE) {
      return std::nullopt;
   } else {
      return token;
   }
}

void Lexer::eatWhitespace() {
   while (true) {
      if (cur() && llvm::isSpace(*cur())) {
         // Simple whitespace
         eat();
         continue;
      } else if (peek(2) == "//") {
         // Line comment
         while (cur() && cur() != '\n') {
            eat();
         }

         if (!cur()) {
            // End of input
            return;
         }

         assert(cur() == '\n');
         eat();
         continue;
      }

      break;
   }
}

Token Lexer::nextToken() {
   eatWhitespace();
   if (!cur()) {
      return Token{Token::END_OF_FILE, locFrom(currentPos(), 0)};
   }

   if (llvm::isAlpha(*cur())) {
      // Identifier [a-zA-Z][a-zA-Z0-9_']*
      auto startPos = currentPos();
      auto start = _offset;
      eat();
      while (cur() && (llvm::isAlnum(*cur()) || cur() == '_' || cur() == '\'')) {
         eat();
      }

      auto end = _offset;
      auto loc = locFromTo(startPos, currentPos());
      auto body = _buffer.slice(start, end);
      if (auto keyword = tokenForKeyword(body)) {
         return Token{*keyword, loc, body};
      }

      return Token{Token::IDENT, loc, body};
   }

   if (llvm::isDigit(*cur())) {
      // Number
      auto startPos = currentPos();
      auto start = _offset;
      eat();
      while (cur() && llvm::isDigit(*cur())) {
         eat();
      }

      auto kind = Token::INT;
      if (cur() == '.') {
         kind = Token::FLOAT;
         eat();
         while (cur() && llvm::isDigit(*cur())) {
            eat();
         }
      }

      auto end = _offset;
      auto loc = locFromTo(startPos, currentPos());
      auto body = _buffer.slice(start, end);
      return Token{kind, loc, body};
   }

   auto two = peek(2);
   auto locTwo = locFrom(currentPos(), 2);
#define TWO(c, t)                           \
   if (two == c) {                          \
      eat();                                \
      eat();                                \
      return Token{Token::t, locTwo, *two}; \
   }

   TWO("->", ARROW)
   TWO("+=", ACCUM)
   TWO("==", EQUAL)
   TWO("!=", NOT_EQUAL)
   TWO("<=", LEQ)
   TWO(">=", GEQ)
#undef TWO

   auto one = _buffer.substr(_offset, 1);
   auto locOne = locFrom(currentPos(), 1);
#define ONE(c, t)                          \
   if (tryEat(c)) {                        \
      return Token{Token::t, locOne, one}; \
   }

   ONE('(', LPAREN)
   ONE(')', RPAREN)
   ONE('{', LBRACE)
   ONE('}', RBRACE)
   ONE('[', LSBRACKET)
   ONE(']', RSBRACKET)
   ONE('<', LANGLE)
   ONE('>', RANGLE)
   ONE(':', COLON)
   ONE(',', COMMA)
   ONE('.', DOT)
   ONE(';', SEMI)
   ONE('+', PLUS)
   ONE('-', MINUS)
   ONE('*', TIMES)
   ONE('/', DIVIDE)
   ONE('=', ASSIGN)
   ONE('!', NOT)
#undef ONE

   mlir::emitError(locOne) << "invalid input character '" << *cur() << "'";
   return Token{Token::INVALID, locOne, _buffer.substr(_offset, 1)};
}

mlir::LogicalResult Lexer::lex(std::vector<Token>& tokens) {
   while (true) {
      auto token = nextToken();
      tokens.push_back(token);
      if (token.type == Token::END_OF_FILE) {
         break;
      } else if (token.type == Token::INVALID) {
         return mlir::failure();
      }
   }

   return mlir::success();
}

mlir::LogicalResult lex(mlir::MLIRContext* ctx, llvm::StringRef buffer,
                        llvm::StringRef filename, int startLine, int startCol,
                        std::vector<Token>& tokens) {
   Lexer lexer(ctx, buffer, filename, startLine, startCol);
   return lexer.lex(tokens);
}

} // namespace graphalg
