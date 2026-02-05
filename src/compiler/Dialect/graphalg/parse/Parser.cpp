#include <optional>
#include <utility>
#include <vector>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgCast.h"
#include "graphalg/GraphAlgDialect.h"
#include "graphalg/GraphAlgOps.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"
#include "graphalg/parse/Lexer.h"
#include "graphalg/parse/Parser.h"
#include "llvm/ADT/StringMap.h"

namespace graphalg {

namespace {

/** Maps dimension names in the program text to dimension symbols. */
class DimMapper {
   private:
   mlir::MLIRContext* _ctx;
   llvm::DenseMap<llvm::StringRef, DimAttr> _nameToDim;
   llvm::DenseMap<DimAttr, llvm::StringRef> _dimToName;

   public:
   DimMapper(mlir::MLIRContext* ctx);
   DimAttr getOrAllocate(llvm::StringRef s);
   llvm::StringRef getName(DimAttr dim) const;
};

class TypeFormatter {
   private:
   const DimMapper& _dimMapper;

   std::string _type;

   void formatScalar(mlir::Type t);
   void formatColumnVector(MatrixType t);
   void formatMatrix(MatrixType t);

   public:
   TypeFormatter(const DimMapper& dimMapper) : _dimMapper(dimMapper) {}

   void format(mlir::Type t);

   std::string take() { return std::move(_type); }
};

class Parser {
   private:
   llvm::ArrayRef<Token> _tokens;
   std::size_t _offset = 0;

   mlir::ModuleOp _module;
   mlir::OpBuilder _builder;

   // Value and location where the assignment happened
   struct VariableAssignment {
      mlir::Value value;
      std::optional<mlir::Location> loc;
   };
   llvm::ScopedHashTable<llvm::StringRef, VariableAssignment> _symbolTable;
   using VariableScope =
      llvm::ScopedHashTableScope<llvm::StringRef, VariableAssignment>;

   DimMapper _dimMapper;

   // Expected return type for the current function.
   mlir::Type _expectedReturnType;

   Token cur() { return _tokens[_offset]; }

   void eat() {
      _offset++;
      // NOTE: We should never eat the end of file token
      assert(_offset < _tokens.size() && "No next token");
   }

   mlir::ParseResult eatOrError(Token::Kind kind) {
      if (cur().type == kind) {
         eat();
         return mlir::success();
      } else {
         return mlir::emitError(cur().loc) << "expected " << Token::kindName(kind);
      }
   }

   std::string typeToString(mlir::Type t);
   std::string dimsToString(std::pair<DimAttr, DimAttr> dims);

   mlir::LogicalResult assign(mlir::Location loc, llvm::StringRef name,
                              mlir::Value value);
   bool isVarDefined(llvm::StringRef name) { return _symbolTable.count(name); }

   DimAttr inferDim(mlir::Value v, mlir::Location refLoc);

   mlir::ParseResult parseIdent(llvm::StringRef& s);
   mlir::ParseResult parseFuncRef(mlir::func::FuncOp& funcOp);
   mlir::ParseResult parseType(mlir::Type& t);
   mlir::ParseResult parseDim(DimAttr& t);

   mlir::Type tryParseSemiring();
   mlir::ParseResult parseSemiring(mlir::Type& s);

   mlir::ParseResult parseProgram();
   mlir::ParseResult parseFunction();
   mlir::ParseResult parseParams(llvm::SmallVectorImpl<llvm::StringRef>& names,
                                 llvm::SmallVectorImpl<mlir::Type>& types,
                                 llvm::SmallVectorImpl<mlir::Location>& locs);

   mlir::ParseResult parseBlock();
   mlir::ParseResult parseStmt();
   mlir::ParseResult parseStmtFor();
   mlir::ParseResult parseStmtReturn();
   mlir::ParseResult parseStmtAssign();
   mlir::ParseResult parseStmtAccum(mlir::Location baseLoc,
                                    llvm::StringRef baseName);

   struct ParsedMask {
      mlir::Value mask = nullptr;
      ;
      bool complement = false;

      bool isNone() const { return !mask; }
   };
   mlir::ParseResult parseMask(ParsedMask& mask);
   mlir::Value applyMask(mlir::Location baseLoc, mlir::Value base,
                         mlir::Location maskLoc, const ParsedMask& mask,
                         mlir::Location exprLoc, mlir::Value expr);

   enum class ParsedFill {
      /** A = v */
      NONE,
      /** A[:] = v */
      VECTOR,
      /** A[:, :] = v */
      MATRIX,
   };
   mlir::ParseResult parseFill(ParsedFill& fill);
   mlir::Value applyFill(mlir::Location baseLoc, mlir::Value base,
                         mlir::Location fillLoc, ParsedFill fill,
                         mlir::Location exprLoc, mlir::Value expr);

   struct ParsedRange {
      mlir::Value begin;
      mlir::Value end;
      DimAttr dim;
   };
   mlir::ParseResult parseRange(ParsedRange& r);

   mlir::ParseResult parseExpr(mlir::Value& v, int minPrec = 1);

   mlir::ParseResult parseBinaryOp(BinaryOp& op);

   /**
   * Check input types and build \c MatMulOp.
   *
   * @return \c nullptr if type checking fails.
   */
   mlir::Value buildMatMul(mlir::Location loc, mlir::Value lhs, mlir::Value rhs);

   /**
   * Check input types and build \c ElementWiseOp.
   *
   * @param mustBeScalar Whether we parsed (a <op> b) or (a (.<op>) b).
   *
   * @return \c nullptr if type checking fails.
   */
   mlir::Value buildElementWise(mlir::Location loc, mlir::Value lhs, BinaryOp op,
                                mlir::Value rhs, bool mustBeScalar);

   /**
   * Check input types and build \c ElementWiseApplyOp.
   *
   * @return \c nullptr if type checking fails.
   */
   mlir::Value buildElementWiseApply(mlir::Location loc, mlir::Value lhs,
                                     mlir::func::FuncOp funcOp, mlir::Value rhs);

   /**
   * Builds \c TransposeOp, \c NrowsOp, \c NcolsOp or \c NvalsOp depending on \c
   * property.
   */
   mlir::Value buildDotProperty(mlir::Location loc, mlir::Value value,
                                llvm::StringRef property);

   mlir::ParseResult parseAtom(mlir::Value& v);
   mlir::ParseResult parseAtomMatrix(mlir::Value& v);
   mlir::ParseResult parseAtomVector(mlir::Value& v);
   mlir::ParseResult parseAtomCast(mlir::Value& v);
   mlir::ParseResult parseAtomZero(mlir::Value& v);
   mlir::ParseResult parseAtomOne(mlir::Value& v);
   mlir::ParseResult parseAtomApply(mlir::Value& v);
   mlir::ParseResult parseAtomSelect(mlir::Value& v);
   mlir::ParseResult parseAtomReduceRows(mlir::Value& v);
   mlir::ParseResult parseAtomReduceCols(mlir::Value& v);
   mlir::ParseResult parseAtomReduce(mlir::Value& v);
   mlir::ParseResult parseAtomPickAny(mlir::Value& v);
   mlir::ParseResult parseAtomDiag(mlir::Value& v);
   mlir::ParseResult parseAtomTril(mlir::Value& v);
   mlir::ParseResult parseAtomTriu(mlir::Value& v);
   mlir::ParseResult parseAtomNot(mlir::Value& v);
   mlir::ParseResult parseAtomNeg(mlir::Value& v);

   mlir::ParseResult parseLiteral(mlir::Type ring, mlir::Value& v);

   public:
   Parser(llvm::ArrayRef<Token> tokens, mlir::ModuleOp module)
      : _tokens(tokens), _module(module), _builder(module.getContext())
        /* , _stringPool(_stringAllocator)  */
        ,
        _dimMapper(module.getContext()) {}

   mlir::LogicalResult parse();
};

} // namespace

DimMapper::DimMapper(mlir::MLIRContext* ctx) : _ctx(ctx) {
   // The dialect must be loaded before we can use dimension symbols
   ctx->getOrLoadDialect<GraphAlgDialect>();

   // Mapping for the special '1' dimension
   auto oneDim = DimAttr::getOne(ctx);
   _nameToDim["1"] = oneDim;
   _dimToName[oneDim] = "1";
}

DimAttr DimMapper::getOrAllocate(llvm::StringRef name) {
   auto it = _nameToDim.find(name);
   if (it != _nameToDim.end()) {
      // Already defined
      return it->second;
   }

   // New dim
   auto dim = DimAttr::newAbstract(_ctx);
   _nameToDim[name] = dim;
   _dimToName[dim] = name;
   return dim;
}

llvm::StringRef DimMapper::getName(DimAttr dim) const {
   assert(_dimToName.contains(dim));
   return _dimToName.at(dim);
}

void TypeFormatter::formatScalar(mlir::Type t) {
   if (t.isInteger(/*width=*/1)) {
      _type += "bool";
   } else if (t.isInteger(/*width=*/64)) {
      _type += "int";
   } else if (t.isF64()) {
      _type += "real";
   } else if (llvm::isa<DimType>(t)) {
      _type += "dim";
   } else if (llvm::isa<TropI64Type>(t)) {
      _type += "trop_int";
   } else if (llvm::isa<TropF64Type>(t)) {
      _type += "trop_real";
   } else if (llvm::isa<TropMaxI64Type>(t)) {
      _type += "trop_max_int";
   } else {
      _type += "!!! UNKNOWN TYPE (";
      _type += t.getAbstractType().getName();
      _type += ") !!!";
   }
}

void TypeFormatter::formatColumnVector(MatrixType t) {
   assert(t.isColumnVector());
   _type += "Vector<";

   // Rows
   _type += _dimMapper.getName(t.getRows());
   _type += ", ";

   formatScalar(t.getSemiring());
   _type += ">";
}

void TypeFormatter::formatMatrix(MatrixType t) {
   if (t.isScalar()) {
      return formatScalar(t.getSemiring());
   } else if (t.isColumnVector()) {
      return formatColumnVector(t);
   }

   _type += "Matrix<";

   // Rows
   _type += _dimMapper.getName(t.getRows());
   _type += ", ";

   // Columns
   _type += _dimMapper.getName(t.getCols());
   _type += ", ";

   formatScalar(t.getSemiring());
   _type += ">";
}

void TypeFormatter::format(mlir::Type t) {
   if (auto mat = llvm::dyn_cast<MatrixType>(t)) {
      formatMatrix(mat);
   } else {
      formatScalar(t);
   }
}

std::string Parser::typeToString(mlir::Type type) {
   TypeFormatter fmt(_dimMapper);
   fmt.format(type);
   return fmt.take();
}

std::string Parser::dimsToString(std::pair<DimAttr, DimAttr> dims) {
   auto [r, c] = dims;
   return "(" + _dimMapper.getName(r).str() + " x " +
      _dimMapper.getName(c).str() + ")";
}

mlir::LogicalResult Parser::assign(mlir::Location loc, llvm::StringRef name,
                                   mlir::Value value) {
   auto previous = _symbolTable.lookup(name);
   if (previous.value && previous.value.getType() != value.getType()) {
      auto diag = mlir::emitError(loc)
         << "cannot assign value of type "
         << typeToString(value.getType())
         << " to previously defined variable of type "
         << typeToString(previous.value.getType());
      diag.attachNote(previous.loc) << "previous assigment was here";
      return diag;
   }

   _symbolTable.insert(name, {value, loc});
   return mlir::success();
}

DimAttr Parser::inferDim(mlir::Value v, mlir::Location refLoc) {
   auto dimOp = v.getDefiningOp<CastDimOp>();
   if (!dimOp) {
      auto diag = mlir::emitError(refLoc) << "not a dimension type";
      diag.attachNote(v.getLoc()) << "defined here";
      return nullptr;
   }

   return dimOp.getInput();
}

mlir::ParseResult Parser::parseIdent(llvm::StringRef& s) {
   if (cur().type != Token::IDENT) {
      return mlir::emitError(cur().loc) << "expected identifier";
   }

   s = cur().body;
   return eatOrError(Token::IDENT);
}

mlir::ParseResult Parser::parseFuncRef(mlir::func::FuncOp& funcOp) {
   auto loc = cur().loc;
   llvm::StringRef name;
   if (parseIdent(name)) {
      return mlir::failure();
   }

   funcOp =
      llvm::dyn_cast_if_present<mlir::func::FuncOp>(_module.lookupSymbol(name));
   if (!funcOp) {
      return mlir::emitError(loc) << "unknown function '" << name << "'";
   }

   return mlir::success();
}

mlir::ParseResult Parser::parseType(mlir::Type& t) {
   auto* ctx = _builder.getContext();
   if (auto ring = tryParseSemiring()) {
      t = MatrixType::scalarOf(ring);
      return mlir::success();
   } else if (cur().type == Token::IDENT && cur().body == "Matrix") {
      // Matrix
      DimAttr rows;
      DimAttr cols;
      mlir::Type ring;
      if (eatOrError(Token::IDENT) || eatOrError(Token::LANGLE) ||
          parseDim(rows) || eatOrError(Token::COMMA) || parseDim(cols) ||
          eatOrError(Token::COMMA) || parseSemiring(ring) ||
          eatOrError(Token::RANGLE)) {
         return mlir::failure();
      }

      t = MatrixType::get(ctx, rows, cols, ring);
      return mlir::success();
   } else if (cur().type == Token::IDENT && cur().body == "Vector") {
      DimAttr rows;
      mlir::Type ring;
      if (eatOrError(Token::IDENT) || eatOrError(Token::LANGLE) ||
          parseDim(rows) || eatOrError(Token::COMMA) || parseSemiring(ring) ||
          eatOrError(Token::RANGLE)) {
         return mlir::failure();
      }

      t = MatrixType::get(ctx, rows, DimAttr::getOne(ctx), ring);
      return mlir::success();
   }

   return mlir::emitError(cur().loc)
      << "expected type such as 'int', 'Matrix<..>' or 'Vector<..>'";
}

mlir::ParseResult Parser::parseDim(DimAttr& t) {
   if (cur().type == Token::INT && cur().body == "1") {
      t = DimAttr::getOne(_builder.getContext());
      return eatOrError(Token::INT);
   } else if (cur().type == Token::IDENT) {
      t = _dimMapper.getOrAllocate(cur().body);
      return eatOrError(Token::IDENT);
   }

   return mlir::emitError(cur().loc)
      << "expected a dimension symbol such as 's' or '1'";
}

mlir::Type Parser::tryParseSemiring() {
   if (cur().type != Token::IDENT) {
      return nullptr;
   }

   auto name = cur().body;
   auto* ctx = _builder.getContext();
   auto ring = llvm::StringSwitch<mlir::Type>(name)
                  .Case("bool", SemiringTypes::forBool(ctx))
                  .Case("int", SemiringTypes::forInt(ctx))
                  .Case("real", SemiringTypes::forReal(ctx))
                  .Case("trop_int", SemiringTypes::forTropInt(ctx))
                  .Case("trop_real", SemiringTypes::forTropReal(ctx))
                  .Case("trop_max_int", SemiringTypes::forTropMaxInt(ctx))
                  .Default(nullptr);
   if (ring) {
      (void) eatOrError(Token::IDENT);
   }

   return ring;
}

mlir::ParseResult Parser::parseSemiring(mlir::Type& s) {
   s = tryParseSemiring();
   return mlir::success(!!s);
}

mlir::ParseResult Parser::parseProgram() {
   _builder.setInsertionPointToStart(_module.getBody());
   while (cur().type == Token::FUNC) {
      if (mlir::failed(parseFunction())) {
         return mlir::failure();
      }
   }

   if (cur().type != Token::END_OF_FILE) {
      auto diag = mlir::emitError(cur().loc)
         << "invalid top-level definition, expected keyword 'func'";
      diag.attachNote() << "only function definitions are allowed here";
      return diag;
   }

   return mlir::success();
}

static bool hasReturn(mlir::Block& block) {
   return block.mightHaveTerminator() &&
      llvm::isa<mlir::func::ReturnOp>(block.getTerminator());
}

mlir::ParseResult Parser::parseFunction() {
   llvm::StringRef name;
   llvm::SmallVector<llvm::StringRef> paramNames;
   llvm::SmallVector<mlir::Type> paramTypes;
   llvm::SmallVector<mlir::Location> paramLocs;
   mlir::Type returnType;
   auto loc = cur().loc;
   if (eatOrError(Token::FUNC) || parseIdent(name) ||
       parseParams(paramNames, paramTypes, paramLocs) ||
       eatOrError(Token::ARROW) || parseType(returnType)) {
      return mlir::failure();
   }

   if (auto* previousDef = _module.lookupSymbol(name)) {
      auto diag = mlir::emitError(loc)
         << "duplicate definition of function '" << name << "'";
      diag.attachNote(previousDef->getLoc()) << "original definition here";
      return diag;
   }

   // Create the new op.
   auto funcType = _builder.getFunctionType(paramTypes, {returnType});
   auto funcOp = _builder.create<mlir::func::FuncOp>(loc, name, funcType);

   // Populate the function body.
   mlir::OpBuilder::InsertionGuard guard(_builder);
   auto& entryBlock = funcOp.getFunctionBody().emplaceBlock();
   _builder.setInsertionPointToStart(&entryBlock);
   VariableScope functionScope(_symbolTable);
   for (const auto& [name, type, loc] :
        llvm::zip_equal(paramNames, paramTypes, paramLocs)) {
      auto arg = entryBlock.addArgument(type, loc);
      if (mlir::failed(assign(loc, name, arg))) {
         return mlir::failure();
      }
   }

   // Set expected return type for this function
   _expectedReturnType = returnType;

   if (mlir::failed(parseBlock())) {
      return mlir::failure();
   }

   // Check for return statement
   if (!hasReturn(entryBlock)) {
      return mlir::emitError(loc) << "function must have a return statement";
   }

   return mlir::success();
}

mlir::ParseResult
Parser::parseParams(llvm::SmallVectorImpl<llvm::StringRef>& names,
                    llvm::SmallVectorImpl<mlir::Type>& types,
                    llvm::SmallVectorImpl<mlir::Location>& locs) {
   if (eatOrError(Token::LPAREN)) {
      return mlir::failure();
   }

   if (cur().type == Token::RPAREN) {
      // No parameters
      return eatOrError(Token::RPAREN);
   }

   // First parameter
   auto& name = names.emplace_back();
   auto& type = types.emplace_back();
   auto loc = cur().loc;
   locs.emplace_back(loc);
   if (parseIdent(name) || eatOrError(Token::COLON) || parseType(type)) {
      return mlir::failure();
   }

   llvm::SmallDenseMap<llvm::StringRef, mlir::Location> previousParams = {
      {name, loc}};

   while (cur().type != Token::RPAREN) {
      // More parameters
      if (eatOrError(Token::COMMA)) {
         return mlir::failure();
      }

      auto& name = names.emplace_back();
      auto& type = types.emplace_back();
      auto loc = cur().loc;
      locs.emplace_back(loc);
      if (parseIdent(name) || eatOrError(Token::COLON) || parseType(type)) {
         return mlir::failure();
      }

      // Check for duplicate parameter names.
      if (previousParams.contains(name)) {
         auto diag = mlir::emitError(loc)
            << "duplicate parameter name '" << name << "'";
         diag.attachNote(previousParams.at(name)) << "previous definition here";
         return diag;
      }

      previousParams.insert({name, loc});
   }

   return eatOrError(Token::RPAREN);
}

mlir::ParseResult Parser::parseBlock() {
   if (eatOrError(Token::LBRACE)) {
      return mlir::failure();
   }

   while (cur().type != Token::RBRACE && cur().type != Token::END_OF_FILE) {
      if (parseStmt()) {
         return mlir::failure();
      }

      // Check if there are statements after a return
      if (hasReturn(*_builder.getInsertionBlock()) &&
          cur().type != Token::RBRACE) {
         return mlir::emitError(cur().loc)
            << "statement after return is not allowed";
      }
   }

   if (eatOrError(Token::RBRACE)) {
      return mlir::failure();
   }

   return mlir::success();
}

mlir::ParseResult Parser::parseStmt() {
   switch (cur().type) {
      case Token::FOR:
         return parseStmtFor();
      case Token::RETURN:
         return parseStmtReturn();
      case Token::IDENT:
         return parseStmtAssign();
      default:
         return mlir::emitError(cur().loc) << "invalid start for statement";
   }
}

static void
findModifiedBindingsInBlock(llvm::ArrayRef<Token> tokens, std::size_t offset,
                            llvm::SmallVectorImpl<llvm::StringRef>& bindings) {
   llvm::SmallDenseSet<llvm::StringRef> uniqueBindings;
   if (tokens[offset].type != Token::LBRACE) {
      // Block must start with {
      return;
   }

   offset++;
   std::size_t depth = 1;
   for (; offset + 1 < tokens.size() && depth > 0; offset++) {
      auto cur = tokens[offset];
      switch (cur.type) {
         case Token::IDENT: {
            auto name = cur.body;
            auto next = tokens[offset + 1];
            if (next.type == Token::ASSIGN || next.type == Token::ACCUM) {
               // An assignment or accumulate op
               auto [_, newlyAdded] = uniqueBindings.insert(name);
               if (newlyAdded) {
                  bindings.emplace_back(name);
               }
            }
         }
         case Token::LBRACE: {
            // Enter nested block.
            depth++;
            break;
         }
         case Token::RBRACE: {
            // End of block.
            depth--;
            break;
         }
         default:
            // Skip
            break;
      }
   }
}

mlir::ParseResult Parser::parseStmtFor() {
   auto loc = cur().loc;
   llvm::StringRef iterVarName;
   ParsedRange range;
   if (eatOrError(Token::FOR) || parseIdent(iterVarName) ||
       eatOrError(Token::IN) || parseRange(range)) {
      return mlir::failure();
   }

   // Find the variables modified inside the loop.
   llvm::SmallVector<llvm::StringRef> varNames;
   findModifiedBindingsInBlock(_tokens, _offset, varNames);
   // Only the variables that exist outside the loop are proper loop variables.
   auto removeIt = llvm::remove_if(
      varNames, [&](llvm::StringRef name) { return !isVarDefined(name); });
   varNames.erase(removeIt, varNames.end());
   llvm::SmallVector<mlir::Value> initArgs;
   llvm::SmallVector<mlir::Type> varTypes;
   for (auto name : varNames) {
      auto [value, _] = _symbolTable.lookup(name);
      assert(!!value);

      initArgs.emplace_back(value);
      varTypes.emplace_back(value.getType());
   }

   // Create the for op.
   mlir::Region* bodyRegion;
   mlir::Region* untilRegion;
   mlir::ValueRange results;
   if (range.dim) {
      auto forOp = _builder.create<ForDimOp>(loc, varTypes, initArgs, range.dim);
      bodyRegion = &forOp.getBody();
      untilRegion = &forOp.getUntil();
      results = forOp->getResults();
   } else {
      assert(range.begin && range.end);
      auto forOp = _builder.create<ForConstOp>(loc, varTypes, initArgs,
                                               range.begin, range.end);
      bodyRegion = &forOp.getBody();
      untilRegion = &forOp.getUntil();
      results = forOp->getResults();
   }

   // Create body.
   assert(bodyRegion->getBlocks().empty());
   auto& bodyBlock = bodyRegion->emplaceBlock();

   {
      // Builder and variable scope for this block
      VariableScope bodyScope(_symbolTable);
      mlir::OpBuilder::InsertionGuard guard(_builder);
      _builder.setInsertionPointToStart(&bodyBlock);
      // Define the iteration variable
      auto iterVar = bodyBlock.addArgument(
         MatrixType::scalarOf(SemiringTypes::forInt(_builder.getContext())),
         loc);
      if (mlir::failed(assign(loc, iterVarName, iterVar))) {
         return mlir::failure();
      }

      // Map variables to their values in the body block
      for (const auto& [name, type] : llvm::zip_equal(varNames, varTypes)) {
         auto var = bodyBlock.addArgument(type, loc);
         if (mlir::failed(assign(loc, name, var))) {
            return mlir::failure();
         }
      }

      // Parse body block.
      if (mlir::failed(parseBlock())) {
         return mlir::failure();
      }

      // yield the new values for the variables
      llvm::SmallVector<mlir::Value> yieldInputs;
      for (const auto& name : varNames) {
         auto [newValue, _] = _symbolTable.lookup(name);
         assert(!!newValue);
         yieldInputs.emplace_back(newValue);
      }
      _builder.create<YieldOp>(loc, yieldInputs);
   }

   if (cur().type == Token::UNTIL) {
      auto loc = cur().loc;
      eat();

      // Add until block
      assert(untilRegion->getBlocks().empty());
      auto& untilBlock = untilRegion->emplaceBlock();

      // Builder and variable scope for this block
      VariableScope bodyScope(_symbolTable);
      mlir::OpBuilder::InsertionGuard guard(_builder);
      _builder.setInsertionPointToStart(&untilBlock);
      // Define the iteration variable
      auto iterVar = untilBlock.addArgument(
         MatrixType::scalarOf(_builder.getI64Type()), loc);
      if (mlir::failed(assign(loc, iterVarName, iterVar))) {
         return mlir::failure();
      }

      // Map variables to their values in the until block
      for (const auto& [name, type] : llvm::zip_equal(varNames, varTypes)) {
         auto var = untilBlock.addArgument(type, loc);
         if (mlir::failed(assign(loc, name, var))) {
            return mlir::failure();
         }
      }

      // Parse condition expression.
      mlir::Value result;
      loc = cur().loc;
      if (parseExpr(result) || eatOrError(Token::SEMI)) {
         return mlir::failure();
      }

      auto resultType = llvm::dyn_cast<MatrixType>(result.getType());
      if (!resultType || !resultType.isScalar() || !resultType.isBoolean()) {
         return mlir::emitError(loc)
            << "loop condition does not produce a boolean scalar, got "
            << typeToString(resultType);
      }

      _builder.create<YieldOp>(loc, result);
   }

   // Use the updated variables from the block
   assert(results.size() == varNames.size());
   assert(results.getTypes() == varTypes);
   for (auto [name, value] : llvm::zip_equal(varNames, results)) {
      if (mlir::failed(assign(loc, name, value))) {
         return mlir::failure();
      }
   }

   return mlir::success();
}

mlir::ParseResult Parser::parseStmtReturn() {
   auto loc = cur().loc;

   // Check if return is inside a loop
   auto* parentOp = _builder.getInsertionBlock()->getParentOp();
   if (llvm::isa<ForConstOp, ForDimOp>(parentOp)) {
      return mlir::emitError(loc)
         << "return statement inside a loop is not allowed";
   }

   // The parser should not create any other nested ops apart from for loops, so
   // we should be at the top-level of a function scope.
   assert(llvm::isa<mlir::func::FuncOp>(parentOp) &&
          "return outside of function body");

   mlir::Value returnValue;
   if (eatOrError(Token::RETURN) || parseExpr(returnValue) ||
       eatOrError(Token::SEMI)) {
      return mlir::failure();
   }

   // Check return type matches
   if (returnValue.getType() != _expectedReturnType) {
      return mlir::emitError(loc)
         << "return type mismatch: expected "
         << typeToString(_expectedReturnType) << ", but got "
         << typeToString(returnValue.getType());
   }

   _builder.create<mlir::func::ReturnOp>(loc, returnValue);
   return mlir::success();
}

mlir::ParseResult Parser::parseStmtAssign() {
   auto loc = cur().loc;
   llvm::StringRef baseName;
   if (parseIdent(baseName)) {
      return mlir::failure();
   }

   if (cur().type == Token::ACCUM) {
      return parseStmtAccum(loc, baseName);
   }

   ParsedMask mask;
   auto maskLoc = cur().loc;
   if (cur().type == Token::LANGLE) {
      if (mlir::failed(parseMask(mask))) {
         return mlir::failure();
      }
   }

   auto fill = ParsedFill::NONE;
   auto fillLoc = cur().loc;
   if (cur().type == Token::LSBRACKET) {
      if (mlir::failed(parseFill(fill))) {
         return mlir::failure();
      }
   }

   if (eatOrError(Token::ASSIGN)) {
      return mlir::failure();
   }

   auto exprLoc = cur().loc;
   mlir::Value expr;
   if (parseExpr(expr) || eatOrError(Token::SEMI)) {
      return mlir::failure();
   }

   auto [baseValue, _] = _symbolTable.lookup(baseName);
   if (!baseValue) {
      // New variable
      if (fill != ParsedFill::NONE || !mask.isNone()) {
         // Cannot fill or mask if base was not already defined.
         return mlir::emitError(loc) << "undefined variable '" << baseName << "'";
      }

      // Regular assignment.
      return assign(loc, baseName, expr);
   }

   expr = applyFill(loc, baseValue, fillLoc, fill, exprLoc, expr);
   if (!expr) {
      return mlir::failure();
   }

   expr = applyMask(loc, baseValue, maskLoc, mask, exprLoc, expr);
   if (!expr) {
      return mlir::failure();
   }

   return assign(loc, baseName, expr);
}

mlir::ParseResult Parser::parseMask(ParsedMask& mask) {
   if (eatOrError(Token::LANGLE)) {
      return mlir::failure();
   }

   if (cur().type == Token::NOT) {
      eat();
      mask.complement = true;
   }

   auto loc = cur().loc;
   llvm::StringRef name;
   if (parseIdent(name)) {
      return mlir::failure();
   }

   auto [maskValue, _] = _symbolTable.lookup(name);
   if (!maskValue) {
      return mlir::emitError(loc) << "undefined variable '" << name << "'";
   }

   mask.mask = maskValue;
   return eatOrError(Token::RANGLE);
}

mlir::Value Parser::applyMask(mlir::Location baseLoc, mlir::Value base,
                              mlir::Location maskLoc, const ParsedMask& mask,
                              mlir::Location exprLoc, mlir::Value expr) {
   if (mask.isNone()) {
      // no mask to apply
      return expr;
   }

   auto baseType = llvm::cast<MatrixType>(base.getType());
   auto maskType = llvm::cast<MatrixType>(mask.mask.getType());
   auto exprType = llvm::cast<MatrixType>(expr.getType());

   if (baseType != exprType) {
      auto diag = mlir::emitError(baseLoc)
         << "base type does not match the value to assign";
      diag.attachNote(baseLoc) << "base type: " << typeToString(baseType);
      diag.attachNote(exprLoc) << "expression type: " << typeToString(exprType);
      return nullptr;
   } else if (!maskType.isBoolean()) {
      // TODO: Widen this to allow any semiring (using implicit cast)
      mlir::emitError(maskLoc)
         << "mask is not a boolean matrix: " << typeToString(maskType);
      return nullptr;
   }

   auto baseDims = baseType.getDims();
   auto maskDims = maskType.getDims();
   if (baseDims != maskDims) {
      auto diag = mlir::emitError(baseLoc)
         << "base dimensions do not match the dimensions of the mask";
      diag.attachNote(baseLoc) << "base dimension: " << dimsToString(baseDims);
      diag.attachNote(maskLoc) << "mask dimensions: " << dimsToString(maskDims);
      return nullptr;
   }

   return _builder.create<MaskOp>(maskLoc, base, mask.mask, expr,
                                  mask.complement);
}

mlir::ParseResult Parser::parseFill(ParsedFill& fill) {
   // [:
   if (eatOrError(Token::LSBRACKET) || eatOrError(Token::COLON)) {
      return mlir::failure();
   }

   if (cur().type == Token::COMMA) {
      fill = ParsedFill::MATRIX;

      // ,:]
      if (eatOrError(Token::COMMA) || eatOrError(Token::COLON) ||
          eatOrError(Token::RSBRACKET)) {
         return mlir::failure();
      }

      return mlir::success();
   } else {
      fill = ParsedFill::VECTOR;
      return eatOrError(Token::RSBRACKET);
   }
}

mlir::Value Parser::applyFill(mlir::Location baseLoc, mlir::Value base,
                              mlir::Location fillLoc, ParsedFill fill,
                              mlir::Location exprLoc, mlir::Value expr) {
   if (fill == ParsedFill::NONE) {
      // No fill to apply
      return expr;
   }

   auto baseType = llvm::cast<MatrixType>(base.getType());
   auto exprType = llvm::cast<MatrixType>(expr.getType());
   if (!exprType.isScalar()) {
      auto diag = mlir::emitError(exprLoc) << "fill expression is not a scalar";
      return nullptr;
   }

   auto baseRing = baseType.getSemiring();
   auto exprRing = exprType.getSemiring();
   if (baseRing != exprRing) {
      auto diag = mlir::emitError(baseLoc)
         << "base and fill expression have different semirings";
      diag.attachNote(exprLoc)
         << "fill expression has semiring " << typeToString(exprRing);
      diag.attachNote(baseLoc)
         << "base matrix has semiring " << typeToString(baseRing);
      return nullptr;
   }

   if (fill == ParsedFill::VECTOR && !baseType.isColumnVector()) {
      auto diag = mlir::emitError(fillLoc)
         << "vector fill [:] used with non-vector base";
      diag.attachNote(baseLoc) << "base has type " << typeToString(baseType);
      return nullptr;
   } else if (fill == ParsedFill::MATRIX && baseType.isColumnVector()) {
      auto diag = mlir::emitError(fillLoc)
         << "matrix fill [:, :] used with column vector base";
      diag.attachNote(baseLoc) << "base has type " << typeToString(baseType);
      return nullptr;
   }

   return _builder.create<BroadcastOp>(fillLoc, baseType, expr);
}

mlir::ParseResult Parser::parseStmtAccum(mlir::Location baseLoc,
                                         llvm::StringRef baseName) {
   mlir::Value expr;
   if (eatOrError(Token::ACCUM) || parseExpr(expr) || eatOrError(Token::SEMI)) {
      return mlir::failure();
   }

   auto [baseValue, _] = _symbolTable.lookup(baseName);
   if (!baseValue) {
      return mlir::emitError(baseLoc) << "undefined variable";
   } else if (baseValue.getType() != expr.getType()) {
      return mlir::emitError(baseLoc)
         << "type of base does not match the expression to accumulate: ("
         << typeToString(baseValue.getType()) << " vs. "
         << typeToString(expr.getType()) << ")";
   }

   // Rewrite a += b; to a = a (.+) b;
   auto result = mlir::Value(
      _builder.create<ElementWiseOp>(baseLoc, baseValue, BinaryOp::ADD, expr));
   return assign(baseLoc, baseName, result);
}

mlir::ParseResult Parser::parseRange(ParsedRange& r) {
   auto exprLoc = cur().loc;
   mlir::Value expr;
   if (parseExpr(expr)) {
      return mlir::failure();
   }

   if (cur().type == Token::COLON) {
      // Const range
      auto beginLoc = exprLoc;
      r.begin = expr;

      auto endLoc = cur().loc;
      if (eatOrError(Token::COLON) || parseExpr(r.end)) {
         return mlir::failure();
      }

      // Check that begin is an integer scalar
      auto intScalarType =
         MatrixType::scalarOf(SemiringTypes::forInt(_builder.getContext()));
      if (r.begin.getType() != intScalarType) {
         return mlir::emitError(beginLoc)
            << "loop range start must be an integer, but got "
            << typeToString(r.begin.getType());
      }

      // Check that end is an integer scalar
      if (r.end.getType() != intScalarType) {
         return mlir::emitError(endLoc)
            << "loop range end must be an integer, but got "
            << typeToString(r.end.getType());
      }

      return mlir::success();
   } else {
      r.dim = inferDim(expr, exprLoc);
      if (!r.dim) {
         return mlir::failure();
      }

      return mlir::success();
   }
}

static int precedenceForOp(Token::Kind op) {
   switch (op) {
      // NOTE: 1 for ewise
      case Token::EQUAL:
      case Token::NOT_EQUAL:
      case Token::LANGLE:
      case Token::RANGLE:
      case Token::LEQ:
      case Token::GEQ:
         return 2;
      case Token::PLUS:
      case Token::MINUS:
         return 3;
      case Token::TIMES:
      case Token::DIVIDE:
         return 4;
      case Token::DOT:
         return 5;
      default:
         // Not an op with precedence
         return 0;
   }
}

mlir::ParseResult Parser::parseExpr(mlir::Value& v, int minPrec) {
   mlir::Value atomLhs;
   if (parseAtom(atomLhs)) {
      return mlir::failure();
   }

   while (true) {
      bool ewise = cur().type == Token::LPAREN && _offset + 1 < _tokens.size() &&
         _tokens[_offset + 1].type == Token::DOT;
      int prec = ewise ? 1 : precedenceForOp(cur().type);
      if (!prec || prec < minPrec) {
         break;
      }

      int nextMinPrec = prec + 1;
      if (cur().type == Token::DOT) {
         // Matrix property
         eat();
         if (cur().type != Token::IDENT) {
            return mlir::emitError(cur().loc)
               << "expected matrix property such as 'nrows'";
         }

         auto loc = cur().loc;
         auto property = cur().body;
         eat();

         atomLhs = buildDotProperty(loc, atomLhs, property);
         if (!atomLhs) {
            return mlir::failure();
         }
      } else if (ewise) {
         eat(); // '('
         eat(); // '.'
         if (cur().type == Token::IDENT) {
            // element-wise function
            auto loc = cur().loc;
            mlir::func::FuncOp funcOp;
            mlir::Value atomRhs;
            if (parseFuncRef(funcOp) || eatOrError(Token::RPAREN) ||
                parseExpr(atomRhs, nextMinPrec)) {
               return mlir::failure();
            }

            atomLhs = buildElementWiseApply(loc, atomLhs, funcOp, atomRhs);
            if (!atomLhs) {
               return mlir::failure();
            }
         } else {
            // element-wise binop
            auto loc = cur().loc;
            BinaryOp binop;
            mlir::Value atomRhs;
            if (parseBinaryOp(binop) || eatOrError(Token::RPAREN) ||
                parseExpr(atomRhs, nextMinPrec)) {
               return mlir::failure();
            }

            atomLhs = buildElementWise(loc, atomLhs, binop, atomRhs,
                                       /*mustBeScalar=*/false);
            if (!atomLhs) {
               return mlir::failure();
            }
         }
      } else {
         // Binary operator
         auto loc = cur().loc;
         BinaryOp binop;
         mlir::Value atomRhs;
         if (parseBinaryOp(binop) || parseExpr(atomRhs, nextMinPrec)) {
            return mlir::failure();
         }

         if (binop == BinaryOp::MUL) {
            // Matmul special case
            atomLhs = buildMatMul(loc, atomLhs, atomRhs);
            if (!atomLhs) {
               return mlir::failure();
            }
         } else {
            atomLhs = buildElementWise(loc, atomLhs, binop, atomRhs,
                                       /*mustBeScalar=*/true);
            if (!atomLhs) {
               return mlir::failure();
            }
         }
      }
   }

   v = atomLhs;
   return mlir::success();
}

mlir::ParseResult Parser::parseBinaryOp(BinaryOp& op) {
   switch (cur().type) {
      case Token::PLUS:
         op = BinaryOp::ADD;
         break;
      case Token::MINUS:
         op = BinaryOp::SUB;
         break;
      case Token::TIMES:
         op = BinaryOp::MUL;
         break;
      case Token::DIVIDE:
         op = BinaryOp::DIV;
         break;
      case Token::EQUAL:
         op = BinaryOp::EQ;
         break;
      case Token::NOT_EQUAL:
         op = BinaryOp::NE;
         break;
      case Token::LANGLE:
         op = BinaryOp::LT;
         break;
      case Token::RANGLE:
         op = BinaryOp::GT;
         break;
      case Token::LEQ:
         op = BinaryOp::LE;
         break;
      case Token::GEQ:
         op = BinaryOp::GE;
         break;
      default:
         return mlir::emitError(cur().loc) << "expected a binary operator";
   }

   eat();
   return mlir::success();
}

mlir::Value Parser::buildMatMul(mlir::Location loc, mlir::Value lhs,
                                mlir::Value rhs) {
   auto lhsType = llvm::cast<MatrixType>(lhs.getType());
   auto rhsType = llvm::cast<MatrixType>(rhs.getType());
   if (lhsType.getSemiring() != rhsType.getSemiring()) {
      auto diag = mlir::emitError(loc)
         << "incompatible semirings for matrix multiply";
      diag.attachNote(lhs.getLoc())
         << "left side has semiring " << typeToString(lhsType.getSemiring());
      diag.attachNote(rhs.getLoc())
         << "right side has semiring " << typeToString(rhsType.getSemiring());
      return nullptr;
   }

   if (lhsType.getCols() == rhsType.getRows()) {
      return _builder.create<MatMulOp>(loc, lhs, rhs);
   } else if (lhsType.isColumnVector() &&
              lhsType.getRows() == rhsType.getRows()) {
      // Special case: Allow implicit column to row vector transpose.
      return _builder.create<VecMatMulOp>(loc, lhs, rhs);
   }

   auto diag = mlir::emitError(loc)
      << "incompatible dimensions for matrix multiply";
   diag.attachNote(lhs.getLoc())
      << "left side has dimensions " << dimsToString(lhsType.getDims());
   diag.attachNote(rhs.getLoc())
      << "right side has dimensions " << dimsToString(rhsType.getDims());
   return nullptr;
}

mlir::Value Parser::buildElementWise(mlir::Location loc, mlir::Value lhs,
                                     BinaryOp op, mlir::Value rhs,
                                     bool mustBeScalar) {
   auto lhsType = llvm::cast<MatrixType>(lhs.getType());
   auto rhsType = llvm::cast<MatrixType>(rhs.getType());

   // Check that semirings match
   if (lhsType.getSemiring() != rhsType.getSemiring()) {
      auto diag = mlir::emitError(loc) << "operands have different semirings";
      diag.attachNote(lhs.getLoc())
         << "left operand has semiring " << typeToString(lhsType.getSemiring());
      diag.attachNote(rhs.getLoc())
         << "right operand has semiring " << typeToString(rhsType.getSemiring());
      return nullptr;
   }

   if (mustBeScalar) {
      // Syntax a + b instead of a (.+) b, so require operands to be scalars.
      if (!lhsType.isScalar() || !rhsType.isScalar()) {
         auto diag = mlir::emitError(loc)
            << "operands are not scalar. Did you mean to use "
               "element-wise (a (.f) b) syntax?";
         diag.attachNote(lhs.getLoc())
            << "left operand has dimensions " << dimsToString(lhsType.getDims());
         diag.attachNote(rhs.getLoc())
            << "right operand has dimensions " << dimsToString(rhsType.getDims());
         return nullptr;
      }
   } else {
      // Dimensions must match
      if (lhsType.getDims() != rhsType.getDims()) {
         auto diag = mlir::emitError(loc) << "operands have different dimensions";
         diag.attachNote(lhs.getLoc())
            << "left operand has dimensions " << dimsToString(lhsType.getDims());
         diag.attachNote(rhs.getLoc())
            << "right operand has dimensions " << dimsToString(rhsType.getDims());
         return nullptr;
      }
   }

   // Additional validation for specific operators
   if (op == BinaryOp::SUB) {
      // Subtraction only supports int and real semirings
      auto* ctx = _builder.getContext();
      auto semiring = lhsType.getSemiring();
      if (semiring != SemiringTypes::forInt(ctx) &&
          semiring != SemiringTypes::forReal(ctx)) {
         auto diag = mlir::emitError(loc)
            << "subtraction is only supported for int and real semirings";
         diag.attachNote(lhs.getLoc())
            << "operands have semiring " << typeToString(semiring);
         return nullptr;
      }
   } else if (op == BinaryOp::DIV) {
      // Division only supports real semiring
      auto* ctx = _builder.getContext();
      auto semiring = lhsType.getSemiring();
      if (semiring != SemiringTypes::forReal(ctx)) {
         auto diag = mlir::emitError(loc)
            << "division is only supported for real semiring";
         diag.attachNote(lhs.getLoc())
            << "operands have semiring " << typeToString(semiring);
         return nullptr;
      }
   } else if (op == BinaryOp::LT || op == BinaryOp::GT || op == BinaryOp::LE ||
              op == BinaryOp::GE) {
      // Ordered compare only supports int, real semirings
      auto* ctx = _builder.getContext();
      auto semiring = lhsType.getSemiring();
      if (semiring != SemiringTypes::forInt(ctx) &&
          semiring != SemiringTypes::forReal(ctx)) {
         auto diag = mlir::emitError(loc) << "ordered compare is only supported "
                                             "for int and real semirings";
         diag.attachNote(lhs.getLoc())
            << "operands have semiring " << typeToString(semiring);
         return nullptr;
      }
   }

   return _builder.create<ElementWiseOp>(loc, lhs, op, rhs);
}

mlir::Value Parser::buildElementWiseApply(mlir::Location loc, mlir::Value lhs,
                                          mlir::func::FuncOp funcOp,
                                          mlir::Value rhs) {
   // Validate element-wise function application
   auto funcType = funcOp.getFunctionType();

   // Check that function takes exactly 2 parameters
   if (funcType.getNumInputs() != 2) {
      auto diag = mlir::emitError(loc)
         << "element-wise function application requires a function "
            "with 2 parameters, but got "
         << funcType.getNumInputs();
      diag.attachNote(funcOp.getLoc()) << "function defined here";
      return nullptr;
   }

   auto lhsType = llvm::cast<MatrixType>(lhs.getType());
   auto rhsType = llvm::cast<MatrixType>(rhs.getType());
   auto param0Type = llvm::cast<MatrixType>(funcType.getInput(0));
   auto param1Type = llvm::cast<MatrixType>(funcType.getInput(1));

   // Check that function parameters are scalars.
   if (!param0Type.isScalar() || !param1Type.isScalar()) {
      auto diag = mlir::emitError(loc)
         << "element-wise function application requires function "
            "parameters to be scalars";
      diag.attachNote(funcOp.getLoc())
         << "first parameter has type " << typeToString(param0Type);
      diag.attachNote(funcOp.getLoc())
         << "second parameter has type " << typeToString(param1Type);
      return nullptr;
   }

   // Check that operand dimensions match
   if (lhsType.getDims() != rhsType.getDims()) {
      auto diag = mlir::emitError(loc) << "operands have different dimensions";
      diag.attachNote(lhs.getLoc())
         << "left operand has dimensions " << dimsToString(lhsType.getDims());
      diag.attachNote(rhs.getLoc())
         << "right operand has dimensions " << dimsToString(rhsType.getDims());
      return nullptr;
   }

   // Check that operand semirings match function parameter semirings
   if (lhsType.getSemiring() != param0Type.getSemiring()) {
      auto diag = mlir::emitError(loc)
         << "left operand semiring does not match first parameter type";
      diag.attachNote(lhs.getLoc())
         << "left operand has semiring " << typeToString(lhsType.getSemiring());
      diag.attachNote(funcOp.getLoc()) << "first parameter has semiring "
                                       << typeToString(param0Type.getSemiring());
      return nullptr;
   }

   if (rhsType.getSemiring() != param1Type.getSemiring()) {
      auto diag =
         mlir::emitError(loc)
         << "right operand semiring does not match second parameter type";
      diag.attachNote(rhs.getLoc())
         << "right operand has semiring " << typeToString(rhsType.getSemiring());
      diag.attachNote(funcOp.getLoc()) << "second parameter has semiring "
                                       << typeToString(param1Type.getSemiring());
      return nullptr;
   }

   return _builder.create<ApplyElementWiseOp>(loc, funcOp, lhs, rhs);
}

mlir::Value Parser::buildDotProperty(mlir::Location loc, mlir::Value value,
                                     llvm::StringRef property) {
   if (property == "T") {
      return _builder.create<TransposeOp>(loc, value);
   } else if (property == "nrows") {
      auto matType = llvm::cast<MatrixType>(value.getType());
      return _builder.create<CastDimOp>(loc, matType.getRows());
   } else if (property == "ncols") {
      auto matType = llvm::cast<MatrixType>(value.getType());
      return _builder.create<CastDimOp>(loc, matType.getCols());
   } else if (property == "nvals") {
      return _builder.create<NValsOp>(loc, value);
   } else {
      mlir::emitError(loc) << "invalid matrix property";
      return nullptr;
   }
}

mlir::ParseResult Parser::parseAtom(mlir::Value& v) {
   auto loc = cur().loc;
   switch (cur().type) {
      case Token::LPAREN: {
         // (<expr>)
         if (eatOrError(Token::LPAREN) || parseExpr(v) ||
             eatOrError(Token::RPAREN)) {
            return mlir::failure();
         }

         return mlir::success();
      }
      case Token::IDENT: {
         // <ring>(<literal>)
         if (auto ring = tryParseSemiring()) {
            // e.g. int(42)
            if (eatOrError(Token::LPAREN) || parseLiteral(ring, v) ||
                eatOrError(Token::RPAREN)) {
               return mlir::failure();
            }

            return mlir::success();
         }

         llvm::StringRef name;
         if (parseIdent(name)) {
            return mlir::failure();
         }

         if (name == "Matrix") {
            return parseAtomMatrix(v);
         }

         if (name == "Vector") {
            return parseAtomVector(v);
         }

         if (name == "cast") {
            return parseAtomCast(v);
         }

         if (name == "zero") {
            return parseAtomZero(v);
         }

         if (name == "one") {
            return parseAtomOne(v);
         }

         if (name == "apply") {
            return parseAtomApply(v);
         }

         if (name == "select") {
            return parseAtomSelect(v);
         }

         if (name == "reduceRows") {
            return parseAtomReduceRows(v);
         }

         if (name == "reduceCols") {
            return parseAtomReduceCols(v);
         }

         if (name == "reduce") {
            return parseAtomReduce(v);
         }

         if (name == "pickAny") {
            return parseAtomPickAny(v);
         }

         if (name == "diag") {
            return parseAtomDiag(v);
         }

         // TODO: Make a separate extension
         if (name == "tril") {
            return parseAtomTril(v);
         }

         // TODO: Make a separate extension
         if (name == "triu") {
            return parseAtomTriu(v);
         }

         auto var = _symbolTable.lookup(name);
         if (!var.value) {
            return mlir::emitError(loc) << "unrecognized variable";
         }

         v = var.value;
         return mlir::success();
      }
      case Token::NOT:
         return parseAtomNot(v);
      case Token::MINUS:
         return parseAtomNeg(v);
      default:
         return mlir::emitError(cur().loc) << "invalid expression";
   }
}

static std::optional<llvm::APInt> parseInt(llvm::StringRef s) {
   // The largest possible 64-bit signed integer has 19 characters.
   constexpr int maxCharacters = 19;
   assert(std::to_string(std::numeric_limits<std::int64_t>::max()).size() ==
          maxCharacters);
   if (s.size() <= maxCharacters) {
      // 128 bits is enough for any 19 character radix 10 integer.
      llvm::APInt v(128, s, 10);
      if (v.getSignificantBits() <= 64) {
         // Fits in 64 bits.
         return v.trunc(64);
      }
   }

   return std::nullopt;
}

static std::optional<llvm::APFloat> parseFloat(llvm::StringRef s) {
   llvm::APFloat v(llvm::APFloat::IEEEdouble());
   auto result = v.convertFromString(s, llvm::APFloat::rmNearestTiesToEven);
   // Note: decimal literals may not be representable in exact form in IEEE
   // double format.
   auto allowedStatusMask = llvm::APFloat::opOK | llvm::APFloat::opInexact;
   if (result.takeError() || (*result & (~allowedStatusMask)) != 0) {
      return std::nullopt;
   }

   return v;
}

mlir::ParseResult Parser::parseAtomMatrix(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Type ring;
   mlir::Value rowsExpr;
   mlir::Value colsExpr;
   if (eatOrError(Token::LANGLE) || parseSemiring(ring) ||
       eatOrError(Token::RANGLE) || eatOrError(Token::LPAREN) ||
       parseExpr(rowsExpr) || eatOrError(Token::COMMA) || parseExpr(colsExpr) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto rows = inferDim(rowsExpr, loc);
   auto cols = inferDim(colsExpr, loc);
   if (!rows || !cols) {
      return mlir::failure();
   }

   v = _builder.create<ConstantMatrixOp>(
      loc, _builder.getType<MatrixType>(rows, cols, ring),
      llvm::cast<SemiringTypeInterface>(ring).addIdentity());
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomVector(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Type ring;
   mlir::Value rowsExpr;
   if (eatOrError(Token::LANGLE) || parseSemiring(ring) ||
       eatOrError(Token::RANGLE) || eatOrError(Token::LPAREN) ||
       parseExpr(rowsExpr) || eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto rows = inferDim(rowsExpr, loc);
   if (!rows) {
      return mlir::failure();
   }

   auto* ctx = _builder.getContext();
   v = _builder.create<ConstantMatrixOp>(
      loc, MatrixType::get(ctx, rows, DimAttr::getOne(ctx), ring),
      llvm::cast<SemiringTypeInterface>(ring).addIdentity());
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomCast(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Type ring;
   mlir::Value expr;
   if (eatOrError(Token::LANGLE) || parseSemiring(ring) ||
       eatOrError(Token::RANGLE) || eatOrError(Token::LPAREN) ||
       parseExpr(expr) || eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto exprType = llvm::cast<MatrixType>(expr.getType());
   auto* dialect = _builder.getContext()->getLoadedDialect<GraphAlgDialect>();
   if (!dialect->isCastLegal(exprType.getSemiring(), ring)) {
      return mlir::emitError(loc)
         << "invalid cast from " << typeToString(exprType.getSemiring())
         << " to " << typeToString(ring);
   }

   v = _builder.create<CastOp>(loc,
                               _builder.getType<MatrixType>(
                                  exprType.getRows(), exprType.getCols(), ring),
                               expr);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomZero(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Type ring;
   if (eatOrError(Token::LPAREN) || parseSemiring(ring) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto value = llvm::cast<SemiringTypeInterface>(ring).addIdentity();
   v = _builder.create<LiteralOp>(loc, value);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomOne(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Type ring;
   if (eatOrError(Token::LPAREN) || parseSemiring(ring) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto value = llvm::cast<SemiringTypeInterface>(ring).mulIdentity();
   v = _builder.create<LiteralOp>(loc, value);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomApply(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::func::FuncOp func;
   llvm::SmallVector<mlir::Value, 2> args(1);
   if (eatOrError(Token::LPAREN) || parseFuncRef(func) ||
       eatOrError(Token::COMMA) || parseExpr(args[0])) {
      return mlir::failure();
   }

   if (cur().type == Token::COMMA) {
      // Have a second arg.
      auto& arg = args.emplace_back();
      if (eatOrError(Token::COMMA) || parseExpr(arg)) {
         return mlir::failure();
      }
   }

   if (eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   // Validate function signature
   auto funcType = func.getFunctionType();
   size_t numFuncArgs = funcType.getNumInputs();

   if (args.size() == 1) {
      // Unary apply: function must have exactly 1 argument
      if (numFuncArgs != 1) {
         auto diag = mlir::emitError(loc)
            << "apply() with 1 matrix argument requires a function "
               "with 1 parameter, but got "
            << numFuncArgs;
         diag.attachNote(func.getLoc()) << "function defined here";
         return mlir::failure();
      }
   } else {
      // Binary apply: function must have exactly 2 arguments
      if (numFuncArgs != 2) {
         auto diag = mlir::emitError(loc)
            << "apply() with 2 arguments requires a function with 2 "
               "parameters, but got "
            << numFuncArgs;
         diag.attachNote(func.getLoc()) << "function defined here";
         return mlir::failure();
      }

      // Second argument must be a scalar
      auto arg1Type = llvm::cast<MatrixType>(args[1].getType());
      if (!arg1Type.isScalar()) {
         auto diag = mlir::emitError(loc)
            << "second argument to apply() must be a scalar";
         diag.attachNote(args[1].getLoc())
            << "argument has type " << typeToString(arg1Type);
         return mlir::failure();
      }
   }

   // Check that all function parameters are scalars
   for (size_t i = 0; i < numFuncArgs; i++) {
      auto paramType = llvm::dyn_cast<MatrixType>(funcType.getInput(i));
      if (!paramType || !paramType.isScalar()) {
         auto diag = mlir::emitError(loc)
            << "apply() requires function parameters to be scalars";
         diag.attachNote(func.getLoc()) << "parameter " << i << " has type "
                                        << typeToString(funcType.getInput(i));
         return mlir::failure();
      }
   }

   if (args.size() == 1) {
      v = _builder.create<ApplyUnaryOp>(loc, func, args[0]);
   } else {
      assert(args.size() == 2);
      v = _builder.create<ApplyBinaryOp>(loc, func, args[0], args[1]);
   }

   return mlir::success();
}

mlir::ParseResult Parser::parseAtomSelect(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::func::FuncOp func;
   llvm::SmallVector<mlir::Value, 2> args(1);
   if (eatOrError(Token::LPAREN) || parseFuncRef(func) ||
       eatOrError(Token::COMMA) || parseExpr(args[0])) {
      return mlir::failure();
   }

   if (cur().type == Token::COMMA) {
      // Have a second arg.
      auto& arg = args.emplace_back();
      if (eatOrError(Token::COMMA) || parseExpr(arg)) {
         return mlir::failure();
      }
   }

   if (eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   // Validate function signature
   auto funcType = func.getFunctionType();
   size_t numFuncArgs = funcType.getNumInputs();

   if (args.size() == 1) {
      // Unary select: function must have exactly 1 argument
      if (numFuncArgs != 1) {
         auto diag = mlir::emitError(loc)
            << "select() with 1 matrix argument requires a function "
               "with 1 parameter, but got "
            << numFuncArgs;
         diag.attachNote(func.getLoc()) << "function defined here";
         return mlir::failure();
      }
   } else {
      // Binary select: function must have exactly 2 arguments
      if (numFuncArgs != 2) {
         auto diag = mlir::emitError(loc)
            << "select() with 2 arguments requires a function with 2 "
               "parameters, but got "
            << numFuncArgs;
         diag.attachNote(func.getLoc()) << "function defined here";
         return mlir::failure();
      }

      // Second argument must be a scalar
      auto arg1Type = llvm::cast<MatrixType>(args[1].getType());
      if (!arg1Type.isScalar()) {
         auto diag = mlir::emitError(loc)
            << "second argument to select() must be a scalar";
         diag.attachNote(args[1].getLoc())
            << "argument has type " << typeToString(arg1Type);
         return mlir::failure();
      }
   }

   // Check that all function parameters are scalars
   for (size_t i = 0; i < numFuncArgs; i++) {
      auto paramType = llvm::dyn_cast<MatrixType>(funcType.getInput(i));
      if (!paramType || !paramType.isScalar()) {
         auto diag = mlir::emitError(loc)
            << "select() requires function parameters to be scalars";
         diag.attachNote(func.getLoc()) << "parameter " << i << " has type "
                                        << typeToString(funcType.getInput(i));
         return mlir::failure();
      }
   }

   // Check that function returns a boolean scalar
   if (funcType.getNumResults() != 1) {
      auto diag = mlir::emitError(loc)
         << "select() requires function to return exactly one value";
      diag.attachNote(func.getLoc()) << "function defined here";
      return mlir::failure();
   }

   auto returnType = llvm::dyn_cast<MatrixType>(funcType.getResult(0));
   auto* ctx = _builder.getContext();
   auto expectedReturnType = MatrixType::scalarOf(SemiringTypes::forBool(ctx));
   if (!returnType || returnType != expectedReturnType) {
      auto diag = mlir::emitError(loc)
         << "select() requires function to return bool";
      diag.attachNote(func.getLoc())
         << "function returns " << typeToString(funcType.getResult(0));
      return mlir::failure();
   }

   if (args.size() == 1) {
      v = _builder.create<SelectUnaryOp>(loc, func.getSymName(), args[0]);
   } else {
      assert(args.size() == 2);
      v = _builder.create<SelectBinaryOp>(loc, func.getSymName(), args[0],
                                          args[1]);
   }

   return mlir::success();
}

mlir::ParseResult Parser::parseAtomReduceRows(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Value arg;
   if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto inputType = llvm::cast<MatrixType>(arg.getType());
   auto* ctx = _builder.getContext();
   auto resultType = MatrixType::get(
      ctx, inputType.getRows(), DimAttr::getOne(ctx), inputType.getSemiring());
   v = _builder.create<ReduceOp>(loc, resultType, arg);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomReduceCols(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Value arg;
   if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto inputType = llvm::cast<MatrixType>(arg.getType());
   auto* ctx = _builder.getContext();
   auto resultType = MatrixType::get(
      ctx, DimAttr::getOne(ctx), inputType.getCols(), inputType.getSemiring());
   v = _builder.create<ReduceOp>(loc, resultType, arg);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomReduce(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Value arg;
   if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto inputType = llvm::cast<MatrixType>(arg.getType());
   v = _builder.create<ReduceOp>(loc, inputType.asScalar(), arg);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomPickAny(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Value arg;
   if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   v = _builder.create<PickAnyOp>(loc, arg);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomDiag(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Value arg;
   if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   auto argType = llvm::cast<MatrixType>(arg.getType());
   if (!argType.isColumnVector() && !argType.isRowVector()) {
      auto diag = mlir::emitError(loc)
         << "diag() requires a row or column vector";
      diag.attachNote(arg.getLoc())
         << "argument has type " << typeToString(argType);
      return mlir::failure();
   }

   v = _builder.create<DiagOp>(loc, arg);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomTril(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Value arg;
   if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   v = _builder.create<TrilOp>(loc, arg);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomTriu(mlir::Value& v) {
   auto loc = cur().loc;
   mlir::Value arg;
   if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
       eatOrError(Token::RPAREN)) {
      return mlir::failure();
   }

   v = _builder.create<TriuOp>(loc, arg);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomNot(mlir::Value& v) {
   auto loc = cur().loc;
   if (eatOrError(Token::NOT) || parseAtom(v)) {
      return mlir::failure();
   }

   // Check that NOT is only used with scalar types
   auto vType = llvm::cast<MatrixType>(v.getType());
   if (!vType.isScalar()) {
      auto diag = mlir::emitError(loc)
         << "not operator is only supported for scalar bool type";
      diag.attachNote(v.getLoc()) << "operand has type " << typeToString(vType);
      return mlir::failure();
   }

   // Check that NOT is only used with bool semiring
   auto semiring = vType.getSemiring();
   auto* ctx = _builder.getContext();
   if (semiring != SemiringTypes::forBool(ctx)) {
      auto diag = mlir::emitError(loc)
         << "not operator is only supported for bool type";
      diag.attachNote(v.getLoc())
         << "operand has semiring " << typeToString(semiring);
      return mlir::failure();
   }

   v = _builder.create<NotOp>(loc, v);
   return mlir::success();
}

mlir::ParseResult Parser::parseAtomNeg(mlir::Value& v) {
   auto loc = cur().loc;
   if (eatOrError(Token::MINUS) || parseAtom(v)) {
      return mlir::failure();
   }

   // Check that negation is only used with scalar types
   auto vType = llvm::cast<MatrixType>(v.getType());
   if (!vType.isScalar()) {
      auto diag = mlir::emitError(loc)
         << "negation is only supported for scalar types";
      diag.attachNote(v.getLoc()) << "operand has type " << typeToString(vType);
      return mlir::failure();
   }

   // Check that negation is only used with int or real semirings
   auto semiring = vType.getSemiring();
   auto* ctx = _builder.getContext();
   if (semiring != SemiringTypes::forInt(ctx) &&
       semiring != SemiringTypes::forReal(ctx)) {
      auto diag = mlir::emitError(loc)
         << "negation is only supported for int and real types";
      diag.attachNote(v.getLoc())
         << "operand has semiring " << typeToString(semiring);
      return mlir::failure();
   }

   v = _builder.create<NegOp>(loc, v);
   return mlir::success();
}

mlir::ParseResult Parser::parseLiteral(mlir::Type ring, mlir::Value& v) {
   auto* ctx = _builder.getContext();
   mlir::TypedAttr attr;
   if (ring == SemiringTypes::forBool(ctx)) {
      if (cur().type == Token::FALSE) {
         attr = _builder.getBoolAttr(false);
      } else if (cur().type == Token::TRUE) {
         attr = _builder.getBoolAttr(true);
      } else {
         return mlir::emitError(cur().loc) << "expected 'true' or 'false'";
      }
   } else if (ring == SemiringTypes::forInt(ctx) ||
              ring == SemiringTypes::forTropInt(ctx) ||
              ring == SemiringTypes::forTropMaxInt(ctx)) {
      if (cur().type != Token::INT) {
         return mlir::emitError(cur().loc) << "expected an integer value";
      }

      if (auto v = parseInt(cur().body)) {
         auto intAttr = _builder.getIntegerAttr(SemiringTypes::forInt(ctx), *v);
         if (intAttr.getType() == ring) {
            attr = intAttr;
         } else {
            attr = TropIntAttr::get(ctx, ring, intAttr);
         }
      } else {
         return mlir::emitError(cur().loc) << "integer does not fit in 64 bits";
      }
   } else if (ring == SemiringTypes::forReal(ctx) ||
              ring == SemiringTypes::forTropReal(ctx)) {
      if (cur().type != Token::FLOAT) {
         return mlir::emitError(cur().loc) << "expected a floating-point value";
      }

      if (auto v = parseFloat(cur().body)) {
         auto floatAttr = _builder.getFloatAttr(SemiringTypes::forReal(ctx), *v);
         if (floatAttr.getType() == ring) {
            attr = floatAttr;
         } else {
            attr = TropFloatAttr::get(ctx, ring, floatAttr);
         }
      } else {
         return mlir::emitError(cur().loc) << "invalid floating-point literal";
      }
   } else {
      llvm_unreachable("Invalid semiring");
   }

   assert(!!attr);
   // Eat the literal token
   eat();
   v = _builder.create<LiteralOp>(cur().loc, MatrixType::scalarOf(ring), attr);
   return mlir::success();
}

mlir::LogicalResult Parser::parse() {
   auto* ctx = _builder.getContext();
   ctx->getOrLoadDialect<graphalg::GraphAlgDialect>();
   ctx->getOrLoadDialect<mlir::func::FuncDialect>();
   return parseProgram();
}

mlir::LogicalResult parse(llvm::StringRef program, mlir::ModuleOp moduleOp) {
   llvm::StringRef filename = "<unknown>";
   int startLine = 0;
   int startCol = 0;
   auto loc = llvm::dyn_cast<mlir::FileLineColLoc>(moduleOp.getLoc());
   if (loc) {
      filename = loc.getFilename();
      startLine = loc.getLine();
      startCol = loc.getColumn();
   }

   std::vector<Token> tokens;
   if (mlir::failed(lex(moduleOp->getContext(), program, filename, startLine,
                        startCol, tokens))) {
      return mlir::failure();
   }

   Parser parser(tokens, moduleOp);
   if (mlir::failed(parser.parse())) {
      return mlir::failure();
   }

   return mlir::success();
}

} // namespace graphalg
