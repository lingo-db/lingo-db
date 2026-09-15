#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <lingodb/compiler/Dialect/util/UtilOps.h>
#include "lingodb/compiler/helper.h"
#include "llvm/TargetParser/Host.h"
#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
namespace {
using namespace lingodb::compiler::dialect;
class SplitConstLike : public mlir::RewritePattern {
   public:
   SplitConstLike(mlir::MLIRContext* context) : RewritePattern(util::LikeOp::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto likeOp = mlir::cast<util::LikeOp>(op);
      auto loc = op->getLoc();
      int32_t sums = 0;
      if (mlir::DictionaryAttr prefixDictAttr = likeOp.getPrefixAttr()) {
         if (mlir::StringAttr prefixAttr = prefixDictAttr.getAs<mlir::StringAttr>("prefix"))
            sums += prefixAttr.size();
      }
      if (mlir::DictionaryAttr suffixDictAttr = likeOp.getSuffixAttr()) {
         if (mlir::StringAttr suffixAttr = suffixDictAttr.getAs<mlir::StringAttr>("suffix"))
            sums += suffixAttr.size();
      }

      for (mlir::Attribute subpattern: likeOp.getSubpatterns()) {
         auto subpatternAttr = mlir::cast<mlir::DictionaryAttr>(subpattern).getAs<mlir::StringAttr>("subpattern");
         sums += subpatternAttr.size();
      }
      int32_t totalSums = sums;

      struct Step { mlir::Value cond, startIndex, endIndex; };
      mlir::Value sumsConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sums);

      auto createOp = likeOp.getStr().getDefiningOp<util::CreateVarLen>();

      mlir::Value haystack = likeOp.getStr();
      mlir::Value haystackPtr = createOp ? createOp.getRef() : rewriter.create<util::VarLenGetRef>(loc, util::RefType::get(rewriter.getContext(), rewriter.getI8Type()), haystack);
      mlir::Value bytesVal = rewriter.create<util::VarLenGetInlinedString>(loc, rewriter.getIntegerType(128), haystack);

      llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>> inlinedSteps;
      llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>> ptrSteps;

      auto loadFromPtrFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value value) -> mlir::Value {
         mlir::Value ptrPos = opBuilder.create<util::ArrayElementPtrOp>(location, util::RefType::get(opBuilder.getContext(), opBuilder.getI8Type()), value, startIndex);
         return opBuilder.create<util::LoadOp>(location, opBuilder.getI8Type(), ptrPos);
      };

      auto loadFromInlinedFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value value) -> mlir::Value {
         mlir::Type i128 = opBuilder.getIntegerType(128);
         mlir::Value idx = opBuilder.create<mlir::arith::IndexCastUIOp>(location, i128, startIndex);
         mlir::Value eight = opBuilder.create<mlir::arith::ConstantIntOp>(location, 8, i128);
         mlir::Value bitOff = opBuilder.create<mlir::arith::MulIOp>(location, idx, eight);
         mlir::Value shifted = opBuilder.create<mlir::arith::ShRUIOp>(location, value, bitOff);
         return opBuilder.create<mlir::arith::TruncIOp>(location, opBuilder.getI8Type(), shifted);
      };

      auto skipCodepointStep = [](auto loadByteFn, int32_t currentSums, mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value) -> Step {
         mlir::Value byteVal = loadByteFn(opBuilder, location, startIndex, value);
         mlir::Value codePointLength = opBuilder.create<util::GetUTF8CodeLenOp>(location, opBuilder.getIndexType(), byteVal);
         mlir::Value newStartIdx = opBuilder.create<mlir::arith::AddIOp>(location, startIndex, codePointLength);
         mlir::Value currentSumValue = opBuilder.create<mlir::arith::ConstantIndexOp>(location, currentSums);
         mlir::Value additionResult = opBuilder.create<mlir::arith::AddIOp>(location, newStartIdx, currentSumValue);
         mlir::Value isMatch = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::ule, additionResult, endIndex);
         return {isMatch, newStartIdx, endIndex};
      };

      if (mlir::DictionaryAttr prefixDictAttr = likeOp.getPrefixAttr()) {
         std::string prefix = prefixDictAttr.getAs<mlir::StringAttr>("prefix").str();
         llvm::ArrayRef<int> skips = prefixDictAttr.getAs<mlir::DenseI32ArrayAttr>("skip").asArrayRef();

         auto startsWithStep = [](auto startsWithFn, mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr prefixAttr) -> Step {
            mlir::Value isMatch = startsWithFn(opBuilder, location, startIndex, endIndex, value, prefixAttr);
            mlir::Value prefixSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, prefixAttr.size());
            mlir::Value newStartIndex = opBuilder.create<mlir::arith::AddIOp>(location, startIndex, prefixSizeConst);
            return {isMatch, newStartIndex, endIndex};
         };

         auto startsWithPtrFn = []( mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr prefixAttr) -> mlir::Value {
            return opBuilder.create<util::StringStartsWith>(location, opBuilder.getI1Type(), value, prefixAttr, startIndex, endIndex);
         };

         auto startsWithInlineFn = []( mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr prefixAttr) -> mlir::Value {
            return opBuilder.create<util::BytesStartsWith>(location, opBuilder.getI1Type(), value, prefixAttr, startIndex, endIndex);
         };

         int32_t cursor = 0;
         for (int32_t pos: skips) {
            std::string prefixLiteral = prefix.substr(cursor, pos - cursor);
            if (!prefixLiteral.empty()) {
               sums -= prefixLiteral.size();
               mlir::StringAttr prefixAttr = rewriter.getStringAttr(prefixLiteral);
               ptrSteps.push_back([startsWithStep, startsWithPtrFn, prefixAttr, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                  return startsWithStep(startsWithPtrFn, opBuilder, location, startIndex, endIndex, haystackPtr, prefixAttr);
               });
               inlinedSteps.push_back([startsWithStep, startsWithInlineFn, prefixAttr, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                  return startsWithStep(startsWithInlineFn, opBuilder, location, startIndex, endIndex, bytesVal, prefixAttr);
               });
            }
            sums--;
            ptrSteps.push_back([skipCodepointStep, loadFromPtrFn, sums, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return skipCodepointStep(loadFromPtrFn, sums, opBuilder, location, startIndex, endIndex, haystackPtr);
            });
            inlinedSteps.push_back([skipCodepointStep, loadFromInlinedFn, sums, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return skipCodepointStep(loadFromInlinedFn, sums, opBuilder, location, startIndex, endIndex, bytesVal);
            });
            cursor = pos + 1;
         }
         std::string prefixLiteral = prefix.substr(cursor);
         if (!prefixLiteral.empty()) {
            sums -= prefixLiteral.size();
            mlir::StringAttr prefixAttr = rewriter.getStringAttr(prefixLiteral);
            ptrSteps.push_back([startsWithStep, startsWithPtrFn, prefixAttr, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return startsWithStep(startsWithPtrFn, opBuilder, location, startIndex, endIndex, haystackPtr, prefixAttr);
            });
            inlinedSteps.push_back([startsWithStep, startsWithInlineFn, prefixAttr, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return startsWithStep(startsWithInlineFn, opBuilder, location, startIndex, endIndex, bytesVal, prefixAttr);
            });
         }
      }

      if (mlir::DictionaryAttr suffixDictAttr = likeOp.getSuffixAttr()) {
         std::string suffix = suffixDictAttr.getAs<mlir::StringAttr>("suffix").str();
         llvm::ArrayRef<int> skips = suffixDictAttr.getAs<mlir::DenseI32ArrayAttr>("skip").asArrayRef();

         auto moveToCharacterStartStep = [](auto loadByteFn, int32_t currentSums, mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value) -> Step {
            mlir::Value mask = opBuilder.create<mlir::arith::ConstantIntOp>(location, 0xC0, opBuilder.getI8Type());
            mlir::Value cont = opBuilder.create<mlir::arith::ConstantIntOp>(location, 0x80, opBuilder.getI8Type());
            mlir::Value one = opBuilder.create<mlir::arith::ConstantIndexOp>(location, 1);
            mlir::Value endIndexMinusOne = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, one);

            // so on a malformed UTF-8 byte this can go horribly wrong
            // but in the runtime functions we did not have any checks anyway
            auto whileOp = opBuilder.create<mlir::scf::WhileOp>(location, mlir::TypeRange{opBuilder.getIndexType()}, mlir::ValueRange{endIndexMinusOne},
                [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::ValueRange args) {
                   mlir::Value idx = args[0];
                   mlir::Value byte = loadByteFn(opBuilder2, location2, idx, value);
                   mlir::Value masked = opBuilder2.create<mlir::arith::AndIOp>(location2, byte, mask);
                   mlir::Value isCont = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, masked, cont);
                   opBuilder2.create<mlir::scf::ConditionOp>(location2, isCont, idx);
                },
                [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::ValueRange args) {
                   mlir::Value prev = opBuilder2.create<mlir::arith::SubIOp>(location2, args[0], one);
                   opBuilder2.create<mlir::scf::YieldOp>(location2, prev);
                });
            mlir::Value newEndIndex = whileOp.getResult(0);
            mlir::Value currentSumValue = opBuilder.create<mlir::arith::ConstantIndexOp>(location, currentSums);
            mlir::Value additionResult = opBuilder.create<mlir::arith::AddIOp>(location, startIndex, currentSumValue);
            mlir::Value isMatch = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::ule, additionResult, newEndIndex);
            return {isMatch, startIndex, newEndIndex};
         };

         auto endsWithStep = [](auto endsWithFn, mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr suffixAttr) -> Step {
            mlir::Value isMatch = endsWithFn(opBuilder, location, startIndex, endIndex, value, suffixAttr);
            mlir::Value suffixSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, suffixAttr.size());
            mlir::Value newEndIndex =  opBuilder.create<mlir::arith::SubIOp>(location, endIndex, suffixSizeConst);
            return {isMatch, startIndex, newEndIndex};
         };

         auto endsWithPtrFn = []( mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr suffixAttr) -> mlir::Value {
            return opBuilder.create<util::StringEndsWith>(location, opBuilder.getI1Type(), value, suffixAttr, startIndex, endIndex);
         };

         auto endsWithInlineFn = []( mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr suffixAttr) -> mlir::Value {
            return opBuilder.create<util::BytesEndsWith>(location, opBuilder.getI1Type(), value, suffixAttr, startIndex, endIndex);
         };

         int32_t cursor = suffix.size();
         for (int32_t pos : llvm::reverse(skips)) {
            std::string suffixLiteral = suffix.substr(pos + 1, cursor - pos - 1);
            if (!suffixLiteral.empty()) {
               sums -= suffixLiteral.size();
               mlir::StringAttr suffixAttr = rewriter.getStringAttr(suffixLiteral);
               ptrSteps.push_back([endsWithStep, endsWithPtrFn, suffixAttr, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                  return endsWithStep(endsWithPtrFn, opBuilder, location, startIndex, endIndex, haystackPtr, suffixAttr);
               });
               inlinedSteps.push_back([endsWithStep, endsWithInlineFn, suffixAttr, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                  return endsWithStep(endsWithInlineFn, opBuilder, location, startIndex, endIndex, bytesVal, suffixAttr);
               });
            }

            sums--;
            ptrSteps.push_back([moveToCharacterStartStep, loadFromPtrFn, sums, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return moveToCharacterStartStep(loadFromPtrFn, sums, opBuilder, location, startIndex, endIndex, haystackPtr);
            });
            inlinedSteps.push_back([moveToCharacterStartStep, loadFromInlinedFn, sums, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return moveToCharacterStartStep(loadFromInlinedFn, sums, opBuilder, location, startIndex, endIndex, bytesVal);
            });
            cursor = pos;
         }
         if (cursor != 0) {
            std::string suffixLiteral = suffix.substr(0, cursor);
            sums -= suffixLiteral.size();
            mlir::StringAttr suffixAttr = rewriter.getStringAttr(suffixLiteral);
            ptrSteps.push_back([endsWithStep, endsWithPtrFn, suffixAttr, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return endsWithStep(endsWithPtrFn, opBuilder, location, startIndex, endIndex, haystackPtr, suffixAttr);
            });
            inlinedSteps.push_back([endsWithStep, endsWithInlineFn, suffixAttr, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return endsWithStep(endsWithInlineFn, opBuilder, location, startIndex, endIndex, bytesVal, suffixAttr);
            });
         }
      }

      for (mlir::Attribute attr: likeOp.getSubpatterns()) {
         mlir::DictionaryAttr dictAttr =  mlir::cast<mlir::DictionaryAttr>(attr);
         std::string subpattern = dictAttr.getAs<mlir::StringAttr>("subpattern").str();
         llvm::ArrayRef<int> skips = dictAttr.getAs<mlir::DenseI32ArrayAttr>("skip").asArrayRef();

         auto containsStep = [](auto containsFn, int32_t currentSums, mlir::StringAttr subpatternAttr, mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value) -> Step {
            mlir::Value sumsValue = opBuilder.create<mlir::arith::ConstantIndexOp>(location, currentSums);
            mlir::Value earlyStopEndIndex = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, sumsValue);
            auto containsRet = containsFn(opBuilder, location, startIndex, earlyStopEndIndex, value, subpatternAttr);
            mlir::Value subpatternSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, subpatternAttr.size());
            mlir::Value newStartIndex = opBuilder.create<mlir::arith::AddIOp>(location, containsRet.getMatchStart(), subpatternSizeConst);
            return {containsRet.getContains(), newStartIndex, endIndex};
         };

         auto containsPtrFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr subpatternAttr) {
            return opBuilder.create<util::StringContains>(location, opBuilder.getI1Type(), opBuilder.getIndexType(), value, subpatternAttr, startIndex, endIndex);
         };

         auto containsInlineFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr subpatternAttr) {
            return opBuilder.create<util::BytesContains>(location, opBuilder.getI1Type(), opBuilder.getIndexType(), value, subpatternAttr, startIndex, endIndex);
         };

         auto findFirstNonMatchingPtrFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr subpatternAttr) -> mlir::Value {
            return opBuilder.create<util::StringNonMatchIndex>(location, opBuilder.getIndexType(), value, subpatternAttr, startIndex, endIndex);
         };

         auto findFirstNonMatchingInlineFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::StringAttr subpatternAttr) -> mlir::Value {
            return opBuilder.create<util::BytesNonMatchIndex>(location, opBuilder.getIndexType(), value, subpatternAttr, startIndex, endIndex);
         };

         auto memchrPtrFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::Value c) {
            mlir::Value memchrStart = opBuilder.create<util::ArrayElementPtrOp>(location, util::RefType::get(opBuilder.getContext(), opBuilder.getI8Type()), value, startIndex);
            mlir::Value memchrLen = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, startIndex);
            return opBuilder.create<util::RefMemchr>(location, opBuilder.getI1Type(), opBuilder.getIndexType(), memchrStart, memchrLen, c);
         };

         auto memchrInlineFn = [](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value, mlir::Value c) {
            mlir::Value eight = opBuilder.create<mlir::arith::ConstantIndexOp>(location, 8);
            mlir::Value bitOffIdx = opBuilder.create<mlir::arith::MulIOp>(location, startIndex, eight);
            mlir::Value bitOff = opBuilder.create<mlir::arith::IndexCastUIOp>(location, opBuilder.getIntegerType(128), bitOffIdx);
            mlir::Value memchrValue = opBuilder.create<mlir::arith::ShRUIOp>(location, value, bitOff);
            mlir::Value memchrLen = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, startIndex);
            return opBuilder.create<util::InlineMemchr>(location, opBuilder.getI1Type(), opBuilder.getIndexType(), memchrValue, memchrLen, c);
         };

         auto whileStep = [skipCodepointStep](auto containsFn, auto findFirstNotMatchingFn, auto loadByteFn, auto memchrFn, llvm::SmallVector<int32_t> skips, mlir::StringAttr subpatternAttr, int32_t sums, mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex, mlir::Value value) -> Step {
            llvm::StringRef subpattern = subpatternAttr.getValue();
            int32_t subpatternSize = static_cast<int32_t>(subpattern.size());
            mlir::Value sumsConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, sums);
            mlir::Value currentEndIndex = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, sumsConst);

            // guaranteed: subpattern[0] is not an underscore
            int32_t anchorEnd = skips[0];
            mlir::StringAttr anchorAttr = opBuilder.getStringAttr(subpattern.substr(0, anchorEnd));

            mlir::Value statusRunning = opBuilder.create<mlir::arith::ConstantOp>(location, opBuilder.getI8Type(), opBuilder.getI8IntegerAttr(0));
            mlir::Value statusFound = opBuilder.create<mlir::arith::ConstantOp>(location, opBuilder.getI8Type(), opBuilder.getI8IntegerAttr(1));
            mlir::Value statusFailed = opBuilder.create<mlir::arith::ConstantOp>(location, opBuilder.getI8Type(), opBuilder.getI8IntegerAttr(2));

            // the tail: alternating underscores and literals, unrolled at compile time.
            llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>> tailSteps;
            llvm::SmallVector<int32_t> maxNumStepsParsed;
            llvm::SmallVector<bool> isUnderscore;
            llvm::SmallVector<mlir::StringAttr> literals;
            llvm::SmallVector<int32_t> minNumCharsLeft;
            int32_t tailSums = subpatternSize - anchorEnd;
            {
               maxNumStepsParsed.push_back(0);
               minNumCharsLeft.push_back(subpatternSize - anchorEnd);

               int32_t cursorPos = anchorEnd;
               size_t skipIdx = 0;
               while (cursorPos < subpatternSize) {
                  if (skipIdx < skips.size() && skips[skipIdx] == cursorPos) {
                     --tailSums;
                     literals.push_back(opBuilder.getStringAttr("_"));
                     maxNumStepsParsed.push_back(maxNumStepsParsed.back() + 4);
                     isUnderscore.push_back(true);
                     minNumCharsLeft.push_back(minNumCharsLeft.back() - 1);
                     tailSteps.push_back([skipCodepointStep, loadByteFn, tailSums, value](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::Value newStartIndex, mlir::Value newEndIndex) -> Step {
                        return skipCodepointStep(loadByteFn, tailSums, opBuilder2, location2, newStartIndex, newEndIndex, value);
                     });
                     ++cursorPos;
                     ++skipIdx;
                  } else {
                     int32_t subpatternEnd = (skipIdx < skips.size()) ? skips[skipIdx] : subpatternSize;
                     mlir::StringAttr subsubpatternAttr = opBuilder.getStringAttr(subpattern.substr(cursorPos, subpatternEnd - cursorPos));
                     literals.push_back(subsubpatternAttr);
                     maxNumStepsParsed.push_back(maxNumStepsParsed.back() + subsubpatternAttr.size());
                     isUnderscore.push_back(false);
                     minNumCharsLeft.push_back(minNumCharsLeft.back() - subsubpatternAttr.size());
                     tailSteps.push_back([findFirstNotMatchingFn, subsubpatternAttr, value](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::Value currentIndex, mlir::Value newEndIndex) -> Step {
                        mlir::Value literalLen = opBuilder2.create<mlir::arith::ConstantIndexOp>(location2, subsubpatternAttr.size());
                        mlir::Value idx = findFirstNotMatchingFn(opBuilder2, location2, currentIndex, newEndIndex, value, subsubpatternAttr);
                        mlir::Value matched = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, idx, literalLen);
                        mlir::Value newCurrentIndex = opBuilder2.create<mlir::arith::AddIOp>(location2, currentIndex, idx);
                        return {matched, newCurrentIndex, newEndIndex};
                     });
                     tailSums -= subsubpatternAttr.size();
                     cursorPos = subpatternEnd;
                  }
               }
            }

         mlir::Value one = opBuilder.create<mlir::arith::ConstantIndexOp>(location, 1);
         mlir::Value afterAnchorConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, subpatternSize - anchorEnd);
         mlir::Value anchorLenConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, anchorEnd);
         mlir::Value totalLenConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, subpatternSize);

         std::function<llvm::SmallVector<mlir::Value, 2>(mlir::OpBuilder&, mlir::Location, size_t, mlir::Value, mlir::Value)> emitTail =
            [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, size_t i, mlir::Value matchStart, mlir::Value currentIndex) -> llvm::SmallVector<mlir::Value, 2> {
            if (i == tailSteps.size())
               return {statusFound, currentIndex};
            Step step = tailSteps[i](opBuilder2, location2, currentIndex, currentEndIndex);
            auto ifOp = opBuilder2.create<mlir::scf::IfOp>(location2, step.cond,
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  auto res = emitTail(opBuilder3, location3, i + 1, matchStart, step.startIndex);
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{res[0], res[1]});
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  if (isUnderscore[i]) {
                     opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{statusFailed, currentIndex});
                     return;
                  }
                  mlir::Value nonmatchingIdx = opBuilder3.create<mlir::arith::SubIOp>(location3, step.startIndex, currentIndex);
                  llvm::SmallVector<uint8_t> bytes(literals[i].begin(), literals[i].end());
                  auto tensorType = mlir::RankedTensorType::get({static_cast<int64_t>(bytes.size())}, opBuilder.getI8Type());
                  auto dataAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<uint8_t>(bytes));
                  mlir::Value constArrayOp = opBuilder3.create<util::CreateConstArrayOp>(location3, util::RefType::get(opBuilder3.getContext(), opBuilder3.getI8Type()), dataAttr, opBuilder3.getI64IntegerAttr(8));
                  mlir::Value c = opBuilder3.create<util::GetConstArrayAtOp>(location3, opBuilder3.getI8Type(), constArrayOp, nonmatchingIdx);
                  mlir::Value nextPos = opBuilder3.create<mlir::arith::AddIOp>(location3, step.startIndex, one);
                  mlir::Value minNumCharsLeftConst = opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, minNumCharsLeft[i]);
                  mlir::Value memchrEndIndex = opBuilder3.create<mlir::arith::SubIOp>(location3, step.endIndex, minNumCharsLeftConst);
                  memchrEndIndex = opBuilder3.create<mlir::arith::AddIOp>(location3, memchrEndIndex, nonmatchingIdx);
                  memchrEndIndex = opBuilder3.create<mlir::arith::AddIOp>(location3, memchrEndIndex, one);
                  auto memchrRes = memchrFn(opBuilder3, location3, nextPos, memchrEndIndex, value, c);

                  auto ifFoundOp = opBuilder3.create<mlir::scf::IfOp>(location3, memchrRes.getFound(),
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                        mlir::Value candidate = opBuilder4.create<mlir::arith::AddIOp>(location4, memchrRes.getPos(), nextPos);
                        mlir::Value subtract = opBuilder4.create<mlir::arith::ConstantIndexOp>(location4, maxNumStepsParsed[i] + anchorEnd);
                        subtract = opBuilder4.create<mlir::arith::AddIOp>(location4, subtract, nonmatchingIdx);
                        mlir::Value wouldUnderflow = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::ult, candidate, subtract);
                        mlir::Value difference = opBuilder4.create<mlir::arith::SubIOp>(location4, candidate, subtract);
                        mlir::Value zero = opBuilder4.create<mlir::arith::ConstantIndexOp>(location4, 0);
                        candidate = opBuilder4.create<mlir::arith::SelectOp>(location4, wouldUnderflow, zero, difference);
                        mlir::Value baseline = opBuilder4.create<mlir::arith::AddIOp>(location4, matchStart, one);
                        candidate = opBuilder4.create<mlir::arith::MaxUIOp>(location4, candidate, baseline);
                        opBuilder4.create<mlir::scf::YieldOp>(location4, mlir::ValueRange{statusRunning, candidate});
                     },
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                        // byte never occurs again -> no later alignment can match
                        opBuilder4.create<mlir::scf::YieldOp>(location4, mlir::ValueRange{statusFailed, currentIndex});
                     });

                  opBuilder3.create<mlir::scf::YieldOp>(location3, ifFoundOp.getResults());
               });
            return {ifOp.getResult(0), ifOp.getResult(1)};
         };


         auto whileOp = opBuilder.create<mlir::scf::WhileOp>(location,
            mlir::TypeRange{opBuilder.getI8Type(), opBuilder.getIndexType()},
            mlir::ValueRange{statusRunning, startIndex},
            [&](mlir::OpBuilder& builder, mlir::Location loc2, mlir::ValueRange args) {
               mlir::Value cond = builder.create<mlir::arith::CmpIOp>(loc2, mlir::arith::CmpIPredicate::eq, args[0], statusRunning);
               builder.create<mlir::scf::ConditionOp>(loc2, cond, args);
            },
            [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::ValueRange args) {
               mlir::Value cursor = args[1];
               mlir::Value sumValue = opBuilder2.create<mlir::arith::AddIOp>(location2, totalLenConst, cursor);
               mlir::Value enough = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::ule, sumValue, currentEndIndex);
               auto ifEnough = opBuilder2.create<mlir::scf::IfOp>(location2, enough,
                  [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                     mlir::Value earlyStopEndIndex = opBuilder3.create<mlir::arith::SubIOp>(location3, currentEndIndex, afterAnchorConst);
                     auto found = containsFn(opBuilder3, location3, cursor, earlyStopEndIndex, value, anchorAttr);
                     auto ifFound = opBuilder3.create<mlir::scf::IfOp>(location3, found.getContains(),
                        [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                           mlir::Value matchStart = found.getMatchStart();
                           mlir::Value currentIndex = opBuilder4.create<mlir::arith::AddIOp>(location4, matchStart, anchorLenConst);
                           auto result = emitTail(opBuilder4, location4, 0, matchStart, currentIndex);
                           opBuilder4.create<mlir::scf::YieldOp>(location4, mlir::ValueRange{result[0], result[1]});
                        },
                        [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                           opBuilder4.create<mlir::scf::YieldOp>(location4, mlir::ValueRange{statusFailed, cursor});
                        });
                     opBuilder3.create<mlir::scf::YieldOp>(location3, ifFound.getResults());
                  },
                  [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                     opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{statusFailed, cursor});
                  });
               opBuilder2.create<mlir::scf::YieldOp>(location2, ifEnough.getResults());
            });

            mlir::Value cond = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, whileOp.getResult(0), statusFound);
            return {cond, whileOp.getResult(1), endIndex};
         };

         // Case 1: no underscores in my middle pattern
         if (skips.empty()) {
            sums -= subpattern.size();
            mlir::StringAttr subpatternAttr = rewriter.getStringAttr(subpattern);

            ptrSteps.push_back([containsStep, containsPtrFn, sums, subpatternAttr, haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return containsStep(containsPtrFn, sums, subpatternAttr, opBuilder, location, startIndex, endIndex, haystackPtr);
            });

            inlinedSteps.push_back([containsStep, containsInlineFn, sums, subpatternAttr, bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
               return containsStep(containsInlineFn, sums, subpatternAttr, opBuilder, location, startIndex, endIndex, bytesVal);
            });
         } else {
            // Case 2: there are some underscores in my middle pattern
            int32_t pos = 0;
            int32_t subpatternSize = static_cast<int32_t>(subpattern.size());
            int32_t skipsSize = static_cast<int32_t>(skips.size());
            while (pos < subpatternSize && pos < skipsSize && skips[pos] == pos) {
               --sums;
               ptrSteps.push_back([skipCodepointStep, loadFromPtrFn, sums, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                  return skipCodepointStep(loadFromPtrFn, sums, opBuilder, location, startIndex, endIndex, haystackPtr);
               });
               inlinedSteps.push_back([skipCodepointStep, loadFromInlinedFn, sums, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                  return skipCodepointStep(loadFromInlinedFn, sums, opBuilder, location, startIndex, endIndex, bytesVal);
               });
               ++pos;
            }


            // Case 2.1: there are characters that are non-underscores in my middle pattern
            if (pos != subpatternSize) {
               mlir::StringAttr subpatternAttr = rewriter.getStringAttr(subpattern.substr(pos));
               sums -= subpatternAttr.size();
               // Case 2.1.1: after the initial underscores, we have no more underscores => we can use regular contains
               if (pos == skipsSize) {
                  ptrSteps.push_back([containsStep, containsPtrFn, sums, subpatternAttr, haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                     return containsStep(containsPtrFn, sums, subpatternAttr, opBuilder, location, startIndex, endIndex, haystackPtr);
                  });

                  inlinedSteps.push_back([containsStep, containsInlineFn, sums, subpatternAttr, bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                     return containsStep(containsInlineFn, sums, subpatternAttr, opBuilder, location, startIndex, endIndex, bytesVal);
                  });
               }
               // Case 2.1.2: after the initial underscores, we have underscores somewhere in the pattern
               else {
                  llvm::SmallVector<int32_t> updatedSkips;
                  for (size_t i = pos; i < skips.size(); ++i)
                     updatedSkips.push_back(skips[i] - pos);
                  ptrSteps.push_back([whileStep, containsPtrFn, findFirstNonMatchingPtrFn, loadFromPtrFn, memchrPtrFn, updatedSkips, subpatternAttr, sums, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                     return whileStep(containsPtrFn, findFirstNonMatchingPtrFn, loadFromPtrFn, memchrPtrFn, updatedSkips, subpatternAttr, sums, opBuilder, location, startIndex, endIndex, haystackPtr);
                  });
                  inlinedSteps.push_back([whileStep, containsInlineFn, findFirstNonMatchingInlineFn, loadFromInlinedFn, memchrInlineFn, updatedSkips, subpatternAttr, sums, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
                     return whileStep(containsInlineFn, findFirstNonMatchingInlineFn, loadFromInlinedFn, memchrInlineFn, updatedSkips, subpatternAttr, sums, opBuilder, location, startIndex, endIndex, bytesVal);
                  });
               }
            }
         }
      }

      std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, size_t, mlir::Value, mlir::Value, llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>>&)> emit = [&](mlir::OpBuilder& opBuilder, mlir::Location location, size_t i, mlir::Value startIndex, mlir::Value endIndex, llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>>& steps) -> mlir::Value {
         if (i == steps.size()) {
            if (likeOp.getForceExact()) {
               return opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, startIndex, endIndex);
            } else {
               return opBuilder.create<mlir::arith::ConstantIntOp>(location, true, opBuilder.getI1Type());
            }
         }

         Step step = steps[i](opBuilder, location, startIndex, endIndex);
         return opBuilder.create<mlir::scf::IfOp>(location, step.cond, [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, i + 1, step.startIndex, step.endIndex, steps));
         }, [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            opBuilder2.create<mlir::scf::YieldOp>(location2, opBuilder2.create<mlir::arith::ConstantIntOp>(location2, false, opBuilder2.getI1Type()).getResult());
         }).getResult(0);
      };

      mlir::Value startIndex = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value endIndex = rewriter.create<util::VarLenGetLen>(loc, rewriter.getIndexType(), haystack);
      mlir::Value canMatch = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, endIndex, sumsConst);
      auto ifBlock = rewriter.create<mlir::scf::IfOp>(loc,canMatch,
         [&](mlir::OpBuilder& opBuilder, mlir::Location location) {
            mlir::Value partialResult;
            if (createOp || totalSums > 12) {
               partialResult = emit(opBuilder, location, 0, startIndex, endIndex, ptrSteps);
            } else {
               mlir::Value inlineThreshold = opBuilder.create<mlir::arith::ConstantIndexOp>(location, 12);
               mlir::Value isNotInlined = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::ugt, endIndex, inlineThreshold);

               auto ifOp = opBuilder.create<mlir::scf::IfOp>(location, isNotInlined,
                  [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                     opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, 0, startIndex, endIndex, ptrSteps));
                  },
                  [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                     opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, 0, startIndex, endIndex, inlinedSteps));
                  });
               partialResult = ifOp.getResult(0);
            }
            opBuilder.create<mlir::scf::YieldOp>(location, partialResult);
         },
         [&](mlir::OpBuilder& opBuilder, mlir::Location location) {
            opBuilder.create<mlir::scf::YieldOp>(location, opBuilder.create<mlir::arith::ConstantIntOp>(location, false, 1).getResult());
         });

      mlir::Value result = ifBlock.getResult(0);
      rewriter.replaceOp(op, result);
      return mlir::success(true);
   }
};


class SpecializeStringContains : public mlir::RewritePattern {
   private:
   static void preprocessSubpattern(const uint8_t* subpattern, size_t size, int32_t& period, int32_t& maxSuffix) {
      if (size == 1) {
         maxSuffix = 0;
         period = 0;
         return;
      }

      if (size == 2) {
         if (subpattern[0] == subpattern[1]) {
            maxSuffix = -1;
            period = (3 << 1) | 1;
         } else {
            maxSuffix = 0;
            period = (1 << 1) | 0;
         }
         return;
      }

      // https://en.wikipedia.org/wiki/Two-way_string-matching_algorithm
      auto computeMaxSuffix = [&]<bool Reverse>() noexcept -> std::pair<int32_t, int32_t> {
         auto compare = [](uint8_t a, uint8_t b) -> int {
            int result = (a > b) - (b > a);
            if constexpr (Reverse) {
               return -result;
            } else {
               return result;
            }
         };

         int32_t currentPeriod = 1;
         int32_t maxSuffixIndex = -1;
         int32_t periodTestIndex = 1;
         int32_t maxSuffixTestIndex = 0;
         int32_t length = size;

         while (maxSuffixTestIndex + periodTestIndex < length) {
            int compareVal = compare(subpattern[maxSuffixTestIndex + periodTestIndex], subpattern[maxSuffixIndex + periodTestIndex]);
            if (compareVal < 0) {
               maxSuffixTestIndex += periodTestIndex;
               periodTestIndex = 1;
               currentPeriod = maxSuffixTestIndex - maxSuffixIndex;
            } else if (compareVal == 0) {
               if (periodTestIndex == currentPeriod) {
                  maxSuffixTestIndex += currentPeriod;
                  periodTestIndex = 1;
               } else {
                  ++periodTestIndex;
               }
            } else {
               maxSuffixIndex = maxSuffixTestIndex;
               ++maxSuffixTestIndex;
               currentPeriod = 1;
               periodTestIndex = 1;
            }
         }
         return std::pair<int32_t, int32_t>{maxSuffixIndex, currentPeriod};
      };

      auto findCriticalFactorization = [&]() {
         auto [maxSuffixIndex1, period1] = computeMaxSuffix.template operator()<false>();
         auto [maxSuffixIndex2, period2] = computeMaxSuffix.template operator()<true>();
         return maxSuffixIndex1 > maxSuffixIndex2 ? std::pair<int, int>{maxSuffixIndex1, period1} : std::pair<int, int>{maxSuffixIndex2, period2};
      };

      auto result = findCriticalFactorization();
      maxSuffix = result.first;
      period = result.second;

      // function SMALL-PERIOD from http://monge.univ-mlv.fr/~mac/Articles-PDF/CP-1991-jacm.pdf
      // l represents the maxSuffix and p the period
      // x[1] x[2] ... x[l] is a suffix of x[l + 1] x[l + 2] ... x[l + p] if and only if x[1] x[2] ... x[l] == x[p + 1] x[p + 2] ... x[l + p]
      // we adjust this for 0-indexing rather than 0-indexing
      bool equal = !memcmp(subpattern, subpattern + period, maxSuffix + 1);

      if (!equal) {
         // Proposition 5.2 from http://monge.univ-mlv.fr/~mac/Articles-PDF/CP-1991-jacm.pdf
         // l represents the maxSuffix
         // adjusted because of 0-indexing rather than 1-indexing used in the paper
         // here, the period actually represents the value q, used for long-periods
         int32_t candidatePeriod1 = maxSuffix + 1;
         int32_t candidatePeriod2 = size - maxSuffix - 1;
         period = std::max(candidatePeriod1, candidatePeriod2) + 1;
      }
      period = (period << 1) | equal;
   }

   public:
   SpecializeStringContains(mlir::MLIRContext* context) : RewritePattern(util::StringContains::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto containsOp = mlir::cast<util::StringContains>(op);
      llvm::StringRef needle = containsOp.getSubpattern();
      auto loc = containsOp.getLoc();
      auto i1Type = rewriter.getI1Type();
      auto indexType = rewriter.getIndexType();
      auto i8Type = rewriter.getI8Type();
      auto i8RefType = util::RefType::get(rewriter.getContext(), rewriter.getI8Type());

      mlir::Value haystackPtr = containsOp.getStr();
      mlir::Value startIdx = containsOp.getStartIndex();
      mlir::Value endIdx = containsOp.getEndIndex();
      mlir::Value stringLen = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, startIdx);
      mlir::Value found;
      mlir::Value pos;
      if (needle.size() == 1) {
         mlir::Value memchrStart = rewriter.create<util::ArrayElementPtrOp>(loc, i8RefType, haystackPtr, startIdx);
         mlir::Value c = rewriter.create<mlir::arith::ConstantIntOp>(loc, needle[0], i8Type);
         auto result = rewriter.create<util::RefMemchr>(loc, i1Type, indexType, memchrStart, stringLen, c);
         found = result.getResult(0);
         pos = rewriter.create<mlir::arith::AddIOp>(loc, result.getResult(1), startIdx);
      } else {
         mlir::Value statusRunning = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(0));
         mlir::Value statusFound   = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(1));
         mlir::Value statusFailed  = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(2));
         mlir::Value oneConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         mlir::Value needleLenMinusOne  = rewriter.create<mlir::arith::ConstantIndexOp>(loc, needle.size() - 1);
         mlir::Value maxWhileIndex = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, needleLenMinusOne);
         mlir::Value needleLenConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, needle.size());


         auto needleShape = mlir::RankedTensorType::get({static_cast<int64_t>(needle.size())}, rewriter.getI8Type());
         auto needleAttr = mlir::DenseElementsAttr::get(needleShape, llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t*>(needle.data()), needle.size()));
         mlir::Value needleGlobal = rewriter.create<util::CreateConstArrayOp>(loc, util::RefType::get(rewriter.getI8Type()),needleAttr, rewriter.getI64IntegerAttr(1));

         auto twoWaySearch = [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            auto whileOp = opBuilder2.create<mlir::scf::WhileOp>(location2, mlir::TypeRange{indexType, i8Type},
               mlir::ValueRange{startIdx, statusRunning},
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value status = args[1];
                  mlir::Value running = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, status, statusRunning);
                  opBuilder3.create<mlir::scf::ConditionOp>(location3, running, args);
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value i = args[0];
                  mlir::Value memchrStart = opBuilder3.create<util::ArrayElementPtrOp>(location3, i8RefType, haystackPtr, i);
                  mlir::Value memchrLen = opBuilder3.create<mlir::arith::SubIOp>(location3, indexType, maxWhileIndex, i);
                  mlir::Value c = opBuilder3.create<mlir::arith::ConstantIntOp>(location3, needle[0], i8Type);
                  auto memchrRes = opBuilder3.create<util::RefMemchr>(location3, i1Type, indexType, memchrStart, memchrLen, c);
                  mlir::Value foundPartialMatch = memchrRes.getResult(0);
                  mlir::Value indexMatch = memchrRes.getResult(1);
                  indexMatch = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, i, indexMatch);

                  auto outer = opBuilder3.create<mlir::scf::IfOp>(location3, foundPartialMatch,
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                        mlir::Value nextNeedleCharConst = opBuilder4.create<mlir::arith::ConstantOp>(location4, opBuilder4.getI8IntegerAttr(needle[1]));
                        mlir::Value nextPos = opBuilder4.create<mlir::arith::AddIOp>(location4, indexType, indexMatch, oneConst);
                        mlir::Value ptrPos = opBuilder4.create<util::ArrayElementPtrOp>(location4, i8RefType, haystackPtr, nextPos);
                        mlir::Value nextByte = opBuilder4.create<util::LoadOp>(location4, i8Type, ptrPos);
                        mlir::Value isMatch = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::eq, nextByte, nextNeedleCharConst);

                        auto inner = opBuilder4.create<mlir::scf::IfOp>(location4, isMatch,
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                              opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{indexMatch, statusFound});
                           },
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                              mlir::Value atEnd;
                              if (needle[0] == needle[1]) {
                                 nextPos = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, nextPos, oneConst);
                                 atEnd = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::uge, nextPos, maxWhileIndex);
                              } else {
                                 atEnd = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::eq, nextPos, maxWhileIndex);
                              }
                              mlir::Value nextStatus = opBuilder5.create<mlir::arith::SelectOp>(location5, atEnd, statusFailed, statusRunning);
                                 opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{nextPos, nextStatus});
                        });

                        opBuilder4.create<mlir::scf::YieldOp>(location4, inner.getResults());
                     },
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                        opBuilder4.create<mlir::scf::YieldOp>(location4, mlir::ValueRange{i, statusFailed});
                  });
                  opBuilder3.create<mlir::scf::YieldOp>(location3, outer.getResults());
            });

            int32_t period;
            int32_t maxSuffix;
            preprocessSubpattern(reinterpret_cast<const uint8_t*>(needle.data()), needle.size(), period, maxSuffix);
            bool isEqual = period & 1;
            period >>= 1;
            mlir::Value mayContinue = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, whileOp.getResult(1), statusFound);
            auto ifBlock = opBuilder2.create<mlir::scf::IfOp>(location2, mayContinue,
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  mlir::Value periodConst= opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, period);
                  mlir::Value maxSufConst= opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, maxSuffix + 1);
                  mlir::Value resetPtrConst= opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, needle.size() - period);
                  mlir::Value minusOneIndex = opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, -1);
                  mlir::Value zeroConst = opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, 0);

                  auto newWhileOp = opBuilder3.create<mlir::scf::WhileOp>(location3, mlir::TypeRange{indexType, indexType, i8Type}, mlir::ValueRange{whileOp.getResult(0), zeroConst, statusRunning},
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4, mlir::ValueRange args) {
                        mlir::Value running = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::eq, args[2], statusRunning);
                        mlir::Value inRange = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::ult, args[0], maxWhileIndex);
                        opBuilder4.create<mlir::scf::ConditionOp>(location4, opBuilder4.create<mlir::arith::AndIOp>(location4, running, inRange), args);
                     },
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4, mlir::ValueRange args) {
                        mlir::Value offset = args[0];
                        mlir::Value lastPtr = args[1];
                        mlir::Value startPos;
                        if (!isEqual) {
                           startPos = maxSufConst;
                        } else {
                           startPos = opBuilder4.create<mlir::arith::MaxUIOp>(location4, lastPtr, maxSufConst);
                        }
                        auto fwd = opBuilder4.create<mlir::scf::WhileOp>(location4, mlir::TypeRange{indexType}, mlir::ValueRange{startPos},
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5, mlir::ValueRange args2) {
                              mlir::Value currentPos = args2[0];
                              mlir::Value inBounds = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::ult, currentPos, needleLenConst);
                              // short-circuit: only load when pos is in bounds
                              auto guard = opBuilder5.create<mlir::scf::IfOp>(location5, inBounds,
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    mlir::Value needleByte = opBuilder6.create<util::GetConstArrayAtOp>(location6, i8Type, needleGlobal, currentPos);
                                    mlir::Value haystackPos = opBuilder6.create<mlir::arith::AddIOp>(location6, indexType, currentPos, offset);
                                    mlir::Value currentHaystackPtr = opBuilder6.create<util::ArrayElementPtrOp>(location6, i8RefType, haystackPtr, haystackPos);
                                    mlir::Value haystackByte = opBuilder6.create<util::LoadOp>(location6, i8Type, currentHaystackPtr);
                                    mlir::Value areEqual = opBuilder6.create<mlir::arith::CmpIOp>(location6, mlir::arith::CmpIPredicate::eq, needleByte, haystackByte);
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{areEqual});
                                 },
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    mlir::Value falseConst = opBuilder6.create<mlir::arith::ConstantOp>(location6, opBuilder6.getBoolAttr(false));
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{falseConst});
                              });

                              opBuilder5.create<mlir::scf::ConditionOp>(location5, guard.getResult(0), args2);
                           },

                           // after: pos++
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5, mlir::ValueRange args2) {
                              mlir::Value next = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, args2[0], oneConst);
                              opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{next});
                        });
                        mlir::Value forwardComplete = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::eq, fwd.getResult(0), needleLenConst);
                        auto branchingPaths = opBuilder4.create<mlir::scf::IfOp>(location4, forwardComplete,
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                              mlir::Value maxSufMinusOne = opBuilder5.create<mlir::arith::ConstantIndexOp>(location5, maxSuffix);
                              auto bwd = opBuilder5.create<mlir::scf::WhileOp>(location5, mlir::TypeRange{indexType}, mlir::ValueRange{maxSufMinusOne},
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6, mlir::ValueRange args2) {
                                    mlir::Value currentPos = args2[0];
                                    mlir::Value inBounds;
                                    if (isEqual) {
                                       inBounds = opBuilder6.create<mlir::arith::CmpIOp>(location6, mlir::arith::CmpIPredicate::sge, currentPos, lastPtr);
                                    } else {
                                       inBounds = opBuilder6.create<mlir::arith::CmpIOp>(location6, mlir::arith::CmpIPredicate::ne, currentPos, minusOneIndex);
                                    }
                                    auto guard = opBuilder6.create<mlir::scf::IfOp>(location6, inBounds,
                                       [&](mlir::OpBuilder& opBuilder7, mlir::Location location7) {
                                          mlir::Value needleByte = opBuilder7.create<util::GetConstArrayAtOp>(location7, i8Type, needleGlobal, currentPos);
                                          mlir::Value haystackPos = opBuilder7.create<mlir::arith::AddIOp>(location7, indexType, currentPos, offset);
                                          mlir::Value currentHaystackPtr = opBuilder7.create<util::ArrayElementPtrOp>(location7, i8RefType, haystackPtr, haystackPos);
                                          mlir::Value haystackByte = opBuilder7.create<util::LoadOp>(location7, i8Type, currentHaystackPtr);
                                          mlir::Value areEqual = opBuilder7.create<mlir::arith::CmpIOp>(location7, mlir::arith::CmpIPredicate::eq, needleByte, haystackByte);
                                          opBuilder7.create<mlir::scf::YieldOp>(location7, mlir::ValueRange{areEqual});
                                       },
                                       [&](mlir::OpBuilder& opBuilder7, mlir::Location location7) {
                                          mlir::Value falseConst = opBuilder7.create<mlir::arith::ConstantOp>(location7, opBuilder7.getBoolAttr(false));
                                          opBuilder7.create<mlir::scf::YieldOp>(location7, mlir::ValueRange{falseConst});
                                    });
                                    opBuilder6.create<mlir::scf::ConditionOp>(location6, guard.getResult(0), args2);
                              },
                              [&](mlir::OpBuilder& opBuilder6, mlir::Location location6, mlir::ValueRange args2) {
                                 mlir::Value prev = opBuilder6.create<mlir::arith::SubIOp>(location6, indexType, args2[0], oneConst);
                                 opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{prev});
                              });

                              mlir::Value stillMismatch;
                              if (isEqual) {
                                 stillMismatch = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::sge, bwd.getResult(0), lastPtr);
                              } else {
                                 stillMismatch = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::ne, bwd.getResult(0), minusOneIndex);
                              }

                              auto inner = opBuilder5.create<mlir::scf::IfOp>(location5, stillMismatch,
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    mlir::Value next = opBuilder6.create<mlir::arith::AddIOp>(location6, indexType, offset, periodConst);
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{next, resetPtrConst, statusRunning});
                                 },
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{offset, lastPtr, statusFound});
                              });

                              opBuilder5.create<mlir::scf::YieldOp>(location5, inner.getResults());
                        },
                        [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                           mlir::Value addOne = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, fwd.getResult(0), oneConst);
                           mlir::Value delta = opBuilder5.create<mlir::arith::SubIOp>(location5, indexType, addOne, maxSufConst);
                           mlir::Value next = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, offset, delta);
                           opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{next, zeroConst, statusRunning});
                        });
                        opBuilder4.create<mlir::scf::YieldOp>(location4, branchingPaths.getResults());
                  });
                  mlir::Value innerFound = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, newWhileOp.getResult(2), statusFound);
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{innerFound, newWhileOp.getResult(0)});
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{opBuilder3.create<mlir::arith::ConstantOp>(location3, opBuilder3.getBoolAttr(false)), whileOp.getResult(0)});
            });
            return ifBlock;
         };

         mlir::Value const16 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 16);
         auto simdSearch = [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            mlir::Value needleVec;
            auto v16i8 = mlir::VectorType::get({16}, i8Type);

            {
               uint8_t buf[16] = {0};
               memcpy(buf, needle.data(), needle.size());
               auto needleVecAttr = mlir::DenseElementsAttr::get(v16i8, llvm::ArrayRef<uint8_t>(buf, 16));
               needleVec =  opBuilder2.create<mlir::arith::ConstantOp>(location2, needleVecAttr);
            }

            mlir::Value lastFullMatchIndex = opBuilder2.create<mlir::arith::ConstantOp>(location2, rewriter.getI32IntegerAttr(16 - needle.size()));
            // _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED
            auto cmpistriFlags = opBuilder2.getI8IntegerAttr(0x0C);
            mlir::Value endIdxMinus16 = opBuilder2.create<mlir::arith::SubIOp>(location2, endIdx, const16);
            auto whileOp = opBuilder2.create<mlir::scf::WhileOp>(location2, mlir::TypeRange{indexType, i8Type}, mlir::ValueRange{startIdx, statusRunning},
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value offset = args[0];
                  mlir::Value status = args[1];
                  mlir::Value running = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, status, statusRunning);
                  mlir::Value inBounds = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::ule, offset, endIdxMinus16);
                  mlir::Value cond = opBuilder3.create<mlir::arith::AndIOp>(location3, running, inBounds);
                  opBuilder3.create<mlir::scf::ConditionOp>(location3, cond, args);
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value offset = args[0];
                  mlir::Value currentHaystackPtr = opBuilder3.create<util::ArrayElementPtrOp>(location3, i8RefType, haystackPtr, offset);
                  mlir::Value haystackVec = opBuilder3.create<util::LoadVectorOp>(location3, v16i8, currentHaystackPtr, opBuilder3.getI64IntegerAttr(1));
                  mlir::Value cmpistriRes = opBuilder3.create<util::CmpistriOp>(location3, opBuilder3.getI32Type(), needleVec, haystackVec, cmpistriFlags);
                  mlir::Value mayReturn = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::ule, cmpistriRes, lastFullMatchIndex);
                  mlir::Value asIndex = opBuilder3.create<mlir::arith::IndexCastOp>(location3, indexType, cmpistriRes);
                  mlir::Value matchStart = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, offset, asIndex);
                  mlir::Value nextStatus = opBuilder3.create<mlir::arith::SelectOp>(location3, mayReturn, statusFound, statusRunning);
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{matchStart, nextStatus});
            });

            auto isFinished = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, whileOp.getResult(1), statusFound);
            auto ifOp = opBuilder2.create<mlir::scf::IfOp>(location2,isFinished,
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  mlir::Value trueConst = opBuilder3.create<mlir::arith::ConstantOp>(location3, opBuilder3.getBoolAttr(true));
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{trueConst, whileOp.getResult(0)});
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  mlir::Value offset = endIdxMinus16;
                  mlir::Value currentHaystackPtr = opBuilder3.create<util::ArrayElementPtrOp>(location3, i8RefType, haystackPtr, offset);
                  mlir::Value haystackVec = opBuilder3.create<util::LoadVectorOp>(location3, v16i8, currentHaystackPtr, opBuilder3.getI64IntegerAttr(1));
                  mlir::Value cmpistriRes = opBuilder3.create<util::CmpistriOp>(location3,opBuilder3.getI32Type(),  needleVec, haystackVec, cmpistriFlags);
                  mlir::Value mayReturn = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::ule, cmpistriRes, lastFullMatchIndex);
                  mlir::Value asIndex = opBuilder3.create<mlir::arith::IndexCastOp>(location3, indexType, cmpistriRes);
                  mlir::Value matchStart = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, offset, asIndex);
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{mayReturn, matchStart});
            });
            return ifOp;
         };

         auto canUseSSE42 = []() {
            auto features = llvm::sys::getHostCPUFeatures();
            auto it = features.find("sse4.2");
            return it != features.end() && it->second;
         };


         if (!canUseSSE42() || needle.size() > 12) {
            auto result = twoWaySearch(rewriter, loc);
            found = result.getResult(0);
            pos = result.getResult(1);
         } else {
            mlir::Value isLongEnough = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, stringLen, const16);
            auto ifOp = rewriter.create<mlir::scf::IfOp>(loc,isLongEnough,
               [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                  auto result = simdSearch(opBuilder2, location2);
                  opBuilder2.create<mlir::scf::YieldOp>(location2, mlir::ValueRange{result.getResult(0), result.getResult(1)});
               },
               [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                  auto result = twoWaySearch(opBuilder2, location2);
                  opBuilder2.create<mlir::scf::YieldOp>(location2, mlir::ValueRange{result.getResult(0), result.getResult(1)});
               });
            found = ifOp.getResult(0);
            pos = ifOp.getResult(1);
         }
      }
      rewriter.replaceOp(containsOp, mlir::ValueRange{found, pos});
      return mlir::success();
   }
};

class SpecializeBytesContains : public mlir::RewritePattern {
   public:
   SpecializeBytesContains(mlir::MLIRContext* context) : RewritePattern(util::BytesContains::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto containsOp = mlir::cast<util::BytesContains>(op);
      llvm::StringRef needle = containsOp.getSubpattern();
      auto loc = containsOp.getLoc();
      auto i1Type = rewriter.getI1Type();
      auto indexType = rewriter.getIndexType();
      auto i8Type = rewriter.getI8Type();
      auto i128Type = rewriter.getIntegerType(128);

      mlir::Value haystackInlined = containsOp.getStr();
      mlir::Value startIdx = containsOp.getStartIndex();
      mlir::Value endIdx = containsOp.getEndIndex();
      mlir::Value stringLen = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, startIdx);
      mlir::Value found;
      mlir::Value pos;
      if (needle.size() == 1) {
         mlir::Value eight = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 8);
         mlir::Value bitOffIdx = rewriter.create<mlir::arith::MulIOp>(loc, startIdx, eight);
         mlir::Value bitOff = rewriter.create<mlir::arith::IndexCastOp>(loc, i128Type, bitOffIdx);
         mlir::Value shiftedString = rewriter.create<mlir::arith::ShRUIOp>(loc, haystackInlined, bitOff);
         mlir::Value needleConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, needle[0], rewriter.getI8Type());
         auto result = rewriter.create<util::InlineMemchr>(loc, i1Type, indexType, shiftedString, stringLen, needleConst);
         found = result.getResult(0);
         pos = rewriter.create<mlir::arith::AddIOp>(loc, indexType, startIdx, result.getResult(1));
      } else {
          uint64_t needleBits[2] = {0};
          uint8_t* needleBitsU8 = reinterpret_cast<uint8_t*>(needleBits);
          uint64_t maskBits[2] = {0};
          uint8_t* maskBitsU8 = reinterpret_cast<uint8_t*>(maskBits);

          mlir::Value statusRunning = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(0));
          mlir::Value statusFound   = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(1));
          mlir::Value statusFailed  = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(2));
          mlir::Value oneConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
          mlir::Value needleLenMinusOne  = rewriter.create<mlir::arith::ConstantIndexOp>(loc, needle.size() - 1);
          mlir::Value maxWhileIndex = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, needleLenMinusOne);

          for (size_t j = 0; j < needle.size(); ++j) {
             needleBitsU8[j] = static_cast<uint8_t>(needle[j]);
             maskBitsU8[j] = 0xFF;
          }
          llvm::APInt maskAP(128, llvm::ArrayRef<uint64_t>(maskBits, 2));
          auto maskAttr =  rewriter.getIntegerAttr(i128Type, maskAP);
          mlir::Value maskVal = rewriter.create<mlir::arith::ConstantOp>(loc, i128Type, maskAttr);

          llvm::APInt needleAP(128, llvm::ArrayRef<uint64_t>(needleBits, 2));
          auto needleAttr =  rewriter.getIntegerAttr(i128Type, needleAP);
          mlir::Value needleVal= rewriter.create<mlir::arith::ConstantOp>(loc, i128Type, needleAttr);
          mlir::Value eightConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 8);

          auto loop = rewriter.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{indexType, i8Type}, mlir::ValueRange{startIdx, statusRunning},
             [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::ValueRange args) {
                mlir::Value running = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, args[1], statusRunning);
                mlir::Value inRange = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::ult, args[0], maxWhileIndex);
                opBuilder2.create<mlir::scf::ConditionOp>(location2, opBuilder2.create<mlir::arith::AndIOp>(location2, running, inRange), args);
             },

             [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::ValueRange args) {
                mlir::Value offset = args[0];

                mlir::Value bitOffIdx = opBuilder2.create<mlir::arith::MulIOp>(location2, indexType, offset, eightConst);
                mlir::Value bitOff= opBuilder2.create<mlir::arith::IndexCastOp>(location2, i128Type, bitOffIdx);
                mlir::Value shifted = opBuilder2.create<mlir::arith::ShRUIOp>(location2, haystackInlined, bitOff);
                mlir::Value remaining = opBuilder2.create<mlir::arith::SubIOp>(location2, indexType, maxWhileIndex, offset);

                mlir::Value needleFirstCharConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, needle[0], rewriter.getI8Type());
                auto memchr = opBuilder2.create<util::InlineMemchr>(location2, i1Type, indexType, shifted, remaining, needleFirstCharConst);
                mlir::Value foundCurrent = memchr.getResult(0);
                mlir::Value hitAbs = opBuilder2.create<mlir::arith::AddIOp>(location2, indexType, offset, memchr.getResult(1));

                auto step = opBuilder2.create<mlir::scf::IfOp>(
                   location2,
                   foundCurrent,
                   // candidate at hitAbs: mask-compare the full needle
                   [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                      mlir::Value hitBitsIdx = opBuilder3.create<mlir::arith::MulIOp>(location3, indexType, hitAbs, eightConst);
                      mlir::Value hitBits = opBuilder3.create<mlir::arith::IndexCastOp>(location3, i128Type, hitBitsIdx);
                      mlir::Value window = opBuilder3.create<mlir::arith::ShRUIOp>(location3, haystackInlined, hitBits);
                      mlir::Value masked = opBuilder3.create<mlir::arith::AndIOp>(location3, window, maskVal);
                      mlir::Value isMatch = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, masked, needleVal);

                      mlir::Value nextOffset = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, hitAbs, oneConst);
                      mlir::Value next = opBuilder3.create<mlir::arith::SelectOp>(location3, isMatch, hitAbs, nextOffset);
                      mlir::Value nextStatus = opBuilder3.create<mlir::arith::SelectOp>(location3, isMatch, statusFound, statusRunning);
                      opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{next, nextStatus});
                   },
                   [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                      opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{offset, statusFailed});
                   });

                opBuilder2.create<mlir::scf::YieldOp>(location2, step.getResults());
             });
          found = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, loop.getResult(1), statusFound);
          pos = loop.getResult(0);
      }
      rewriter.replaceOp(containsOp, mlir::ValueRange{found, pos});
      return mlir::success();
   }
};

class SpecializeStringNonMatchIndex : public mlir::RewritePattern {
   public:
   SpecializeStringNonMatchIndex(mlir::MLIRContext* context) : RewritePattern(util::StringNonMatchIndex::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto nonMatchOp = mlir::cast<util::StringNonMatchIndex>(op);
      auto loc = op->getLoc();
      llvm::StringRef literal = nonMatchOp.getSubpattern();
      auto indexType = rewriter.getIndexType();

      mlir::Value haystackPtr = nonMatchOp.getStr();
      mlir::Value startIdx = nonMatchOp.getStartIndex();

      int64_t literalSize = static_cast<int64_t>(literal.size());

      std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, int64_t)> emit =
         [&](mlir::OpBuilder& opBuilder, mlir::Location location, int64_t i) -> mlir::Value {
         if (i == literalSize)
            return opBuilder.create<mlir::arith::ConstantIndexOp>(location, literalSize);

         mlir::Value offset = opBuilder.create<mlir::arith::ConstantIndexOp>(location, i);
         mlir::Value pos = opBuilder.create<mlir::arith::AddIOp>(location, startIdx, offset);
         mlir::Value currentHaystackPtr = opBuilder.create<util::ArrayElementPtrOp>(location, util::RefType::get(rewriter.getI8Type()), haystackPtr, pos);
         if (i == literalSize - 1) {
            mlir::Value needleConst = opBuilder.create<mlir::arith::ConstantIntOp>(location, static_cast<uint8_t>(literal[literalSize - 1]), rewriter.getI8Type());
            mlir::Value haystackByte = opBuilder.create<util::LoadOp>(location, rewriter.getI8Type(), currentHaystackPtr);
            mlir::Value isEqual = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, needleConst, haystackByte);
            auto ifOp = opBuilder.create<mlir::scf::IfOp>(location, isEqual,
               [&emit, i](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                  opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, i + 1));
               },
               [i](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                  mlir::Value mismatchAt = opBuilder2.create<mlir::arith::ConstantIndexOp>(location2, i);
                  opBuilder2.create<mlir::scf::YieldOp>(location2, mismatchAt);
               });
            return ifOp.getResult(0);
         }

         int32_t numBytes;
         if (i + 8 <= literalSize)
            numBytes = 8;
         else if (i + 4 <= literalSize)
            numBytes = 4;
         else numBytes = 2;

         uint64_t bitWidth = 8 * numBytes;

         mlir::Type intType = rewriter.getIntegerType(bitWidth);
         util::RefType intRefType = util::RefType::get(rewriter.getContext(), intType);
         int64_t bits = 0;
         memcpy(&bits, literal.data() + i, numBytes);
         mlir::Value literalBytes = opBuilder.create<mlir::arith::ConstantIntOp>(location, bits, intType);
         currentHaystackPtr = opBuilder.create<util::GenericMemrefCastOp>(location, intRefType, currentHaystackPtr);
         mlir::Value haystackBytes = opBuilder.create<util::UnalignedLoadOp>(location, intType, currentHaystackPtr);
         mlir::Value maskValue = opBuilder.create<mlir::arith::XOrIOp>(location, haystackBytes, literalBytes);
         mlir::Value zero = opBuilder.create<mlir::arith::ConstantIntOp>(location, 0, intType);
         mlir::Value isEqual = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, maskValue, zero);
         auto ifOp = opBuilder.create<mlir::scf::IfOp>(location, isEqual,
            [&emit, i, numBytes](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
               opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, i + numBytes));
            },
            [&maskValue, intType, i](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
               mlir::Value trailingZeros = opBuilder2.create<mlir::LLVM::CountTrailingZerosOp>(location2, intType,  maskValue, false);
               mlir::Value tzIndex = opBuilder2.create<mlir::arith::IndexCastOp>(location2, opBuilder2.getIndexType(), trailingZeros);
               mlir::Value const8 = opBuilder2.create<mlir::arith::ConstantIndexOp>(location2, 8);
               mlir::Value mismatchAt = opBuilder2.create<mlir::arith::DivUIOp>(location2, tzIndex, const8);
               mlir::Value offsetConst = opBuilder2.create<mlir::arith::ConstantIndexOp>(location2, i);
               mismatchAt = opBuilder2.create<mlir::arith::AddIOp>(location2, offsetConst, mismatchAt);
               opBuilder2.create<mlir::scf::YieldOp>(location2, mismatchAt);
            });
         return ifOp.getResult(0);
      };

      rewriter.replaceOp(op, emit(rewriter, loc, 0));
      return mlir::success();
   }
};

class SpecializeBytesNonMatchIndex : public mlir::RewritePattern {
   public:
   SpecializeBytesNonMatchIndex(mlir::MLIRContext* context) : RewritePattern(util::BytesNonMatchIndex::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto nonMatchOp = mlir::cast<util::BytesNonMatchIndex>(op);
      auto loc = op->getLoc();
      llvm::StringRef literal = nonMatchOp.getSubpattern();
      auto indexType = rewriter.getIndexType();

      mlir::Value eight = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 8);
      mlir::Value bitOffIdx = rewriter.create<mlir::arith::MulIOp>(loc, nonMatchOp.getStartIndex(), eight);
      mlir::Value bitOff = rewriter.create<mlir::arith::IndexCastUIOp>(loc, rewriter.getIntegerType(128), bitOffIdx);
      mlir::Value haystackValue = rewriter.create<mlir::arith::ShRUIOp>(loc, nonMatchOp.getStr(), bitOff);
      mlir::Value first64 = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI64Type(), haystackValue);
      uint64_t needleInt1 = 0;
      uint64_t mask1 = 0;
      memcpy(&needleInt1, literal.data(), literal.size() <= 8 ? literal.size() : 8);
      memset(&mask1, 0xFF,  literal.size() <= 8 ? literal.size() : 8);
      mlir::Value needleIntValue1 = rewriter.create<mlir::arith::ConstantIntOp>(loc, needleInt1, rewriter.getI64Type());

      mlir::Value maskValue = rewriter.create<mlir::arith::XOrIOp>(loc, first64, needleIntValue1);
      mlir::Value needleMaskValue1 = rewriter.create<mlir::arith::ConstantIntOp>(loc, mask1, rewriter.getI64Type());
      maskValue = rewriter.create<mlir::arith::AndIOp>(loc, maskValue, needleMaskValue1);
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, rewriter.getI64Type());
      mlir::Value isEqual = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, maskValue, zero);
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, isEqual,
         [&literal, &haystackValue](mlir::OpBuilder& opBuilder, mlir::Location location) {
            if (literal.size() <= 8) {
               mlir::Value mismatchAt = opBuilder.create<mlir::arith::ConstantIndexOp>(location, literal.size());
               opBuilder.create<mlir::scf::YieldOp>(location, mismatchAt);
            } else {
               uint32_t needleInt2 = 0;
               uint32_t mask2 = 0;
               memcpy(&needleInt2, literal.data() + 8, literal.size() - 8);
               memset(&mask2, 0xFF, literal.size() - 8);
               mlir::Value needleIntValue2 = opBuilder.create<mlir::arith::ConstantIntOp>(location, needleInt2, opBuilder.getI32Type());
               mlir::Value shift = opBuilder.create<mlir::arith::ConstantOp>(location, opBuilder.getIntegerAttr(opBuilder.getIntegerType(128), 64));
               mlir::Value shiftedHaystack = opBuilder.create<mlir::arith::ShRUIOp>(location, haystackValue, shift);
               mlir::Value bytes8to11 = opBuilder.create<mlir::arith::TruncIOp>(location, opBuilder.getI32Type(), shiftedHaystack);
               mlir::Value maskValue = opBuilder.create<mlir::arith::XOrIOp>(location, bytes8to11, needleIntValue2);
               mlir::Value haystackMaskValue2 = opBuilder.create<mlir::arith::ConstantIntOp>(location, mask2, opBuilder.getI32Type());
               maskValue = opBuilder.create<mlir::arith::AndIOp>(location, maskValue, haystackMaskValue2);
               mlir::Value zero = opBuilder.create<mlir::arith::ConstantIntOp>(location, 0, opBuilder.getI32Type());
               mlir::Value isEqual = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, maskValue, zero);
               auto innerIfOp = opBuilder.create<mlir::scf::IfOp>(location, isEqual,
                  [&literal](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                     mlir::Value mismatchAt = opBuilder2.create<mlir::arith::ConstantIndexOp>(location2, literal.size());
                     opBuilder2.create<mlir::scf::YieldOp>(location2, mismatchAt);
                  },
                  [&maskValue](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                     mlir::Value trailingZeros = opBuilder2.create<mlir::LLVM::CountTrailingZerosOp>(location2, opBuilder2.getI32Type(),  maskValue, false);
                     mlir::Value tzIndex = opBuilder2.create<mlir::arith::IndexCastOp>(location2, opBuilder2.getIndexType(), trailingZeros);
                     mlir::Value const8 = opBuilder2.create<mlir::arith::ConstantIndexOp>(location2, 8);
                     mlir::Value mismatchAt = opBuilder2.create<mlir::arith::DivUIOp>(location2, tzIndex, const8);
                     mismatchAt = opBuilder2.create<mlir::arith::AddIOp>(location2, mismatchAt, const8);
                     opBuilder2.create<mlir::scf::YieldOp>(location2, mismatchAt);
                  });
               opBuilder.create<mlir::scf::YieldOp>(location, innerIfOp.getResult(0));
            }
         },
         [&maskValue](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            mlir::Value trailingZeros = opBuilder2.create<mlir::LLVM::CountTrailingZerosOp>(location2, opBuilder2.getI64Type(),  maskValue, false);
            mlir::Value tzIndex = opBuilder2.create<mlir::arith::IndexCastOp>(location2, opBuilder2.getIndexType(), trailingZeros);
            mlir::Value const8 = opBuilder2.create<mlir::arith::ConstantIndexOp>(location2, 8);
            mlir::Value mismatchAt = opBuilder2.create<mlir::arith::DivUIOp>(location2, tzIndex, const8);
            opBuilder2.create<mlir::scf::YieldOp>(location2, mismatchAt);
      });
      mlir::Value index = ifOp.getResult(0);
      rewriter.replaceOp(op, index);
      return mlir::success();
   }
};
class PrepareLowering : public mlir::PassWrapper<PrepareLowering, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "util-prepare-lowering"; }
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::scf::SCFDialect>();
      registry.insert<mlir::arith::ArithDialect>();
      registry.insert<mlir::LLVM::LLVMDialect>();
   }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLowering)
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.add<SplitConstLike>(&getContext());
         patterns.add<SpecializeStringContains>(&getContext());
         patterns.add<SpecializeBytesContains>(&getContext());
         patterns.add<SpecializeStringNonMatchIndex>(&getContext());
         patterns.add<SpecializeBytesNonMatchIndex>(&getContext());
         if (lingodb::compiler::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }

   }
};
} // namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::util::createPrepareLoweringPass() { return std::make_unique<PrepareLowering>(); } // NOLINT(misc-use-internal-linkage)
