#include "catch2/catch_all.hpp"

#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/BetaEstimator.h"

#include <cmath>
#include <vector>

using namespace lingodb::compiler::dialect::relalg::betaestimator;

namespace {

constexpr double kLn2 = 0.6931471805599453;

struct BitmapOracle {
   size_t sampleSize = 0;
   std::vector<std::vector<bool>> bitmaps;

   std::optional<size_t> eval(llvm::ArrayRef<size_t> positive,
                              llvm::ArrayRef<size_t> negatedGroup) const {
      size_t count = 0;
      for (size_t t = 0; t < sampleSize; ++t) {
         bool keep = true;
         for (size_t p : positive) {
            if (!bitmaps[p][t]) {
               keep = false;
               break;
            }
         }
         if (keep && !negatedGroup.empty()) {
            bool negAll = true;
            for (size_t p : negatedGroup) {
               if (!bitmaps[p][t]) {
                  negAll = false;
                  break;
               }
            }
            if (negAll) keep = false;
         }
         if (keep) ++count;
      }
      return count;
   }

   CountOracle make() const {
      return [this](llvm::ArrayRef<size_t> p, llvm::ArrayRef<size_t> n) { return eval(p, n); };
   }
};

} // namespace

TEST_CASE("BetaEstimator: single predicate, non-0-TS uses traditional k/m") {
   BitmapOracle o;
   o.sampleSize = 100;
   o.bitmaps.assign(1, std::vector<bool>(100, false));
   for (size_t i = 0; i < 50; ++i) o.bitmaps[0][i] = true;

   auto r = estimateSelectivity(1, 100, o.make());
   REQUIRE(r.has_value());
   CHECK(*r == Catch::Approx(0.5));
}

TEST_CASE("BetaEstimator: single predicate, k=1 uses Patch 1 (ln(2)/m)") {
   BitmapOracle o;
   o.sampleSize = 100;
   o.bitmaps.assign(1, std::vector<bool>(100, false));
   o.bitmaps[0][0] = true;

   auto r = estimateSelectivity(1, 100, o.make());
   REQUIRE(r.has_value());
   CHECK(*r == Catch::Approx(kLn2 / 100.0));
}

TEST_CASE("BetaEstimator: single predicate, k=0 yields ~ln(2)/(2m)") {
   BitmapOracle o;
   o.sampleSize = 100;
   o.bitmaps.assign(1, std::vector<bool>(100, false));

   auto r = estimateSelectivity(1, 100, o.make());
   REQUIRE(r.has_value());
   CHECK(*r == Catch::Approx(kLn2 / 200.0));
}

TEST_CASE("BetaEstimator: multi-predicate non-0-TS uses k/m on the full conjunction") {
   BitmapOracle o;
   o.sampleSize = 100;
   o.bitmaps.assign(2, std::vector<bool>(100, false));
   for (size_t i = 0; i < 50; ++i) o.bitmaps[0][i] = true; // tuples 0..49
   for (size_t i = 0; i < 75; ++i) o.bitmaps[1][i] = true; // tuples 0..74, conj = 0..49

   auto r = estimateSelectivity(2, 100, o.make());
   REQUIRE(r.has_value());
   CHECK(*r == Catch::Approx(0.5)); // 50/100
}

TEST_CASE("BetaEstimator: 0-TS with anti-correlated predicates is smaller than AVI") {
   // pred0 = tuples 0..49, pred1 = tuples 50..99. Disjoint.
   BitmapOracle o;
   o.sampleSize = 100;
   o.bitmaps.assign(2, std::vector<bool>(100, false));
   for (size_t i = 0; i < 50; ++i) o.bitmaps[0][i] = true;
   for (size_t i = 50; i < 100; ++i) o.bitmaps[1][i] = true;

   auto r = estimateSelectivity(2, 100, o.make());
   REQUIRE(r.has_value());
   // AVI would be 0.5 * 0.5 = 0.25; the paper's estimator should recognise the
   // anti-correlation and produce a much smaller selectivity.
   CHECK(*r > 0.0);
   CHECK(*r < 0.05);
}

TEST_CASE("BetaEstimator: 0-TS with sample size bigger => smaller estimate") {
   // Two anti-correlated predicates at two sample sizes; bigger sample should
   // let the estimator express stronger anti-correlation => smaller value.
   auto estimate = [](size_t n) {
      BitmapOracle o;
      o.sampleSize = n;
      o.bitmaps.assign(2, std::vector<bool>(n, false));
      for (size_t i = 0; i < n / 2; ++i) o.bitmaps[0][i] = true;
      for (size_t i = n / 2; i < n; ++i) o.bitmaps[1][i] = true;
      return *estimateSelectivity(2, n, o.make());
   };
   double small = estimate(100);
   double large = estimate(1000);
   CHECK(large < small);
}

TEST_CASE("BetaEstimator: 0-TS bounds – result always in (0,1]") {
   BitmapOracle o;
   o.sampleSize = 200;
   o.bitmaps.assign(3, std::vector<bool>(200, false));
   // pred0 = 0..99, pred1 = 0..49 (conj with 0 = 0..49), pred2 = 100..199.
   // full conj: pred0∧pred1∧pred2 is empty.
   for (size_t i = 0; i < 100; ++i) o.bitmaps[0][i] = true;
   for (size_t i = 0; i < 50; ++i) o.bitmaps[1][i] = true;
   for (size_t i = 100; i < 200; ++i) o.bitmaps[2][i] = true;

   auto r = estimateSelectivity(3, 200, o.make());
   REQUIRE(r.has_value());
   CHECK(*r > 0.0);
   CHECK(*r <= 1.0);
}

TEST_CASE("BetaEstimator: all-zero predicate returns non-zero") {
   // When even an individual predicate has 0 hits, Beta still produces a
   // positive selectivity (it does not collapse to 0).
   BitmapOracle o;
   o.sampleSize = 100;
   o.bitmaps.assign(2, std::vector<bool>(100, false));
   for (size_t i = 0; i < 30; ++i) o.bitmaps[0][i] = true;
   // pred1 has no hits at all.

   auto r = estimateSelectivity(2, 100, o.make());
   REQUIRE(r.has_value());
   CHECK(*r > 0.0);
}

TEST_CASE("BetaEstimator: degenerate input – m=0 returns nullopt") {
   BitmapOracle o;
   o.sampleSize = 0;
   o.bitmaps.assign(1, std::vector<bool>{});

   auto r = estimateSelectivity(1, 0, o.make());
   CHECK(!r.has_value());
}

TEST_CASE("BetaEstimator: degenerate input – 0 predicates returns 1.0") {
   BitmapOracle o;
   o.sampleSize = 100;
   auto r = estimateSelectivity(0, 100, o.make());
   REQUIRE(r.has_value());
   CHECK(*r == Catch::Approx(1.0));
}
