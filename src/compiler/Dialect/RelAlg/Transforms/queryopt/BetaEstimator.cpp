#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/BetaEstimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/tools/minima.hpp>

namespace lingodb::compiler::dialect::relalg::betaestimator {
namespace {

constexpr double kLn2 = 0.6931471805599453;

// Eq. 9: beta shape parameters for a sample of size m with k qualifying tuples.
std::pair<double, double> shapeParams(size_t k, size_t m) {
   if (k >= 1) {
      return {static_cast<double>(k) + 1.0 / 3.0,
              static_cast<double>(m) - static_cast<double>(k) + 1.0 / 3.0};
   }
   return {0.634, static_cast<double>(m)};
}

// Tight bounds on the conditional selectivity (Alg. 1, Lines 2-3).
std::pair<double, double> getBounds(size_t k, size_t m) {
   double lo = (k >= 1) ? (static_cast<double>(k) - 1.0) / m : 0.0;
   double up = std::min((static_cast<double>(k) + 1.0) / static_cast<double>(m), 1.0);
   return {lo, up};
}

// Point estimate used at Alg. 2 Line 10 for a single predicate.
// k qualifying of m with the paper's 0-TS / full-sample corner-case adjustments.
double pointEstimate(size_t k, size_t m) {
   if (m == 0) return 0.0;
   if (k == 0) return kLn2 / (2.0 * static_cast<double>(m));
   if (k == m) return 1.0 - kLn2 / (2.0 * static_cast<double>(m));
   return static_cast<double>(k) / static_cast<double>(m);
}

double betaCdf(double x, double a, double b) {
   if (x <= 0.0) return 0.0;
   if (x >= 1.0) return 1.0;
   return boost::math::ibeta(a, b, x);
}

std::vector<size_t> unionIndices(llvm::ArrayRef<size_t> a, llvm::ArrayRef<size_t> b) {
   std::vector<size_t> r(a.begin(), a.end());
   for (size_t x : b) {
      if (std::find(r.begin(), r.end(), x) == r.end()) r.push_back(x);
   }
   return r;
}

// Algorithm 1: balanceEstimate.
// Returns an estimate of p(A|B), adjusted by balancing the beta distributions
// modeling the certainty of p(A|B) and p(A|!B).
std::optional<double> balanceEstimate(
   llvm::ArrayRef<size_t> subsetA,
   llvm::ArrayRef<size_t> subsetB,
   size_t m,
   const CountOracle& count) {
   auto mBOpt = count(subsetB, {});
   if (!mBOpt) return std::nullopt;
   size_t mB = *mBOpt;

   // Alg. 1, Line 16: Cnt(P_B, S) ∈ {0, |S|} means nothing to balance; fall
   // back to the traditional sample-based estimate of p(A).
   if (mB == 0 || mB == m) {
      auto mAOpt = count(subsetA, {});
      if (!mAOpt) return std::nullopt;
      return static_cast<double>(*mAOpt) / static_cast<double>(m);
   }

   size_t mNotB = m - mB;

   auto unionAB = unionIndices(subsetA, subsetB);
   auto kABOpt = count(unionAB, {});
   auto kANotBOpt = count(subsetA, subsetB);
   if (!kABOpt || !kANotBOpt) return std::nullopt;
   size_t kAB = *kABOpt;
   size_t kANotB = *kANotBOpt;

   auto [a1, b1] = shapeParams(kAB, mB);
   auto [a2, b2] = shapeParams(kANotB, mNotB);
   auto [zabLo, zabUp] = getBounds(kAB, mB);
   auto [znbLo, znbUp] = getBounds(kANotB, mNotB);

   double pB = static_cast<double>(mB) / static_cast<double>(m);
   double pNotB = 1.0 - pB;

   // p(A) is held at the value implied by the initial (pre-balance) point
   // estimates of p(A|B) and p(A|!B), so that Eq. 11 remains satisfied while
   // the CDFs (Eq. 12) are balanced.
   double zabInit = pointEstimate(kAB, mB);
   double znbInit = pointEstimate(kANotB, mNotB);
   double pA = pB * zabInit + pNotB * znbInit;

   // Pick the variable whose interval is tighter (Alg. 1, Line 22).
   bool searchAB = (zabUp - zabLo) < (znbUp - znbLo);

   auto targetAB = [&](double zAB) -> double {
      if (zAB <= 0.0 || zAB >= 1.0) return std::numeric_limits<double>::infinity();
      double zSub = (pA - pB * zAB) / pNotB;
      if (zSub <= 0.0 || zSub >= 1.0) return std::numeric_limits<double>::infinity();
      double cAB = betaCdf(zAB, a1, b1);
      double cSub = betaCdf(zSub, a2, b2);
      if (cAB <= 0.0 || cSub <= 0.0) return std::numeric_limits<double>::infinity();
      return std::max(cAB / cSub, cSub / cAB);
   };
   auto targetNotB = [&](double zNotB) -> double {
      if (zNotB <= 0.0 || zNotB >= 1.0) return std::numeric_limits<double>::infinity();
      double zSub = (pA - pNotB * zNotB) / pB;
      if (zSub <= 0.0 || zSub >= 1.0) return std::numeric_limits<double>::infinity();
      double cNotB = betaCdf(zNotB, a2, b2);
      double cSub = betaCdf(zSub, a1, b1);
      if (cNotB <= 0.0 || cSub <= 0.0) return std::numeric_limits<double>::infinity();
      return std::max(cNotB / cSub, cSub / cNotB);
   };

   constexpr int kBrentBits = 20;
   boost::uintmax_t maxIter = 100;
   double result;
   if (searchAB) {
      auto minimum = boost::math::tools::brent_find_minima(
         targetAB, zabLo, zabUp, kBrentBits, maxIter);
      result = minimum.first;
   } else {
      auto minimum = boost::math::tools::brent_find_minima(
         targetNotB, znbLo, znbUp, kBrentBits, maxIter);
      // Back-substitute via Eq. 10 to recover z_A|B.
      result = (pA - pNotB * minimum.first) / pB;
   }
   return std::clamp(result, 0.0, 1.0);
}

} // namespace

std::optional<double> estimateSelectivity(
   size_t numPredicates,
   size_t m,
   const CountOracle& count) {
   if (numPredicates == 0) return 1.0;
   if (m == 0) return std::nullopt;

   std::vector<size_t> all(numPredicates);
   for (size_t i = 0; i < numPredicates; ++i) all[i] = i;

   // Alg. 2 Line 4-5: traditional estimate if the full conjunction has any
   // qualifying tuples. Patch 1 substitutes ln(2)/m for k=1 (1-TS).
   auto kAllOpt = count(all, {});
   if (!kAllOpt) return std::nullopt;
   size_t kAll = *kAllOpt;
   if (kAll >= 1) {
      if (kAll == 1) return kLn2 / static_cast<double>(m);
      return static_cast<double>(kAll) / static_cast<double>(m);
   }

   // Phase 1: extend the visited prefix predicate-by-predicate until a 0-TS
   // appears; when it does, use balanceEstimate to adjust the conditional.
   double selectivity = 1.0;
   std::vector<size_t> visited;
   bool exitPhase1 = false;
   for (size_t i = 0; i < numPredicates && !exitPhase1; ++i) {
      size_t pNew = i;

      std::vector<size_t> visPlus = visited;
      visPlus.push_back(pNew);
      auto visNewOpt = count(visPlus, {});
      if (!visNewOpt) return std::nullopt;

      if (*visNewOpt > 0) {
         // Alg. 2, Line 11-12.
         visited.push_back(pNew);
         continue;
      }

      // visited ∧ p_new is 0-TS; inspect visited and residual counts.
      size_t visCount;
      if (visited.empty()) {
         visCount = m;
      } else {
         auto vOpt = count(visited, {});
         if (!vOpt) return std::nullopt;
         visCount = *vOpt;
      }

      std::vector<size_t> residual;
      residual.reserve(numPredicates - i);
      for (size_t j = i; j < numPredicates; ++j) residual.push_back(j);
      auto resOpt = count(residual, {});
      if (!resOpt) return std::nullopt;
      size_t resCount = *resOpt;

      if (visCount == 0 && resCount > 0) {
         // Alg. 2, Line 13-14.
         exitPhase1 = true;
         break;
      }

      auto pNewCountOpt = count(llvm::ArrayRef<size_t>(&pNew, 1), {});
      if (!pNewCountOpt) return std::nullopt;
      double pB = pointEstimate(*pNewCountOpt, m);

      if (visCount == 0) {
         // Alg. 2, Line 15-16: AVI (visited already dead, can't balance).
         selectivity *= pB;
      } else {
         // Alg. 2, Line 17-19: balanceEstimate.
         auto adj = balanceEstimate(visited, llvm::ArrayRef<size_t>(&pNew, 1), m, count);
         if (!adj) return std::nullopt;
         selectivity *= (*adj) * pB;
      }
      visited.push_back(pNew);
   }

   // Phase 2: residual = predicates not in visited.
   std::vector<size_t> residual;
   {
      std::vector<bool> inVisited(numPredicates, false);
      for (size_t v : visited) inVisited[v] = true;
      for (size_t i = 0; i < numPredicates; ++i) {
         if (!inVisited[i]) residual.push_back(i);
      }
   }

   if (residual.empty()) return selectivity;

   while (!visited.empty()) {
      auto testWithRes = count(unionIndices(visited, residual), {});
      auto testOnly = count(visited, {});
      if (!testWithRes || !testOnly) return std::nullopt;

      if (*testWithRes > 0 && *testOnly > 0) {
         double pAB = static_cast<double>(*testWithRes) / static_cast<double>(*testOnly);
         return pAB * selectivity;
      }

      auto testNotRes = count(visited, residual);
      if (!testNotRes) return std::nullopt;
      if (*testNotRes > 0) {
         auto adj = balanceEstimate(residual, visited, m, count);
         if (!adj) return std::nullopt;
         return (*adj) * selectivity;
      }

      visited.pop_back();
   }

   // Alg. 2, Line 30: AVI fallback over the residual.
   auto resOnly = count(residual, {});
   if (!resOnly) return std::nullopt;
   return selectivity * static_cast<double>(*resOnly) / static_cast<double>(m);
}

} // namespace lingodb::compiler::dialect::relalg::betaestimator
