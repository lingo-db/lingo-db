// Implementation of the μ-estimator from:
// Moerkotte & Hertzschuch, "α to ω: The G(r)eek Alphabet of Sampling", CIDR 2020.
//
// Given a relation of size n, a sample of size m drawn without replacement, and
// k qualifying sample tuples, this computes a better estimate of the true
// number l of qualifying tuples in the full relation via the geometric mean of
// probabilistic lower bound α and upper bound ω.
//
// α, ω are the integer bounds on l such that the probability of observing k
// qualifying samples given l is below ε = 1e-5 outside [α, ω]. They are found
// as roots of f(x) = log P(n,m,k,x) - log ε using Newton's method with the
// derivative approximated from Stirling's formula (Appendix C.6 of the paper).

#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/SampleCorrection.h"

#include <algorithm>
#include <cmath>

namespace lingodb::compiler::dialect::relalg {
namespace {

constexpr double EPS = 1e-5;

// F(x) = log Γ(x+1) ; continuous extension of log(x!)
inline double F(double x) { return std::lgamma(x + 1.0); }

// log P(n,m,k,l) for the hypergeometric distribution
//   P(n,m,k,l) = C(n-l, m-k) · C(l, k) / C(n, m)
double logProb(double n, double m, double k, double l) {
   return F(n - l) + F(l) + F(m) + F(n - m)
      - F(m - k) - F(n - l - m + k) - F(k) - F(l - k) - F(n);
}

// Derivative f'(x) based on the Stirling approximation (Eqn 24), as given in
// Appendix C.6.
double fPrime(double n, double m, double k, double x) {
   const double a = n - m + k; // k - m + n in the paper
   return (a + 0.5) / (a - x)
      - x / (a - x)
      + std::log(a - x)
      + (k - 0.5) / (x - k)
      - x / (x - k)
      - std::log(x - k)
      - (n + 0.5) / (n - x)
      + x / (n - x)
      - std::log(n - x)
      + 0.5 / x + std::log(x) + 1.0;
}

// Initial value l_start for the α root search (C.7).
double initAlpha(double n, double m, double k) {
   if (m < 3.0 * k) return (8.0 * n) / (11.0 * m) * k;
   if (k > 50) return 0.4 * n / m * k;
   if (k > 30) return 0.3 * n / m * k;
   if (k > 20) return 0.2 * n / m * k;
   if (k > 12) return 0.1 * n / m * k;
   // k ∈ [2, 12]: n / (x_k · m) · k with x_k from the paper.
   static const double xk[] = {500, 80, 40, 20, 15, 10, 8, 7, 6, 6, 5};
   const int idx = std::max(0, std::min(10, static_cast<int>(k) - 2));
   return n / (xk[idx] * m) * k;
}

// Initial value l_start for the ω root search (C.7).
double initOmega(double n, double m, double k) {
   const double nmk = n / m * k;
   if (k > 0.95 * m) return std::min(1.01 * nmk, n - 10.0);
   if (k > 0.90 * m) return 1.04 * nmk;
   if (k > 0.80 * m) return 1.06 * nmk;
   if (k > 0.70 * m) return 1.09 * nmk;
   if (k > 0.60 * m) return 1.13 * nmk;
   if (k > 0.40 * m) return 1.17 * nmk;
   if (k > 0.30 * m) return 1.20 * nmk;
   return n / (m * (0.75 / (10.0 + 1.7 * k)));
}

// Newton's method to find the root of f(x) = log P(n,m,k,x) - log ε.
// The valid domain is k < x < n-m+k.
double newton(double n, double m, double k, double lStart) {
   const double logEps = std::log(EPS);
   const double lLo = k + 1e-9;
   const double lHi = n - m + k - 1e-9;
   double l = std::clamp(lStart, lLo, lHi);

   // Relative tolerance: f(x) computed via lgamma carries ~1e-4 of numerical
   // noise that gets amplified by a small f'(x) for large n, so an absolute
   // |Δl|<1e-4 criterion oscillates forever. A relative tolerance converges
   // cleanly without affecting the integer-rounded result.
   for (int i = 0; i < 50; ++i) {
      const double fv = logProb(n, m, k, l) - logEps;
      const double fp = fPrime(n, m, k, l);
      if (!std::isfinite(fp) || std::abs(fp) < 1e-300) break;

      double lNew = l - fv / fp;
      if (lNew <= lLo) lNew = 0.5 * (l + lLo);
      if (lNew >= lHi) lNew = 0.5 * (l + lHi);

      const double tol = std::max(1e-4, 1e-7 * std::max(1.0, l));
      if (std::abs(lNew - l) < tol) {
         l = lNew;
         break;
      }
      l = lNew;
   }
   return l;
}

} // namespace

double muEstimate(double relationSize, double hitsInSample, double sampleSize) {
   const double n = relationSize;
   const double k = hitsInSample;
   const double m = sampleSize;

   if (n <= 0 || m <= 0 || k < 0 || k > m || m > n) return 0.0;

   // Sample covers the full relation — k is the exact count, no correction
   // needed. Applies to small fact tables where the "sample" is the table
   // itself, and to any filter where we've seen every row.
   if (m >= n) return k;

   const double logEps = std::log(EPS);

   // k = 0: α(n,m,0) = 0 exactly; we floor at 1 so the geometric mean is
   // non-degenerate (matches the "Formula" column of Table 2 in the paper).
   // ω(n,m,0) ≈ (√(1 − 2·log ε / (t·n)) − 1)·n with t = m/n (C.3).
   if (k == 0) {
      const double t = m / n;
      const double omega = (std::sqrt(1.0 - 2.0 * logEps / (t * n)) - 1.0) * n;
      return std::sqrt(1.0 * std::floor(omega));
   }

   // k = 1: handle without Lambert W. fPrime is singular at x = k = 1, so we
   // detect α via a closed form (α = 1 whenever m/n ≥ ε) and only call Newton
   // for the rare m/n < ε case and for ω.
   if (k == 1) {
      double alpha;
      if (m / n >= EPS) {
         alpha = 1.0;
      } else {
         const double start = std::max(1.1, n * EPS / m);
         const double x0 = newton(n, m, k, start);
         alpha = std::max(1.0, std::ceil(x0));
      }
      const double x1 = newton(n, m, k, initOmega(n, m, k));
      const double omega = std::floor(x1);
      return std::sqrt(alpha * omega);
   }

   // k ≥ 2: Newton's method for both roots.
   const double x0 = newton(n, m, k, initAlpha(n, m, k));
   const double x1 = newton(n, m, k, initOmega(n, m, k));
   const double alpha = std::max(1.0, std::ceil(x0));
   const double omega = std::floor(x1);
   return std::sqrt(alpha * omega);
}

} // namespace lingodb::compiler::dialect::relalg
