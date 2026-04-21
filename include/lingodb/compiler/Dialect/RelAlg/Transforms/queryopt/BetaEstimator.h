#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_BETAESTIMATOR_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_BETAESTIMATOR_H

#include <cstddef>
#include <functional>
#include <optional>

#include <llvm/ADT/ArrayRef.h>

namespace lingodb::compiler::dialect::relalg::betaestimator {

// Answers: Cnt( (AND p_i for i in positive) AND NOT (AND p_i for i in negatedGroup), S ).
// An empty positive list is interpreted as TRUE; an empty negatedGroup skips the negation.
// Returns nullopt if the count cannot be computed (e.g. unsupported predicate).
using CountOracle = std::function<std::optional<size_t>(
   llvm::ArrayRef<size_t> positive,
   llvm::ArrayRef<size_t> negatedGroup)>;

// Selectivity estimator implementing Algorithms 1 and 2 of
//   Hertzschuch, Moerkotte, Lehner, May, Wolf, Fricke.
//   "Small Selectivities Matter: Lifting the Burden of Empty Samples". SIGMOD '21.
//
// Uses the beta distribution to adjust the selectivity estimate of a conjunctive
// filter on a sample when no sample tuple qualifies (0-TS). Returns the estimated
// selectivity of p_0 ∧ p_1 ∧ ... ∧ p_{numPredicates-1}, or nullopt if the oracle
// could not answer some query or the inputs are ill-formed.
std::optional<double> estimateSelectivity(
   size_t numPredicates,
   size_t sampleSize,
   const CountOracle& count);

} // namespace lingodb::compiler::dialect::relalg::betaestimator

#endif // LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_BETAESTIMATOR_H
