#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_SAMPLECORRECTION_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_SAMPLECORRECTION_H

namespace lingodb::compiler::dialect::relalg {

// μ-estimator from Moerkotte & Hertzschuch, "α to ω: The G(r)eek Alphabet of
// Sampling" (CIDR 2020). Given a relation of size `n`, a sample of size `m`
// drawn without replacement, and `k` qualifying sample tuples, returns
// μ = √(α·ω): the geometric mean of the probabilistic lower/upper bounds on
// the true number of qualifying tuples in the full relation. This corrects
// the bias of the naive k/m · n extrapolation, which collapses to 0 for k=0
// and systematically underestimates for small k.
double muEstimate(double relationSize, double hitsInSample, double sampleSize);

} // namespace lingodb::compiler::dialect::relalg

#endif // LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_SAMPLECORRECTION_H
