// RUN: mlir-db-opt %s --graphalg-semi-naive | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  // A monotone, linear fixpoint over the boolean (idempotent OR) semiring:
  //   M_next = reduce(M (+) reduce(A . M))   -> eligible for semi-naive.
  // Rewritten into semi-naive form: the loop now carries (state, delta) and the
  // body extracts the changed cells via graphalg.delta.
  // CHECK-LABEL: func.func @eligible_bool_linear
  // CHECK: graphalg.for_dim
  // CHECK: graphalg.delta
  func.func @eligible_bool_linear(%A: !graphalg.mat<#dim x #dim x i1>, %init: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i1> {
    %r = graphalg.for_dim range(#dim) init(%init) : !graphalg.mat<#dim x #dim x i1> -> !graphalg.mat<#dim x #dim x i1> body {
    ^bb0(%idx: !graphalg.mat<1 x 1 x i64>, %M: !graphalg.mat<#dim x #dim x i1>):
      %p = graphalg.mxm_join %A, %M : <#dim x #dim x i1>, <#dim x #dim x i1>
      %pr = graphalg.deferred_reduce %p : !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i1>
      %u = graphalg.union %M, %pr : !graphalg.mat<#dim x #dim x i1>, !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i1>
      %ur = graphalg.deferred_reduce %u : !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i1>
      graphalg.yield %ur : !graphalg.mat<#dim x #dim x i1>
    } until {
    }
    return %r : !graphalg.mat<#dim x #dim x i1>
  }

  // Same shape over the tropical (min, idempotent) semiring -> eligible.
  // CHECK-LABEL: func.func @eligible_tropical_linear
  // CHECK: graphalg.for_dim
  // CHECK: graphalg.delta
  func.func @eligible_tropical_linear(%A: !graphalg.mat<#dim x #dim x !graphalg.trop_f64>, %init: !graphalg.mat<#dim x 1 x !graphalg.trop_f64>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> {
    %r = graphalg.for_dim range(#dim) init(%init) : !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> body {
    ^bb0(%idx: !graphalg.mat<1 x 1 x i64>, %M: !graphalg.mat<#dim x 1 x !graphalg.trop_f64>):
      %p = graphalg.mxm_join %A, %M : <#dim x #dim x !graphalg.trop_f64>, <#dim x 1 x !graphalg.trop_f64>
      %pr = graphalg.deferred_reduce %p : !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> <#dim x 1 x !graphalg.trop_f64>
      %u = graphalg.union %M, %pr : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>, !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> <#dim x 1 x !graphalg.trop_f64>
      %ur = graphalg.deferred_reduce %u : !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> <#dim x 1 x !graphalg.trop_f64>
      graphalg.yield %ur : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
    } until {
    }
    return %r : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
  }

  // Non-idempotent additive monoid (plain i64 uses `+`) -> NOT eligible, even
  // though the body is linear and accumulating.
  // CHECK-LABEL: func.func @ineligible_non_idempotent
  // CHECK: graphalg.for_dim
  // CHECK-NOT: graphalg.delta
  func.func @ineligible_non_idempotent(%A: !graphalg.mat<#dim x #dim x i64>, %init: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
    %r = graphalg.for_dim range(#dim) init(%init) : !graphalg.mat<#dim x #dim x i64> -> !graphalg.mat<#dim x #dim x i64> body {
    ^bb0(%idx: !graphalg.mat<1 x 1 x i64>, %M: !graphalg.mat<#dim x #dim x i64>):
      %p = graphalg.mxm_join %A, %M : <#dim x #dim x i64>, <#dim x #dim x i64>
      %pr = graphalg.deferred_reduce %p : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64>
      %u = graphalg.union %M, %pr : !graphalg.mat<#dim x #dim x i64>, !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64>
      %ur = graphalg.deferred_reduce %u : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64>
      graphalg.yield %ur : !graphalg.mat<#dim x #dim x i64>
    } until {
    }
    return %r : !graphalg.mat<#dim x #dim x i64>
  }

  // Non-linear body: the state is multiplied by itself (M . M, degree 2), which
  // does not distribute over the semiring add -> NOT eligible.
  // CHECK-LABEL: func.func @ineligible_non_linear
  // CHECK: graphalg.for_dim
  // CHECK-NOT: graphalg.delta
  func.func @ineligible_non_linear(%init: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i1> {
    %r = graphalg.for_dim range(#dim) init(%init) : !graphalg.mat<#dim x #dim x i1> -> !graphalg.mat<#dim x #dim x i1> body {
    ^bb0(%idx: !graphalg.mat<1 x 1 x i64>, %M: !graphalg.mat<#dim x #dim x i1>):
      %p = graphalg.mxm_join %M, %M : <#dim x #dim x i1>, <#dim x #dim x i1>
      %pr = graphalg.deferred_reduce %p : !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i1>
      %u = graphalg.union %M, %pr : !graphalg.mat<#dim x #dim x i1>, !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i1>
      %ur = graphalg.deferred_reduce %u : !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i1>
      graphalg.yield %ur : !graphalg.mat<#dim x #dim x i1>
    } until {
    }
    return %r : !graphalg.mat<#dim x #dim x i1>
  }
}
