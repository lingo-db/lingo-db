#include "catch2/catch_all.hpp"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/baseline/utils.hpp"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

using namespace lingodb::execution::baseline;

TEST_CASE("TupleHelper::getElementOffset and sizeAndPadding with real MLIR types") {
   mlir::MLIRContext ctx;

   auto i8  = mlir::IntegerType::get(&ctx, 8);
   auto i32 = mlir::IntegerType::get(&ctx, 32);
   auto f64 = mlir::Float64Type::get(&ctx);
   auto idx = mlir::IndexType::get(&ctx);

   // If you have a custom RefType, create it here, otherwise use a placeholder
   // For demonstration, use i32 as a stand-in for RefType
   auto ref = i32;

   auto tuple = mlir::TupleType::get(&ctx, {i8, i32, f64, ref, idx});
   TupleHelper helper(tuple);

   SECTION("getElementOffset returns correct offsets") {
      auto offset0 = helper.getElementOffset(0);
      auto offset1 = helper.getElementOffset(1);
      auto offset2 = helper.getElementOffset(2);
      auto offset3 = helper.getElementOffset(3);
      auto offset4 = helper.getElementOffset(4);

      REQUIRE(offset0 == 0);
      REQUIRE(offset1 > offset0);
      REQUIRE(offset2 > offset1);
      REQUIRE(offset3 > offset2);
      REQUIRE(offset4 > offset3);
   }

   SECTION("sizeAndPadding returns correct size and alignment") {
      auto [size, align] = helper.sizeAndPadding();
      REQUIRE(size > 0);
      REQUIRE(align > 0);
      auto lastOffset = helper.getElementOffset(4);
      REQUIRE(size >= lastOffset + sizeof(uint64_t));
   }
}