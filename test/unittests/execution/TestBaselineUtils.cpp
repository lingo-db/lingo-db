#include "catch2/catch_all.hpp"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/baseline/utils.hpp"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

using namespace lingodb::execution::baseline;

TEST_CASE("TupleHelper::getElementOffset and sizeAndPadding with real MLIR types") {
   mlir::MLIRContext ctx;
   mlir::DialectRegistry registry;
   registry.insert<dialect::util::UtilDialect>();
   ctx.appendDialectRegistry(registry);
   ctx.loadDialect<dialect::util::UtilDialect>();

   auto i8 = mlir::IntegerType::get(&ctx, 8);
   auto i32 = mlir::IntegerType::get(&ctx, 32);
   auto f64 = mlir::Float64Type::get(&ctx);
   auto idx = mlir::IndexType::get(&ctx);
   auto ref = dialect::util::RefType::get(&ctx);

   auto t1 = mlir::TupleType::get(&ctx, {i8, i32, f64, ref, idx});
   auto t2 = mlir::TupleType::get(&ctx, {i8, ref, i8});
   auto t3 = mlir::TupleType::get(&ctx, {ref, idx});
   auto t4 = mlir::TupleType::get(&ctx, {t1, t2, i8});
   auto t5_h = mlir::TupleType::get(&ctx, {i8});
   auto t5 = mlir::TupleType::get(&ctx, {t5_h, i8});

   SECTION("getElementOffset returns correct offsets") {
      TupleHelper helper(t1);
      REQUIRE(helper.getElementOffset(0) == 0);
      REQUIRE(helper.getElementOffset(1) == 4);
      REQUIRE(helper.getElementOffset(2) == 8);
      REQUIRE(helper.getElementOffset(3) == 16);
      REQUIRE(helper.getElementOffset(4) == 24);

      helper = TupleHelper(t2);
      REQUIRE(helper.getElementOffset(0) == 0);
      REQUIRE(helper.getElementOffset(1) == 8);
      REQUIRE(helper.getElementOffset(2) == 16);

      helper = TupleHelper(t3);
      REQUIRE(helper.getElementOffset(0) == 0);
      REQUIRE(helper.getElementOffset(1) == 8);

      helper = TupleHelper(t4);
      REQUIRE(helper.getElementOffset(0) == 0);
      REQUIRE(helper.getElementOffset(1) == 32);
      REQUIRE(helper.getElementOffset(2) == 56);

      helper = TupleHelper(t5);
      REQUIRE(helper.getElementOffset(0) == 0);
      REQUIRE(helper.getElementOffset(1) == 1);
   }

   SECTION("sizeAndPadding returns correct size and alignment") {
      TupleHelper helper(t1);
      auto [size, align] = helper.sizeAndPadding();
      REQUIRE(size == 32);
      REQUIRE(align == 8);

      helper = TupleHelper(t2);
      std::tie(size, align) = helper.sizeAndPadding();
      REQUIRE(size == 24);
      REQUIRE(align == 8);

      helper = TupleHelper(t3);
      std::tie(size, align) = helper.sizeAndPadding();
      REQUIRE(size == 16);
      REQUIRE(align == 8);

      helper = TupleHelper(t4);
      std::tie(size, align) = helper.sizeAndPadding();
      REQUIRE(size == 64);
      REQUIRE(align == 8);

      helper = TupleHelper(t5);
      std::tie(size, align) = helper.sizeAndPadding();
      REQUIRE(size == 2);
      REQUIRE(align == 1);
   }
}
