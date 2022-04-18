#include "frontend/SQL/Parser.h"

int main(int argc, char** argv) {
   mlir::MLIRContext context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<mlir::relalg::RelAlgDialect>();
   mlir::OpBuilder builder(&context);
   frontend::sql::Parser::Schema schema = {{"nation", {{"n_nationkey", builder.getI32Type()}, {"n_name", mlir::db::StringType::get(builder.getContext())}, {"n_regionkey", builder.getI32Type()}, {"n_comment", mlir::db::NullableType::get(builder.getContext(), mlir::db::StringType::get(builder.getContext()))}}},

                                   {"region", {{"r_regionkey", builder.getI32Type()}, {"r_name", mlir::db::StringType::get(builder.getContext())}, {"r_comment", mlir::db::NullableType::get(builder.getContext(), mlir::db::StringType::get(builder.getContext()))}}},

                                   {"part", {{"p_partkey", builder.getI32Type()}, {"p_name", mlir::db::StringType::get(builder.getContext())}, {"p_mfgr", mlir::db::StringType::get(builder.getContext())}, {"p_brand", mlir::db::StringType::get(builder.getContext())}, {"p_type", mlir::db::StringType::get(builder.getContext())}, {"p_size", builder.getI32Type()}, {"p_container", mlir::db::StringType::get(builder.getContext())}, {"p_retailprice", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"p_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"supplier", {{"s_suppkey", builder.getI32Type()}, {"s_name", mlir::db::StringType::get(builder.getContext())}, {"s_address", mlir::db::StringType::get(builder.getContext())}, {"s_nationkey", builder.getI32Type()}, {"s_phone", mlir::db::StringType::get(builder.getContext())}, {"s_acctbal", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"s_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"partsupp", {{"ps_partkey", builder.getI32Type()}, {"ps_suppkey", builder.getI32Type()}, {"ps_availqty", builder.getI32Type()}, {"ps_supplycost", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"ps_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"customer", {{"c_custkey", builder.getI32Type()}, {"c_name", mlir::db::StringType::get(builder.getContext())}, {"c_address", mlir::db::StringType::get(builder.getContext())}, {"c_nationkey", builder.getI32Type()}, {"c_phone", mlir::db::StringType::get(builder.getContext())}, {"c_acctbal", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"c_mktsegment", mlir::db::StringType::get(builder.getContext())}, {"c_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"orders", {{"o_orderkey", builder.getI32Type()}, {"o_custkey", builder.getI32Type()}, {"o_orderstatus", mlir::db::CharType::get(builder.getContext(), 1)}, {"o_totalprice", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"o_orderdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"o_orderpriority", mlir::db::StringType::get(builder.getContext())}, {"o_clerk", mlir::db::StringType::get(builder.getContext())}, {"o_shippriority", builder.getI32Type()}, {"o_comment", mlir::db::StringType::get(builder.getContext())}}},

                                   {"lineitem", {{"l_orderkey", builder.getI32Type()}, {"l_partkey", builder.getI32Type()}, {"l_suppkey", builder.getI32Type()}, {"l_linenumber", builder.getI32Type()}, {"l_quantity", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_extendedprice", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_discount", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_tax", mlir::db::DecimalType::get(builder.getContext(), 15, 2)}, {"l_returnflag", mlir::db::CharType::get(builder.getContext(), 1)}, {"l_linestatus", mlir::db::CharType::get(builder.getContext(), 1)}, {"l_shipdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"l_commitdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"l_receiptdate", mlir::db::DateType::get(builder.getContext(), mlir::db::DateUnitAttr::day)}, {"l_shipinstruct", mlir::db::StringType::get(builder.getContext())}, {"l_shipmode", mlir::db::StringType::get(builder.getContext())}, {"l_comment", mlir::db::StringType::get(builder.getContext())}}},
                                   {"assistenten", {{"persnr", builder.getI64Type()}, {"name", mlir::db::StringType::get(builder.getContext())}, {"fachgebiet", mlir::db::StringType::get(builder.getContext())}, {"boss", builder.getI64Type()}}},
                                   {"hoeren", {{"matrnr", builder.getI64Type()}, {"vorlnr", builder.getI64Type()}}},
                                   {"studenten", {{"matrnr", builder.getI64Type()}, {"name", mlir::db::StringType::get(builder.getContext())}, {"semester", builder.getI64Type()}}},
                                   {"professoren", {{"persnr", builder.getI64Type()}, {"name", mlir::db::StringType::get(builder.getContext())}, {"rang", mlir::db::StringType::get(builder.getContext())}, {"raum", builder.getI64Type()}}},
                                   {"vorlesungen", {{"vorlnr", builder.getI64Type()}, {"titel", mlir::db::StringType::get(builder.getContext())}, {"sws", builder.getI64Type()}, {"gelesenvon", builder.getI64Type()}}},
                                   {"voraussetzen", {{"vorgaenger", builder.getI64Type()}, {"nachfolger", builder.getI64Type()}}},
                                   {"pruefen", {{"matrnr", builder.getI64Type()}, {"vorlnr", builder.getI64Type()}, {"persnr", builder.getI64Type()}, {"note", mlir::db::DecimalType::get(builder.getContext(), 2, 1)}}}};

   std::string filename = std::string(argv[1]);
   std::ifstream istream{filename};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   frontend::sql::Parser translator(buffer.str(), schema, &context);
   mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

   builder.setInsertionPointToStart(moduleOp.getBody());
   mlir::FuncOp funcOp = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {mlir::dsa::TableType::get(builder.getContext())}));
   funcOp.body().push_back(new mlir::Block);
   builder.setInsertionPointToStart(&funcOp.body().front());
   mlir::Value val = translator.translate(builder);

   builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), val);
   mlir::OpPrintingFlags flags;
   flags.assumeVerified();
   moduleOp->print(llvm::outs(), flags);
   return 0;
}