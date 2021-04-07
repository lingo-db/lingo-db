#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include <llvm/Support/ErrorOr.h>
#include <mlir/Dialect/DB/IR/DBDialect.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code EC = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << EC.message() << "\n";
      return -1;
   }

   // Parse the input mlir.
   llvm::SourceMgr sourceMgr;
   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
   module = mlir::parseSourceFile(sourceMgr, &context);
   if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
   }
   return 0;
}

class ToSQL {
   mlir::MLIRContext* context;
   mlir::ModuleOp moduleOp;
   std::unordered_map<mlir::Operation*, std::string> values;
   std::unordered_map<mlir::Operation*, std::string> operators;

   std::string operator_name(mlir::Operation* op) {
      return "op_" + std::to_string((size_t) op);
   }
   std::string attribute_name(mlir::relalg::RelationalAttribute& attr) {
      auto& attributeManager = context->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      auto [scope, name] = attributeManager.getName(&attr);
      return "attr_" + scope + "___" + name;
   }
   std::string resolve_val(mlir::Value v) {
      mlir::Operation* definingOp = v.getDefiningOp();
      if (!definingOp || !values.count(definingOp))
         return "<unknown value>";
      return values[definingOp];
   }
   void handleBinOp(std::stringstream& output, std::string op, mlir::Value a, mlir::Value b) {
      output << "(" << resolve_val(a) << " " << op << " " << resolve_val(b) << ")";
   }
   void joinstr(std::stringstream& output, std::string op, mlir::ValueRange vr) {
      auto first = true;
      for (auto v : vr) {
         if (first) {
            first = false;
         } else {
            output << " " << op << " ";
         }
         output << resolve_val(v);
      }
   }
   void handleOtherOp(mlir::Operation* op) {
      using namespace mlir::db;
      using namespace mlir::relalg;
      using namespace mlir;
      std::stringstream output;
      ::llvm::TypeSwitch<mlir::Operation*, void>(op)
         .Case<mlir::db::ConstantOp>([&](mlir::db::ConstantOp op) {
            mlir::Type t = op.getType();
            mlir::Attribute val = op.getValue();
            if (t.isa<mlir::db::DateType>()) {
               output << "date(";
            } else if (t.isa<mlir::db::IntervalType>()) {
               output << "interval ";
            }
            if (auto strAttr = val.dyn_cast_or_null<mlir::StringAttr>()) {
               if (t.isa<mlir::db::IntType>() || t.isa<mlir::db::DecimalType>() || t.isa<mlir::db::FloatType>()) {
                  output << std::string(strAttr.getValue());

               } else {
                  output << "'" << std::string(strAttr.getValue()) << "'";
               }
            } else if (auto intAttr = val.dyn_cast_or_null<mlir::IntegerAttr>()) {
               output << intAttr.getInt();
            }
            if (t.isa<mlir::db::DateType>()) {
               output << ")";
            }
         })
         .Case<GetAttrOp>([&](GetAttrOp op) {
            output << attribute_name(op.attr().getRelationalAttribute());
         })
         .Case<mlir::db::AndOp>([&](mlir::db::AndOp op) {
            joinstr(output, "and", op.vals());
         })
         .Case<mlir::db::OrOp>([&](mlir::db::OrOp op) {
            joinstr(output, "or", op.vals());
         })
         .Case<DateSubOp>([&](DateSubOp op) {
            handleBinOp(output, "-", op.left(), op.right());
         })
         .Case<AddOp>([&](AddOp op) {
            handleBinOp(output, "+", op.lhs(), op.rhs());
         })
         .Case<SubOp>([&](SubOp op) {
            handleBinOp(output, "-", op.lhs(), op.rhs());
         })
         .Case<MulOp>([&](MulOp op) {
            handleBinOp(output, "*", op.lhs(), op.rhs());
         })
         .Case<DivOp>([&](DivOp op) {
            handleBinOp(output, "/", op.lhs(), op.rhs());
         })
         .Case<CmpOp>([&](CmpOp op) {
            std::string pred = "<unknown pred>";
            switch (op.predicate()) {
               case DBCmpPredicate::eq:
                  pred = "=";
                  break;
               case DBCmpPredicate::neq:
                  pred = "!=";
                  break;
               case DBCmpPredicate::lt:
                  pred = "<";
                  break;
               case DBCmpPredicate::lte:
                  pred = "<=";
                  break;
               case DBCmpPredicate::gt:
                  pred = ">";
                  break;
               case DBCmpPredicate::gte:
                  pred = ">=";
                  break;
               case DBCmpPredicate::like:
                  pred = "like";
                  break;
            }
            handleBinOp(output, pred, op.lhs(), op.rhs());
         })
         .Case<AggrFuncOp>([&](AggrFuncOp op) {
            std::string fnname(mlir::relalg::stringifyAggrFunc(op.fn()));
            auto attr = attribute_name(op.attr().getRelationalAttribute());
            std::string distinct = "";
            if (auto projop = mlir::dyn_cast_or_null<ProjectionOp>(op.rel().getDefiningOp())) {
               if (projop.set_semantic() == SetSemantic::distinct) {
                  distinct = "distinct ";
               }
            }
            output << fnname << "(" << distinct << attr << ")";
         })
         .Case<CountRowsOp>([&](CountRowsOp op) {
            output << "count(*)";
         })
         .Case<CastOp>([&](CastOp op) {
            output << resolve_val(op.val());
         })
         .Case<NotOp>([&](NotOp op) {
            output << "not " << resolve_val(op.vals());
         })
         .Case<mlir::db::IfOp>([&](mlir::db::IfOp op) {
            output << " case \n";
            output << " when " << resolve_val(op.condition());
            output << " then " << resolve_val(mlir::dyn_cast_or_null<mlir::db::YieldOp>(op.thenRegion().front().getTerminator()).getOperand(0));
            if (!op.elseRegion().empty()) {
               output << " else " << resolve_val(mlir::dyn_cast_or_null<mlir::db::YieldOp>(op.elseRegion().front().getTerminator()).getOperand(0));
            }
            output << "\n end";
         })
         .Case<mlir::db::DateExtractOp>([&](mlir::db::DateExtractOp op) {
            output << "extract(" << op.unit().str() << " from " << resolve_val(op.date()) << " )";
         })
         .Case<MaterializeOp>([&](MaterializeOp op) {
            std::vector<std::string> attrs;
            for (auto attr : op.attrs()) {
               attrs.push_back(attribute_name(attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>().getRelationalAttribute()));
            }
            output << "select ";
            auto first = true;
            for (auto attr : attrs) {
               if (first) {
                  first = false;
               } else {
                  output << ", ";
               }
               output << attr;
            }
            output << "\n from " << operator_name(op.rel().getDefiningOp());
         })
         .Case<mlir::relalg::ReturnOp, mlir::ReturnOp, AddAttrOp, mlir::db::YieldOp>([&](mlir::Operation* others) {

         })
         .Default([&](mlir::Operation* others) {
            llvm::dbgs() << "could not translate db op:\n";
            others->dump();
            output << "<unknown value (op)>";
         });
      values.insert({op, output.str()});
   }

   public:
   ToSQL(mlir::MLIRContext* context, mlir::ModuleOp moduleOp) : context(context), moduleOp(moduleOp) {}
   std::string run() {
      mlir::FuncOp func = mlir::dyn_cast_or_null<mlir::FuncOp>(&moduleOp.getRegion().front().front());
      std::stringstream total_output;
      total_output << "with ";
      bool addComma = false;
      func->walk([&](mlir::Operation* operation) {
         if (mlir::isa<mlir::FuncOp>(operation))
            return;
         if (auto op = mlir::dyn_cast_or_null<Operator>(operation)) {
            if (!addComma) {
               addComma = true;
            } else {
               total_output << ", ";
            }
            std::stringstream output;

            auto op_name = operator_name(op.getOperation());
            if (auto const_rel_op = mlir::dyn_cast_or_null<mlir::relalg::ConstRelationOp>(op.getOperation())) {
               output << op_name << "(" << attribute_name(const_rel_op.attributes()[0].dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>().getRelationalAttribute()) << ")";
               output << " as ( values ";
               auto first = true;
               for (auto val : const_rel_op.values()) {
                  if (first) {
                     first = false;
                  } else {
                     output << ", ";
                  }
                  output << "(";
                  if (auto strAttr = val.dyn_cast_or_null<mlir::StringAttr>()) {
                     output << "'" << std::string(strAttr.getValue()) << "'";
                  } else if (auto intAttr = val.dyn_cast_or_null<mlir::IntegerAttr>()) {
                     output << intAttr.getInt();
                  }
                  output << ")";
               }
               output << ")";
               total_output << output.str();
            } else {
               ::llvm::TypeSwitch<mlir::Operation*, void>(op.getOperation())
                  .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp op) {
                     output << "select ";
                     auto first = true;
                     for (auto mapping : op.columns()) {
                        auto [column_name, attr] = mapping;
                        auto relation_def_attr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << column_name.str() << " as " << attribute_name(relation_def_attr.getRelationalAttribute());
                     }
                     output << "\nfrom " << std::string(op.table_identifier());
                  })
                  .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp op) {
                     output << "select * \nfrom " << operator_name(op.rel().getDefiningOp()) << "\n";
                     output << "where ";
                     output << resolve_val(mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator()).getOperand(0));
                  })
                  .Case<mlir::relalg::SingleJoinOp>([&](mlir::relalg::SingleJoinOp op) {
                     output << " select * from " << operator_name(op.left().getDefiningOp()) << " left outer join " << operator_name(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolve_val(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::InnerJoinOp>([&](mlir::relalg::InnerJoinOp op) {
                    output << " select * from " << operator_name(op.left().getDefiningOp()) << " inner join " << operator_name(op.right().getDefiningOp()) << " ";
                    auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                    if (returnop->getNumOperands() > 0) {
                       output << " on " << resolve_val(returnop.getOperand(0));
                    } else {
                       output << " on true";
                    }
                  })
                  .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) {
                     output << " select * from " << operator_name(op.left().getDefiningOp()) << " " << std::string(mlir::relalg::stringifyJoinDirection(op.join_direction())) << " outer join " << operator_name(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolve_val(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) {
                    output << " select * from " << operator_name(op.left().getDefiningOp()) << " full outer join " << operator_name(op.right().getDefiningOp()) << " ";
                    auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                    if (returnop->getNumOperands() > 0) {
                       output << " on " << resolve_val(returnop.getOperand(0));
                    } else {
                       output << " on true";
                    }
                  })
                  .Case<mlir::relalg::SemiJoinOp>([&](mlir::relalg::SemiJoinOp op) {
                     output << " select * from " << operator_name(op.left().getDefiningOp()) << " where exists(select * from " << operator_name(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolve_val(returnop.getOperand(0));
                     }
                     output << ")";
                  })
                  .Case<mlir::relalg::MarkJoinOp>([&](mlir::relalg::MarkJoinOp op) {
                     output << " select *, exists(select * from " << operator_name(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolve_val(returnop.getOperand(0));
                     }
                     output << ") as " << attribute_name(op.markattr().getRelationalAttribute()) << " from " << operator_name(op.left().getDefiningOp());
                  })
                  .Case<mlir::relalg::AntiSemiJoinOp>([&](mlir::relalg::AntiSemiJoinOp op) {
                     output << " select * from " << operator_name(op.left().getDefiningOp()) << " where not exists(select * from " << operator_name(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolve_val(returnop.getOperand(0));
                     }
                     output << ")";
                  })
                  .Case<mlir::relalg::CrossProductOp>([&](mlir::relalg::CrossProductOp op) {
                     output << "select * \nfrom " << operator_name(op.left().getDefiningOp()) << ", " << operator_name(op.right().getDefiningOp()) << "\n";
                  })
                  .Case<mlir::relalg::MapOp>([&](mlir::relalg::MapOp op) {
                     std::vector<std::pair<std::string, std::string>> mappings;
                     op->walk([&](mlir::relalg::AddAttrOp addAttrOp) {
                        auto attr_name = attribute_name(addAttrOp.attr().getRelationalAttribute());
                        auto attr_val = resolve_val(addAttrOp.val());
                        mappings.push_back({attr_name, attr_val});
                     });
                     output << "select ";
                     auto first = true;
                     for (auto mapping : mappings) {
                        auto [name, val] = mapping;
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << val << " as " << name;
                     }
                     output << ", * \n from " << operator_name(op.rel().getDefiningOp()) << "\n";
                  })
                  .Case<mlir::relalg::AggregationOp>([&](mlir::relalg::AggregationOp op) {
                     std::vector<std::pair<std::string, std::string>> mappings;
                     std::vector<std::string> group_by_attrs;
                     for (auto attr : op.group_by_attrs()) {
                        group_by_attrs.push_back(attribute_name(attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>().getRelationalAttribute()));
                     }
                     op->walk([&](mlir::relalg::AddAttrOp addAttrOp) {
                        auto attr_name = attribute_name(addAttrOp.attr().getRelationalAttribute());
                        auto attr_val = resolve_val(addAttrOp.val());
                        mappings.push_back({attr_name, attr_val});
                     });
                     output << "select ";
                     auto first = true;
                     for (auto attr : group_by_attrs) {
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << attr;
                     }
                     for (auto mapping : mappings) {
                        auto [name, val] = mapping;
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << val << " as " << name;
                     }
                     output << "\n from " << operator_name(op.rel().getDefiningOp()) << "\n";
                     if (!group_by_attrs.empty()) {
                        output << " group by ";
                        first = true;
                        for (auto attr : group_by_attrs) {
                           if (first) {
                              first = false;
                           } else {
                              output << ", ";
                           }
                           output << attr;
                        }
                     }
                  })
                  .Case<mlir::relalg::ProjectionOp>([&](mlir::relalg::ProjectionOp op) {
                     if (!op.rel().getDefiningOp()) {
                        // do not emit projection if in e.g. aggregation
                        return;
                     }
                     std::vector<std::string> attrs;
                     for (auto attr : op.attrs()) {
                        attrs.push_back(attribute_name(attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>().getRelationalAttribute()));
                     }
                     output << "select ";
                     if (op.set_semantic() == mlir::relalg::SetSemantic::distinct) {
                        output << "distinct ";
                     }
                     auto first = true;
                     for (auto attr : attrs) {
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << attr;
                     }
                     output << "\n from " << operator_name(op.rel().getDefiningOp());
                  })
                  .Case<mlir::relalg::RenamingOp>([&](mlir::relalg::RenamingOp op) {
                     llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8> already_printed;
                     std::vector<std::string> attrs;
                     output << "select ";
                     auto first = true;

                     for (auto attr : op.attributes()) {
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        auto def = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
                        auto ref = def.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0].dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>();
                        already_printed.insert(&def.getRelationalAttribute());
                        output << attribute_name(ref.getRelationalAttribute()) << " as " << attribute_name(def.getRelationalAttribute());
                     }
                     for (auto attr : op.getAvailableAttributes()) {
                        if (!already_printed.contains(attr)) {
                           if (first) {
                              first = false;
                           } else {
                              output << ", ";
                           }
                           output << attribute_name(*attr);
                        }
                     }
                     output << "\n from " << operator_name(op.rel().getDefiningOp());
                  })
                  .Case<mlir::relalg::LimitOp>([&](mlir::relalg::LimitOp op) {
                     std::vector<std::string> attrs;
                     output << operators[op.rel().getDefiningOp()] << " limit " << op.rows();
                  })

                  .Case<mlir::relalg::SortOp>([&](mlir::relalg::SortOp op) {
                     std::vector<std::string> order_by_attrs;
                     for (auto attr : op.sortspecs()) {
                        auto sortspec = attr.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>();
                        auto sortspecifier = std::string(mlir::relalg::stringifySortSpec(sortspec.getSortSpec()));
                        order_by_attrs.push_back(attribute_name(sortspec.getAttr().getRelationalAttribute()) + " " + sortspecifier);
                     }

                     output << "select * \n from " << operator_name(op.rel().getDefiningOp()) << "\n order by ";
                     auto first = true;
                     for (auto attr : order_by_attrs) {
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << attr;
                     }
                  })
                  .Default([&](Operator others) {
                     llvm::dbgs() << "could not translate:\n";
                     others.dump();
                  });
               if (output.str().empty()) {
                  addComma = false;
               } else {
                  operators.insert({operation,output.str()});
                  total_output << op_name << " as (";
                  total_output << output.str();
                  total_output << ") ";
               }
            }
         } else {
            handleOtherOp(operation);
         }
      });
      mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(func.getBody().front().getTerminator());
      total_output << resolve_val(returnOp.getOperand(0));
      return total_output.str();
   }
};

int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   mlir::MLIRContext context;
   context.appendDialectRegistry(registry);
   mlir::OwningModuleRef module;
   llvm::SourceMgr sourceMgr;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(context, module))
      return error;
   llvm::outs()<<ToSQL(&context,module.get()).run();
}