#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>
#include <list>
#include <queue>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
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

   std::string operatorName(mlir::Operation* op) {
      return "op_" + std::to_string((size_t) op);
   }
   std::string attributeName(mlir::relalg::RelationalAttribute& attr) {
      auto& attributeManager = context->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      auto [scope, name] = attributeManager.getName(&attr);
      return "attr_" + scope + "___" + name;
   }
   std::string resolveVal(mlir::Value v) {
      mlir::Operation* definingOp = v.getDefiningOp();
      if (!definingOp || !values.count(definingOp))
         return "<unknown value>";
      return values[definingOp];
   }
   void handleBinOp(std::stringstream& output, std::string op, mlir::Value a, mlir::Value b) {
      output << "(" << resolveVal(a) << " " << op << " " << resolveVal(b) << ")";
   }
   void joinstr(std::stringstream& output, std::string op, mlir::ValueRange vr) {
      auto first = true;
      for (auto v : vr) {
         if (first) {
            first = false;
         } else {
            output << " " << op << " ";
         }
         output << resolveVal(v);
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
            output << attributeName(op.attr().getRelationalAttribute());
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
            handleBinOp(output, "+", op.left(), op.right());
         })
         .Case<SubOp>([&](SubOp op) {
            handleBinOp(output, "-", op.left(), op.right());
         })
         .Case<MulOp>([&](MulOp op) {
            handleBinOp(output, "*", op.left(), op.right());
         })
         .Case<DivOp>([&](DivOp op) {
            handleBinOp(output, "/", op.left(), op.right());
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
            handleBinOp(output, pred, op.left(), op.right());
         })
         .Case<AggrFuncOp>([&](AggrFuncOp op) {
            std::string fnname(mlir::relalg::stringifyAggrFunc(op.fn()));
            auto attr = attributeName(op.attr().getRelationalAttribute());
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
            output << resolveVal(op.val());
         })
         .Case<NotOp>([&](NotOp op) {
            output << "not " << resolveVal(op.val());
         })
         .Case<mlir::db::IfOp>([&](mlir::db::IfOp op) {
            output << " case \n";
            output << " when " << resolveVal(op.condition());
            output << " then " << resolveVal(mlir::dyn_cast_or_null<mlir::db::YieldOp>(op.thenRegion().front().getTerminator()).getOperand(0));
            if (!op.elseRegion().empty()) {
               output << " else " << resolveVal(mlir::dyn_cast_or_null<mlir::db::YieldOp>(op.elseRegion().front().getTerminator()).getOperand(0));
            }
            output << "\n end";
         })
         .Case<mlir::db::DateExtractOp>([&](mlir::db::DateExtractOp op) {
            output << "extract(" << mlir::db::stringifyExtractableTimeUnitAttr(op.unit()).str() << " from " << resolveVal(op.val()) << " )";
         })
         .Case<MaterializeOp>([&](MaterializeOp op) {
            std::vector<std::string> attrs;
            for (auto attr : op.attrs()) {
               attrs.push_back(attributeName(attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>().getRelationalAttribute()));
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
            output << "\n from " << operatorName(op.rel().getDefiningOp());
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
      std::stringstream totalOutput;
      totalOutput << "with ";
      bool addComma = false;
      func->walk([&](mlir::Operation* operation) {
         if (mlir::isa<mlir::FuncOp>(operation))
            return;
         if (auto op = mlir::dyn_cast_or_null<Operator>(operation)) {
            if (!addComma) {
               addComma = true;
            } else {
               totalOutput << ", ";
            }
            std::stringstream output;

            auto opName = operatorName(op.getOperation());
            if (auto constRelOp = mlir::dyn_cast_or_null<mlir::relalg::ConstRelationOp>(op.getOperation())) {
               output << opName << "(" << attributeName(constRelOp.attributes()[0].dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>().getRelationalAttribute()) << ")";
               output << " as ( values ";
               auto first = true;
               for (auto val : constRelOp.values()) {
                  if (first) {
                     first = false;
                  } else {
                     output << ", ";
                  }
                  auto first2 = true;
                  output << "(";
                  auto row=val.cast<mlir::ArrayAttr>();
                  for(auto entryAttr:row.getValue()) {
                     if (first2) {
                        first2 = false;
                     } else {
                        output << ", ";
                     }
                     if (auto strAttr = entryAttr.dyn_cast_or_null<mlir::StringAttr>()) {
                        output << "'" << std::string(strAttr.getValue()) << "'";
                     } else if (auto intAttr = entryAttr.dyn_cast_or_null<mlir::IntegerAttr>()) {
                        output << intAttr.getInt();
                     }
                  }
                  output << ")";
               }
               output << ")";
               totalOutput << output.str();
            } else {
               ::llvm::TypeSwitch<mlir::Operation*, void>(op.getOperation())
                  .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp op) {
                     output << "select ";
                     auto first = true;
                     for (auto mapping : op.columns()) {
                        auto [column_name, attr] = mapping;
                        auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << column_name.str() << " as " << attributeName(relationDefAttr.getRelationalAttribute());
                     }
                     output << "\nfrom " << std::string(op.table_identifier());
                  })
                  .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp op) {
                     output << "select * \nfrom " << operatorName(op.rel().getDefiningOp()) << "\n";
                     output << "where ";
                     output << resolveVal(mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator()).getOperand(0));
                  })
                  .Case<mlir::relalg::SingleJoinOp>([&](mlir::relalg::SingleJoinOp op) {
                     output << " select * from " << operatorName(op.left().getDefiningOp()) << " left outer join " << operatorName(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::InnerJoinOp>([&](mlir::relalg::InnerJoinOp op) {
                     output << " select * from " << operatorName(op.left().getDefiningOp()) << " inner join " << operatorName(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) {
                     output << " select * from " << operatorName(op.left().getDefiningOp()) << " " << std::string(mlir::relalg::stringifyJoinDirection(op.join_direction())) << " outer join " << operatorName(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) {
                     output << " select * from " << operatorName(op.left().getDefiningOp()) << " full outer join " << operatorName(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::SemiJoinOp>([&](mlir::relalg::SemiJoinOp op) {
                     output << " select * from " << operatorName(op.left().getDefiningOp()) << " where exists(select * from " << operatorName(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolveVal(returnop.getOperand(0));
                     }
                     output << ")";
                  })
                  .Case<mlir::relalg::MarkJoinOp>([&](mlir::relalg::MarkJoinOp op) {
                     output << " select *, exists(select * from " << operatorName(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolveVal(returnop.getOperand(0));
                     }
                     output << ") as " << attributeName(op.markattr().getRelationalAttribute()) << " from " << operatorName(op.left().getDefiningOp());
                  })
                  .Case<mlir::relalg::AntiSemiJoinOp>([&](mlir::relalg::AntiSemiJoinOp op) {
                     output << " select * from " << operatorName(op.left().getDefiningOp()) << " where not exists(select * from " << operatorName(op.right().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.predicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolveVal(returnop.getOperand(0));
                     }
                     output << ")";
                  })
                  .Case<mlir::relalg::CrossProductOp>([&](mlir::relalg::CrossProductOp op) {
                     output << "select * \nfrom " << operatorName(op.left().getDefiningOp()) << ", " << operatorName(op.right().getDefiningOp()) << "\n";
                  })
                  .Case<mlir::relalg::MapOp>([&](mlir::relalg::MapOp op) {
                     std::vector<std::pair<std::string, std::string>> mappings;
                     op->walk([&](mlir::relalg::AddAttrOp addAttrOp) {
                        auto attrName = attributeName(addAttrOp.attr().getRelationalAttribute());
                        auto attrVal = resolveVal(addAttrOp.val());
                        mappings.push_back({attrName, attrVal});
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
                     output << ", * \n from " << operatorName(op.rel().getDefiningOp()) << "\n";
                  })
                  .Case<mlir::relalg::AggregationOp>([&](mlir::relalg::AggregationOp op) {
                     std::vector<std::pair<std::string, std::string>> mappings;
                     std::vector<std::string> groupByAttrs;
                     for (auto attr : op.group_by_attrs()) {
                        groupByAttrs.push_back(attributeName(attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>().getRelationalAttribute()));
                     }
                     op->walk([&](mlir::relalg::AddAttrOp addAttrOp) {
                        auto attrName = attributeName(addAttrOp.attr().getRelationalAttribute());
                        auto attrVal = resolveVal(addAttrOp.val());
                        mappings.push_back({attrName, attrVal});
                     });
                     output << "select ";
                     auto first = true;
                     for (auto attr : groupByAttrs) {
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
                     output << "\n from " << operatorName(op.rel().getDefiningOp()) << "\n";
                     if (!groupByAttrs.empty()) {
                        output << " group by ";
                        first = true;
                        for (auto attr : groupByAttrs) {
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
                        attrs.push_back(attributeName(attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>().getRelationalAttribute()));
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
                     output << "\n from " << operatorName(op.rel().getDefiningOp());
                  })
                  .Case<mlir::relalg::RenamingOp>([&](mlir::relalg::RenamingOp op) {
                     llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8> alreadyPrinted;
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
                        alreadyPrinted.insert(&def.getRelationalAttribute());
                        output << attributeName(ref.getRelationalAttribute()) << " as " << attributeName(def.getRelationalAttribute());
                     }
                     for (auto* attr : op.getAvailableAttributes()) {
                        if (!alreadyPrinted.contains(attr)) {
                           if (first) {
                              first = false;
                           } else {
                              output << ", ";
                           }
                           output << attributeName(*attr);
                        }
                     }
                     output << "\n from " << operatorName(op.rel().getDefiningOp());
                  })
                  .Case<mlir::relalg::LimitOp>([&](mlir::relalg::LimitOp op) {
                     std::vector<std::string> attrs;
                     output << operators[op.rel().getDefiningOp()] << " limit " << op.rows();
                  })

                  .Case<mlir::relalg::SortOp>([&](mlir::relalg::SortOp op) {
                     std::vector<std::string> orderByAttrs;
                     for (auto attr : op.sortspecs()) {
                        auto sortspec = attr.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>();
                        auto sortspecifier = std::string(mlir::relalg::stringifySortSpec(sortspec.getSortSpec()));
                        orderByAttrs.push_back(attributeName(sortspec.getAttr().getRelationalAttribute()) + " " + sortspecifier);
                     }

                     output << "select * \n from " << operatorName(op.rel().getDefiningOp()) << "\n order by ";
                     auto first = true;
                     for (auto attr : orderByAttrs) {
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
                  operators.insert({operation, output.str()});
                  totalOutput << opName << " as (";
                  totalOutput << output.str();
                  totalOutput << ") ";
               }
            }
         } else {
            handleOtherOp(operation);
         }
      });
      mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(func.getBody().front().getTerminator());
      totalOutput << resolveVal(returnOp.getOperand(0));
      return totalOutput.str();
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
   llvm::outs() << ToSQL(&context, module.get()).run();
}