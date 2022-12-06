#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <iostream>
#include <list>
#include <queue>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return -1;
   }

   // Parse the input mlir.
   llvm::SourceMgr sourceMgr;
   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
   module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
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
   std::string attributeName(const mlir::tuples::Column& attr) {
      auto& columnManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto [scope, name] = columnManager.getName(&attr);
      return "attr_" + scope + "___" + name;
   }
   std::string resolveVal(mlir::Value v) {
      mlir::Operation* definingOp = v.getDefiningOp();
      if (!definingOp || !values.count(definingOp))
         return "<unknown value>";
      return values[definingOp];
   }
   void handleBinOp(std::stringstream& output, std::string op, mlir::Value a, mlir::Value b) {
      output << "( (" << resolveVal(a) << ") " << op << " (" << resolveVal(b) << ") )";
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
               if (t.isa<mlir::IntegerType>() || t.isa<mlir::db::DecimalType>() || t.isa<mlir::FloatType>()) {
                  output << std::string(strAttr.getValue());

               } else if (auto intervalType = t.dyn_cast_or_null<mlir::db::IntervalType>()) {
                  if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
                     output << "'" << std::string(strAttr.getValue()) << "' month";
                  } else {
                     int64_t valAsInt = std::stoll(std::string(strAttr.getValue()));
                     int64_t valInDays = valAsInt;
                     output << "'" << std::to_string(valInDays) << "' day";
                  }

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
         .Case<tuples::GetColumnOp>([&](tuples::GetColumnOp op) {
            output << attributeName(op.getAttr().getColumn());
         })
         .Case<mlir::db::AndOp>([&](mlir::db::AndOp op) {
            output << "(";
            joinstr(output, "and", op.getVals());
            output << ")";
         })
         .Case<mlir::db::OrOp>([&](mlir::db::OrOp op) {
            output << "(";
            joinstr(output, "or", op.getVals());
            output << ")";
         })
         .Case<mlir::db::BetweenOp>([&](mlir::db::BetweenOp op) {
            output << resolveVal(op.getVal()) << " between " << resolveVal(op.getLower()) << " and " << resolveVal(op.getUpper()) << " ";
         })
         .Case<mlir::db::OneOfOp>([&](mlir::db::OneOfOp op) {
            output << resolveVal(op.getVal()) << " in (";
            bool first = true;
            for (auto v : op.getVals()) {
               if (first) {
                  first = false;
               } else {
                  output << ", ";
               }
               output << resolveVal(v);
            }
            output << ") ";
         })
         .Case<mlir::db::DeriveTruth>([&](mlir::db::DeriveTruth op) {
            output << resolveVal(op.getVal());
         })

         .Case<AddOp>([&](AddOp op) {
            handleBinOp(output, "+", op.getLeft(), op.getRight());
         })
         .Case<SubOp>([&](SubOp op) {
            handleBinOp(output, "-", op.getLeft(), op.getRight());
         })
         .Case<MulOp>([&](MulOp op) {
            handleBinOp(output, "*", op.getLeft(), op.getRight());
         })
         .Case<DivOp>([&](DivOp op) {
            handleBinOp(output, "/", op.getLeft(), op.getRight());
         })
         .Case<CmpOp>([&](CmpOp op) {
            std::string pred = "<unknown pred>";
            switch (op.getPredicate()) {
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
               case DBCmpPredicate::isa:
                  handleBinOp(output, "=", op.getLeft(), op.getRight());
                  auto left=resolveVal(op.getLeft());
                  auto right=resolveVal(op.getRight());
                  output << "( (" << left << ") " << "=" << " (" << right << ") ) or ( ("<< left <<") is null and ("<< right << ") is null )";
                  return;
            }
            handleBinOp(output, pred, op.getLeft(), op.getRight());
         })
         .Case<AggrFuncOp>([&](AggrFuncOp op) {
            std::string getFnname(mlir::relalg::stringifyAggrFunc(op.getFn()));
            auto attr = attributeName(op.getAttr().getColumn());
            std::string distinct = "";
            if (auto projop = mlir::dyn_cast_or_null<ProjectionOp>(op.getRel().getDefiningOp())) {
               if (projop.getSetSemantic() == SetSemantic::distinct) {
                  distinct = "distinct ";
               }
            }
            output << getFnname << "(" << distinct << attr << ")";
         })
         .Case<CountRowsOp>([&](CountRowsOp op) {
            output << "count(*)";
         })
         .Case<CastOp>([&](CastOp op) {
            output << resolveVal(op.getVal());
         })
         .Case<NotOp>([&](NotOp op) {
            output << "not " << resolveVal(op.getVal());
         })
         .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp op) {
            output << " case \n";
            output << " when " << resolveVal(op.getCondition());
            output << " then " << resolveVal(mlir::dyn_cast_or_null<mlir::scf::YieldOp>(op.getThenRegion().front().getTerminator()).getOperand(0));
            if (!op.getElseRegion().empty()) {
               output << " else " << resolveVal(mlir::dyn_cast_or_null<mlir::scf::YieldOp>(op.getElseRegion().front().getTerminator()).getOperand(0));
            }
            output << "\n end";
         })
         .Case<mlir::db::RuntimeCall>([&](mlir::db::RuntimeCall op) {
            if (op.getFn() == "ExtractFromDate") {
               output << "extract(" << resolveVal(op.getArgs()[0]) << " from " << resolveVal(op.getArgs()[1]) << " )";
            }
            if (op.getFn().startswith("DateAdd")) {
               handleBinOp(output, "+", op.getArgs()[0], op.getArgs()[1]);
            }
            if (op.getFn().startswith("DateSubtract")) {
               handleBinOp(output, "-", op.getArgs()[0], op.getArgs()[1]);
            }
            if (op.getFn().endswith("Like")) {
               handleBinOp(output, "like", op.getArgs()[0], op.getArgs()[1]);
            }
         })
         .Case<MaterializeOp>([&](MaterializeOp op) {
            std::vector<std::string> attrs;
            for (auto attr : op.getCols()) {
               attrs.push_back(attributeName(attr.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>().getColumn()));
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
            output << "\n from " << operatorName(op.getRel().getDefiningOp());
         })
         .Case<mlir::tuples::ReturnOp, mlir::func::ReturnOp, mlir::scf::YieldOp>([&](mlir::Operation* others) {

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
      mlir::func::FuncOp func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(&moduleOp.getRegion().front().front());
      std::stringstream totalOutput;
      totalOutput << "with ";
      bool addComma = false;
      func->walk([&](mlir::Operation* operation) {
         if (mlir::isa<mlir::func::FuncOp>(operation))
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
               output << opName << "(" << attributeName(constRelOp.getColumns()[0].dyn_cast_or_null<mlir::tuples::ColumnDefAttr>().getColumn()) << ")";
               output << " as ( values ";
               auto first = true;
               for (auto val : constRelOp.getValues()) {
                  if (first) {
                     first = false;
                  } else {
                     output << ", ";
                  }
                  auto first2 = true;
                  output << "(";
                  auto row = val.cast<mlir::ArrayAttr>();
                  for (auto entryAttr : row.getValue()) {
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
                     for (auto mapping : op.getColumns()) {
                        auto columnName = mapping.getName();
                        auto attr = mapping.getValue();
                        auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        output << columnName.str() << " as " << attributeName(relationDefAttr.getColumn());
                     }
                     output << "\nfrom " << std::string(op.getTableIdentifier());
                  })
                  .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp op) {
                     output << "select * \nfrom " << operatorName(op.getRel().getDefiningOp()) << "\n";
                     output << "where ";
                     output << resolveVal(mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator()).getOperand(0));
                  })
                  .Case<mlir::relalg::SingleJoinOp>([&](mlir::relalg::SingleJoinOp op) {
                     output << " select ";
                     llvm::SmallPtrSet<mlir::tuples::Column*, 8> alreadyPrinted;
                     std::vector<std::string> attrs;
                     auto first = true;

                     for (auto attr : op.getMapping()) {
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        auto def = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
                        auto ref = def.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0].dyn_cast_or_null<mlir::tuples::ColumnRefAttr>();
                        alreadyPrinted.insert(&def.getColumn());
                        output << attributeName(ref.getColumn()) << " as " << attributeName(def.getColumn());
                     }
                     for (const auto* attr : op.getAvailableColumns()) {
                        if (!alreadyPrinted.contains(attr)) {
                           if (first) {
                              first = false;
                           } else {
                              output << ", ";
                           }
                           output << attributeName(*attr);
                        }
                     }
                     output << " from " << operatorName(op.getLeft().getDefiningOp()) << " left outer join " << operatorName(op.getRight().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::InnerJoinOp>([&](mlir::relalg::InnerJoinOp op) {
                     output << " select * from " << operatorName(op.getLeft().getDefiningOp()) << " inner join " << operatorName(op.getRight().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) {
                     output << " select ";
                     llvm::SmallPtrSet<mlir::tuples::Column*, 8> alreadyPrinted;
                     std::vector<std::string> attrs;
                     auto first = true;

                     for (auto attr : op.getMapping()) {
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        auto def = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
                        auto ref = def.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0].dyn_cast_or_null<mlir::tuples::ColumnRefAttr>();
                        alreadyPrinted.insert(&def.getColumn());
                        output << attributeName(ref.getColumn()) << " as " << attributeName(def.getColumn());
                     }
                     for (const auto* attr : op.getAvailableColumns()) {
                        if (!alreadyPrinted.contains(attr)) {
                           if (first) {
                              first = false;
                           } else {
                              output << ", ";
                           }
                           output << attributeName(*attr);
                        }
                     }
                     output << " from " << operatorName(op.getLeft().getDefiningOp()) << " left outer join " << operatorName(op.getRight().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) {
                     output << " select * from " << operatorName(op.getLeft().getDefiningOp()) << " full outer join " << operatorName(op.getRight().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " on " << resolveVal(returnop.getOperand(0));
                     } else {
                        output << " on true";
                     }
                  })
                  .Case<mlir::relalg::SemiJoinOp>([&](mlir::relalg::SemiJoinOp op) {
                     output << " select * from " << operatorName(op.getLeft().getDefiningOp()) << " where exists(select * from " << operatorName(op.getRight().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolveVal(returnop.getOperand(0));
                     }
                     output << ")";
                  })
                  .Case<mlir::relalg::MarkJoinOp>([&](mlir::relalg::MarkJoinOp op) {
                     output << " select *, exists(select * from " << operatorName(op.getRight().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolveVal(returnop.getOperand(0));
                     }
                     output << ") as " << attributeName(op.getMarkattr().getColumn()) << " from " << operatorName(op.getLeft().getDefiningOp());
                  })
                  .Case<mlir::relalg::AntiSemiJoinOp>([&](mlir::relalg::AntiSemiJoinOp op) {
                     output << " select * from " << operatorName(op.getLeft().getDefiningOp()) << " where not exists(select * from " << operatorName(op.getRight().getDefiningOp()) << " ";
                     auto returnop = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicate().front().getTerminator());
                     if (returnop->getNumOperands() > 0) {
                        output << " where " << resolveVal(returnop.getOperand(0));
                     }
                     output << ")";
                  })
                  .Case<mlir::relalg::CrossProductOp>([&](mlir::relalg::CrossProductOp op) {
                     output << "select * \nfrom " << operatorName(op.getLeft().getDefiningOp()) << ", " << operatorName(op.getRight().getDefiningOp()) << "\n";
                  })
                  .Case<mlir::relalg::MapOp>([&](mlir::relalg::MapOp op) {
                     std::vector<std::pair<std::string, std::string>> mappings;
                     auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(op.getLambdaBlock().getTerminator());
                     size_t i = 0;
                     for (auto col : op.getComputedCols()) {
                        auto attrName = attributeName(col.cast<mlir::tuples::ColumnDefAttr>().getColumn());
                        auto attrVal = resolveVal(returnOp.getResults()[i++]);
                        mappings.push_back({attrName, attrVal});
                     }
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
                     output << ", * \n from " << operatorName(op.getRel().getDefiningOp()) << "\n";
                  })
                  .Case<mlir::relalg::AggregationOp>([&](mlir::relalg::AggregationOp op) {
                     std::vector<std::pair<std::string, std::string>> mappings;
                     std::vector<std::string> groupByAttrs;
                     for (auto attr : op.getGroupByCols()) {
                        groupByAttrs.push_back(attributeName(attr.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>().getColumn()));
                     }
                     auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(op.getAggrFunc().front().getTerminator());
                     size_t i = 0;
                     for (auto col : op.getComputedCols()) {
                        auto attrName = attributeName(col.cast<mlir::tuples::ColumnDefAttr>().getColumn());
                        auto attrVal = resolveVal(returnOp.getResults()[i++]);
                        mappings.push_back({attrName, attrVal});
                     }
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
                     output << "\n from " << operatorName(op.getRel().getDefiningOp()) << "\n";
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
                     if (!op.getRel().getDefiningOp()) {
                        // do not emit projection if in e.g. aggregation
                        return;
                     }
                     std::vector<std::string> attrs;
                     for (auto attr : op.getCols()) {
                        attrs.push_back(attributeName(attr.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>().getColumn()));
                     }
                     output << "select ";
                     if (op.getSetSemantic() == mlir::relalg::SetSemantic::distinct) {
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
                     output << "\n from " << operatorName(op.getRel().getDefiningOp());
                  })
                  .Case<mlir::relalg::RenamingOp>([&](mlir::relalg::RenamingOp op) {
                     llvm::SmallPtrSet<mlir::tuples::Column*, 8> alreadyPrinted;
                     std::vector<std::string> attrs;
                     output << "select ";
                     auto first = true;

                     for (auto attr : op.getColumns()) {
                        if (first) {
                           first = false;
                        } else {
                           output << ", ";
                        }
                        auto def = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
                        auto ref = def.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0].dyn_cast_or_null<mlir::tuples::ColumnRefAttr>();
                        alreadyPrinted.insert(&def.getColumn());
                        output << attributeName(ref.getColumn()) << " as " << attributeName(def.getColumn());
                     }
                     for (const auto* attr : op.getAvailableColumns()) {
                        if (!alreadyPrinted.contains(attr)) {
                           if (first) {
                              first = false;
                           } else {
                              output << ", ";
                           }
                           output << attributeName(*attr);
                        }
                     }
                     output << "\n from " << operatorName(op.getRel().getDefiningOp());
                  })
                  .Case<mlir::relalg::LimitOp>([&](mlir::relalg::LimitOp op) {
                     std::vector<std::string> attrs;
                     output << operators[op.getRel().getDefiningOp()] << " limit " << op.getMaxRows();
                  })

                  .Case<mlir::relalg::SortOp>([&](mlir::relalg::SortOp op) {
                     std::vector<std::string> orderByAttrs;
                     for (auto attr : op.getSortspecs()) {
                        auto sortspec = attr.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>();
                        auto sortspecifier = std::string(mlir::relalg::stringifySortSpec(sortspec.getSortSpec()));
                        orderByAttrs.push_back(attributeName(sortspec.getAttr().getColumn()) + " " + sortspecifier);
                     }

                     output << "select * \n from " << operatorName(op.getRel().getDefiningOp()) << "\n order by ";
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
      mlir::func::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(func.getBody().front().getTerminator());
      totalOutput << resolveVal(returnOp.getOperand(0));
      return totalOutput.str();
   }
};

int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::tuples::TupleStreamDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();

   mlir::MLIRContext context;
   context.appendDialectRegistry(registry);
   mlir::OwningOpRef<mlir::ModuleOp> module;
   llvm::SourceMgr sourceMgr;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(context, module))
      return error;
   llvm::outs() << ToSQL(&context, module.get()).run();
}