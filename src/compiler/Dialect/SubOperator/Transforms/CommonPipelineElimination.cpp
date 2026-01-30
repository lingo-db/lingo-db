#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "lingodb/runtime/DatasourceRestrictionProperty.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include <queue>

namespace {
using namespace lingodb::compiler::dialect;

class CommonPipelineEliminationPass : public mlir::PassWrapper<CommonPipelineEliminationPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonPipelineEliminationPass)
   virtual llvm::StringRef getArgument() const override { return "subop-common-pipeline-elimination"; }
   struct InsertingPipeline {
      lingodb::runtime::ExternalDatasourceProperty dataSource;
      subop::ScanOp scanOp;
      subop::InsertOp insertOp;
      subop::GenericCreateOp createOp;
      bool inGroup;
   };
   std::vector<std::string> getInsertedColumns(subop::MemberManager& memberManager, InsertingPipeline& pipeline) {
      llvm::DenseMap<subop::Member, std::string> memberToColName;
      for (auto& mapping : pipeline.dataSource.mapping) {
         auto member = memberManager.lookupMember(mapping.memberName);
         memberToColName[member] = mapping.identifier;
      }
      llvm::DenseMap<tuples::Column*, std::string> colToColName;
      for (auto [member, col] : pipeline.scanOp.getMapping().getMapping()) {
         colToColName[&col.getColumn()] = memberToColName[member];
      }

      llvm::DenseMap<subop::Member, std::string> insertMemberToColName;
      for (auto [member, col] : pipeline.insertOp.getMapping().getMapping()) {
         insertMemberToColName[member] = colToColName[&col.getColumn()];
      }
      std::vector<std::string> res;
      for (auto member : mlir::cast<subop::MultiMapType>(pipeline.insertOp.getState().getType()).getMembers().getMembers()) {
         res.push_back(insertMemberToColName[member]);
      }
      return res;
   }
   void runOnOperation() override {
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();
      auto& memberManager = getContext().getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

      std::unordered_map<std::string, std::vector<InsertingPipeline>> pipelines;
      getOperation()->walk([&](subop::ScanOp scanOp) {
         if (auto tableType = mlir::dyn_cast<subop::TableType>(scanOp.getState().getType())) {
            std::vector<mlir::Operation*> users(scanOp->getUsers().begin(), scanOp->getUsers().end());
            if (users.size() != 1) {
               return;
            }
            auto insertOp = mlir::dyn_cast_or_null<subop::InsertOp>(users[0]);
            if (!insertOp) {
               return;
            }
            auto createOp = mlir::dyn_cast_or_null<subop::GenericCreateOp>(insertOp.getState().getDefiningOp());
            if (!createOp) {
               return;
            }
            if (!mlir::isa<subop::MultiMapType>(createOp.getType())) {
               return;
            }
            bool hasOtherInsertions = false;
            for (auto* user : createOp->getUsers()) {
               if (mlir::isa<subop::InsertOp>(user) && user != insertOp.getOperation()) {
                  hasOtherInsertions = true;
                  break;
               }
            }
            if (hasOtherInsertions) {
               return;
            }
            if (auto getExternalOp = mlir::dyn_cast_or_null<subop::GetExternalOp>(scanOp.getState().getDefiningOp())) {
               auto dataSource = lingodb::utility::deserializeFromHexString<lingodb::runtime::ExternalDatasourceProperty>(getExternalOp.getDescr());
               pipelines[dataSource.tableName].push_back({dataSource,
                                                          scanOp,
                                                          insertOp,
                                                          createOp,
                                                          false});
            }
         }
      });
      std::vector<std::vector<InsertingPipeline>> toMerge;
      for (auto& [tableName, pipelineList] : pipelines) {
         if (pipelineList.size() < 2) {
            continue;
         }
         // sort list of pipelines by "insertOp" order
         std::sort(pipelineList.begin(), pipelineList.end(), [](const InsertingPipeline& a, const InsertingPipeline& b) {
            return a.insertOp->isBeforeInBlock(b.insertOp);
         });
         for (size_t i = 0; i < pipelineList.size(); i++) {
            auto& curr = pipelineList[i];
            if (curr.inGroup) {
               continue;
            }
            auto insertedCols = getInsertedColumns(memberManager, curr);
            std::vector<InsertingPipeline> group;
            //find other pipelines that can be merged with curr
            for (size_t j = i + 1; j < pipelineList.size(); j++) {
               auto& other = pipelineList[j];
               if (other.inGroup) {
                  continue;
               }
               if (curr.dataSource.filterDescriptions == other.dataSource.filterDescriptions) {
                  std::unordered_set<std::string> colsCurr;
                  std::unordered_set<std::string> colsOther;
                  for (auto& mapping : curr.dataSource.mapping) {
                     colsCurr.insert(mapping.identifier);
                  }
                  for (auto& mapping : other.dataSource.mapping) {
                     colsOther.insert(mapping.identifier);
                  }
                  if (colsCurr != colsOther) {
                     continue;
                  }
                  if (insertedCols != getInsertedColumns(memberManager, other)) {
                     continue;
                  }

                  group.push_back(other);
                  other.inGroup = true;
               }
            }
            if (!group.empty()) {
               group.insert(group.begin(), curr);
               curr.inGroup = true;
               toMerge.push_back(group);
            }
         }
      }
      //print groups for now (full ops)
      for (auto& group : toMerge) {
         auto& first = group[0];
         auto firstType = mlir::cast<subop::MultiMapType>(first.insertOp.getState().getType());
         mlir::TypeConverter typeConverter;
         typeConverter.addConversion([&](subop::MultiMapType) {
            return firstType;
         });
         typeConverter.addConversion([&](subop::MultiMapEntryRefType t) {
            return subop::MultiMapEntryRefType::get(t.getContext(), firstType);
         });
         typeConverter.addConversion([&](subop::LookupEntryRefType lookupRefType) {
            return subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<subop::LookupAbleState>(typeConverter.convertType(lookupRefType.getState())));
         });

         typeConverter.addConversion([&](subop::ListType listType) {
            return subop::ListType::get(listType.getContext(), mlir::cast<subop::StateEntryReference>(typeConverter.convertType(listType.getT())));
         });
         typeConverter.addConversion([&](subop::OptionalType optionalType) {
            return subop::OptionalType::get(optionalType.getContext(), mlir::cast<subop::StateEntryReference>(typeConverter.convertType(optionalType.getT())));
         });
         subop::SubOpStateUsageTransformer transformer(columnUsageAnalysis, &getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
            return typeConverter.convertType(type);
         });

         for (size_t i = 1; i < group.size(); i++) {
            auto& other = group[i];
            auto otherType = mlir::cast<subop::MultiMapType>(other.insertOp.getState().getType());
            llvm::DenseMap<subop::Member, subop::Member> memberMapping;
            for (auto [firstMember, otherMember] : llvm::zip(firstType.getMembers().getMembers(), otherType.getMembers().getMembers())) {
               memberMapping[otherMember] = firstMember;
            }
            transformer.mapMembers(memberMapping);
            std::vector<std::pair<size_t, mlir::Operation*>> uses;
            for (auto& use : other.createOp->getUses()) {
               if (use.getOwner() != other.insertOp.getOperation()) {
                  uses.push_back({use.getOperandNumber(), use.getOwner()});
               }
            }
            for (auto [opId, user] : uses) {
               transformer.updateUse(user->getOpOperand(opId), firstType);
               user->setOperand(opId, first.insertOp.getState());
            }
            other.insertOp.erase();
            other.scanOp.erase();
            other.createOp.erase();
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createCommonPiplineEliminationPass() { return std::make_unique<CommonPipelineEliminationPass>(); }