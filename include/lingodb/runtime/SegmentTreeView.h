#ifndef LINGODB_RUNTIME_SEGMENTTREEVIEW_H
#define LINGODB_RUNTIME_SEGMENTTREEVIEW_H
#include "GrowingBuffer.h"

#include <cstdint>
#include <type_traits>

#include <stdlib.h>
namespace lingodb::runtime {

class SegmentTreeView {
   using CreateInitialStateFn = std::add_pointer<void(uint8_t* newState, uint8_t* entry)>::type;
   using CombineStatesFn = std::add_pointer<void(uint8_t* newState, uint8_t* left, uint8_t* right)>::type;
   CreateInitialStateFn createInitialStateFn;
   CombineStatesFn combineStatesFn;
   size_t stateTypeSize;
   struct TreeNode {
      size_t leftMin;
      size_t rightMax;
      TreeNode* left = nullptr;
      TreeNode* right = nullptr;
      uint8_t* state;
   };
   TreeNode* storage;
   size_t storageCntr=0;
   size_t numEntries;
   TreeNode* buildRecursively(std::vector<uint8_t*>& entryPointers, size_t from, size_t to);
   void lookupRecursively(TreeNode* t, uint8_t* result, size_t from, size_t to, bool& first);
   TreeNode* root;
   void destroyNode(TreeNode* node);
   TreeNode* allocate();
   uint8_t* allocateState();
   uint8_t* stateStorage;
   size_t stateCntr=0;
   public:
   void lookup(uint8_t* result, size_t from, size_t to);
   static SegmentTreeView* build(runtime::ExecutionContext* executionContext, Buffer buffer, size_t typeSize, void (*createInitialStateFn)(unsigned char*, unsigned char*), void (*combineStatesFn)(unsigned char*, unsigned char*, unsigned char*), size_t stateTypeSize);
   ~SegmentTreeView();
};
} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_SEGMENTTREEVIEW_H
