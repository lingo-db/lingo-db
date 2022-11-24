#ifndef RUNTIME_SEGMENTTREEVIEW_H
#define RUNTIME_SEGMENTTREEVIEW_H
#include "GrowingBuffer.h"

#include <cstdint>
#include <type_traits>

#include <stdlib.h>
namespace runtime {

class SegmentTreeView {
   using CreateInitialStateFn = std::add_pointer<void(uint8_t* newState, uint8_t* entry)>::type;
   using CreateDefaultStateFn = std::add_pointer<void(uint8_t* newState)>::type;
   using CombineStatesFn = std::add_pointer<void(uint8_t* newState, uint8_t* left, uint8_t* right)>::type;
   CreateInitialStateFn createInitialStateFn;
   CreateDefaultStateFn createDefaultStateFn;
   CombineStatesFn combineStatesFn;
   size_t stateTypeSize;
   struct TreeNode {
      size_t leftMin;
      size_t rightMax;
      TreeNode* left = nullptr;
      TreeNode* right = nullptr;
      uint8_t* state;
   };
   TreeNode* buildRecursively(std::vector<uint8_t*>& entryPointers, size_t from, size_t to);
   void lookup(uint8_t* result, size_t from, size_t to);
   void lookupRecursively(TreeNode* t, uint8_t* result, size_t from, size_t to);
   TreeNode* root;
   static SegmentTreeView* build(Buffer buffer, size_t typeSize, CreateInitialStateFn createInitialStateFn, CreateDefaultStateFn createDefaultStateFn, CombineStatesFn combineStatesFn, size_t stateTypeSize);
};
} // namespace runtime

#endif // RUNTIME_SEGMENTTREEVIEW_H
