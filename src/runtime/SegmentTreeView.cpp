#include "lingodb/runtime/SegmentTreeView.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/utility/Tracer.h"
#include <cassert>
#include <cstring>
#include <iostream>
namespace {
utility::Tracer::Event buildEvent("SegmentTree", "build");
} // end namespace
namespace lingodb::runtime {
SegmentTreeView::TreeNode* SegmentTreeView::allocate() {
   assert(storageCntr < numEntries);
   return &storage[storageCntr++];
}
uint8_t* SegmentTreeView::allocateState() {
   assert(stateCntr < numEntries);
   return &stateStorage[stateTypeSize * stateCntr++];
}
//from: inclusive,to: inclusive
SegmentTreeView::TreeNode* SegmentTreeView::buildRecursively(std::vector<uint8_t*>& entryPointers, size_t from, size_t to) {
   if (from == to) {
      TreeNode* leaf = allocate();
      leaf->leftMin = from;
      leaf->rightMax = to;
      leaf->state = allocateState();
      createInitialStateFn(leaf->state, entryPointers[from]);
      uint64_t init = 0;
      memcpy(&init, leaf->state, sizeof(init));
      return leaf;
   } else {
      size_t mid = from + (to - from) / 2;
      TreeNode* left = buildRecursively(entryPointers, from, mid);
      TreeNode* right = buildRecursively(entryPointers, mid + 1, to);
      TreeNode* inner = allocate();
      inner->leftMin = from;
      inner->rightMax = to;
      inner->left = left;
      inner->right = right;
      inner->state = allocateState();
      combineStatesFn(inner->state, left->state, right->state);
      return inner;
   }
}
void SegmentTreeView::destroyNode(lingodb::runtime::SegmentTreeView::TreeNode* node) {
   if (node == nullptr) {
      return;
   }
   destroyNode(node->left);
   destroyNode(node->right);
   delete node;
}
void SegmentTreeView::lookup(uint8_t* result, size_t from, size_t to) {
   if (from > to) {
      throw std::runtime_error("from must be <= to");
   }
   if (!root) {
      throw std::runtime_error("can not perform lookup on empty segment tree");
   }
   bool first = true;
   lookupRecursively(root, result, from, to, first);
}
void SegmentTreeView::lookupRecursively(lingodb::runtime::SegmentTreeView::TreeNode* t, uint8_t* result, size_t from, size_t to, bool& first) {
   if (from <= t->leftMin && to >= t->rightMax) {
      //direct match
      if (first) {
         memcpy(result, t->state, stateTypeSize);
         first = false;
      } else {
         combineStatesFn(result, result, t->state);
      }
   } else if (from <= t->rightMax && to >= t->leftMin) {
      //partial match
      lookupRecursively(t->left, result, from, to, first);
      lookupRecursively(t->right, result, from, to, first);
   }
}
SegmentTreeView* SegmentTreeView::build(lingodb::runtime::ExecutionContext* executionContext, Buffer buffer, size_t typeSize, void (*createInitialStateFn)(unsigned char*, unsigned char*), void (*combineStatesFn)(unsigned char*, unsigned char*, unsigned char*), size_t stateTypeSize) {
   utility::Tracer::Trace trace(buildEvent);
   std::vector<uint8_t*> entryPointers;
   auto numElements = buffer.numElements / typeSize;
   for (size_t i = 0; i < numElements; i++) {
      entryPointers.push_back(&buffer.ptr[i * typeSize]);
   }
   if (stateTypeSize % 8 != 0) {
      stateTypeSize += 8 - stateTypeSize % 8;
   }
   auto* view = new SegmentTreeView;
   executionContext->registerState({view, [](void* ptr) { delete reinterpret_cast<SegmentTreeView*>(ptr); }});
   view->stateTypeSize = stateTypeSize;
   view->numEntries = numElements * 2 - 1;
   view->storage = lingodb::runtime::FixedSizedBuffer<TreeNode>::createZeroed(view->numEntries);
   view->stateStorage = lingodb::runtime::FixedSizedBuffer<uint8_t>::createZeroed(view->numEntries * stateTypeSize);
   view->combineStatesFn = combineStatesFn;
   view->createInitialStateFn = createInitialStateFn;

   if (entryPointers.empty()) {
      view->root = nullptr;
   } else {
      view->root = view->buildRecursively(entryPointers, 0, entryPointers.size() - 1);
   }
   return view;
}
SegmentTreeView::~SegmentTreeView() {
   FixedSizedBuffer<TreeNode>::deallocate(storage, numEntries);
}
} // namespace lingodb::runtime