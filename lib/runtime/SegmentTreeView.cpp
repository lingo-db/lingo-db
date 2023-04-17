#include "runtime/SegmentTreeView.h"
#include <cstring>
#include <iostream>
namespace runtime {
//from: inclusive,to: inclusive
SegmentTreeView::TreeNode* SegmentTreeView::buildRecursively(std::vector<uint8_t*>& entryPointers, size_t from, size_t to) {
   if (from == to) {
      TreeNode* leaf = new TreeNode;
      leaf->leftMin = from;
      leaf->rightMax = to;
      leaf->state = (uint8_t*) aligned_alloc(8, stateTypeSize);
      createInitialStateFn(leaf->state, entryPointers[from]);
      uint64_t init = 0;
      memcpy(&init, leaf->state, sizeof(init));
      return leaf;
   } else {
      size_t mid = from + (to - from) / 2;
      TreeNode* left = buildRecursively(entryPointers, from, mid);
      TreeNode* right = buildRecursively(entryPointers, mid + 1, to);
      TreeNode* inner = new TreeNode;
      inner->leftMin = from;
      inner->rightMax = to;
      inner->left = left;
      inner->right = right;
      inner->state = (uint8_t*) aligned_alloc(8, stateTypeSize);
      combineStatesFn(inner->state, left->state, right->state);
      return inner;
   }
}
void SegmentTreeView::destroyNode(runtime::SegmentTreeView::TreeNode* node) {
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
void SegmentTreeView::lookupRecursively(runtime::SegmentTreeView::TreeNode* t, uint8_t* result, size_t from, size_t to, bool& first) {
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
SegmentTreeView* SegmentTreeView::build(runtime::ExecutionContext* executionContext, Buffer buffer, size_t typeSize, void (*createInitialStateFn)(unsigned char*, unsigned char*), void (*combineStatesFn)(unsigned char*, unsigned char*, unsigned char*), size_t stateTypeSize) {
   std::vector<uint8_t*> entryPointers;
   auto numElements = buffer.numElements / typeSize;
   for (size_t i = 0; i < numElements; i++) {
      entryPointers.push_back(&buffer.ptr[i * typeSize]);
   }
   auto* view = new SegmentTreeView;
   executionContext->registerState({view, [](void* ptr) { delete reinterpret_cast<SegmentTreeView*>(ptr); }});
   view->combineStatesFn = combineStatesFn;
   view->createInitialStateFn = createInitialStateFn;
   view->stateTypeSize = stateTypeSize;
   if (entryPointers.empty()) {
      view->root = nullptr;
   } else {
      view->root = view->buildRecursively(entryPointers, 0, entryPointers.size() - 1);
   }
   return view;
}
} // namespace runtime