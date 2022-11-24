#include "runtime/SegmentTreeView.h"
namespace runtime {
//from: inclusive,to: inclusive
SegmentTreeView::TreeNode* SegmentTreeView::buildRecursively(std::vector<uint8_t*>& entryPointers, size_t from, size_t to) {
   if (from == to) {
      TreeNode* leaf = new TreeNode;
      leaf->leftMin = from;
      leaf->rightMax = to;
      leaf->state = (uint8_t*) aligned_alloc(8, stateTypeSize);
      createInitialStateFn(leaf->state, entryPointers[from]);
      return leaf;
   } else {
      size_t mid = from + (to - from) / 2;
      TreeNode* left = buildRecursively(entryPointers, from, mid);
      TreeNode* right = buildRecursively(entryPointers, from, mid);
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
void SegmentTreeView::lookup(uint8_t* result, size_t from, size_t to) {
   createDefaultStateFn(result);
   lookupRecursively(root, result, from, to);
}
void SegmentTreeView::lookupRecursively(runtime::SegmentTreeView::TreeNode* t, uint8_t* result, size_t from, size_t to) {
   if (from <= t->leftMin && to >= t->rightMax) {
      //direct match
      combineStatesFn(result, result, t->state);
   } else if (from <= t->rightMax && to >= t->leftMin) {
      //partial match
      lookupRecursively(t->left, result, from, to);
      lookupRecursively(t->right, result, from, to);
   }
}
SegmentTreeView* SegmentTreeView::build(Buffer buffer, size_t typeSize, void (*createInitialStateFn)(unsigned char*, unsigned char*), void (*createDefaultStateFn)(unsigned char*), void (*combineStatesFn)(unsigned char*, unsigned char*, unsigned char*), size_t stateTypeSize) {
   std::vector<uint8_t*> entryPointers;
   auto numElements = buffer.numElements / typeSize;
   for (size_t i = 0; i < numElements; i++) {
      entryPointers.push_back(&buffer.ptr[i * typeSize]);
   }
   auto* view = new SegmentTreeView;
   view->combineStatesFn = combineStatesFn;
   view->createInitialStateFn = createInitialStateFn;
   view->createDefaultStateFn = createDefaultStateFn;
   if (entryPointers.empty()) {
      view->root = new TreeNode;
      view->root->leftMin = 0;
      view->root->rightMax = 0;
      view->root->state = (uint8_t*) aligned_alloc(8, stateTypeSize);
      createDefaultStateFn(view->root->state);
   } else {
      view->root = view->buildRecursively(entryPointers, 0, entryPointers.size() - 1);
   }
   return view;
}
} // namespace runtime