#ifndef DB_DIALECTS_UTILS_H
#define DB_DIALECTS_UTILS_H
namespace mlir::relalg {
class node_set {
   public:
   sul::dynamic_bitset<> storage;
   class bit_iterator
      : public std::iterator<std::forward_iterator_tag, size_t> {
      typedef bit_iterator iterator;
      sul::dynamic_bitset<>& bitset;
      size_t pos;

      public:
      bit_iterator(sul::dynamic_bitset<>& bitset, size_t pos) : bitset(bitset), pos(pos) {}
      ~bit_iterator() {}

      iterator operator++(int) /* postfix */ { return bit_iterator(bitset, bitset.find_next(pos)); }
      iterator& operator++() /* prefix */ {
         pos = bitset.find_next(pos);
         return *this;
      }
      reference operator*() { return pos; }
      pointer operator->() { return &pos; }
      bool operator==(const iterator& rhs) const { return pos == rhs.pos; }
      bool operator!=(const iterator& rhs) const { return pos != rhs.pos; }
   };

   public:
   node_set() {
   }
   node_set(size_t size) : storage(size) {}
   node_set negate() const {
      node_set res = *this;
      size_t pos = res.find_first();
      size_t flip_len = res.storage.size() - pos - 1;
      if (flip_len) {
         res.storage.flip(pos + 1, flip_len);
      }
      return res;
   }
   static node_set ones(size_t size) {
      node_set res(size);
      res.storage.set();
      return res;
   }
   static node_set fill_until(size_t num_nodes, size_t n) {
      auto res = node_set(num_nodes);
      res.storage.set(0, n + 1, true);
      return res;
   }

   static node_set single(size_t num_nodes, size_t pos) {
      auto res = node_set(num_nodes);
      res.set(pos);
      return res;
   }
   bool is_subset_of(const node_set& S) const {
      return storage.is_subset_of(S.storage);
   }
   bool intersects(const node_set& rhs) const {
      return storage.intersects(rhs.storage);
   }
   void set(size_t pos) {
      storage.set(pos);
   }
   auto begin() {
      return bit_iterator(storage, storage.find_first());
   }
   auto end() {
      return bit_iterator(storage, storage.npos);
   }
   size_t find_first() const {
      return storage.find_first();
   }
   bool operator==(const node_set& rhs) const { return storage == rhs.storage; }
   bool operator!=(const node_set& rhs) const { return storage != rhs.storage; }
   bool operator<(const node_set& rhs) const { return storage < rhs.storage; }

   bool any() const {
      return storage.any();
   }
   node_set& operator|=(const node_set& rhs) {
      storage |= rhs.storage;
      return *this;
   }
   node_set& operator&=(const node_set& rhs) {
      storage &= rhs.storage;
      return *this;
   }
   node_set operator&(
      const node_set& rhs) {
      node_set result = *this;
      result &= rhs;
      return result;
   }
   node_set operator~() const {
      node_set res = flip();
      return res;
   }
   node_set operator|(
      const node_set& rhs) {
      node_set result = *this;
      result |= rhs;
      return result;
   }
   bool valid() {
      return !storage.empty();
   }
   size_t count() {
      return storage.count();
   }
   node_set flip() const {
      node_set res = *this;
      res.storage.flip();
      return res;
   }
   void iterateSubsets(std::function<void(node_set)> fn) const {
      if (!storage.any()) return;
      node_set S = *this;
      auto S1 = S & S.negate();
      while (S1 != S) {
         fn(S1);
         auto S1flipped = S1.flip();
         auto S2 = S & S1flipped;
         S1 = S & S2.negate();
      }
      fn(S);
   }
   size_t hash() const {
      size_t res = 0;
      for (size_t i = 0; i < storage.num_blocks(); i++) {
         res ^= storage.data()[i];
      }
      return res;
   }
};
struct hash_node_set {
   size_t operator()(const node_set& bitset) const {
      return bitset.hash();
   }
};

}
#endif //DB_DIALECTS_UTILS_H
