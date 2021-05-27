#ifndef RUNTIME_HELPERS_H
#define RUNTIME_HELPERS_H
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

#define EXPORT extern "C" __attribute__((visibility("default")))

namespace runtime {
template <class T, size_t N>
class MemRef {
   protected:
   T* allocatedPointer;
   T* pointer;
   size_t offset;
   size_t dimensions[N];
   size_t strides[N];

   public:
   MemRef(T* pointer) : allocatedPointer(pointer), pointer(pointer), offset(0) {}
   MemRef(T* pointer, std::initializer_list<size_t> dimensions) : allocatedPointer(pointer), pointer(pointer), offset(0) {
      size_t curr = 0;
      for (auto dim : dimensions) {
         this->dimensions[curr] = dim;
      }
   }
};
template <class T>
class Pointer : public MemRef<T, 0> {
   public:
   Pointer(T* ptr) : MemRef<T, 0>(ptr) {}
   T& operator*() { return *this->pointer; }
   T* operator->() { return this->pointer; }
   T& operator[](size_t pos) { return this->pointer[pos]; }
   T* get() { return this->pointer; }
   T& ref() { return *this->pointer; }
};
class ByteRange : public MemRef<uint8_t, 1> {
   public:
   ByteRange(uint8_t* ptr, size_t bytes) : MemRef<uint8_t, 1>(ptr, {bytes}) {}
   uint8_t* getPtr() {
      return pointer;
   }
   size_t getSize() {
      return dimensions[0];
   }
};
class String : public MemRef<char, 1> {
   public:
   String(char* ptr, size_t len) : MemRef<char, 1>(ptr, {len}) {}
   operator std::string() { return std::string(pointer, dimensions[0]); }
   std::string str() { return std::string(pointer, dimensions[0]); }
   size_t len() {
      return dimensions[0];
   }
   char* data(){
      return pointer;
   }
};
template<class T1,class T2>
class Pair{
   T1 first;
   T2 second;

   public:
   Pair(T1 first,T2 second){}
};
} // end namespace runtime
#endif // RUNTIME_HELPERS_H
