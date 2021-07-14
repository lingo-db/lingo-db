#ifndef RUNTIME_HELPERS_H
#define RUNTIME_HELPERS_H
#include "string.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
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
   char* data() {
      return pointer;
   }
};
class Str {
   char* pointer;
   size_t size;

   public:
   Str(char* ptr, size_t len) : pointer(ptr), size(len) {}
   operator std::string() { return std::string(pointer, size); }
   std::string str() { return std::string(pointer, size); }
   size_t len() {
      return size;
   }
   char* data() {
      return pointer;
   }
};
class Bytes {
   uint8_t* pointer;
   size_t size;

   public:
   Bytes(uint8_t* ptr, size_t bytes) : pointer(ptr), size(bytes) {}
   uint8_t* getPtr() {
      return pointer;
   }
   size_t getSize() {
      return size;
   }
};
template <class T1, class T2>
class Pair {
   T1 first;
   T2 second;

   public:
   Pair(T1 first, T2 second) : first(first), second(second) {}
};
template <class T1, class T2, class T3>
class Triple {
   T1 first;
   T2 second;
   T3 third;

   public:
   Triple(T1 first, T2 second, T3 third) : first(first), second(second), third(third) {}
};
template <class T>
class ObjectBuffer {
   class Part {
      public:
      size_t len;
      size_t capacity;
      uint8_t* data;

      Part() : len(0), capacity(0) {
         data = nullptr;
      }
      Part(size_t capacity) : len(0), capacity(capacity) {
         data = new uint8_t[capacity];
         memset(data, 0, capacity);
      }
      inline bool fits(size_t required) {
         return required <= (capacity - len);
      }
      inline T* alloc(size_t size) {
         uint8_t* ptr = reinterpret_cast<uint8_t*>(data + len);
         this->len += size;
         return (T*) ptr;
      }

      size_t getCapacity() const {
         return capacity;
      }
      ~Part(){
         delete[] data;
      }
   };

   size_t objSize;
   std::vector<Part*> parts;

   public:
   class RangeIterator : public std::iterator<std::forward_iterator_tag, T> {
      public:
      // Default constructor
      RangeIterator(std::vector<Part*>& parts) : parts(parts), part(parts.size()), offset(0), objSize(0) {}
      // Constructor
      explicit RangeIterator(std::vector<Part*>& parts, size_t part, size_t offset, size_t objSize) : parts(parts), part(part), offset(offset), objSize(objSize) {
      }
      // Destructor
      ~RangeIterator() = default;

      // Postfix increment
      RangeIterator operator++(int) {
         RangeIterator copy = *this;
         this->operator++();
         return copy;
      }
      // Prefix increment
      RangeIterator& operator++() {
         offset += objSize;
         if (offset + objSize > (parts[part]->len)) {
            part++;
            offset = 0;
         }
         return *this;
      }
      bool valid() {
         return part < parts.size() && offset + objSize <= parts[part]->len;
      }
      // Reference
      T& operator*() { return *operator->(); }
      // Pointer
      T* operator->() { return (T*) &parts[part]->data[offset]; }
      // Equality
      bool operator==(const RangeIterator& other) const { return part == other.part && offset == other.offset; }
      // Inequality
      bool operator!=(const RangeIterator& other) const { return part != other.part || offset != other.offset; }

      public:
      std::vector<Part*>& parts;
      size_t part;
      size_t offset;
      size_t objSize;
   };
   inline T* alloc() {
      auto* currPart = parts[parts.size() - 1];
      if (!currPart->fits(objSize)) {
         size_t newSize = currPart->getCapacity() * 2;
         parts.push_back(new Part(newSize));
         currPart = parts[parts.size() - 1];
      }
      return (T*) currPart->alloc(objSize);
   }
   ObjectBuffer(size_t objSize) : objSize(objSize) {
      parts.push_back(new Part(1024 * objSize));
   }
   RangeIterator end() { return RangeIterator(parts); }
   RangeIterator begin() { return RangeIterator(parts, 0, 0, objSize); }
   RangeIterator* beginPtr() {
      return new RangeIterator(parts, 0, 0, objSize);
   }
   ~ObjectBuffer(){
      for(auto *part:parts){
         delete part;
      }
   }
};
class VarLenBuffer {
   class Part {
      size_t len;
      size_t capacity;
      uint8_t* data;

      public:
      Part() : len(0), capacity(0) {
         data = nullptr;
      }
      Part(size_t capacity) : len(0), capacity(capacity) {
         data = new uint8_t[capacity];
         memset(data, 0, capacity);
      }
      inline bool fits(size_t required) {
         return required <= (capacity - len);
      }
      inline ByteRange insert(ByteRange toInsert) {
         uint8_t* ptr = reinterpret_cast<uint8_t*>(data + len);
         memcpy(data + len, toInsert.getPtr(), toInsert.getSize());
         len += toInsert.getSize();
         return ByteRange(ptr, toInsert.getSize());
      }
      inline Bytes insert(Bytes toInsert) {
         uint8_t* ptr = reinterpret_cast<uint8_t*>(data + len);
         memcpy(data + len, toInsert.getPtr(), toInsert.getSize());
         len += toInsert.getSize();
         return Bytes(ptr, toInsert.getSize());
      }
      size_t getCapacity() const {
         return capacity;
      }
      ~Part(){
         delete[] data;
      }
   };
   std::vector<Part*> parts;

   public:
   inline ByteRange persist(ByteRange string) {
      auto* currPart = parts[parts.size() - 1];
      if (!currPart->fits(string.getSize())) {
         size_t newSize = std::max(currPart->getCapacity(), string.getSize()) * 2;
         parts.push_back(new Part(newSize));
         currPart = parts[parts.size() - 1];
      }
      return currPart->insert(string);
   }
   inline Bytes persist(Bytes string) {
      auto* currPart = parts[parts.size() - 1];
      if (!currPart->fits(string.getSize())) {
         size_t newSize = std::max(currPart->getCapacity(), string.getSize()) * 2;
         parts.push_back(new Part(newSize));
         currPart = parts[parts.size() - 1];
      }
      return currPart->insert(string);
   }

   VarLenBuffer() {
      parts.push_back(new Part(100000));
   }
   ~VarLenBuffer(){
      for(auto *part:parts){
         delete part;
      }
   }
};
struct ResizableBuffer {
   uint8_t* ptr;
   size_t len;
   ResizableBuffer(size_t len) {
      //std::cout<<"allocating:"<<len<<std::endl;
      ptr = new uint8_t[len];
      this->len = len;
   }
   ResizableBuffer() {
      ptr = nullptr;
      len = 0;
   }
   void resize(size_t newLen) {
      if (newLen > len) {
         //std::cout<<"allocating:"<<newLen<<std::endl;
         uint8_t* newPtr = new uint8_t[newLen];
         if (len > 0) {
            memcpy(newPtr, ptr, len);
            delete[] ptr;
         }
         ptr = newPtr;
         len = newLen;
      }
   }
   void zero(){
      if(len>0){
         memset(ptr,0,len);
      }
   }
   ~ResizableBuffer() {
      if (len > 0) {
         delete[] ptr;
      }
   }
};
template <class T>
struct Vec {
   ResizableBuffer buffer;
   Vec() : buffer() {
   }
   Vec(size_t initialSize) : buffer(initialSize*sizeof(T)) {
   }
   void resize(size_t newSize) {
      buffer.resize(newSize*sizeof(T));
   }

   T& operator[](size_t n) {
      return ((T*)buffer.ptr)[n];
   }
   size_t size() {
      return buffer.len/sizeof(T);
   }
   void zero() {
      buffer.zero();
   }
};
struct Vector {
   std::vector<uint8_t> values;
   runtime::VarLenBuffer varLenBuffer;
};
} // end namespace runtime
#endif // RUNTIME_HELPERS_H
