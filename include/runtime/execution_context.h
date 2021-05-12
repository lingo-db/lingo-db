#ifndef RUNTIME_EXECUTION_CONTEXT_H
#define RUNTIME_EXECUTION_CONTEXT_H
#include "database.h"
namespace runtime {
class ExecutionContext {
   public:
   int id;
   std::unique_ptr<Database> db;
};
template <class T, size_t N>
class MemRef {
   protected:
   T* allocatedPointer;
   T* pointer;
   size_t offset;
   size_t dimensions[N];
   size_t strides[N];
   public:
   MemRef(T* pointer): allocatedPointer(pointer),pointer(pointer),offset(0){}
};
template <class T>
class Pointer : public MemRef<T, 0> {
   public:
   Pointer(T* ptr):MemRef<T,0>(ptr){}
   T& operator*() { return *this->pointer; }
   T* operator->() { return this->pointer; }
   T& operator[](size_t pos) { return this->pointer[pos]; }
   T* get(){return this->pointer;}
};
class String : public MemRef<char, 1> {
   public:
   operator std::string() { return std::string(pointer, dimensions[0]); }
   std::string str() { return std::string(pointer, dimensions[0]); }
};
} // end namespace runtime

#endif // RUNTIME_EXECUTION_CONTEXT_H
