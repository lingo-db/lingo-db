#include "lingodb/runtime/StructRuntime.h"

using namespace lingodb::runtime;

Struct* Struct::create(size_t totalSize) {
   return createRefCounted<Struct>(totalSize, StorageClass::REFCOUNTED);
}

uint8_t* Struct::data() {
   return values.data();
}

void Struct::cleanupUse(Struct* s) {
   if (s && s->storageClass == StorageClass::REFCOUNTED) {
      decRefCount<Struct>(s, [](Struct*) {});
   }
}

void Struct::addUse(Struct* s) {
   if (s && s->storageClass == StorageClass::REFCOUNTED) {
      incRefCount<Struct>(s);
   }
}

Struct* Struct::promoteToGlobal(Struct* s) {
   if (!s) return nullptr;
   if (s->storageClass == StorageClass::GLOBAL) return s;
   auto* ns = new Struct(s->values.size(), StorageClass::GLOBAL);
   std::ranges::copy(s->values, ns->values.begin());
   getCurrentExecutionContext()->registerState({ns, [](void* p) { delete static_cast<Struct*>(p); }});
   if (s->storageClass == StorageClass::REFCOUNTED) {
      cleanupUse(s);
   }
   return ns;
}
