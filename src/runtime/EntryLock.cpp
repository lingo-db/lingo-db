#include "lingodb/runtime/EntryLock.h"
#include <new>

namespace lingodb::runtime {
void EntryLock::initialize(EntryLock* lock) {
   new (lock) EntryLock();
}

void EntryLock::lock(EntryLock* lock) {
   while (lock->m.test_and_set(std::memory_order_acquire))
#if defined(__cpp_lib_atomic_wait) && __cpp_lib_atomic_wait >= 201907L
      // Since C++20, locks can be acquired only after notification in the unlock,
      // avoiding any unnecessary spinning.
      // Note that even though wait guarantees it returns only after the value has
      // changed, the lock is acquired after the next condition check.
      lock->m.wait(true, std::memory_order_relaxed)
#endif
         ;
}
void EntryLock::unlock(EntryLock* lock) {
   lock->m.clear(std::memory_order_release);
#if defined(__cpp_lib_atomic_wait) && __cpp_lib_atomic_wait >= 201907L
   lock->m.notify_one();
#endif
}

} // namespace lingodb::runtime
