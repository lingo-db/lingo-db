#ifndef LINGODB_RUNTIME_ENTRYLOCK_H
#define LINGODB_RUNTIME_ENTRYLOCK_H
#include <atomic>

namespace lingodb::runtime {
class EntryLock {
  std::atomic_flag m{};

  public:
  static void lock(EntryLock* lock);
  static void unlock(EntryLock* lock);
  static void initialize(EntryLock* lock);
};
static_assert(sizeof(EntryLock) <= 8, "SpinLock is too big");

} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_ENTRYLOCK_H
