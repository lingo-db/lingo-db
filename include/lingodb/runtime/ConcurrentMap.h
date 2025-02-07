#ifndef LINGODB_RUNTIME_CONCURRENTMAP_H
#define LINGODB_RUNTIME_CONCURRENTMAP_H
#include <mutex>
#include <unordered_map>

namespace lingodb::runtime {
// A map with mutex when reading and writing. It is suitable for low concurrency. High concurrency 
// may have performance issue because mutex always lock the whole map. For High concurrency, 
// an alternative with mutex locking only buckets should be considered.
template <typename T, typename P>
class ConcurrentMap
{
    std::unordered_map<T, P> m;
    mutable std::mutex mutex;

public:
    bool contains(const T& key) {
        std::lock_guard<std::mutex> _(mutex);
        return m.contains(key);
    }
    P& get(const T& key) {
        std::lock_guard<std::mutex> _(mutex);
        return m.get(key);
    }
    bool insert(const T& key, const P& val) {
        std::lock_guard<std::mutex> _(mutex);
        const bool ok = m.insert({key, val}).second;
        return ok;
    }
    void erase(const T& key) {
        std::lock_guard<std::mutex> _(mutex);
        m.erase(key);
    }
    void clear() {
        std::lock_guard<std::mutex> _(mutex);
        m.clear();
    }
    size_t size() const {
        std::lock_guard<std::mutex> _(mutex);
        return m.size();
    }
    void forEach(std::function<void(const T& key, const P& val)> f) {
        std::lock_guard<std::mutex> _(mutex);
        for (auto p : m) {
            f(p.first, p.second);
        }
    }
};
} // namespace lingodb::runtime
#endif // LINGODB_RUNTIME_CONCURRENTMAP_H
