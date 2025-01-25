#include <unordered_map>
#include <mutex>

// A map with mutex when reading and writing. It is suitable for low concurrency. High concurrency 
// may have performance issue because mutex always lock the whole map. For High concurrency, 
// an alternative with mutex locking only buckets should be considered.
template <typename T, typename P>
class ConcurrentMap
{
private:
    std::unordered_map<T, P> m;
    std::mutex mutex;

public:
    bool contains(const T& key) {
        std::unique_lock<std::mutex> lock(mutex);
        return m.contains(key);
    }
    P& get(const T& key) {
        std::unique_lock<std::mutex> lock(mutex);
        return m.get(key);
    }
    bool insert(const T& key, const P& val) {
        std::unique_lock<std::mutex> lock(mutex);
        const bool ok = m.insert({key, val}).second;
        return ok;
    }
    void erase(const T& key) {
        std::unique_lock<std::mutex> lock(mutex);
        m.erase(key);
    }
    void clear() {
        std::unique_lock<std::mutex> lock(mutex);
        m.clear();
    }
    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex);
        return m.size();
    }
    void forEach(std::function<void(const T& key, const P& val)> f) {
        std::unique_lock<std::mutex> lock(mutex);
        for (auto p : m) {
            f(p.first, p.second);
        }
    }
};