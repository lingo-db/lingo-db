#include "lingodb/runtime/PerfectHashTable.h"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstring>

// TODO INDENT
// Universal hash function: h(x) = ((a*x + b) mod p) mod r
size_t lingodb::runtime::PerfectHashView::universalHash(const std::string& key, const HashParams& params) const {
    size_t hash = 0;
    const uint32_t prime = 0x7FFFFFFF; // 2^31 - 1
    
    for (char c : key) {
        hash = (hash * params.a + static_cast<uint8_t>(c)) % prime;
    }
    hash = (hash * params.b) % prime;
    return hash % this->r;
}

// Hash function to map a key to its perfect hash index
size_t lingodb::runtime::PerfectHashView::hash(const std::string& key) const {
    size_t h1 = universalHash(key, auxHashParams[0]);
    size_t h2 = universalHash(key, auxHashParams[1]);
    return (h1 + g[h2]);
}

lingodb::runtime::PerfectHashView* lingodb::runtime::PerfectHashView::build(FlexibleBuffer* lkvalues, FlexibleBuffer* gvalues) {
    std::vector<std::string> empty;
    auto* ph = new PerfectHashView(empty);
    // TODO destroy
    printf("#### PerfectHashView::build %d %d\n", lkvalues->getLen(), gvalues->getLen());
    ph->tableSize = lkvalues->getLen();
    // For FCH, r is typically set to m² where m is number of keys
    ph->r = std::max(size_t(16), size_t(ph->tableSize) * size_t(ph->tableSize));
    ph->lookupTable.resize(ph->tableSize);
    ph->g.resize(ph->r, std::numeric_limits<size_t>::max());

    size_t aIdx = 0;
    gvalues->iterate([&](uint8_t* entryRawPtr) {
        if (aIdx == 0) std::memcpy(&ph->auxHashParams[0].a, entryRawPtr, sizeof(uint32_t));
        else if (aIdx == 1) std::memcpy(&ph->auxHashParams[0].b, entryRawPtr, sizeof(uint32_t));
        else if (aIdx == 2) std::memcpy(&ph->auxHashParams[1].a, entryRawPtr, sizeof(uint32_t));
        else if (aIdx == 3) std::memcpy(&ph->auxHashParams[1].b, entryRawPtr, sizeof(uint32_t));
        else {
            uint64_t displ;
            std::memcpy(&displ, entryRawPtr, sizeof(displ));
            uint16_t idx = displ >> 32;
            displ -= uint64_t(idx) << 32;
            ph->g[idx] = ph->g[displ];
        }
        aIdx++;
    });
    printf("  #### PerfectHashView::build 1\n");

    lkvalues->iterate([&](uint8_t* entryRawPtr) {
        printf("  #### PerfectHashView::build lkvalues\n");
        LKEntry lk;
        if (entryRawPtr == nullptr) {
            lk.empty = true;
        } else {
            lk.empty = false;
            printf("    #### PerfectHashView::build lkvalues b\n");
            std::memcpy(&lk.key, entryRawPtr, sizeof(lk.key));
            printf("    #### PerfectHashView::build lkvalues a %s\n", lk.key.str().c_str());
            lk.hashvalue = ph->hash(lk.key.str());
            printf("    #### PerfectHashView::build lkvalues h\n");
        }
        ph->lookupTable.push_back(lk);
    });

    printf("  #### PerfectHashView::build done\n");
    return ph;
}

// TODO INDENT
lingodb::runtime::PerfectHashView* lingodb::runtime::PerfectHashView::buildPerfectHash(const std::vector<std::string>& keySet) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1, 0x7FFFFFFE);
    
    // Initialize auxiliary hash functions with random parameters
    std::vector<HashParams> auxHashParams(2);
    auxHashParams[0] = {dist(gen), dist(gen)};
    auxHashParams[1] = {dist(gen), dist(gen)};
    
    PerfectHashView* ph = new PerfectHashView(keySet);
    // Initialize displacement array with default values
    auto& g = ph->g;
    // For FCH, r is typically set to m² where m is number of keys
    ph->r = std::max(size_t(16), size_t(ph->tableSize) * size_t(ph->tableSize));
    g.resize(ph->r, std::numeric_limits<size_t>::max());
    
    // Used to track hash collisions during construction
    auto tableSize = ph->tableSize;
    std::vector<bool> occupied(tableSize, false);
    
    // Process each key to build the perfect hash function
    for (const auto& key : keySet) {
        
        // Calculate the two auxiliary hashes
        uint64_t h1 = ph->universalHash(key, auxHashParams[0]);
        uint64_t h2 = ph->universalHash(key, auxHashParams[1]);
        
        // Calculate the FCH index: (h1 + g[h2]) % tableSize
        size_t idx = (h1 + g[h2]) % tableSize;
        
        // If there's a collision, resolve it by adjusting g[h2]
        if (occupied[idx]) {
            // Try different values for g[h2] until collision is resolved
            bool found = false;
            for (int d = 0; d < tableSize && !found; ++d) {
                g[h2] = d;
                idx = (h1 + g[h2]) % tableSize;
                if (!occupied[idx]) {
                    found = true;
                }
            }
            
            // If we couldn't resolve collisions, restart with new hash functions
            if (!found) {
                std::fill(g.begin(), g.end(), std::numeric_limits<size_t>::max());
                std::fill(occupied.begin(), occupied.end(), false);
                LKEntry emptyLK;
                emptyLK.empty = false;
                std::fill(ph->lookupTable.begin(), ph->lookupTable.end(), emptyLK);
                std::fill(ph->lookupTableRaw.begin(), ph->lookupTableRaw.end(), std::nullopt);
                // TODO
                // auxHashParams[0] = {dist(gen), dist(gen)};
                // auxHashParams[1] = {dist(gen), dist(gen)};
                return buildPerfectHash(keySet);  // 递归重建
            }
        }
        
        // Mark this index as occupied
        occupied[idx] = true;
        ph->lookupTableRaw[idx] = key;
    }
    return ph;
}