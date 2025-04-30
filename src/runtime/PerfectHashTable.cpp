#include "lingodb/runtime/PerfectHashTable.h"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cassert>

// Universal hash function: h(x) = ((a*x + b) mod p) mod r
size_t lingodb::runtime::PerfectHashView::universalHash(const std::string& key, const HashParams& params) const {
    uint64_t hash = 0;
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
    return (h1 + g[h2]) % tableSize;
}

// TODO REMOVE
mlir::Value lingodb::runtime::PerfectHashView::convertToMLIRValue(PerfectHashView& instance, mlir::OpBuilder& builder) {
    
}

lingodb::runtime::PerfectHashView* lingodb::runtime::PerfectHashView::build(FlexibleBuffer* lkvalues, FlexibleBuffer* gvalues, FlexibleBuffer* auxvalues) {
    std::vector<std::string> empty;
    auto* ph = new PerfectHashView(empty);
    // TODO destroy
    size_t aIdx = 0;
    auxvalues->iterate([&](uint8_t* entryRawPtr) {
        if (aIdx == 0) ph->auxHashParams[0].a = (uint32_t)entryRawPtr;
        else if (aIdx == 1) ph->auxHashParams[0].b = (uint32_t)entryRawPtr;
        else if (aIdx == 2) ph->auxHashParams[1].a = (uint32_t)entryRawPtr;
        else if (aIdx == 3) ph->auxHashParams[1].b = (uint32_t)entryRawPtr;
        aIdx++;
    });
    ph->tableSize = lkvalues->getLen();
    ph->r = std::max(size_t(16), ph->tableSize * ph->tableSize);
    ph->lookupTable.resize(ph->tableSize);
    lkvalues->iterate([&](uint8_t* entryRawPtr) {
        LKEntry* e = (LKEntry*) entryRawPtr;
        ph->lookupTable.push_back(e);
    });

    ph->g.resize(ph->r, -1);
    gvalues->iterate([&](uint8_t* entryRawPtr) {
        GEntry* e = (GEntry*) entryRawPtr;
        ph->g[e->index] = ph->g[e->value];
    });
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
    g.resize(ph->r, -1);
    
    // Used to track hash collisions during construction
    auto tableSize = ph->tableSize;
    std::vector<bool> occupied(tableSize, false);
    
    // Process each key to build the perfect hash function
    for (const auto& key : keySet) {
        
        // Calculate the two auxiliary hashes
        size_t h1 = ph->universalHash(key, auxHashParams[0]);
        size_t h2 = ph->universalHash(key, auxHashParams[1]);
        
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
                std::fill(g.begin(), g.end(), -1);
                std::fill(occupied.begin(), occupied.end(), false);
                std::fill(ph->lookupTable.begin(), ph->lookupTable.end(), std::nullopt);
                auxHashParams[0] = {dist(gen), dist(gen)};
                auxHashParams[1] = {dist(gen), dist(gen)};
                buildPerfectHash(keySet);  // 递归重建
                return;
            }
        }
        
        // Mark this index as occupied
        occupied[idx] = true;
        // TODO REMOVE
        // LKEntry* e = (LKEntry*)ph->lkvalues.insert();
        // auto v = lingodb::runtime::VarLen32::fromString(key);
        // memcpy(e->content, &key, sizeof(v));
        ph->lookupTableRaw[idx] = key;
    }

    // TODO REMOVE
    // for (int i = 0; i < g.size(); i ++) {
    //     if (ph->g[i] != -1) {
    //         GEntry* e = (GEntry*)ph->lkvalues.insert();
    //         e->index = i;
    //         e->value = ph->g[i];
    //     }
    // }
}