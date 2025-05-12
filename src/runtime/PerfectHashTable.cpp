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
lingodb::runtime::PerfectHashView* lingodb::runtime::PerfectHashView::build(FlexibleBuffer* lkvalues, FlexibleBuffer* gvalues) {
    std::vector<std::string> empty;
    auto* ph = new PerfectHashView(empty);
    // TODO destroy
    // printf("#### PerfectHashView::build %d %d\n", lkvalues->getLen(), gvalues->getLen());
    ph->tableSize = lkvalues->getLen() * 2;
    // For FCH, r is typically set to m² where m is number of keys
    ph->r = std::max(size_t(16), size_t(ph->tableSize) * size_t(ph->tableSize) / 4);
    ph->lookupTable = (LKEntry*) malloc(sizeof(LKEntry) * ph->tableSize);
    ph->g.resize(ph->r, 0);

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
            ph->g[idx] = displ;
            printf("!! ph->g[%lu]: %lu\n", idx, displ);
        }
        aIdx++;
    });

    printf("!!build aux %u, %u %u %u %u \n", ph->tableSize, ph->auxHashParams[0].a, ph->auxHashParams[0].b, ph->auxHashParams[1].a, ph->auxHashParams[1].b);

    for (size_t i = 0; i < ph->tableSize; i ++) {
        ph->lookupTable[i].empty = true;
    }

    lkvalues->iterate([&](uint8_t* entryRawPtr) {
        LKEntry lk;
        lk.empty = false;
        std::memcpy(&lk.key, entryRawPtr, sizeof(lk.key));
        size_t h1 = ph->universalHash(lk.key.data(), lk.key.getLen(), ph->auxHashParams[0], false);
        size_t h2 = ph->universalHash(lk.key.data(), lk.key.getLen(), ph->auxHashParams[1], true);

        lk.hashvalue = h1 + ph->g[h2];
        size_t lkIdx = lk.hashvalue % ph->tableSize;
        ph->lookupTable[lkIdx] = lk;
        printf("!! idx %d %s %lu %lu %lu\n", lkIdx, lk.key.str().c_str(), h1, h2, ph->g[h2]);
    });

    // for (int i = 0; i < ph->tableSize; i ++) {
    //     auto e = ph->lookupTable[i];
    //     if (e.empty) continue;
    //     printf("!! lk %d %s %lu\n", i, e.key.str().c_str(), e.hashvalue);
    // }

    // printf("!! build done\n");

    return ph;
}

// TODO INDENT
lingodb::runtime::PerfectHashView* lingodb::runtime::PerfectHashView::buildPerfectHash(const std::vector<std::string>& keySet) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1, 0x7FFFFFFE);

    PerfectHashView* ph = new PerfectHashView(keySet);
    
    // Used to track hash collisions during construction
    size_t tableSize = ph->tableSize;
    std::vector<bool> occupied(tableSize);

    // For FCH, r is typically set to m² where m is number of keys
    ph->r = std::max(size_t(16), size_t(tableSize) * size_t(tableSize) / 4);
    ph->g.resize(ph->r);

    auto tryDisplacement = [&](size_t h1, size_t h2) {
        while (true) {
            int d = dist(gen);
            int h = (h1 + d) % tableSize;
            if (!occupied[h]) {
                return int64_t(d);
            }
        }
        return int64_t(-1);
    };

    // Initialize auxiliary hash functions with random parameters
    HashParams* auxHashParams = ph->auxHashParams;
    size_t gEmpty = std::numeric_limits<size_t>::max();
    std::function<bool()> build = [&]() {
        printf("*** build\n");
        auxHashParams[0] = {dist(gen), dist(gen)};
        auxHashParams[1] = {dist(gen), dist(gen)};
        ph->g.assign(ph->g.size(), gEmpty);
        occupied.assign(occupied.size(), false);
        ph->lookupTableRaw.assign(ph->lookupTableRaw.size(), std::nullopt);
        // Process each key to build the perfect hash function
        for (const auto& key : keySet) {
            
            // Calculate the two auxiliary hashes
            size_t h1 = ph->universalHash(key.data(), key.size(), auxHashParams[0], false);
            size_t h2 = ph->universalHash(key.data(), key.size(), auxHashParams[1], true);
            
            // Calculate the FCH index: (h1 + g[h2]) % tableSize
            size_t displ = std::min(displ = ph->g[h2], size_t(0));
            size_t idx = (h1 + displ) % tableSize;
            
            // If there's a collision, resolve it by adjusting g[h2]
            if (occupied[idx]) {
                int64_t found = -1;
                // TODO MOVE INSIDE tryDisplacement
                if (ph->g[h2] == gEmpty) {
                    // Try different values for g[h2] until collision is resolved
                    found = tryDisplacement(h1, h2);
                }

                
                
                // displacement is already taken or failed to displace. rebuild whole table
                if (found == -1) {
                    return false;
                }
                
                ph->g[h2] = found;
                idx = (h1 + ph->g[h2]) % tableSize;
            } else {
                ph->g[h2] = 0;
            }
            
            // Mark this index as occupied
            occupied[idx] = true;
            ph->lookupTableRaw[idx] = key;

            // printf("<<< idx %d %s %lu %lu %lu\n", idx, key.c_str(), h1, h2, ph->g[h2]);
        }
        return true;
    };
    while (!build()) {}

    printf("~~build aux %u %u %u %u %u \n", tableSize, auxHashParams[0].a, auxHashParams[0].b, auxHashParams[1].a, auxHashParams[1].b);
    for (int i = 0; i < ph->lookupTableRaw.size(); i ++) {
        auto v = ph->lookupTableRaw[i];
        if (v.has_value()) {
            auto s = v.value();
            size_t h1 = ph->universalHash(s.data(), s.size(), auxHashParams[0], false);
            size_t h2 = ph->universalHash(s.data(), s.size(), auxHashParams[1], true);
            size_t idx = (h1 + ph->g[h2]) % tableSize;
            printf("~~~s %s %lu %lu %lu %lu %lu\n", s.c_str(), idx, i, h1, h2, ph->g[h2]);
        }
    }

    for (int i = 0; i < ph->g.size(); i ++) {
        if(ph->g[i] == gEmpty) {
            ph->g[i] = 0;
            continue;
        }
        printf("~~ ph->g[%lu]: %lu\n", i, ph->g[i]);
    }

    return ph;
}