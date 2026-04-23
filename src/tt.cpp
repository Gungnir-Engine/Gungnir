#include "tt.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
    #include <intrin.h>
    #define PREFETCH(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
    #define PREFETCH(p) __builtin_prefetch(p)
#else
    #define PREFETCH(p) ((void)0)
#endif

namespace gungnir {
namespace TT {

// 2-entry clusters (Session 41): each cluster has a depth-preferred slot and
// an always-replace slot. probe checks both; store picks the slot that's most
// useful to overwrite. Reduces collisions and keeps important PV entries.
namespace {
constexpr int CLUSTER_SIZE = 2;
struct Cluster { Entry e[CLUSTER_SIZE]; };
static_assert(sizeof(Cluster) == 32, "Cluster expected to be 32 bytes (cache-line friendly)");

std::vector<Cluster> g_table;
size_t g_mask = 0;          // cluster count is always a power of 2
u8     g_gen  = 0;
}  // namespace

void init(size_t mb) {
    if (mb == 0) mb = 1;
    const size_t bytes = mb * 1024 * 1024;
    size_t clusters = bytes / sizeof(Cluster);
    size_t pow2 = 1;
    while ((pow2 << 1) <= clusters) pow2 <<= 1;
    g_table.assign(pow2, Cluster{});
    g_mask = pow2 - 1;
    g_gen = 0;
}

void clear() {
    if (g_table.empty()) return;
    std::memset(g_table.data(), 0, g_table.size() * sizeof(Cluster));
    g_gen = 0;
}

void new_search() { ++g_gen; }

void prefetch(u64 key) {
    if (g_table.empty()) return;
    PREFETCH(&g_table[key & g_mask]);
}

Entry* probe(u64 key, bool& found) {
    Cluster& c = g_table[key & g_mask];
    // Look for exact key match.
    for (int i = 0; i < CLUSTER_SIZE; ++i) {
        if (c.e[i].key == key && c.e[i].bound != BOUND_NONE) {
            found = true;
            // Refresh generation so this entry survives the next round.
            c.e[i].gen = g_gen;
            return &c.e[i];
        }
    }
    found = false;
    // Return an empty / replaceable slot for the caller to potentially fill on store.
    return &c.e[0];
}

void store(u64 key, Move m, int score, int depth, Bound b) {
    Cluster& c = g_table[key & g_mask];
    // Find best slot to overwrite:
    //   1. Same key — always replace (keeps best info per position).
    //   2. Empty slot.
    //   3. Lowest "priority" slot (older generation + lower depth).
    Entry* slot = &c.e[0];
    int    worst_priority = (slot->bound == BOUND_NONE)
        ? -1000000
        : (int(slot->depth) + (slot->gen == g_gen ? 200 : 0));
    for (int i = 0; i < CLUSTER_SIZE; ++i) {
        Entry& e = c.e[i];
        if (e.key == key) { slot = &e; worst_priority = -2000000; break; }
        const int prio = (e.bound == BOUND_NONE)
            ? -1000000
            : (int(e.depth) + (e.gen == g_gen ? 200 : 0));
        if (prio < worst_priority) { slot = &e; worst_priority = prio; }
    }
    // Same-key with deeper EXACT existing entry: don't downgrade it.
    if (slot->key == key && slot->depth > depth && slot->bound == BOUND_EXACT) return;
    slot->key   = key;
    slot->move  = m;
    slot->score = i16(score);
    slot->depth = i8(std::clamp(depth, -127, 127));
    slot->bound = u8(b);
    slot->gen   = g_gen;
}

}  // namespace TT
}  // namespace gungnir
