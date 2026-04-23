#include "tt.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gungnir {
namespace TT {

namespace {
std::vector<Entry> g_table;
size_t g_mask = 0;          // table size is always a power of 2
u8     g_gen  = 0;
}  // namespace

void init(size_t mb) {
    if (mb == 0) mb = 1;
    const size_t bytes = mb * 1024 * 1024;
    size_t entries = bytes / sizeof(Entry);
    // Round down to power of two.
    size_t pow2 = 1;
    while ((pow2 << 1) <= entries) pow2 <<= 1;
    g_table.assign(pow2, Entry{});
    g_mask = pow2 - 1;
    g_gen = 0;
}

void clear() {
    if (g_table.empty()) return;
    std::memset(g_table.data(), 0, g_table.size() * sizeof(Entry));
    g_gen = 0;
}

void new_search() { ++g_gen; }

Entry* probe(u64 key, bool& found) {
    Entry* e = &g_table[key & g_mask];
    found = (e->key == key) && (e->bound != BOUND_NONE);
    return e;
}

void store(u64 key, Move m, int score, int depth, Bound b) {
    Entry* e = &g_table[key & g_mask];
    // Always-replace, except: don't overwrite a deeper entry for the SAME key
    // with a shallower one (lets PV-line entries survive their own subtree).
    if (e->key == key && e->depth > depth && e->bound == BOUND_EXACT) return;
    e->key   = key;
    e->move  = m;
    e->score = i16(score);
    e->depth = i8(std::clamp(depth, -127, 127));
    e->bound = u8(b);
    e->gen   = g_gen;
}

}  // namespace TT
}  // namespace gungnir
