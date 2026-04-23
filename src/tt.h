// Gungnir — transposition table.
// Flat array of fixed-size entries, indexed by Zobrist key modulo capacity.
// Always-replace policy with a generation counter (current entries can resist
// being overwritten by stale ones — but we keep it simple for now).

#pragma once

#include "move.h"
#include "types.h"

#include <cstddef>

namespace gungnir {

namespace TT {

enum Bound : u8 {
    BOUND_NONE  = 0,
    BOUND_LOWER = 1,  // fail-high — score is at least `score`
    BOUND_UPPER = 2,  // fail-low  — score is at most  `score`
    BOUND_EXACT = 3,
};

struct Entry {
    u64  key;
    Move move;
    i16  score;
    i8   depth;
    u8   bound;
    u8   gen;
};

// Initialize table with `mb` megabytes of capacity. Calling again resizes.
void init(size_t mb = 16);
void clear();
void new_search();              // bump generation (currently unused but reserved)

Entry* probe(u64 key, bool& found);
void   store(u64 key, Move m, int score, int depth, Bound b);

}  // namespace TT

}  // namespace gungnir
