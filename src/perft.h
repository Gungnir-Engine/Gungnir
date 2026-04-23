// Gungnir — perft (performance test).
// Recursively counts leaf nodes at depth `d` from the given position.
// Used exclusively for verifying move-generation correctness against
// well-known position/depth → node-count values.

#pragma once

#include "types.h"

namespace gungnir {

class Position;

// Returns total leaf positions at exactly depth `d`.
u64 perft(Position& pos, int depth);

// Runs perft(d) and prints each depth-1 child's sub-count plus total and timing.
// Useful for bisecting movegen bugs against `stockfish divide`.
void perft_divide(Position& pos, int depth);

// Like perft, but at every make/unmake verifies that the incrementally-
// maintained Zobrist hash matches a from-scratch recomputation, AND that
// unmake restores the previous hash. Aborts on any mismatch with a diagnostic.
u64 perft_hashed(Position& pos, int depth);

}  // namespace gungnir
