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

}  // namespace gungnir
