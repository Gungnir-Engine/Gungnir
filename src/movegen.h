// Gungnir — move generation.
// Pseudo-legal: generates all moves obeying piece movement rules, but may
// leave own king in check.
// Legal: filters pseudo-legal via make/test/unmake. Used for correctness
// (perft). Search will use a faster pin-aware generator later.

#pragma once

#include "move.h"
#include "types.h"

namespace gungnir {

class Position;

struct MoveList {
    static constexpr int MAX = 256;  // upper bound on legal moves from any position
    Move moves[MAX];
    int  size = 0;

    void add(Move m) { moves[size++] = m; }
    Move* begin() { return moves; }
    Move* end()   { return moves + size; }
    const Move* begin() const { return moves; }
    const Move* end()   const { return moves + size; }
};

// Fills `list` with all pseudo-legal moves for the side to move.
void generate_pseudo_legal(const Position& pos, MoveList& list);

// Fills `list` with all legal moves. Takes non-const Position since we
// make/unmake moves during filtering.
void generate_legal(Position& pos, MoveList& list);

}  // namespace gungnir
