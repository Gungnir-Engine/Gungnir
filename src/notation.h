// Gungnir — UCI move notation: print/parse helpers used by perft, search,
// and the UCI loop.

#pragma once

#include "move.h"
#include "position.h"

#include <string>
#include <string_view>

namespace gungnir {

// "e2e4", "e1g1" (castle), "e7e8q" (promotion).
std::string move_to_uci(Move m);

// Parses a UCI move string in the context of `pos` and returns the matching
// legal move (which may be MT_CASTLING, MT_EN_PASSANT, or MT_PROMOTION even
// though the UCI string is just from-to-promo). Returns MOVE_NULL if no legal
// move matches.
Move parse_uci_move(Position& pos, std::string_view uci);

}  // namespace gungnir
