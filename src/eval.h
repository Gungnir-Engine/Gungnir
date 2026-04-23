// Gungnir — static evaluation.
// Tapered PeSTO eval: piece-square tables for both middlegame and endgame
// phases, blended by a piece-weight phase counter (24 = full MG, 0 = full EG).
// Returns centipawns from the side-to-move's perspective.

#pragma once

#include "position.h"

namespace gungnir {

constexpr int VALUE_INF          = 32000;
constexpr int VALUE_MATE         = 30000;
constexpr int VALUE_MATE_IN_MAX  = VALUE_MATE - 256;   // anything past this is a forced mate
constexpr int VALUE_DRAW         = 0;

// Material values (used by MVV-LVA, SEE, etc.). Endgame material values from PeSTO.
extern const int piece_value[PIECE_TYPE_NB];

int evaluate(const Position& pos);

}  // namespace gungnir
