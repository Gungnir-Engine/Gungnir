// Gungnir — precomputed attack tables.
// Non-slider pieces (knight, king, pawn) have simple fixed attack patterns
// that depend only on the piece's square (and color, for pawns).
// Sliders (bishop, rook, queen) will be added in Session 16 via PEXT/magic bitboards.

#pragma once

#include "bitboard.h"

namespace gungnir {

// Fill these tables once at program startup. Safe to call multiple times.
void init_attacks();

// Per-square attack masks. Pawn is also indexed by color since pawns attack
// diagonally forward only. These cover only capture squares; pawn push targets
// are computed via `shift<NORTH>`/`shift<SOUTH>` at use time.
extern Bitboard KnightAttacks[SQ_NB];
extern Bitboard KingAttacks[SQ_NB];
extern Bitboard PawnAttacks[COLOR_NB][SQ_NB];

inline Bitboard knight_attacks(Square s)            { return KnightAttacks[s]; }
inline Bitboard king_attacks(Square s)              { return KingAttacks[s]; }
inline Bitboard pawn_attacks(Color c, Square s)     { return PawnAttacks[c][s]; }

}  // namespace gungnir
