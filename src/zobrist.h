// Gungnir — Zobrist hashing tables.
// 64-bit hash key for chess positions:
//   psq[piece][square]    — one random per (piece, square) pair
//   castling[combo]       — one random per 4-bit castling-rights combination
//   enpassant[file]       — one random per file when an EP target square is set
//   side                  — XORed when it's black to move
//
// Tables are filled once at program startup with a fixed-seed SplitMix64 PRNG,
// so hashes are deterministic across runs (eases debugging + perft hashing).

#pragma once

#include "types.h"

namespace gungnir {

namespace Zobrist {
    extern u64 psq[PIECE_NB][SQ_NB];
    extern u64 castling[16];
    extern u64 enpassant[FILE_NB];
    extern u64 side;
}

void init_zobrist();

}  // namespace gungnir
