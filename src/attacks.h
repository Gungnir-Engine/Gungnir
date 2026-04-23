// Gungnir — precomputed attack tables.
// Non-sliders: fixed per-square tables (knight, king, pawn).
// Sliders (bishop/rook/queen): computed via BMI2 PEXT from precomputed
// dense tables indexed by `pext(occupancy, mask)` — requires CPU with
// BMI2 (Haswell 2013+ Intel, Excavator+ AMD). Our /arch:AVX2 in MSVC
// enables _pext_u64 on compatible targets.

#pragma once

#include "bitboard.h"

#ifdef _MSC_VER
  #include <immintrin.h>
#endif

namespace gungnir {

// Call once at startup to fill all tables.
void init_attacks();

// --- Non-slider tables (simple square → bitboard lookups) ---
extern Bitboard KnightAttacks[SQ_NB];
extern Bitboard KingAttacks[SQ_NB];
extern Bitboard PawnAttacks[COLOR_NB][SQ_NB];

inline Bitboard knight_attacks(Square s)        { return KnightAttacks[s]; }
inline Bitboard king_attacks(Square s)          { return KingAttacks[s]; }
inline Bitboard pawn_attacks(Color c, Square s) { return PawnAttacks[c][s]; }

// --- Slider tables ---
// Per-square entry: mask of relevant blocker bits + pointer into a dense
// shared array of precomputed attacks. Table size per square = 2^popcount(mask).
struct SliderEntry {
    const Bitboard* attacks;  // points into SliderAttackTable
    Bitboard        mask;
};

extern SliderEntry RookSlider[SQ_NB];
extern SliderEntry BishopSlider[SQ_NB];

// Helper: PEXT index into a SliderEntry's dense attack table.
inline u64 pext_u64(u64 source, u64 mask) {
#ifdef _MSC_VER
    return _pext_u64(source, mask);
#else
    return __builtin_ia32_pext_di(source, mask);
#endif
}

inline Bitboard rook_attacks(Square s, Bitboard occ) {
    const auto& e = RookSlider[s];
    return e.attacks[pext_u64(occ, e.mask)];
}

inline Bitboard bishop_attacks(Square s, Bitboard occ) {
    const auto& e = BishopSlider[s];
    return e.attacks[pext_u64(occ, e.mask)];
}

inline Bitboard queen_attacks(Square s, Bitboard occ) {
    return rook_attacks(s, occ) | bishop_attacks(s, occ);
}

// Attacks from a piece type (orthogonal for ROOK, diagonal for BISHOP, both for QUEEN,
// non-slider for KNIGHT/KING; not defined for PAWN — callers handle pawn separately).
inline Bitboard piece_attacks(PieceType pt, Square s, Bitboard occ) {
    switch (pt) {
        case KNIGHT: return knight_attacks(s);
        case BISHOP: return bishop_attacks(s, occ);
        case ROOK:   return rook_attacks(s, occ);
        case QUEEN:  return queen_attacks(s, occ);
        case KING:   return king_attacks(s);
        default:     return 0;
    }
}

}  // namespace gungnir
