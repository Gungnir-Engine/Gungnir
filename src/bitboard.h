// Gungnir — 64-bit bitboard type + elementary ops.
// One bit per square: bit N = square N = (file_of + 8*rank_of).
// A1=bit0, H1=bit7, A8=bit56, H8=bit63.

#pragma once

#include "types.h"

#include <cassert>
#include <string>

#ifdef _MSC_VER
  #include <intrin.h>
  #pragma intrinsic(_BitScanForward64, _BitScanReverse64, __popcnt64)
#endif

namespace gungnir {

using Bitboard = u64;

// ------------- File and rank masks -------------
constexpr Bitboard BB_EMPTY = 0ULL;
constexpr Bitboard BB_FULL  = 0xFFFFFFFFFFFFFFFFULL;

constexpr Bitboard BB_FILE_A = 0x0101010101010101ULL;
constexpr Bitboard BB_FILE_B = BB_FILE_A << 1;
constexpr Bitboard BB_FILE_C = BB_FILE_A << 2;
constexpr Bitboard BB_FILE_D = BB_FILE_A << 3;
constexpr Bitboard BB_FILE_E = BB_FILE_A << 4;
constexpr Bitboard BB_FILE_F = BB_FILE_A << 5;
constexpr Bitboard BB_FILE_G = BB_FILE_A << 6;
constexpr Bitboard BB_FILE_H = BB_FILE_A << 7;

constexpr Bitboard BB_RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard BB_RANK_2 = BB_RANK_1 << (8 * 1);
constexpr Bitboard BB_RANK_3 = BB_RANK_1 << (8 * 2);
constexpr Bitboard BB_RANK_4 = BB_RANK_1 << (8 * 3);
constexpr Bitboard BB_RANK_5 = BB_RANK_1 << (8 * 4);
constexpr Bitboard BB_RANK_6 = BB_RANK_1 << (8 * 5);
constexpr Bitboard BB_RANK_7 = BB_RANK_1 << (8 * 6);
constexpr Bitboard BB_RANK_8 = BB_RANK_1 << (8 * 7);

constexpr Bitboard BB_DARK_SQUARES  = 0xAA55AA55AA55AA55ULL;
constexpr Bitboard BB_LIGHT_SQUARES = ~BB_DARK_SQUARES;

constexpr Bitboard file_bb(File f) { return BB_FILE_A << int(f); }
constexpr Bitboard rank_bb(Rank r) { return BB_RANK_1 << (int(r) * 8); }
constexpr Bitboard square_bb(Square s) { return 1ULL << int(s); }

// ------------- Bit manipulation -------------
constexpr bool test_bit(Bitboard bb, Square s) { return (bb & square_bb(s)) != 0; }

constexpr Bitboard set_bit(Bitboard bb, Square s)   { return bb | square_bb(s); }
constexpr Bitboard clear_bit(Bitboard bb, Square s) { return bb & ~square_bb(s); }
constexpr Bitboard toggle_bit(Bitboard bb, Square s){ return bb ^ square_bb(s); }

// Population count (number of set bits).
inline int popcount(Bitboard bb) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(bb));
#else
    return __builtin_popcountll(bb);
#endif
}

// Least-significant set bit index (as Square). Undefined for bb==0 — assert.
inline Square lsb(Bitboard bb) {
    assert(bb != 0);
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return Square(idx);
#else
    return Square(__builtin_ctzll(bb));
#endif
}

// Most-significant set bit index (as Square). Undefined for bb==0.
inline Square msb(Bitboard bb) {
    assert(bb != 0);
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanReverse64(&idx, bb);
    return Square(idx);
#else
    return Square(63 ^ __builtin_clzll(bb));
#endif
}

// Pop the least-significant bit and return its square index.
// Typical iteration pattern:   while (bb) { Square s = pop_lsb(bb); ... }
inline Square pop_lsb(Bitboard& bb) {
    Square s = lsb(bb);
    bb &= bb - 1;  // clears the lowest set bit
    return s;
}

// Shift all bits one square in a direction. Edge-of-board bits are dropped.
template<Direction D>
constexpr Bitboard shift(Bitboard bb) {
    if constexpr (D == NORTH)       return  bb << 8;
    else if constexpr (D == SOUTH)  return  bb >> 8;
    else if constexpr (D == EAST)       return (bb & ~BB_FILE_H) << 1;
    else if constexpr (D == WEST)       return (bb & ~BB_FILE_A) >> 1;
    else if constexpr (D == NORTH_EAST) return (bb & ~BB_FILE_H) << 9;
    else if constexpr (D == NORTH_WEST) return (bb & ~BB_FILE_A) << 7;
    else if constexpr (D == SOUTH_EAST) return (bb & ~BB_FILE_H) >> 7;
    else if constexpr (D == SOUTH_WEST) return (bb & ~BB_FILE_A) >> 9;
    else return 0;
}

// Pretty-printing for debugging.
std::string bitboard_to_string(Bitboard bb);

}  // namespace gungnir
