// Gungnir — core type definitions.
// Convention: plain enums (not enum class) so arithmetic works without casts.
// Squares, files, ranks are numeric so we can iterate and use them as indices.
//
// Piece encoding matches Stockfish's convention:
//   NO_PIECE = 0
//   White: PAWN=1 ... KING=6
//   Black: PAWN=9 ... KING=14  (color bit = bit 3)
// So `color_of(p) = (p >> 3) & 1` and `type_of(p) = p & 7`.
// Gap at 0, 7, 8, 15 (simplifies array-based lookups indexed by piece).

#pragma once

#include <cstdint>

namespace gungnir {

// Primitive aliases
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// ---------------- Color ----------------
enum Color : int {
    WHITE = 0,
    BLACK = 1,
    COLOR_NB = 2,
};

constexpr Color operator~(Color c) { return Color(c ^ BLACK); }

// ---------------- Piece types ----------------
enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN   = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK   = 4,
    QUEEN  = 5,
    KING   = 6,
    PIECE_TYPE_NB = 7,
};

// ---------------- Pieces (color + type, packed) ----------------
enum Piece : int {
    NO_PIECE = 0,

    W_PAWN   = 1, W_KNIGHT = 2, W_BISHOP = 3,
    W_ROOK   = 4, W_QUEEN  = 5, W_KING   = 6,

    B_PAWN   = 9, B_KNIGHT = 10, B_BISHOP = 11,
    B_ROOK   = 12, B_QUEEN = 13, B_KING  = 14,

    PIECE_NB = 16,
};

constexpr PieceType type_of(Piece p)  { return PieceType(p & 7); }
constexpr Color     color_of(Piece p) { return Color((p >> 3) & 1); }
constexpr Piece     make_piece(Color c, PieceType pt) { return Piece((int(c) << 3) | int(pt)); }

// ---------------- Squares ----------------
enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,
    SQ_NB = 64,
};

enum File : int {
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H,
    FILE_NB = 8,
};

enum Rank : int {
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8,
    RANK_NB = 8,
};

constexpr File  file_of(Square s)       { return File(int(s) & 7); }
constexpr Rank  rank_of(Square s)       { return Rank(int(s) >> 3); }
constexpr Square make_square(File f, Rank r) { return Square(int(r) * 8 + int(f)); }

// Rank as seen from a color's POV (own back rank = RANK_1, promo rank = RANK_8).
constexpr Rank relative_rank(Color c, Rank r) { return Rank(int(r) ^ (int(c) * 7)); }
constexpr Rank relative_rank(Color c, Square s) { return relative_rank(c, rank_of(s)); }

// Mirror a square horizontally (file flip) or vertically (rank flip).
constexpr Square flip_file(Square s) { return Square(int(s) ^ 7); }
constexpr Square flip_rank(Square s) { return Square(int(s) ^ 56); }

// ---------------- Directions (square-index offsets) ----------------
enum Direction : int {
    NORTH =  8,
    SOUTH = -8,
    EAST  =  1,
    WEST  = -1,
    NORTH_EAST = NORTH + EAST,
    NORTH_WEST = NORTH + WEST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
};

// Square arithmetic
constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Square& operator+=(Square& s, Direction d) { return s = s + d; }
constexpr Square& operator-=(Square& s, Direction d) { return s = s - d; }

// Pre-increment, for iteration: `for (Square s = SQ_A1; s < SQ_NONE; ++s)`.
constexpr Square& operator++(Square& s) { return s = Square(int(s) + 1); }
constexpr File&   operator++(File& f)   { return f = File(int(f) + 1); }
constexpr Rank&   operator++(Rank& r)   { return r = Rank(int(r) + 1); }

// Post-increment, returned by value; rarely needed but keeps Square iterable.
inline Square operator++(Square& s, int) { Square old = s; ++s; return old; }

}  // namespace gungnir
