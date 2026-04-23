#include "attacks.h"

namespace gungnir {

Bitboard KnightAttacks[SQ_NB];
Bitboard KingAttacks[SQ_NB];
Bitboard PawnAttacks[COLOR_NB][SQ_NB];

namespace {

// All knight moves as (file_delta, rank_delta) pairs.
constexpr int kKnightOffsets[8][2] = {
    {-2, -1}, {-2,  1}, {-1, -2}, {-1,  2},
    { 1, -2}, { 1,  2}, { 2, -1}, { 2,  1},
};

// All king moves.
constexpr int kKingOffsets[8][2] = {
    {-1, -1}, {-1,  0}, {-1,  1},
    { 0, -1},           { 0,  1},
    { 1, -1}, { 1,  0}, { 1,  1},
};

constexpr bool on_board(int f, int r) {
    return f >= 0 && f < 8 && r >= 0 && r < 8;
}

}  // anonymous namespace

void init_attacks() {
    for (int sq = 0; sq < 64; ++sq) {
        const int f = sq & 7;
        const int r = sq >> 3;

        // Knight
        Bitboard nb = 0;
        for (const auto& off : kKnightOffsets) {
            const int nf = f + off[0], nr = r + off[1];
            if (on_board(nf, nr)) nb |= square_bb(make_square(File(nf), Rank(nr)));
        }
        KnightAttacks[sq] = nb;

        // King
        Bitboard kb = 0;
        for (const auto& off : kKingOffsets) {
            const int nf = f + off[0], nr = r + off[1];
            if (on_board(nf, nr)) kb |= square_bb(make_square(File(nf), Rank(nr)));
        }
        KingAttacks[sq] = kb;

        // Pawn diagonal captures (not pushes).
        // White pawns attack the two squares "north" diagonally;
        // black pawns attack "south" diagonally. Edges drop naturally.
        Bitboard wp = 0, bp = 0;
        if (r < 7) {
            if (f > 0) wp |= square_bb(make_square(File(f - 1), Rank(r + 1)));
            if (f < 7) wp |= square_bb(make_square(File(f + 1), Rank(r + 1)));
        }
        if (r > 0) {
            if (f > 0) bp |= square_bb(make_square(File(f - 1), Rank(r - 1)));
            if (f < 7) bp |= square_bb(make_square(File(f + 1), Rank(r - 1)));
        }
        PawnAttacks[WHITE][sq] = wp;
        PawnAttacks[BLACK][sq] = bp;
    }
}

}  // namespace gungnir
