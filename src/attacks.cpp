#include "attacks.h"

#include <array>

namespace gungnir {

Bitboard KnightAttacks[SQ_NB];
Bitboard KingAttacks[SQ_NB];
Bitboard PawnAttacks[COLOR_NB][SQ_NB];

SliderEntry RookSlider[SQ_NB];
SliderEntry BishopSlider[SQ_NB];

// Storage for all slider attack tables. Exact size is the sum of
// 2^popcount(mask[sq]) across all 64 squares, per piece type.
// Empirically: rook ≈ 102400 u64 entries (~800 KB), bishop ≈ 5248 (~40 KB).
// We over-allocate a safe upper bound so init can run without sizing first.
namespace {
constexpr std::size_t kSliderTableSize = 128 * 1024;  // 1 MB; plenty
Bitboard RookAttackTable[kSliderTableSize];
Bitboard BishopAttackTable[kSliderTableSize];
}  // namespace

namespace {

constexpr int kKnightOffsets[8][2] = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
    { 1, -2}, { 1, 2}, { 2, -1}, { 2, 1},
};

constexpr int kKingOffsets[8][2] = {
    {-1, -1}, {-1,  0}, {-1,  1},
    { 0, -1},           { 0,  1},
    { 1, -1}, { 1,  0}, { 1,  1},
};

constexpr int kRookDirs[4][2]   = { { 1, 0}, {-1, 0}, {0,  1}, {0, -1} };
constexpr int kBishopDirs[4][2] = { { 1, 1}, { 1,-1}, {-1, 1}, {-1,-1} };

constexpr bool on_board(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

// Enumerate slider attack squares in given directions, stopping on blockers.
Bitboard slow_slider_attacks(int sq, Bitboard occ, const int dirs[][2], int ndirs) {
    Bitboard attacks = 0;
    const int f0 = sq & 7, r0 = sq >> 3;
    for (int i = 0; i < ndirs; ++i) {
        int f = f0 + dirs[i][0], r = r0 + dirs[i][1];
        while (on_board(f, r)) {
            const int ns = r * 8 + f;
            attacks |= (1ULL << ns);
            if (occ & (1ULL << ns)) break;
            f += dirs[i][0];
            r += dirs[i][1];
        }
    }
    return attacks;
}

// Relevant blocker mask for a slider on `sq`: squares on the rook/bishop rays
// that MIGHT block the attack. Excludes the sq itself and the far edge in each
// direction (those can never alter the attack set since a blocker there is
// captured/ignored identically regardless).
Bitboard slider_mask(int sq, const int dirs[][2], int ndirs) {
    Bitboard mask = 0;
    const int f0 = sq & 7, r0 = sq >> 3;
    for (int i = 0; i < ndirs; ++i) {
        int f = f0 + dirs[i][0], r = r0 + dirs[i][1];
        // Stop just before the edge.
        while (on_board(f + dirs[i][0], r + dirs[i][1])) {
            mask |= (1ULL << (r * 8 + f));
            f += dirs[i][0];
            r += dirs[i][1];
        }
    }
    return mask;
}

// Iterate every blocker-subset of a mask using the Carry-Rippler trick,
// filling a dense table indexed by pext(subset, mask).
void fill_slider_table(int sq, const int dirs[][2], int ndirs, Bitboard mask,
                       Bitboard* out) {
    Bitboard sub = 0;
    do {
        const u64 idx = pext_u64(sub, mask);
        out[idx] = slow_slider_attacks(sq, sub, dirs, ndirs);
        sub = (sub - mask) & mask;
    } while (sub != 0);
}

void init_nonsliders() {
    for (int sq = 0; sq < 64; ++sq) {
        const int f = sq & 7, r = sq >> 3;

        Bitboard nb = 0;
        for (const auto& off : kKnightOffsets) {
            const int nf = f + off[0], nr = r + off[1];
            if (on_board(nf, nr)) nb |= 1ULL << (nr * 8 + nf);
        }
        KnightAttacks[sq] = nb;

        Bitboard kb = 0;
        for (const auto& off : kKingOffsets) {
            const int nf = f + off[0], nr = r + off[1];
            if (on_board(nf, nr)) kb |= 1ULL << (nr * 8 + nf);
        }
        KingAttacks[sq] = kb;

        Bitboard wp = 0, bp = 0;
        if (r < 7) {
            if (f > 0) wp |= 1ULL << ((r + 1) * 8 + (f - 1));
            if (f < 7) wp |= 1ULL << ((r + 1) * 8 + (f + 1));
        }
        if (r > 0) {
            if (f > 0) bp |= 1ULL << ((r - 1) * 8 + (f - 1));
            if (f < 7) bp |= 1ULL << ((r - 1) * 8 + (f + 1));
        }
        PawnAttacks[WHITE][sq] = wp;
        PawnAttacks[BLACK][sq] = bp;
    }
}

void init_slider_tables() {
    std::size_t rook_off = 0, bishop_off = 0;

    for (int sq = 0; sq < 64; ++sq) {
        // Rook
        const Bitboard rmask = slider_mask(sq, kRookDirs, 4);
        RookSlider[sq].mask    = rmask;
        RookSlider[sq].attacks = RookAttackTable + rook_off;
        fill_slider_table(sq, kRookDirs, 4, rmask, RookAttackTable + rook_off);
        rook_off += (std::size_t(1) << popcount(rmask));

        // Bishop
        const Bitboard bmask = slider_mask(sq, kBishopDirs, 4);
        BishopSlider[sq].mask    = bmask;
        BishopSlider[sq].attacks = BishopAttackTable + bishop_off;
        fill_slider_table(sq, kBishopDirs, 4, bmask, BishopAttackTable + bishop_off);
        bishop_off += (std::size_t(1) << popcount(bmask));
    }
    // Sanity: we should never exceed the table
    // rook_off ≈ 102400, bishop_off ≈ 5248.
}

}  // namespace

void init_attacks() {
    init_nonsliders();
    init_slider_tables();
}

}  // namespace gungnir
