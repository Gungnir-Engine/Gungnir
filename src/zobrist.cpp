#include "zobrist.h"

namespace gungnir {

namespace Zobrist {
    u64 psq[PIECE_NB][SQ_NB];
    u64 castling[16];
    u64 enpassant[FILE_NB];
    u64 side;
}

namespace {

// SplitMix64 — high-quality, deterministic, single-state PRNG. Fine for Zobrist.
u64 splitmix64(u64& state) {
    u64 z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void init_zobrist() {
    u64 seed = 0xDEADBEEFCAFEBABEULL;

    for (int p = 0; p < PIECE_NB; ++p) {
        for (int s = 0; s < SQ_NB; ++s) {
            Zobrist::psq[p][s] = splitmix64(seed);
        }
    }
    // Zero gap entries (NO_PIECE = 0; indices 7, 8, 15 are unused piece codes)
    // so XORing one is identity if it ever happens.
    for (int s = 0; s < SQ_NB; ++s) {
        Zobrist::psq[NO_PIECE][s] = 0;
        Zobrist::psq[7][s]        = 0;
        Zobrist::psq[8][s]        = 0;
        Zobrist::psq[15][s]       = 0;
    }

    for (int i = 0; i < 16; ++i) {
        Zobrist::castling[i] = splitmix64(seed);
    }
    for (int f = 0; f < FILE_NB; ++f) {
        Zobrist::enpassant[f] = splitmix64(seed);
    }
    Zobrist::side = splitmix64(seed);
}

}  // namespace gungnir
