// Gungnir — a UCI chess engine
// Session 15 demo: initialize precomputed attack tables for knight/king/pawn,
// print a few sanity bitboards, and exit. No movegen yet (that's Session 16).

#include "attacks.h"
#include "bitboard.h"
#include "types.h"

#include <iostream>

namespace gungnir {

constexpr const char* kVersion = "0.1.0-dev";

void print_banner() {
    std::cout << "Gungnir v" << kVersion << " — a UCI chess engine\n";
    std::cout << "Session 15: bitboards, types, non-slider attack tables.\n\n";
}

}  // namespace gungnir

int main() {
    using namespace gungnir;

    init_attacks();
    print_banner();

    std::cout << "Knight attacks from e4:\n";
    std::cout << bitboard_to_string(knight_attacks(SQ_E4)) << "\n";

    std::cout << "King attacks from d4:\n";
    std::cout << bitboard_to_string(king_attacks(SQ_D4)) << "\n";

    std::cout << "White pawn attacks from e4:\n";
    std::cout << bitboard_to_string(pawn_attacks(WHITE, SQ_E4)) << "\n";

    std::cout << "Black pawn attacks from e4:\n";
    std::cout << bitboard_to_string(pawn_attacks(BLACK, SQ_E4)) << "\n";

    // Quick structural sanity:
    //   - Knight on e4 attacks 8 squares (in-board)
    //   - Knight on a1 attacks 2 squares (corner)
    //   - King on e1 attacks 5 squares (on the edge, including e2, d1, f1, d2, f2)
    std::cout << "Sanity counts:\n";
    std::cout << "  knight_attacks(e4) popcount = " << popcount(knight_attacks(SQ_E4))
              << " (expect 8)\n";
    std::cout << "  knight_attacks(a1) popcount = " << popcount(knight_attacks(SQ_A1))
              << " (expect 2)\n";
    std::cout << "  king_attacks(e1) popcount   = " << popcount(king_attacks(SQ_E1))
              << " (expect 5)\n";
    std::cout << "  pawn_attacks(W, e2) popcount = " << popcount(pawn_attacks(WHITE, SQ_E2))
              << " (expect 2)\n";
    std::cout << "  pawn_attacks(W, a2) popcount = " << popcount(pawn_attacks(WHITE, SQ_A2))
              << " (expect 1)\n";

    return 0;
}
