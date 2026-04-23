// Gungnir — session 16 driver.
// Usage:
//   gungnir                          → perft(4) from startpos
//   gungnir perft <depth>            → perft depth from startpos
//   gungnir perft <depth> <fen...>   → perft depth from given FEN
//   gungnir divide <depth> [fen...]  → divide (per-move) perft

#include "attacks.h"
#include "perft.h"
#include "position.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace gungnir {
constexpr const char* kVersion = "0.1.0-dev";
void print_banner() {
    std::cout << "Gungnir v" << kVersion << " — a UCI chess engine\n";
    std::cout << "Session 16: sliders via PEXT, Move, Position, movegen, perft.\n\n";
}
}  // namespace gungnir

static std::string join_args(int argc, char** argv, int start) {
    std::string s;
    for (int i = start; i < argc; ++i) {
        if (!s.empty()) s += ' ';
        s += argv[i];
    }
    return s;
}

int main(int argc, char** argv) {
    using namespace gungnir;
    init_attacks();
    print_banner();

    Position pos;

    // Parse args
    std::string mode = "perft";
    int depth = 4;
    std::string fen;

    if (argc >= 2) {
        mode = argv[1];
        if (argc >= 3) {
            try { depth = std::stoi(argv[2]); } catch (...) { depth = 4; }
        }
        if (argc >= 4) {
            fen = join_args(argc, argv, 3);
        }
    }

    if (fen.empty()) pos.set_startpos();
    else if (!pos.set_from_fen(fen)) {
        std::cerr << "Invalid FEN.\n";
        return 1;
    }

    std::cout << pos.to_string() << "\n";

    if (mode == "divide") {
        perft_divide(pos, depth);
    } else {
        // Plain perft — print the count per depth from 1..depth
        for (int d = 1; d <= depth; ++d) {
            const u64 n = perft(pos, d);
            std::cout << "perft(" << d << ") = " << n << "\n";
        }
    }

    return 0;
}
