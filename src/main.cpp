// Gungnir — a UCI chess engine
// v0.0.1: scaffold. Prints the starting position and exits.
// Next sessions add: bitboards, move generation, search, NNUE eval, UCI I/O.

#include <array>
#include <iostream>
#include <string_view>

namespace gungnir {

constexpr const char* kVersion = "0.0.1";

// Standard 8x8 starting position. Uppercase = white, lowercase = black.
// Row 0 = rank 8 (black's home); row 7 = rank 1 (white's home).
constexpr std::array<std::string_view, 8> kStartPos = {
    "rnbqkbnr",
    "pppppppp",
    "........",
    "........",
    "........",
    "........",
    "PPPPPPPP",
    "RNBQKBNR",
};

void print_banner() {
    std::cout << "Gungnir v" << kVersion << " — a UCI chess engine\n";
    std::cout << "(c) 2026 Gungnir-Dev. Session 14: scaffold.\n\n";
}

void print_board(const std::array<std::string_view, 8>& board) {
    for (int r = 0; r < 8; ++r) {
        std::cout << (8 - r) << "  ";
        for (char c : board[r]) std::cout << c << ' ';
        std::cout << '\n';
    }
    std::cout << "\n   a b c d e f g h\n";
}

}  // namespace gungnir

int main() {
    gungnir::print_banner();
    gungnir::print_board(gungnir::kStartPos);
    std::cout << "\nWhite to move.\n";
    return 0;
}
