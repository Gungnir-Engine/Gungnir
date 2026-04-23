#include "bitboard.h"

#include <iomanip>
#include <sstream>

namespace gungnir {

std::string bitboard_to_string(Bitboard bb) {
    std::ostringstream os;
    os << "+---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            Square s = make_square(File(f), Rank(r));
            os << "| " << (test_bit(bb, s) ? 'X' : '.') << " ";
        }
        os << "| " << (r + 1) << "\n";
        os << "+---+---+---+---+---+---+---+---+\n";
    }
    os << "  a   b   c   d   e   f   g   h\n";
    os << "(bb = 0x" << std::hex << std::setw(16) << std::setfill('0') << bb << std::dec << ")\n";
    return os.str();
}

}  // namespace gungnir
