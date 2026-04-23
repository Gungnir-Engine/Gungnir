#include "perft.h"

#include "movegen.h"
#include "position.h"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace gungnir {

u64 perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    MoveList list;
    generate_legal(pos, list);

    if (depth == 1) return u64(list.size);  // shortcut; each legal move = 1 leaf

    u64 total = 0;
    for (int i = 0; i < list.size; ++i) {
        pos.make_move(list.moves[i]);
        total += perft(pos, depth - 1);
        pos.unmake_move(list.moves[i]);
    }
    return total;
}

static std::string move_to_uci(Move m) {
    auto sq_name = [](Square s) {
        std::string out;
        out += char('a' + file_of(s));
        out += char('1' + rank_of(s));
        return out;
    };
    std::string r = sq_name(m.from()) + sq_name(m.to());
    if (m.type() == MT_PROMOTION) {
        switch (m.promo_type()) {
            case KNIGHT: r += 'n'; break;
            case BISHOP: r += 'b'; break;
            case ROOK:   r += 'r'; break;
            case QUEEN:  r += 'q'; break;
            default: break;
        }
    }
    return r;
}

void perft_divide(Position& pos, int depth) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    MoveList list;
    generate_legal(pos, list);

    u64 total = 0;
    for (int i = 0; i < list.size; ++i) {
        pos.make_move(list.moves[i]);
        const u64 sub = depth <= 1 ? 1 : perft(pos, depth - 1);
        pos.unmake_move(list.moves[i]);
        std::cout << move_to_uci(list.moves[i]) << ": " << sub << "\n";
        total += sub;
    }

    const auto t1 = clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const double sec = double(ms) / 1000.0;
    const double nps = sec > 0 ? double(total) / sec : 0.0;

    std::cout << "\nNodes searched: " << total << "\n";
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "NPS:  " << std::fixed << std::setprecision(0) << nps << "\n";
}

}  // namespace gungnir
