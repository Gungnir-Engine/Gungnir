#include "perft.h"

#include "movegen.h"
#include "notation.h"
#include "position.h"

#include <chrono>
#include <cstdlib>
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

u64 perft_hashed(Position& pos, int depth) {
    if (depth == 0) return 1;

    MoveList list;
    generate_legal(pos, list);

    u64 total = 0;
    for (int i = 0; i < list.size; ++i) {
        const Move m = list.moves[i];
        const u64 hash_before = pos.hash();

        pos.make_move(m);
        const u64 expected_after = pos.compute_hash_from_scratch();
        if (pos.hash() != expected_after) {
            std::cerr << "HASH MISMATCH after make: " << move_to_uci(m) << "\n";
            std::cerr << "  Got:      0x" << std::hex << pos.hash() << "\n";
            std::cerr << "  Expected: 0x" << expected_after << std::dec << "\n";
            std::abort();
        }

        total += perft_hashed(pos, depth - 1);

        pos.unmake_move(m);
        if (pos.hash() != hash_before) {
            std::cerr << "HASH MISMATCH after unmake: " << move_to_uci(m) << "\n";
            std::cerr << "  Got:      0x" << std::hex << pos.hash() << "\n";
            std::cerr << "  Expected: 0x" << hash_before << std::dec << "\n";
            std::abort();
        }
    }
    return total;
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
