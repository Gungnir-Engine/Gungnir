// Gungnir — session 20 driver.
// Default (no args): enter UCI loop (so fastchess / GUIs can drive us).
// CLI modes:
//   gungnir perft <depth> [fen...]       → perft 1..depth
//   gungnir divide <depth> [fen...]      → per-move divide
//   gungnir hashtest <depth> [fen...]    → perft + Zobrist invariant check
//   gungnir reptest                      → threefold-repetition self-test
//   gungnir go <depth> [fen...]          → one-shot search at fixed depth
//   gungnir bench [depth]                → search a fixed set of positions

#include "attacks.h"
#include "nnue.h"
#include "notation.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "uci.h"
#include "zobrist.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gungnir {
constexpr const char* kVersion = "0.3.0-dev";
void print_banner() {
    std::cout << "Gungnir v" << kVersion << " — a UCI chess engine\n";
    std::cout << "Sessions 18-20: search, eval, transposition table, UCI.\n\n";
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

static int run_reptest() {
    using namespace gungnir;
    Position pos;
    pos.set_startpos();

    auto status = [&pos](const char* tag) {
        std::cout << tag
                  << "  hash=0x" << std::hex << pos.hash() << std::dec
                  << "  halfmove=" << pos.halfmove()
                  << "  threefold=" << (pos.is_threefold_repetition() ? "true" : "false")
                  << "\n";
    };

    const u64 start_hash = pos.hash();
    status("start          :");

    auto cycle = [&pos]() {
        pos.make_move(Move::make(SQ_G1, SQ_F3));
        pos.make_move(Move::make(SQ_B8, SQ_C6));
        pos.make_move(Move::make(SQ_F3, SQ_G1));
        pos.make_move(Move::make(SQ_C6, SQ_B8));
    };

    cycle();
    status("after 1 cycle  :");
    if (pos.hash() != start_hash) { std::cerr << "FAIL: hash changed.\n"; return 1; }
    if (pos.is_threefold_repetition()) { std::cerr << "FAIL: premature threefold.\n"; return 1; }

    cycle();
    status("after 2 cycles :");
    if (pos.hash() != start_hash) { std::cerr << "FAIL: hash changed.\n"; return 1; }
    if (!pos.is_threefold_repetition()) { std::cerr << "FAIL: missed threefold.\n"; return 1; }

    std::cout << "\nreptest: OK\n";
    return 0;
}

// Bench: search a small set of canonical positions at fixed depth, sum nodes
// and total time. Useful for measuring NPS regressions across changes.
static int run_bench(int depth) {
    using namespace gungnir;
    static const char* kBenchFens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    };
    TT::init(16);
    u64 total_nodes = 0;
    int total_ms = 0;
    Position pos;
    for (const char* fen : kBenchFens) {
        pos.set_from_fen(fen);
        SearchInfo info = search_depth(pos, depth);
        total_nodes += info.nodes;
        total_ms += info.time_ms;
        std::cout << "---\n";
    }
    const u64 ms = total_ms > 0 ? u64(total_ms) : 1;
    std::cout << "\nBench: nodes=" << total_nodes
              << " time=" << total_ms << "ms"
              << " nps=" << (total_nodes * 1000 / ms) << "\n";
    return 0;
}

// Try to auto-load nn-small.nnue from a few standard locations.
// Returns true if loaded.
static bool try_auto_load_nnue() {
    using namespace gungnir;
    static const char* kCandidates[] = {
        "nn-small.nnue",                     // CWD
        "../nn-small.nnue",                  // build/Release/ → build/
        "../../nn-small.nnue",               // build/Release/ → repo root
        "E:/Claude/nn-small.nnue",           // hardcoded fallback
    };
    for (const char* path : kCandidates) {
        if (NNUE::load(path)) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    using namespace gungnir;
    init_zobrist();
    init_attacks();
    try_auto_load_nnue();

    // Default: UCI loop. (Also explicit `gungnir uci`.)
    if (argc < 2 || std::string(argv[1]) == "uci") {
        return uci_loop();
    }

    std::string mode = argv[1];
    if (mode == "reptest") {
        print_banner();
        return run_reptest();
    }
    if (mode == "bench") {
        print_banner();
        int depth = 8;
        if (argc >= 3) {
            try { depth = std::stoi(argv[2]); } catch (...) {}
        }
        return run_bench(depth);
    }

    if (mode == "nnueverify") {
        print_banner();
        if (!NNUE::is_loaded()) {
            std::cerr << "NNUE not loaded — couldn't find nn-small.nnue.\n";
            return 1;
        }
        std::cout << "NNUE loaded.\n";
        std::cout << "  arch hash: 0x" << std::hex << NNUE::arch_hash() << std::dec << "\n";
        std::cout << "  desc:      " << NNUE::description().substr(0, 70) << "\n\n";

        struct TestPos { const char* name; const char* fen; };
        TestPos tests[] = {
            {"startpos (white)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
            {"startpos (black)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"},
            {"KQvK (white)",    "8/8/8/4k3/8/8/8/3QK3 w - - 0 1"},
            {"KQvK (black)",    "8/8/8/4k3/8/8/8/3QK3 b - - 0 1"},
            {"KRvK",            "8/8/8/4k3/8/8/8/3RK3 w - - 0 1"},
            {"Italian",         "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"},
        };
        Position pos;
        for (auto& t : tests) {
            pos.set_from_fen(t.fen);
            int feat_w[40], feat_b[40];
            const int nw = NNUE::features(pos, 0, feat_w);
            const int nb = NNUE::features(pos, 1, feat_b);
            const int raw = NNUE::evaluate(pos);
            const int cp  = raw * 100 / 208;
            std::cout << t.name << "\n"
                      << "  features: white=" << nw << " black=" << nb << "\n"
                      << "  first w-feat=" << feat_w[0] << "  first b-feat=" << feat_b[0] << "\n"
                      << "  raw eval (SF internal cp): " << raw
                      << "  →  cp: " << cp << "\n\n";
        }
        return 0;
    }

    int depth = 4;
    std::string fen;
    if (argc >= 3) {
        try { depth = std::stoi(argv[2]); } catch (...) { depth = 4; }
    }
    if (argc >= 4) fen = join_args(argc, argv, 3);

    print_banner();

    Position pos;
    if (fen.empty()) pos.set_startpos();
    else if (!pos.set_from_fen(fen)) {
        std::cerr << "Invalid FEN.\n";
        return 1;
    }

    std::cout << pos.to_string() << "\n";

    if (mode == "divide") {
        perft_divide(pos, depth);
    } else if (mode == "hashtest") {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        const u64 n = perft_hashed(pos, depth);
        const auto t1 = clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "perft_hashed(" << depth << ") = " << n
                  << "   (" << ms << " ms — hash invariants verified at every node)\n";
    } else if (mode == "go") {
        TT::init(16);
        SearchInfo info = search_depth(pos, depth);
        std::cout << "\nbestmove " << move_to_uci(info.best_move) << "\n";
    } else if (mode == "perft") {
        for (int d = 1; d <= depth; ++d) {
            const u64 n = perft(pos, d);
            std::cout << "perft(" << d << ") = " << n << "\n";
        }
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}
