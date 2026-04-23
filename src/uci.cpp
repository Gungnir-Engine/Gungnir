#include "uci.h"

#include "nnue.h"
#include "notation.h"
#include "position.h"
#include "search.h"
#include "tt.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

namespace gungnir {

namespace {

int g_threads = 1;

void cmd_position(Position& pos, std::istringstream& iss) {
    std::string sub;
    iss >> sub;
    if (sub == "startpos") {
        pos.set_startpos();
        iss >> sub;  // either end of stream or "moves"
    } else if (sub == "fen") {
        std::string fen, part;
        for (int i = 0; i < 6 && (iss >> part); ++i) {
            if (!fen.empty()) fen += ' ';
            fen += part;
        }
        pos.set_from_fen(fen);
        iss >> sub;  // either end of stream or "moves"
    } else {
        return;
    }

    if (sub == "moves") {
        std::string mv;
        while (iss >> mv) {
            const Move m = parse_uci_move(pos, mv);
            if (m == MOVE_NULL) break;
            pos.make_move(m);
        }
    }
    // After (re)setting the position, NNUE accumulator must be rebuilt.
    if (NNUE::is_loaded() && NNUE::is_enabled()) NNUE::refresh(pos);
}

void cmd_go(Position& pos, std::istringstream& iss) {
    SearchLimits limits;
    int wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1, movestogo = 30;
    std::string sub;
    while (iss >> sub) {
        if      (sub == "depth")     iss >> limits.max_depth;
        else if (sub == "movetime")  iss >> movetime;
        else if (sub == "wtime")     iss >> wtime;
        else if (sub == "btime")     iss >> btime;
        else if (sub == "winc")      iss >> winc;
        else if (sub == "binc")      iss >> binc;
        else if (sub == "movestogo") iss >> movestogo;
        else if (sub == "infinite")  limits.infinite = true;
    }

    if (limits.infinite) {
        // No time limit.
    } else if (movetime > 0) {
        limits.soft_ms = limits.hard_ms = movetime;
    } else {
        const int my_time = (pos.stm() == WHITE) ? wtime : btime;
        const int my_inc  = (pos.stm() == WHITE) ? winc  : binc;
        if (my_time > 0) {
            // Conservative budget: divide the remaining time across roughly
            // movestogo moves and add a fraction of the increment. Hard cap
            // at half the remaining time so we never flag.
            const int budget = my_time / std::max(movestogo, 1) + my_inc * 3 / 4;
            limits.soft_ms = std::min(budget, my_time / 4);
            limits.hard_ms = std::min(budget * 4, my_time / 2);
            if (limits.hard_ms < 1) limits.hard_ms = 1;
            if (limits.soft_ms < 1) limits.soft_ms = 1;
        }
    }

    SearchInfo info = (g_threads > 1) ? search_smp(pos, limits, g_threads)
                                       : search(pos, limits);
    if (info.best_move == MOVE_NULL) {
        // No moves searched — emit a recognizable "null" so the GUI doesn't hang.
        std::cout << "bestmove 0000" << std::endl;
    } else {
        std::cout << "bestmove " << move_to_uci(info.best_move) << std::endl;
    }
}

}  // namespace

int uci_loop() {
    // Engines must be unbuffered for the GUI to see responses promptly.
    std::cout.setf(std::ios::unitbuf);

    TT::init(16);
    Position pos;
    pos.set_startpos();

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
        if (!(iss >> token)) continue;

        if (token == "uci") {
            std::cout << "id name Gungnir 0.5-dev" << std::endl;
            std::cout << "id author Gungnir-Engine" << std::endl;
            std::cout << "option name Hash type spin default 16 min 1 max 1024" << std::endl;
            std::cout << "option name Threads type spin default 1 min 1 max 64" << std::endl;
            std::cout << "option name UseNNUE type check default true" << std::endl;
            std::cout << "option name NNUEFile type string default nn-small.nnue" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (token == "setoption") {
            // setoption name <name> value <value>
            std::string sub, name, value;
            iss >> sub;  // "name"
            // Read name tokens until "value"
            while (iss >> sub && sub != "value") {
                if (!name.empty()) name += ' ';
                name += sub;
            }
            std::string vpart;
            while (iss >> vpart) {
                if (!value.empty()) value += ' ';
                value += vpart;
            }
            if (name == "Hash") {
                try { TT::init(size_t(std::stoi(value))); } catch (...) {}
            } else if (name == "Threads") {
                try { g_threads = std::max(1, std::stoi(value)); } catch (...) {}
            } else if (name == "UseNNUE") {
                NNUE::set_enabled(value == "true" || value == "True" || value == "1");
            } else if (name == "NNUEFile") {
                NNUE::load(value);
            }
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "ucinewgame") {
            TT::clear();
            pos.set_startpos();
        } else if (token == "position") {
            cmd_position(pos, iss);
        } else if (token == "go") {
            cmd_go(pos, iss);
        } else if (token == "stop") {
            g_stop.store(true);
        } else if (token == "quit" || token == "exit") {
            break;
        } else if (token == "d" || token == "display") {
            std::cout << pos.to_string();
        }
    }
    return 0;
}

}  // namespace gungnir
