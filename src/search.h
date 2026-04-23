// Gungnir — alpha-beta search with quiescence, iterative deepening,
// aspiration windows, MVV-LVA move ordering, and check extensions.
//
// Session 18 introduced the basic negamax + alpha-beta + material/PST eval.
// Session 19 added qsearch, MVV-LVA, ID, aspiration windows, check extensions.
// Session 20 wires in the transposition table.

#pragma once

#include "eval.h"
#include "move.h"
#include "position.h"

#include <atomic>

namespace gungnir {

struct SearchLimits {
    int  max_depth = 64;        // hard depth cap
    int  soft_ms   = 0;         // stop iterative deepening after this much elapsed (0 = no limit)
    int  hard_ms   = 0;         // abort current iteration if elapsed exceeds this (0 = no limit)
    bool infinite  = false;     // ignore time limits entirely
};

struct SearchInfo {
    int   depth     = 0;
    int   score     = 0;
    Move  best_move = MOVE_NULL;
    Move  pv[64]    = {};
    int   pv_len    = 0;
    u64   nodes     = 0;
    int   time_ms   = 0;
};

// Top-level entry. Iterative deepening with aspiration windows.
// Prints `info ...` lines after each completed iteration.
SearchInfo search(Position& pos, const SearchLimits& limits);

// Convenience wrappers.
SearchInfo search_depth(Position& pos, int depth);     // fixed depth, no time limit
SearchInfo search_movetime(Position& pos, int ms);     // fixed time budget

// External flag the UCI loop sets to interrupt a running search.
extern std::atomic<bool> g_stop;

}  // namespace gungnir
