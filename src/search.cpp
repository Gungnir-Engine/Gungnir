#include "search.h"

#include "movegen.h"
#include "notation.h"
#include "tt.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace gungnir {

std::atomic<bool> g_stop{false};

namespace {

using clock_t_ = std::chrono::steady_clock;

// --- Per-search state (single-threaded; lives only for the duration of a `search()` call).
u64                          g_nodes;
clock_t_::time_point         g_start;
int                          g_hard_ms;
bool                         g_use_time;
bool                         g_aborted;

inline int elapsed_ms() {
    using namespace std::chrono;
    return int(duration_cast<milliseconds>(clock_t_::now() - g_start).count());
}

inline bool should_stop() {
    if (g_stop.load(std::memory_order_relaxed)) return true;
    if (g_use_time && elapsed_ms() >= g_hard_ms) return true;
    return false;
}

// --- Move ordering ----------------------------------------------------------
// Score moves once, then pick-by-best on each iteration of the move loop.

struct ScoredMove { Move move; int score; };

constexpr int SCORE_TT_MOVE     = 1'000'000;
constexpr int SCORE_CAPTURE     =   100'000;  // base; MVV-LVA delta added on top

inline bool is_capture(const Position& pos, Move m) {
    if (m.type() == MT_EN_PASSANT) return true;
    return pos.piece_on(m.to()) != NO_PIECE;
}

inline PieceType victim_type(const Position& pos, Move m) {
    if (m.type() == MT_EN_PASSANT) return PAWN;
    return type_of(pos.piece_on(m.to()));
}

void score_moves(const Position& pos, const MoveList& list,
                 ScoredMove* out, Move tt_move) {
    for (int i = 0; i < list.size; ++i) {
        const Move m = list.moves[i];
        int s = 0;
        if (m == tt_move) {
            s = SCORE_TT_MOVE;
        } else if (is_capture(pos, m)) {
            const PieceType v = victim_type(pos, m);
            const PieceType a = type_of(pos.piece_on(m.from()));
            s = SCORE_CAPTURE + piece_value[v] * 16 - piece_value[a];
        }
        out[i] = {m, s};
    }
}

inline Move pick_next(ScoredMove* a, int n, int start) {
    int best = start;
    for (int i = start + 1; i < n; ++i) {
        if (a[i].score > a[best].score) best = i;
    }
    if (best != start) std::swap(a[start], a[best]);
    return a[start].move;
}

// --- Mate-distance score adjustment for TT ---------------------------------
inline int score_to_tt(int score, int ply) {
    if (score >=  VALUE_MATE_IN_MAX) return score + ply;
    if (score <= -VALUE_MATE_IN_MAX) return score - ply;
    return score;
}
inline int score_from_tt(int score, int ply) {
    if (score >=  VALUE_MATE_IN_MAX) return score - ply;
    if (score <= -VALUE_MATE_IN_MAX) return score + ply;
    return score;
}

// --- Quiescence search -----------------------------------------------------
// Stand-pat + capture extensions. Avoids horizon effects from captures.
int qsearch(Position& pos, int alpha, int beta, int ply) {
    if ((++g_nodes & 2047) == 0 && should_stop()) {
        g_aborted = true;
        return 0;
    }
    if (ply >= 96) return evaluate(pos);

    const bool in_check = pos.in_check();

    MoveList list;
    int best;

    if (in_check) {
        // No stand-pat in check (eval of an in-check position is unreliable).
        // Generate ALL legal moves (evasions) and search them all — same as
        // negamax would, but at qsearch's "no further depth decrement" semantics.
        generate_legal(pos, list);
        if (list.size == 0) return -VALUE_MATE + ply;  // checkmated
        best = -VALUE_INF;
    } else {
        const int stand_pat = evaluate(pos);
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;
        generate_legal(pos, list);
        best = stand_pat;
    }

    ScoredMove scored[256];
    int n = 0;
    for (int i = 0; i < list.size; ++i) {
        const Move m = list.moves[i];
        const bool cap = is_capture(pos, m);
        if (!in_check && !cap) continue;  // out of check: only captures
        int s = 0;
        if (cap) {
            const PieceType v = victim_type(pos, m);
            const PieceType a = type_of(pos.piece_on(m.from()));
            s = piece_value[v] * 16 - piece_value[a];
        }
        scored[n++] = {m, s};
    }

    for (int i = 0; i < n; ++i) {
        const Move m = pick_next(scored, n, i);
        pos.make_move(m);
        const int score = -qsearch(pos, -beta, -alpha, ply + 1);
        pos.unmake_move(m);

        if (g_aborted) return 0;

        if (score > best) {
            best = score;
            if (score > alpha) alpha = score;
        }
        if (alpha >= beta) break;
    }
    return best;
}

// --- Main search -----------------------------------------------------------
constexpr int MAX_EXTENSIONS = 16;

int negamax(Position& pos, int depth, int alpha, int beta, int ply, int extensions,
            Move* pv, int* pv_len) {
    if ((++g_nodes & 2047) == 0 && should_stop()) {
        g_aborted = true;
        return 0;
    }

    *pv_len = 0;

    // Repetition / 50-move draw at non-root nodes.
    if (ply > 0 && (pos.halfmove() >= 100 || pos.is_threefold_repetition())) {
        return VALUE_DRAW;
    }

    // Hard ply cap — safety net against pathological extension chains.
    if (ply >= 96) return evaluate(pos);

    // Drop into qsearch unconditionally at depth 0. qsearch handles in-check
    // positions via evasion-only generation.
    if (depth <= 0) return qsearch(pos, alpha, beta, ply);

    const bool in_check = pos.in_check();
    const bool can_extend = in_check && extensions < MAX_EXTENSIONS;
    const int  next_extensions = extensions + (can_extend ? 1 : 0);

    // --- TT probe ---
    bool found;
    TT::Entry* tte = TT::probe(pos.hash(), found);
    Move tt_move = MOVE_NULL;
    if (found) {
        tt_move = tte->move;
        if (ply > 0 && tte->depth >= depth) {
            const int s = score_from_tt(tte->score, ply);
            if (tte->bound == TT::BOUND_EXACT) return s;
            if (tte->bound == TT::BOUND_LOWER && s >= beta)  return s;
            if (tte->bound == TT::BOUND_UPPER && s <= alpha) return s;
        }
    }

    MoveList list;
    generate_legal(pos, list);

    if (list.size == 0) {
        return in_check ? -VALUE_MATE + ply : VALUE_DRAW;
    }

    ScoredMove scored[256];
    score_moves(pos, list, scored, tt_move);

    Move child_pv[64];
    int  child_pv_len = 0;

    Move best_move = MOVE_NULL;
    int  best = -VALUE_INF;
    int  orig_alpha = alpha;

    for (int i = 0; i < list.size; ++i) {
        const Move m = pick_next(scored, list.size, i);

        pos.make_move(m);
        const int new_depth = depth - 1 + (can_extend ? 1 : 0);  // check extension (capped)
        const int score = -negamax(pos, new_depth, -beta, -alpha, ply + 1,
                                   next_extensions, child_pv, &child_pv_len);
        pos.unmake_move(m);

        if (g_aborted) return 0;

        if (score > best) {
            best = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
                // Update PV: best move + child PV
                pv[0] = m;
                for (int j = 0; j < child_pv_len && j + 1 < 64; ++j) {
                    pv[j + 1] = child_pv[j];
                }
                *pv_len = std::min(child_pv_len + 1, 64);
            }
        }
        if (alpha >= beta) break;
    }

    // --- TT store ---
    TT::Bound bound = TT::BOUND_EXACT;
    if (best <= orig_alpha) bound = TT::BOUND_UPPER;
    else if (best >= beta)  bound = TT::BOUND_LOWER;
    TT::store(pos.hash(), best_move, score_to_tt(best, ply), depth, bound);

    return best;
}

// --- UCI score formatting --------------------------------------------------
void print_score_uci(int score) {
    if (std::abs(score) >= VALUE_MATE_IN_MAX) {
        const int sign = score > 0 ? 1 : -1;
        const int plies = VALUE_MATE - std::abs(score);
        const int moves = (plies + 1) / 2;
        std::cout << "mate " << (sign * moves);
    } else {
        std::cout << "cp " << score;
    }
}

void print_info_line(const SearchInfo& info) {
    const u64 ms = info.time_ms > 0 ? u64(info.time_ms) : 1;
    const u64 nps = info.nodes * 1000 / ms;
    std::cout << "info depth " << info.depth
              << " score "; print_score_uci(info.score);
    std::cout << " nodes " << info.nodes
              << " nps " << nps
              << " time " << info.time_ms
              << " pv";
    for (int i = 0; i < info.pv_len; ++i) std::cout << ' ' << move_to_uci(info.pv[i]);
    std::cout << std::endl;
}

}  // namespace

SearchInfo search(Position& pos, const SearchLimits& limits) {
    g_start = clock_t_::now();
    g_nodes = 0;
    g_aborted = false;
    g_stop.store(false);
    g_use_time = !limits.infinite && limits.hard_ms > 0;
    g_hard_ms = limits.hard_ms;
    TT::new_search();

    SearchInfo result;
    result.best_move = MOVE_NULL;

    int prev_score = 0;
    Move pv[64];
    int  pv_len = 0;

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        // Aspiration window from depth 4 onward.
        int alpha = -VALUE_INF, beta = VALUE_INF;
        int delta = 25;
        if (depth >= 4) {
            alpha = prev_score - delta;
            beta  = prev_score + delta;
        }

        int score;
        while (true) {
            score = negamax(pos, depth, alpha, beta, 0, 0, pv, &pv_len);
            if (g_aborted) break;
            if (score <= alpha) {            // fail low — widen alpha
                beta  = (alpha + beta) / 2;
                alpha = std::max(score - delta, -VALUE_INF);
                delta += delta;
            } else if (score >= beta) {      // fail high — widen beta
                beta = std::min(score + delta,  VALUE_INF);
                delta += delta;
            } else {
                break;
            }
        }

        if (g_aborted) break;

        result.depth     = depth;
        result.score     = score;
        result.best_move = (pv_len > 0) ? pv[0] : MOVE_NULL;
        result.pv_len    = pv_len;
        for (int i = 0; i < pv_len; ++i) result.pv[i] = pv[i];
        result.nodes   = g_nodes;
        result.time_ms = elapsed_ms();
        prev_score = score;

        print_info_line(result);

        // Soft time budget — stop launching deeper iterations once exceeded.
        if (!limits.infinite && limits.soft_ms > 0 && elapsed_ms() >= limits.soft_ms) break;

        // Mate-distance shortcut: if we found a forced mate, no need to search deeper.
        if (std::abs(score) >= VALUE_MATE_IN_MAX) break;
    }

    return result;
}

SearchInfo search_depth(Position& pos, int depth) {
    SearchLimits l;
    l.max_depth = depth;
    l.infinite  = true;
    return search(pos, l);
}

SearchInfo search_movetime(Position& pos, int ms) {
    SearchLimits l;
    l.soft_ms = ms;
    l.hard_ms = ms;
    return search(pos, l);
}

}  // namespace gungnir
