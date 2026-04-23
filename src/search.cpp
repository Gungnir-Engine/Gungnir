#include "search.h"

#include "movegen.h"
#include "nnue.h"
#include "notation.h"
#include "tt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace gungnir {

std::atomic<bool> g_stop{false};

namespace {

using clock_t_ = std::chrono::steady_clock;

// --- Per-search state.
// Shared across threads (set once by main before workers start).
clock_t_::time_point  g_start;
int                   g_hard_ms;
bool                  g_use_time;

// Per-thread (each LazySMP worker has its own copy).
thread_local u64      g_nodes;
thread_local bool     g_aborted;
thread_local bool     g_is_helper = false;   // helper threads suppress info-line output

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
constexpr int SCORE_KILLER1     =    90'000;
constexpr int SCORE_KILLER2     =    80'000;
constexpr int MAX_PLY           = 96;

// --- LMR table (Sessions 27 + 39 retune): reduction = base + log(d) * log(m) / divisor.
// Session 39 tuning: dropped base from 0.75 → 0.50 and divisor from 2.25 → 2.00 —
// slightly more aggressive at shallow depths/late moves where the gain dominates.
int g_lmr_table[64][64];
void init_lmr_table() {
    static bool done = false; if (done) return; done = true;
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            g_lmr_table[d][m] = int(0.50 + std::log(double(d)) * std::log(double(m)) / 2.00);
}

// --- Late move pruning thresholds (Session 28; 39 widened slightly)
constexpr int LMP_THRESHOLD[7] = {0, 4, 7, 12, 19, 28, 40};

// --- Pruning margins (Session 28; 39 retuned)
constexpr int RFP_MARGIN_PER_DEPTH = 75;   // reverse futility (static null) — was 80
constexpr int FUTILITY_MARGIN      = 110;  // standard futility — was 100

// Killers + history persist across the iterative-deepening run (cleared at
// start of each `search()`). All thread_local — each LazySMP worker keeps
// its own copy.
thread_local Move g_killers[MAX_PLY][2];
thread_local int  g_history[COLOR_NB][SQ_NB][SQ_NB];

// --- Capture history (Session 30): [piece-type][to][captured-type]
thread_local int g_cap_history[PIECE_TYPE_NB][SQ_NB][PIECE_TYPE_NB];

// --- Continuation history (Session 30): [prev-piece-type][prev-to][curr-piece-type][curr-to]
// 7*64*7*64 = 200,704 ints = ~785 KB per thread.
thread_local int g_cont_hist[PIECE_TYPE_NB][SQ_NB][PIECE_TYPE_NB][SQ_NB];

// Track previous move's (moving piece type, to-square) per ply for cont-hist lookup.
struct PlyState {
    PieceType prev_piece;
    Square    prev_to;
};
thread_local PlyState g_ply_state[MAX_PLY + 4];

void clear_killers_history() {
    for (int p = 0; p < MAX_PLY; ++p) {
        g_killers[p][0] = MOVE_NULL;
        g_killers[p][1] = MOVE_NULL;
    }
    std::memset(g_history,    0, sizeof(g_history));
    std::memset(g_cap_history,0, sizeof(g_cap_history));
    std::memset(g_cont_hist,  0, sizeof(g_cont_hist));
    for (size_t i = 0; i < sizeof(g_ply_state)/sizeof(g_ply_state[0]); ++i) {
        g_ply_state[i] = {NO_PIECE_TYPE, SQ_A1};
    }
}

inline void update_cap_history(PieceType pt, Square to, PieceType captured, int depth) {
    int& h = g_cap_history[pt][to][captured];
    h += depth * depth;
    if (h > 100000) h /= 2;
}

inline void update_cont_history(int ply, PieceType cur_pt, Square cur_to, int depth) {
    if (ply <= 0) return;
    const PlyState& prev = g_ply_state[ply - 1];
    if (prev.prev_piece == NO_PIECE_TYPE) return;
    int& h = g_cont_hist[prev.prev_piece][prev.prev_to][cur_pt][cur_to];
    h += depth * depth;
    if (h > 100000) h /= 2;
}

inline void update_killer(int ply, Move m) {
    if (g_killers[ply][0] == m) return;
    g_killers[ply][1] = g_killers[ply][0];
    g_killers[ply][0] = m;
}

inline void update_history(Color stm, Move m, int depth) {
    int& h = g_history[stm][m.from()][m.to()];
    h += depth * depth;
    if (h > 100000) {
        // Halve everything in this row to avoid runaway. Cheap and effective.
        for (int s = 0; s < SQ_NB; ++s) g_history[stm][m.from()][s] /= 2;
    }
}

inline bool is_capture(const Position& pos, Move m) {
    if (m.type() == MT_EN_PASSANT) return true;
    return pos.piece_on(m.to()) != NO_PIECE;
}

inline PieceType victim_type(const Position& pos, Move m) {
    if (m.type() == MT_EN_PASSANT) return PAWN;
    return type_of(pos.piece_on(m.to()));
}

void score_moves(const Position& pos, const MoveList& list,
                 ScoredMove* out, Move tt_move, int ply) {
    const Color stm = pos.stm();
    const PlyState& prev = (ply > 0) ? g_ply_state[ply - 1] : g_ply_state[0];
    const bool have_prev = (ply > 0 && prev.prev_piece != NO_PIECE_TYPE);
    for (int i = 0; i < list.size; ++i) {
        const Move m = list.moves[i];
        int s;
        if (m == tt_move) {
            s = SCORE_TT_MOVE;
        } else if (is_capture(pos, m)) {
            const PieceType v = victim_type(pos, m);
            const PieceType a = type_of(pos.piece_on(m.from()));
            s = SCORE_CAPTURE + piece_value[v] * 16 - piece_value[a]
              + g_cap_history[a][m.to()][v] / 8;
        } else if (ply < MAX_PLY && m == g_killers[ply][0]) {
            s = SCORE_KILLER1;
        } else if (ply < MAX_PLY && m == g_killers[ply][1]) {
            s = SCORE_KILLER2;
        } else {
            const PieceType cur_pt = type_of(pos.piece_on(m.from()));
            s = g_history[stm][m.from()][m.to()];
            if (have_prev) {
                s += g_cont_hist[prev.prev_piece][prev.prev_to][cur_pt][m.to()];
            }
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
// Time-check granularity (Session 42): tighter for short time controls so we
// don't overshoot a 5-ms budget. Set in `search()` based on hard_ms.
thread_local u64 g_time_check_mask = 2047;

int qsearch(Position& pos, int alpha, int beta, int ply) {
    if ((++g_nodes & g_time_check_mask) == 0 && should_stop()) {
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
        NNUE::on_make(pos, m);
        const int score = -qsearch(pos, -beta, -alpha, ply + 1);
        NNUE::on_unmake();
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
            bool did_null, Move excluded_move, Move* pv, int* pv_len) {
    if ((++g_nodes & g_time_check_mask) == 0 && should_stop()) {
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
    const bool is_pv_node = (beta - alpha > 1);
    const bool in_singular_search = (excluded_move != MOVE_NULL);

    // --- TT probe ---
    // Skip TT during a singular-extension verification search (would short-circuit
    // the very thing we're verifying).
    bool found = false;
    TT::Entry* tte = nullptr;
    Move tt_move = MOVE_NULL;
    int  tt_score = 0;
    int  tt_depth = 0;
    u8   tt_bound = TT::BOUND_NONE;
    if (!in_singular_search) {
        tte = TT::probe(pos.hash(), found);
        if (found) {
            tt_move  = tte->move;
            tt_score = score_from_tt(tte->score, ply);
            tt_depth = tte->depth;
            tt_bound = tte->bound;
            if (ply > 0 && tt_depth >= depth) {
                if (tt_bound == TT::BOUND_EXACT) return tt_score;
                if (tt_bound == TT::BOUND_LOWER && tt_score >= beta)  return tt_score;
                if (tt_bound == TT::BOUND_UPPER && tt_score <= alpha) return tt_score;
            }
        }
    }

    Move child_pv_buf[64];
    int  child_pv_len = 0;

    // Compute static eval once if we'll need it for pruning decisions.
    int static_eval = 0;
    bool have_static = false;
    if (!in_check) {
        static_eval = evaluate(pos);
        have_static = true;
    }

    // --- Reverse futility / static null pruning (Session 28) ---
    // At low depth in a non-PV node, if static eval is way above beta,
    // assume we're winning and return early.
    if (!is_pv_node && !in_check && depth <= 6
        && have_static && static_eval - depth * RFP_MARGIN_PER_DEPTH >= beta
        && std::abs(beta) < VALUE_MATE_IN_MAX) {
        return static_eval;
    }

    // --- Null-move pruning ---
    if (!is_pv_node && !in_check && !did_null && !in_singular_search && depth >= 3
        && pos.has_non_pawn_material(pos.stm())
        && (!have_static || static_eval >= beta)) {
        const int R = 2 + depth / 4;
        pos.make_null_move();
        NNUE::on_null_make();
        const int score = -negamax(pos, depth - R - 1, -beta, -beta + 1, ply + 1,
                                   extensions, true, MOVE_NULL, child_pv_buf, &child_pv_len);
        NNUE::on_null_unmake();
        pos.unmake_null_move();
        if (g_aborted) return 0;
        if (score >= beta) {
            if (score >= VALUE_MATE_IN_MAX) return beta;
            return score;
        }
    }

    MoveList list;
    generate_legal(pos, list);

    if (list.size == 0) {
        return in_check ? -VALUE_MATE + ply : VALUE_DRAW;
    }

    ScoredMove scored[256];
    score_moves(pos, list, scored, tt_move, ply);

    Move best_move = MOVE_NULL;
    int  best = -VALUE_INF;
    int  orig_alpha = alpha;
    int  quiets_searched = 0;
    Move quiets_seen[64];  // for history malus on quiets that didn't cut

    // Prefetch first move's TT cluster — overlap memory latency with move ordering.
    if (list.size > 0) {
        // Best-effort: we don't know the resulting hash without making the move.
        // Instead prefetch the current cluster (already in cache typically).
        TT::prefetch(pos.hash());
    }

    for (int i = 0; i < list.size; ++i) {
        const Move m = pick_next(scored, list.size, i);
        if (m == excluded_move) continue;  // singular-extension verification skips this move
        const bool capture = is_capture(pos, m);
        const bool is_killer = (ply < MAX_PLY) &&
                               (m == g_killers[ply][0] || m == g_killers[ply][1]);

        // --- Singular extension (Session 29) ---
        // If TT entry suggests this is the only good move (verified via a
        // reduced-depth search excluding it), extend by 1 ply.
        int singular_extension = 0;
        if (!in_singular_search && m == tt_move && depth >= 8
            && tt_depth >= depth - 3 && tt_bound != TT::BOUND_UPPER
            && std::abs(tt_score) < VALUE_MATE_IN_MAX) {
            const int singular_beta = tt_score - depth;  // margin = depth cp
            const int singular_depth = (depth - 1) / 2;
            const int s = negamax(pos, singular_depth, singular_beta - 1, singular_beta,
                                  ply, extensions, did_null, m, child_pv_buf, &child_pv_len);
            if (g_aborted) return 0;
            if (s < singular_beta) singular_extension = 1;
        }

        // --- Late move pruning (Session 28) ---
        // At low depth in a non-PV non-check node, after we've tried the early
        // moves, skip remaining quiet moves entirely.
        if (!is_pv_node && !in_check && depth <= 6 && best > -VALUE_MATE_IN_MAX
            && !capture && !is_killer
            && quiets_searched >= LMP_THRESHOLD[depth]) {
            continue;
        }

        // --- Futility pruning (Session 28) ---
        // At frontier (depth==1), if static eval + margin can't reach alpha,
        // skip non-tactical non-evading moves.
        if (!is_pv_node && !in_check && depth == 1 && !capture
            && have_static && static_eval + FUTILITY_MARGIN <= alpha
            && std::abs(alpha) < VALUE_MATE_IN_MAX) {
            quiets_searched++;
            continue;
        }

        // Track previous-move state for the child's continuation history.
        const PieceType moving_pt = type_of(pos.piece_on(m.from()));
        if (ply < MAX_PLY) g_ply_state[ply] = {moving_pt, m.to()};

        pos.make_move(m);
        NNUE::on_make(pos, m);

        const int new_depth = depth - 1 + (can_extend ? 1 : 0) + singular_extension;

        // --- Late Move Reductions (Session 27) ---
        int reduction = 0;
        if (i >= 2 && new_depth >= 3 && !in_check && !capture && !is_killer) {
            reduction = g_lmr_table[std::min(new_depth, 63)][std::min(i, 63)];
            if (is_pv_node) reduction = std::max(0, reduction - 1);
            // Don't reduce below 1 ply.
            if (reduction >= new_depth) reduction = new_depth - 1;
            if (reduction < 0) reduction = 0;
        }

        // --- PVS (Session 31): first move full window, others zero-window + re-search ---
        int score;
        if (i == 0) {
            score = -negamax(pos, new_depth, -beta, -alpha, ply + 1,
                             next_extensions, false, MOVE_NULL, child_pv_buf, &child_pv_len);
        } else {
            // Zero-window with possible LMR
            score = -negamax(pos, new_depth - reduction, -alpha - 1, -alpha, ply + 1,
                             next_extensions, false, MOVE_NULL, child_pv_buf, &child_pv_len);
            // If LMR succeeded (score > alpha), re-search at full depth zero-window
            if (!g_aborted && reduction > 0 && score > alpha) {
                score = -negamax(pos, new_depth, -alpha - 1, -alpha, ply + 1,
                                 next_extensions, false, MOVE_NULL, child_pv_buf, &child_pv_len);
            }
            // If still > alpha and within window of a PV node, re-search full window
            if (!g_aborted && score > alpha && score < beta && is_pv_node) {
                score = -negamax(pos, new_depth, -beta, -alpha, ply + 1,
                                 next_extensions, false, MOVE_NULL, child_pv_buf, &child_pv_len);
            }
        }

        NNUE::on_unmake();
        pos.unmake_move(m);

        if (g_aborted) return 0;

        if (!capture && quiets_searched < 64) {
            quiets_seen[quiets_searched++] = m;
        }

        if (score > best) {
            best = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
                pv[0] = m;
                for (int j = 0; j < child_pv_len && j + 1 < 64; ++j) {
                    pv[j + 1] = child_pv_buf[j];
                }
                *pv_len = std::min(child_pv_len + 1, 64);
            }
        }
        if (alpha >= beta) {
            // Beta cutoff. Update killers + history (quiet) or capture-history.
            if (!capture && ply < MAX_PLY) {
                update_killer(ply, m);
                update_history(pos.stm(), m, depth);
                update_cont_history(ply, moving_pt, m.to(), depth);
                // History malus on earlier quiets that didn't cut.
                for (int q = 0; q < quiets_searched - 1; ++q) {
                    Move qm = quiets_seen[q];
                    int& h = g_history[pos.stm()][qm.from()][qm.to()];
                    h -= depth * depth;
                    if (h < -100000) h = -100000;
                }
            } else if (capture) {
                const PieceType v = victim_type(pos, m);
                update_cap_history(moving_pt, m.to(), v, depth);
            }
            break;
        }
    }

    // --- TT store --- (skip during singular-extension verification search)
    if (!in_singular_search) {
        TT::Bound bound = TT::BOUND_EXACT;
        if (best <= orig_alpha) bound = TT::BOUND_UPPER;
        else if (best >= beta)  bound = TT::BOUND_LOWER;
        TT::store(pos.hash(), best_move, score_to_tt(best, ply), depth, bound);
    }

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
    if (g_is_helper) return;  // only the main thread prints
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
    init_lmr_table();
    clear_killers_history();
    if (NNUE::is_loaded() && NNUE::is_enabled()) NNUE::refresh(pos);

    // Time-check granularity (Session 42): tighter on short budgets, looser on long.
    // Mask values must be 2^k − 1 so `(nodes & mask) == 0` fires every 2^k nodes.
    if (limits.infinite || limits.hard_ms <= 0) {
        g_time_check_mask = 8191;  // ~8K nodes between checks (rare time pressure)
    } else if (limits.hard_ms <= 50) {
        g_time_check_mask = 255;   // very tight: check every 256 nodes
    } else if (limits.hard_ms <= 500) {
        g_time_check_mask = 1023;  // medium: 1K nodes
    } else {
        g_time_check_mask = 2047;  // long: 2K nodes (default)
    }

    SearchInfo result;
    result.best_move = MOVE_NULL;

    int prev_score = 0;
    Move pv[64];
    int  pv_len = 0;

    // --- Time-management state (Session 32) ---
    int  best_move_changes  = 0;
    Move prev_best_move     = MOVE_NULL;
    int  prev_iter_time     = 0;

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        // Aspiration window (Sessions 19 + 40): start tight from depth 4, widen
        // adaptively on fails. Smaller initial delta + 1.5x growth (instead of 2x)
        // gives faster convergence in the common case where score is near prev.
        int alpha = -VALUE_INF, beta = VALUE_INF;
        int delta = 18;            // tighter than before (was 25)
        if (depth >= 4) {
            alpha = prev_score - delta;
            beta  = prev_score + delta;
        }
        int aspiration_fails = 0;

        int score;
        while (true) {
            score = negamax(pos, depth, alpha, beta, 0, 0, false, MOVE_NULL, pv, &pv_len);
            if (g_aborted) break;
            if (score <= alpha) {            // fail low — widen alpha
                beta  = (alpha + beta) / 2;
                alpha = std::max(score - delta, -VALUE_INF);
                ++aspiration_fails;
                // Adaptive growth: small fails grow 1.5x, repeated fails grow 2x.
                delta = (aspiration_fails >= 3) ? delta * 2 : delta + delta / 2;
            } else if (score >= beta) {      // fail high — widen beta
                beta = std::min(score + delta,  VALUE_INF);
                ++aspiration_fails;
                delta = (aspiration_fails >= 3) ? delta * 2 : delta + delta / 2;
            } else {
                break;
            }
        }

        if (g_aborted) break;

        const int iter_time = elapsed_ms() - (result.time_ms);

        result.depth     = depth;
        result.score     = score;
        Move new_best    = (pv_len > 0) ? pv[0] : MOVE_NULL;
        if (depth > 1 && new_best != prev_best_move) ++best_move_changes;
        result.best_move = new_best;
        result.pv_len    = pv_len;
        for (int i = 0; i < pv_len; ++i) result.pv[i] = pv[i];
        result.nodes   = g_nodes;
        result.time_ms = elapsed_ms();
        prev_score      = score;
        prev_best_move  = new_best;
        prev_iter_time  = iter_time > 0 ? iter_time : 1;

        print_info_line(result);

        // Mate-distance shortcut: if we found a forced mate, no need to search deeper.
        if (std::abs(score) >= VALUE_MATE_IN_MAX) break;

        // --- Time management (Session 32) ---
        if (limits.infinite || limits.soft_ms <= 0) continue;

        const int elapsed = elapsed_ms();
        // Predict next iteration as ~3x current iteration (conservative).
        // Stability multipliers:
        //  - best_move_changes: each change in the last few iterations widens budget by ~25%.
        //  - if best move is unchanged AND score didn't move much, tighten budget by 30%.
        double budget_mult = 1.0;
        if (best_move_changes > 0) budget_mult += 0.25 * std::min(best_move_changes, 4);
        if (best_move_changes == 0 && std::abs(score - prev_score) < 30) budget_mult *= 0.70;
        const int adjusted_soft = int(limits.soft_ms * budget_mult);

        // Hard cap: if we'd exceed adjusted_soft after next iter, stop now.
        const int predicted_next = prev_iter_time * 3;
        if (elapsed + predicted_next > adjusted_soft) break;
        if (elapsed >= adjusted_soft) break;
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

// --- LazySMP (Sessions 33+34) ---
// Spawn N-1 helper threads + use the calling thread as the main worker.
// All threads share TT and the time budget; each has its own per-thread
// search state (killers/history/NNUE accumulator) via thread_local.
SearchInfo search_smp(Position& pos, const SearchLimits& limits, int n_threads) {
    if (n_threads <= 1) return search(pos, limits);

    g_start = std::chrono::steady_clock::now();
    g_stop.store(false);
    g_use_time = !limits.infinite && limits.hard_ms > 0;
    g_hard_ms  = limits.hard_ms;
    TT::new_search();

    std::vector<std::thread> helpers;
    helpers.reserve(n_threads - 1);
    std::vector<Position> helper_positions(n_threads - 1, pos);

    // Helper threads: same search but suppress info output to keep stdout clean.
    // (We mark this by passing a sentinel through SearchLimits — the simplest
    // way is to silently run search() and discard their output via a flag.)
    for (int i = 0; i < n_threads - 1; ++i) {
        helpers.emplace_back([&, i]() {
            g_is_helper = true;
            search(helper_positions[i], limits);
            g_is_helper = false;
        });
    }

    // Main thread also runs search.
    SearchInfo main_result = search(pos, limits);

    // Signal helpers to stop and join.
    g_stop.store(true);
    for (auto& th : helpers) th.join();

    return main_result;
}

}  // namespace gungnir
