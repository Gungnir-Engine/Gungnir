// Gungnir — Stockfish HalfKAv2_hm NNUE loader + forward pass.
// Loads SF17 small-net (.nnue) and computes the same eval Stockfish does.
//
// Ported from the JS implementation in chess.html (Sessions 12–13). All math
// is integer (int32) — matches SF's portable (non-SIMD) reference path so the
// outputs agree to within ±2 cp on every position.
//
// Key differences vs the JS port: Gungnir uses Stockfish's square convention
// directly (a1=0, h8=63), so all the `^ 56` flips disappear.

#pragma once

#include "move.h"
#include "position.h"

#include <cstddef>
#include <string>

namespace gungnir {
namespace NNUE {

// Load a .nnue file. Returns true on success. Subsequent loads replace.
bool load(const std::string& path);

bool is_loaded();
void set_enabled(bool on);
bool is_enabled();

// --- Incremental accumulator (Session 24) ---
// `refresh` rebuilds both perspectives from scratch — call it whenever the
// position is set fresh (after FEN parse or any non-incremental change).
void refresh(const Position& pos);

// Call AFTER pos.make_move(m) but BEFORE recursing. Pushes a new accumulator
// entry derived from the previous one + this move's feature delta. King moves
// trigger a full rebuild of that side's accumulator (because the king bucket
// and orient changes affect every feature on that perspective).
void on_make(const Position& pos, Move m);

// Call AFTER pos.unmake_move(m) (or before — order doesn't matter, this just
// pops the accumulator stack).
void on_unmake();

// Null-move support: pushes an unchanged copy on the stack.
void on_null_make();
void on_null_unmake();

// Forward pass — returns Stockfish-internal centipawns from STM POV.
// Caller should only call this when is_loaded() && is_enabled().
int evaluate(const Position& pos);

// Diagnostics
const std::string& description();
unsigned long file_version();
unsigned long arch_hash();

// Returns active HalfKAv2_hm feature indices for one perspective (0=white, 1=black).
// Up to 32 entries (one per piece on the board, kings included).
int features(const Position& pos, int perspective, int* out);

}  // namespace NNUE
}  // namespace gungnir
