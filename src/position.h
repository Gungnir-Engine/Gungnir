// Gungnir — Position: full chess game state.
// Represents the board via 12 piece-bitboards + per-color + all-pieces bitboards,
// plus metadata (side to move, castling rights, en passant target, halfmove
// clock, fullmove number). Handles FEN parse/serialize, make/unmake moves,
// and "is king in check" helper.
//
// Zobrist hashing is Session 17.

#pragma once

#include "move.h"
#include "bitboard.h"

#include <string>
#include <string_view>

namespace gungnir {

// Castling rights, bit-packed. Each bit = one side-right combination.
enum CastlingRights : u8 {
    NO_CASTLING  = 0,
    WHITE_OO     = 1 << 0,
    WHITE_OOO    = 1 << 1,
    BLACK_OO     = 1 << 2,
    BLACK_OOO    = 1 << 3,
    ALL_CASTLING = WHITE_OO | WHITE_OOO | BLACK_OO | BLACK_OOO,
};

// StateInfo holds the "reversible" parts of position state needed to undo a move.
// Pushed onto Position's undo stack on make_move, popped on unmake_move.
struct StateInfo {
    u8     castling;
    Square ep_square;       // SQ_NONE if no ep available
    int    halfmove;
    Piece  captured;        // NO_PIECE if the move wasn't a capture
    // (Zobrist hash goes here once we add it in Session 17.)
};

class Position {
public:
    Position() { clear(); }

    void clear();
    void set_startpos();

    // FEN I/O
    bool set_from_fen(std::string_view fen);
    std::string fen() const;
    std::string to_string() const;  // ASCII board with metadata for debugging

    // Move execution
    void make_move(Move m);
    void unmake_move(Move m);

    // Queries
    Color stm() const { return stm_; }
    Piece piece_on(Square s) const { return board_[s]; }
    bool  empty_at(Square s) const { return board_[s] == NO_PIECE; }

    Bitboard pieces() const { return occupancy_[WHITE] | occupancy_[BLACK]; }
    Bitboard pieces(Color c) const { return occupancy_[c]; }
    Bitboard pieces(PieceType pt) const { return by_type_[pt]; }
    Bitboard pieces(Color c, PieceType pt) const { return by_type_[pt] & occupancy_[c]; }

    Square   king_square(Color c) const;

    u8     castling() const { return state_.castling; }
    Square ep_square() const { return state_.ep_square; }
    int    halfmove() const  { return state_.halfmove; }
    int    fullmove() const  { return fullmove_; }

    // True if any piece of color `by` attacks square `s` given the current
    // board occupancy.
    bool square_attacked(Square s, Color by) const;

    // True if the side-to-move's king is in check.
    bool in_check() const;

private:
    // Internal helpers
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

    // Board state
    Piece     board_[SQ_NB];       // piece on each square (NO_PIECE if empty)
    Bitboard  by_type_[PIECE_TYPE_NB];  // per-piece-type bitboard (both colors combined)
    Bitboard  occupancy_[COLOR_NB];     // all pieces of that color

    Color   stm_;
    int     fullmove_;
    StateInfo state_;

    // Undo stack: one StateInfo per made move.
    StateInfo history_[1024];
    int       history_size_;
};

}  // namespace gungnir
