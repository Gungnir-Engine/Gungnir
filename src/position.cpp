#include "position.h"

#include "attacks.h"

#include <cassert>
#include <sstream>

namespace gungnir {

namespace {

// Castling-rights update on move: mask to AND with when the given square is
// the source or destination of any move. E.g., moving from/to a1 clears
// white's queenside right (WHITE_OOO).
constexpr u8 kCastlingLostMask[SQ_NB] = {
    // rank 1 (a1..h1)
    u8(~WHITE_OOO), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    u8(~(WHITE_OO | WHITE_OOO)), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(~WHITE_OO),
    // ranks 2..7 untouched
    u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    // rank 8 (a8..h8)
    u8(~BLACK_OOO), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(ALL_CASTLING),
    u8(~(BLACK_OO | BLACK_OOO)), u8(ALL_CASTLING), u8(ALL_CASTLING), u8(~BLACK_OO),
};

Piece piece_from_char(char c) {
    switch (c) {
        case 'P': return W_PAWN;   case 'p': return B_PAWN;
        case 'N': return W_KNIGHT; case 'n': return B_KNIGHT;
        case 'B': return W_BISHOP; case 'b': return B_BISHOP;
        case 'R': return W_ROOK;   case 'r': return B_ROOK;
        case 'Q': return W_QUEEN;  case 'q': return B_QUEEN;
        case 'K': return W_KING;   case 'k': return B_KING;
        default:  return NO_PIECE;
    }
}

char char_from_piece(Piece p) {
    static constexpr char kChars[] = {
        ' ',
        'P', 'N', 'B', 'R', 'Q', 'K', ' ',
        ' ',
        'p', 'n', 'b', 'r', 'q', 'k', ' ',
    };
    return kChars[int(p)];
}

}  // namespace

void Position::clear() {
    for (int i = 0; i < SQ_NB; ++i) board_[i] = NO_PIECE;
    for (int i = 0; i < PIECE_TYPE_NB; ++i) by_type_[i] = 0;
    occupancy_[WHITE] = occupancy_[BLACK] = 0;
    stm_ = WHITE;
    fullmove_ = 1;
    state_ = StateInfo{0, NO_CASTLING, SQ_NONE, 0, NO_PIECE};
    history_size_ = 0;
}

void Position::put_piece(Piece p, Square s) {
    assert(board_[s] == NO_PIECE);
    board_[s] = p;
    const Bitboard bb = square_bb(s);
    by_type_[type_of(p)] |= bb;
    occupancy_[color_of(p)] |= bb;
}

void Position::remove_piece(Square s) {
    const Piece p = board_[s];
    assert(p != NO_PIECE);
    const Bitboard bb = square_bb(s);
    by_type_[type_of(p)] ^= bb;
    occupancy_[color_of(p)] ^= bb;
    board_[s] = NO_PIECE;
}

void Position::move_piece(Square from, Square to) {
    const Piece p = board_[from];
    assert(p != NO_PIECE);
    assert(board_[to] == NO_PIECE);
    const Bitboard mask = square_bb(from) | square_bb(to);
    by_type_[type_of(p)] ^= mask;
    occupancy_[color_of(p)] ^= mask;
    board_[from] = NO_PIECE;
    board_[to] = p;
}

Square Position::king_square(Color c) const {
    const Bitboard k = by_type_[KING] & occupancy_[c];
    assert(k != 0);
    return lsb(k);
}

bool Position::square_attacked(Square s, Color by) const {
    const Bitboard occ = pieces();

    // Pawn: we use PawnAttacks[opposite-of-attacker][s] to get the squares
    // from which a pawn of color `by` would attack `s`.
    if (pawn_attacks(~by, s) & pieces(by, PAWN)) return true;

    if (knight_attacks(s) & pieces(by, KNIGHT)) return true;
    if (king_attacks(s)   & pieces(by, KING))   return true;

    const Bitboard bq = pieces(by, BISHOP) | pieces(by, QUEEN);
    if (bishop_attacks(s, occ) & bq) return true;

    const Bitboard rq = pieces(by, ROOK) | pieces(by, QUEEN);
    if (rook_attacks(s, occ) & rq) return true;

    return false;
}

bool Position::in_check() const {
    return square_attacked(king_square(stm_), ~stm_);
}

void Position::set_startpos() {
    set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Position::set_from_fen(std::string_view fen) {
    clear();

    // FEN has 6 fields: board, stm, castling, ep, halfmove, fullmove
    // Minimal tolerant parse — doesn't reject ill-formed FENs aggressively.
    std::istringstream iss{std::string(fen)};
    std::string board_str, stm_str, castling_str, ep_str;
    int hm = 0, fm = 1;
    iss >> board_str >> stm_str >> castling_str >> ep_str;
    if (iss) iss >> hm;
    if (iss) iss >> fm;

    // Board: rank 8 first in the FEN, ranks separated by '/'.
    int r = 7, f = 0;
    for (char c : board_str) {
        if (c == '/') {
            if (f != 8) return false;
            --r;
            f = 0;
        } else if (c >= '1' && c <= '8') {
            f += (c - '0');
        } else {
            const Piece p = piece_from_char(c);
            if (p == NO_PIECE) return false;
            if (f < 0 || f > 7 || r < 0 || r > 7) return false;
            put_piece(p, make_square(File(f), Rank(r)));
            ++f;
        }
    }

    stm_ = (stm_str == "w") ? WHITE : BLACK;

    state_.castling = NO_CASTLING;
    if (castling_str != "-") {
        for (char c : castling_str) {
            switch (c) {
                case 'K': state_.castling |= WHITE_OO;  break;
                case 'Q': state_.castling |= WHITE_OOO; break;
                case 'k': state_.castling |= BLACK_OO;  break;
                case 'q': state_.castling |= BLACK_OOO; break;
                default: break;
            }
        }
    }

    state_.ep_square = SQ_NONE;
    if (ep_str.size() >= 2 && ep_str != "-") {
        const int ff = ep_str[0] - 'a';
        const int rr = ep_str[1] - '1';
        if (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) {
            state_.ep_square = make_square(File(ff), Rank(rr));
        }
    }

    state_.halfmove = hm;
    state_.captured = NO_PIECE;
    fullmove_ = fm;
    history_size_ = 0;
    state_.hash = compute_hash_from_scratch();
    return true;
}

u64 Position::compute_hash_from_scratch() const {
    u64 h = 0;
    for (Square s = SQ_A1; s < SQ_NONE; ++s) {
        const Piece p = board_[s];
        if (p != NO_PIECE) h ^= Zobrist::psq[p][s];
    }
    h ^= Zobrist::castling[state_.castling];
    if (state_.ep_square != SQ_NONE) {
        h ^= Zobrist::enpassant[file_of(state_.ep_square)];
    }
    if (stm_ == BLACK) h ^= Zobrist::side;
    return h;
}

bool Position::is_threefold_repetition() const {
    // Two prior occurrences (plus the current one) = threefold.
    // Positions can only repeat with the same side to move, so step by 2 plies.
    // The halfmove clock resets on captures + pawn moves, both of which
    // permanently alter material/structure — so no repetition can reach further
    // back than `halfmove` plies.
    int matches = 0;
    for (int back = 4; back <= state_.halfmove && back <= history_size_; back += 2) {
        if (history_[history_size_ - back].hash == state_.hash) {
            if (++matches >= 2) return true;
        }
    }
    return false;
}

std::string Position::fen() const {
    std::ostringstream os;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board_[make_square(File(f), Rank(r))];
            if (p == NO_PIECE) { ++empty; continue; }
            if (empty) { os << empty; empty = 0; }
            os << char_from_piece(p);
        }
        if (empty) os << empty;
        if (r > 0) os << '/';
    }
    os << ' ' << (stm_ == WHITE ? 'w' : 'b') << ' ';
    if (state_.castling == NO_CASTLING) os << '-';
    else {
        if (state_.castling & WHITE_OO)  os << 'K';
        if (state_.castling & WHITE_OOO) os << 'Q';
        if (state_.castling & BLACK_OO)  os << 'k';
        if (state_.castling & BLACK_OOO) os << 'q';
    }
    os << ' ';
    if (state_.ep_square == SQ_NONE) os << '-';
    else os << char('a' + file_of(state_.ep_square)) << char('1' + rank_of(state_.ep_square));
    os << ' ' << state_.halfmove << ' ' << fullmove_;
    return os.str();
}

std::string Position::to_string() const {
    std::ostringstream os;
    os << "+---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            Piece p = board_[make_square(File(f), Rank(r))];
            os << "| " << (p == NO_PIECE ? ' ' : char_from_piece(p)) << " ";
        }
        os << "| " << (r + 1) << "\n+---+---+---+---+---+---+---+---+\n";
    }
    os << "  a   b   c   d   e   f   g   h\n";
    os << "FEN: " << fen() << "\n";
    return os.str();
}

void Position::make_move(Move m) {
    assert(history_size_ < 1024);
    history_[history_size_++] = state_;  // save reversible state for unmake

    const Square from = m.from();
    const Square to = m.to();
    const MoveType mt = m.type();
    const Piece moving = board_[from];
    const PieceType moving_type = type_of(moving);
    const Color us = stm_;
    const Color them = ~us;

    Piece captured = NO_PIECE;

    // --- Hash: XOR out the OLD ep + castling bits before any state mutation.
    // We'll XOR in the NEW values after the state is updated below, and toggle
    // `side` once at the end. Piece-on-square XORs happen alongside their
    // respective move-type branches.
    u64 h = state_.hash;
    if (state_.ep_square != SQ_NONE) {
        h ^= Zobrist::enpassant[file_of(state_.ep_square)];
    }
    h ^= Zobrist::castling[state_.castling];

    if (mt == MT_EN_PASSANT) {
        // Pawn captures the pawn behind the ep target square.
        const Square cap_sq = Square(int(to) + (us == WHITE ? -8 : 8));
        captured = board_[cap_sq];
        remove_piece(cap_sq);
        move_piece(from, to);
        h ^= Zobrist::psq[moving][from] ^ Zobrist::psq[moving][to];
        h ^= Zobrist::psq[captured][cap_sq];
    } else if (mt == MT_CASTLING) {
        // to square is the king's destination; we also move the rook.
        // Determine rook squares by the king's final square.
        Square rook_from, rook_to;
        if (to == SQ_G1) { rook_from = SQ_H1; rook_to = SQ_F1; }
        else if (to == SQ_C1) { rook_from = SQ_A1; rook_to = SQ_D1; }
        else if (to == SQ_G8) { rook_from = SQ_H8; rook_to = SQ_F8; }
        else { rook_from = SQ_A8; rook_to = SQ_D8; }
        const Piece rook = board_[rook_from];
        move_piece(from, to);
        move_piece(rook_from, rook_to);
        h ^= Zobrist::psq[moving][from] ^ Zobrist::psq[moving][to];
        h ^= Zobrist::psq[rook][rook_from] ^ Zobrist::psq[rook][rook_to];
    } else {
        // Normal move or promotion.
        if (board_[to] != NO_PIECE) {
            captured = board_[to];
            remove_piece(to);
            h ^= Zobrist::psq[captured][to];
        }
        if (mt == MT_PROMOTION) {
            const Piece promoted = make_piece(us, m.promo_type());
            remove_piece(from);
            put_piece(promoted, to);
            h ^= Zobrist::psq[moving][from];
            h ^= Zobrist::psq[promoted][to];
        } else {
            move_piece(from, to);
            h ^= Zobrist::psq[moving][from] ^ Zobrist::psq[moving][to];
        }
    }

    // Update castling rights: any touch of the rook/king squares.
    state_.castling &= kCastlingLostMask[from] & kCastlingLostMask[to];

    // Update en passant: set if this was a double pawn push.
    state_.ep_square = SQ_NONE;
    if (moving_type == PAWN && std::abs(int(to) - int(from)) == 16) {
        state_.ep_square = Square((int(from) + int(to)) / 2);
    }

    // Halfmove clock
    if (moving_type == PAWN || captured != NO_PIECE) {
        state_.halfmove = 0;
    } else {
        state_.halfmove += 1;
    }

    state_.captured = captured;

    // --- Hash: XOR in the NEW ep + castling bits, and toggle side.
    h ^= Zobrist::castling[state_.castling];
    if (state_.ep_square != SQ_NONE) {
        h ^= Zobrist::enpassant[file_of(state_.ep_square)];
    }
    h ^= Zobrist::side;
    state_.hash = h;

    // Fullmove: increment after black's move.
    if (us == BLACK) ++fullmove_;

    stm_ = them;
}

void Position::make_null_move() {
    assert(history_size_ < 1024);
    assert(!in_check());  // null-move illegal if in check
    history_[history_size_++] = state_;

    u64 h = state_.hash;
    if (state_.ep_square != SQ_NONE) {
        h ^= Zobrist::enpassant[file_of(state_.ep_square)];
    }
    h ^= Zobrist::side;

    state_.ep_square = SQ_NONE;
    state_.captured  = NO_PIECE;
    state_.halfmove += 1;
    state_.hash      = h;

    if (stm_ == BLACK) ++fullmove_;
    stm_ = ~stm_;
}

void Position::unmake_null_move() {
    stm_ = ~stm_;
    if (stm_ == BLACK) --fullmove_;
    state_ = history_[--history_size_];
}

void Position::unmake_move(Move m) {
    const Square from = m.from();
    const Square to = m.to();
    const MoveType mt = m.type();
    const Color us = ~stm_;   // side that moved
    stm_ = us;

    const StateInfo saved = history_[--history_size_];

    if (mt == MT_EN_PASSANT) {
        move_piece(to, from);  // pawn back
        const Square cap_sq = Square(int(to) + (us == WHITE ? -8 : 8));
        put_piece(state_.captured, cap_sq);
    } else if (mt == MT_CASTLING) {
        Square rook_from, rook_to;
        if (to == SQ_G1) { rook_from = SQ_H1; rook_to = SQ_F1; }
        else if (to == SQ_C1) { rook_from = SQ_A1; rook_to = SQ_D1; }
        else if (to == SQ_G8) { rook_from = SQ_H8; rook_to = SQ_F8; }
        else { rook_from = SQ_A8; rook_to = SQ_D8; }
        move_piece(to, from);
        move_piece(rook_to, rook_from);
    } else {
        if (mt == MT_PROMOTION) {
            remove_piece(to);
            put_piece(make_piece(us, PAWN), from);
        } else {
            move_piece(to, from);
        }
        if (state_.captured != NO_PIECE) {
            put_piece(state_.captured, to);
        }
    }

    state_ = saved;
    if (us == BLACK) --fullmove_;
}

}  // namespace gungnir
