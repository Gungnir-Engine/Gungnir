#include "movegen.h"

#include "attacks.h"
#include "bitboard.h"
#include "position.h"

namespace gungnir {

namespace {

// Add the four promotion moves for a pawn landing on `to` from `from`.
inline void add_promotions(MoveList& list, Square from, Square to) {
    list.add(Move::make(from, to, MT_PROMOTION, KNIGHT));
    list.add(Move::make(from, to, MT_PROMOTION, BISHOP));
    list.add(Move::make(from, to, MT_PROMOTION, ROOK));
    list.add(Move::make(from, to, MT_PROMOTION, QUEEN));
}

template <Color Us>
void gen_pawn_moves(const Position& pos, MoveList& list) {
    constexpr Color Them = (Us == WHITE) ? BLACK : WHITE;
    constexpr Direction Up     = (Us == WHITE) ? NORTH      : SOUTH;
    constexpr Direction UpLeft = (Us == WHITE) ? NORTH_WEST : SOUTH_EAST;
    constexpr Direction UpRight= (Us == WHITE) ? NORTH_EAST : SOUTH_WEST;
    constexpr Bitboard Rank3    = (Us == WHITE) ? BB_RANK_3 : BB_RANK_6;
    constexpr Bitboard Rank8    = (Us == WHITE) ? BB_RANK_8 : BB_RANK_1;

    const Bitboard pawns = pos.pieces(Us, PAWN);
    const Bitboard empty = ~pos.pieces();
    const Bitboard enemy = pos.pieces(Them);

    // Single push (all empty squares ahead)
    Bitboard push1 = shift<Up>(pawns) & empty;
    // Double push from starting rank
    Bitboard push2 = shift<Up>(push1 & Rank3) & empty;

    // Split into promotion / non-promotion
    Bitboard push1_promo   = push1 & Rank8;
    Bitboard push1_nopromo = push1 & ~Rank8;

    while (push1_nopromo) {
        Square to = pop_lsb(push1_nopromo);
        Square from = Square(int(to) - int(Up));
        list.add(Move::make(from, to));
    }
    while (push1_promo) {
        Square to = pop_lsb(push1_promo);
        Square from = Square(int(to) - int(Up));
        add_promotions(list, from, to);
    }
    while (push2) {
        Square to = pop_lsb(push2);
        Square from = Square(int(to) - 2 * int(Up));
        list.add(Move::make(from, to));
    }

    // Diagonal captures (both directions)
    Bitboard capL = shift<UpLeft>(pawns)  & enemy;
    Bitboard capR = shift<UpRight>(pawns) & enemy;

    Bitboard capL_promo = capL & Rank8, capL_nopromo = capL & ~Rank8;
    Bitboard capR_promo = capR & Rank8, capR_nopromo = capR & ~Rank8;

    while (capL_nopromo) {
        Square to = pop_lsb(capL_nopromo);
        Square from = Square(int(to) - int(UpLeft));
        list.add(Move::make(from, to));
    }
    while (capR_nopromo) {
        Square to = pop_lsb(capR_nopromo);
        Square from = Square(int(to) - int(UpRight));
        list.add(Move::make(from, to));
    }
    while (capL_promo) {
        Square to = pop_lsb(capL_promo);
        Square from = Square(int(to) - int(UpLeft));
        add_promotions(list, from, to);
    }
    while (capR_promo) {
        Square to = pop_lsb(capR_promo);
        Square from = Square(int(to) - int(UpRight));
        add_promotions(list, from, to);
    }

    // En passant
    if (pos.ep_square() != SQ_NONE) {
        // Pawns of color Us that can capture onto the ep square are exactly
        // those attacking it from the opposite color's perspective.
        Bitboard epers = pawn_attacks(Them, pos.ep_square()) & pawns;
        while (epers) {
            Square from = pop_lsb(epers);
            list.add(Move::make(from, pos.ep_square(), MT_EN_PASSANT));
        }
    }
}

// Knight, Bishop, Rook, Queen: generic "attacks excluding own pieces".
void gen_piece_moves(const Position& pos, PieceType pt, MoveList& list) {
    const Color us = pos.stm();
    const Bitboard occ = pos.pieces();
    const Bitboard own = pos.pieces(us);

    Bitboard from_bb = pos.pieces(us, pt);
    while (from_bb) {
        Square from = pop_lsb(from_bb);
        Bitboard targets = piece_attacks(pt, from, occ) & ~own;
        while (targets) {
            Square to = pop_lsb(targets);
            list.add(Move::make(from, to));
        }
    }
}

// King: non-castle moves plus castling if available and path clear/safe.
template <Color Us>
void gen_king_moves(const Position& pos, MoveList& list) {
    const Square king = pos.king_square(Us);
    const Bitboard own = pos.pieces(Us);

    // Regular king moves (castling-squares safety not checked here; those
    // are filtered at legality stage via king-not-in-check test).
    Bitboard targets = king_attacks(king) & ~own;
    while (targets) {
        Square to = pop_lsb(targets);
        list.add(Move::make(king, to));
    }

    // Castling. We verify: rights available, path empty, king not currently
    // in check, king doesn't pass through attacked square, king doesn't land
    // on attacked square. The "doesn't land on attacked" is redundant with
    // the legal filter but we check upfront to avoid generating obviously
    // illegal castlings.
    constexpr Square KingFrom   = (Us == WHITE) ? SQ_E1 : SQ_E8;
    constexpr Square KingToOO   = (Us == WHITE) ? SQ_G1 : SQ_G8;
    constexpr Square KingToOOO  = (Us == WHITE) ? SQ_C1 : SQ_C8;
    constexpr Square RookFromOO = (Us == WHITE) ? SQ_H1 : SQ_H8;
    constexpr Square RookFromOOOsq = (Us == WHITE) ? SQ_A1 : SQ_A8;
    constexpr u8 OO_right  = (Us == WHITE) ? WHITE_OO  : BLACK_OO;
    constexpr u8 OOO_right = (Us == WHITE) ? WHITE_OOO : BLACK_OOO;
    constexpr Color Them = (Us == WHITE) ? BLACK : WHITE;

    // King must be on its starting square (in standard chess, redundant if
    // rights invariants are maintained, but cheap safety).
    if (king != KingFrom) return;
    if (pos.square_attacked(king, Them)) return;  // can't castle out of check

    // King-side
    if (pos.castling() & OO_right) {
        // Squares the king passes / lands on:
        Square f_sq = (Us == WHITE) ? SQ_F1 : SQ_F8;
        Square g_sq = KingToOO;
        if (pos.empty_at(f_sq) && pos.empty_at(g_sq)
            && !pos.square_attacked(f_sq, Them)
            && !pos.square_attacked(g_sq, Them)) {
            (void)RookFromOO;  // silence unused
            list.add(Move::make(king, g_sq, MT_CASTLING));
        }
    }
    // Queen-side
    if (pos.castling() & OOO_right) {
        Square b_sq = (Us == WHITE) ? SQ_B1 : SQ_B8;
        Square c_sq = KingToOOO;
        Square d_sq = (Us == WHITE) ? SQ_D1 : SQ_D8;
        // d and c must be empty AND safe; b must be empty (not safe — king doesn't go there)
        if (pos.empty_at(b_sq) && pos.empty_at(c_sq) && pos.empty_at(d_sq)
            && !pos.square_attacked(c_sq, Them)
            && !pos.square_attacked(d_sq, Them)) {
            (void)RookFromOOOsq;
            list.add(Move::make(king, c_sq, MT_CASTLING));
        }
    }
}

}  // anonymous namespace

void generate_pseudo_legal(const Position& pos, MoveList& list) {
    list.size = 0;
    if (pos.stm() == WHITE) gen_pawn_moves<WHITE>(pos, list);
    else                     gen_pawn_moves<BLACK>(pos, list);

    gen_piece_moves(pos, KNIGHT, list);
    gen_piece_moves(pos, BISHOP, list);
    gen_piece_moves(pos, ROOK,   list);
    gen_piece_moves(pos, QUEEN,  list);

    if (pos.stm() == WHITE) gen_king_moves<WHITE>(pos, list);
    else                     gen_king_moves<BLACK>(pos, list);
}

void generate_legal(Position& pos, MoveList& list) {
    MoveList pseudo;
    generate_pseudo_legal(pos, pseudo);
    list.size = 0;
    for (int i = 0; i < pseudo.size; ++i) {
        Move m = pseudo.moves[i];
        pos.make_move(m);
        // After make_move, pos.stm() = opponent. Ask: is our (just-moved) king attacked?
        Color us = Color(pos.stm() ^ 1);
        if (!pos.square_attacked(pos.king_square(us), pos.stm())) {
            list.add(m);
        }
        pos.unmake_move(m);
    }
}

}  // namespace gungnir
