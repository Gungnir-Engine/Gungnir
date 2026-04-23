#include "notation.h"

#include "movegen.h"

namespace gungnir {

std::string move_to_uci(Move m) {
    auto sq_name = [](Square s) {
        std::string out;
        out += char('a' + file_of(s));
        out += char('1' + rank_of(s));
        return out;
    };
    std::string r = sq_name(m.from()) + sq_name(m.to());
    if (m.type() == MT_PROMOTION) {
        switch (m.promo_type()) {
            case KNIGHT: r += 'n'; break;
            case BISHOP: r += 'b'; break;
            case ROOK:   r += 'r'; break;
            case QUEEN:  r += 'q'; break;
            default: break;
        }
    }
    return r;
}

Move parse_uci_move(Position& pos, std::string_view uci) {
    if (uci.size() < 4) return MOVE_NULL;
    const int ff = uci[0] - 'a';
    const int fr = uci[1] - '1';
    const int tf = uci[2] - 'a';
    const int tr = uci[3] - '1';
    if (ff < 0 || ff > 7 || fr < 0 || fr > 7 || tf < 0 || tf > 7 || tr < 0 || tr > 7) {
        return MOVE_NULL;
    }
    const Square from = make_square(File(ff), Rank(fr));
    const Square to   = make_square(File(tf), Rank(tr));
    PieceType promo = NO_PIECE_TYPE;
    if (uci.size() >= 5) {
        switch (uci[4]) {
            case 'n': promo = KNIGHT; break;
            case 'b': promo = BISHOP; break;
            case 'r': promo = ROOK;   break;
            case 'q': promo = QUEEN;  break;
            default: break;
        }
    }

    MoveList list;
    generate_legal(pos, list);
    for (int i = 0; i < list.size; ++i) {
        const Move m = list.moves[i];
        if (m.from() != from || m.to() != to) continue;
        if (m.type() == MT_PROMOTION) {
            if (m.promo_type() == promo) return m;
        } else {
            return m;
        }
    }
    return MOVE_NULL;
}

}  // namespace gungnir
