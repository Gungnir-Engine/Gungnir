// Gungnir — 16-bit packed Move type.
// Layout:
//   bits  0- 5: from square (0..63)
//   bits  6-11: to square   (0..63)
//   bits 12-13: promotion piece type relative to KNIGHT
//                 0 = KNIGHT, 1 = BISHOP, 2 = ROOK, 3 = QUEEN
//              (only meaningful if move_type == PROMOTION)
//   bits 14-15: move type
//                 0 = NORMAL (quiet or regular capture, no promotion/EP/castle)
//                 1 = PROMOTION
//                 2 = EN_PASSANT
//                 3 = CASTLING
//
// 16 bits total. Fits in a u16 so a MoveList can be a small array.
// Distinguishing "captures" is done at use-site by looking at the target square;
// storing capture info in the Move itself isn't necessary.

#pragma once

#include "types.h"

namespace gungnir {

enum MoveType : u32 {
    MT_NORMAL     = 0,
    MT_PROMOTION  = 1,
    MT_EN_PASSANT = 2,
    MT_CASTLING   = 3,
};

class Move {
public:
    constexpr Move() : data_(0) {}
    constexpr explicit Move(u16 d) : data_(d) {}

    static constexpr Move make(Square from, Square to,
                               MoveType mt = MT_NORMAL,
                               PieceType promo = KNIGHT) {
        const u16 promo_bits = u16((int(promo) - int(KNIGHT)) & 0x3);
        return Move(u16(int(from))
                  | u16(int(to) << 6)
                  | u16(promo_bits << 12)
                  | u16(int(mt) << 14));
    }

    constexpr Square    from()      const { return Square(data_ & 0x3F); }
    constexpr Square    to()        const { return Square((data_ >> 6) & 0x3F); }
    constexpr MoveType  type()      const { return MoveType((data_ >> 14) & 0x3); }
    constexpr PieceType promo_type() const { return PieceType(KNIGHT + ((data_ >> 12) & 0x3)); }

    constexpr u16  raw()     const { return data_; }
    constexpr bool is_null() const { return data_ == 0; }

    constexpr bool operator==(Move o) const { return data_ == o.data_; }
    constexpr bool operator!=(Move o) const { return data_ != o.data_; }

private:
    u16 data_;
};

constexpr Move MOVE_NULL = Move(0);

}  // namespace gungnir
