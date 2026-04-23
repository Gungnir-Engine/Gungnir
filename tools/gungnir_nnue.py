"""
gungnir_nnue.py — HalfKAv2_hm feature extraction + model definition
matching Gungnir's C++ forward pass byte-for-byte.

This is the shared library that tools/train_halfka.py will consume.

Constants, tables, and architecture exactly mirror src/nnue.cpp's small-net
path. If anything here diverges, the trained .nnue will load but compute
wrong evals.
"""
import numpy as np

# ============================================================================
# Constants (match src/nnue.cpp)
# ============================================================================

FT_INPUT_DIM   = 22528           # HalfKAv2_hm
FT_OUTPUT_DIM  = 128             # small net
PSQT_BUCKETS   = 8
LAYER_STACKS   = 8
L2             = 15              # fc_0 outputs used (16th is skip)
FC0_OUT        = L2 + 1          # 16
FC1_IN_PAD     = 32
FC1_OUT        = 32
FC2_IN_PAD     = 32

# Piece codes (Gungnir convention — bit 3 = color)
NO_PIECE = 0
W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING = 1, 2, 3, 4, 5, 6
B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING = 9, 10, 11, 12, 13, 14

PS_NB        = 11 * 64           # 704
PS_W_PAWN    = 0  * 64
PS_B_PAWN    = 1  * 64
PS_W_KNIGHT  = 2  * 64
PS_B_KNIGHT  = 3  * 64
PS_W_BISHOP  = 4  * 64
PS_B_BISHOP  = 5  * 64
PS_W_ROOK    = 6  * 64
PS_B_ROOK    = 7  * 64
PS_W_QUEEN   = 8  * 64
PS_B_QUEEN   = 9  * 64
PS_KING      = 10 * 64

# PieceSquareIndex[perspective][piece_code 0..15]
PIECE_SQUARE_INDEX = np.zeros((2, 16), dtype=np.int64)
PIECE_SQUARE_INDEX[0] = [
    0,
    PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, 0,
    0,
    PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, 0,
]
PIECE_SQUARE_INDEX[1] = [
    0,
    PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, 0,
    0,
    PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, 0,
]

# Orient: white a-d files -> XOR 7 (horizontal flip), e-h -> 0 (identity).
#         black a-d -> XOR 63 (flip both), e-h -> XOR 56 (flip rank).
ORIENT = np.zeros((2, 64), dtype=np.int64)
for r in range(8):
    for f in range(8):
        sq = r * 8 + f
        ORIENT[0, sq] = 7 if f < 4 else 0
        ORIENT[1, sq] = 63 if f < 4 else 56

# King buckets (match src/nnue.cpp tables exactly).
_KING_BUCKETS_WHITE = np.array([
    [28, 29, 30, 31, 31, 30, 29, 28],  # rank 1 (white home)
    [24, 25, 26, 27, 27, 26, 25, 24],
    [20, 21, 22, 23, 23, 22, 21, 20],
    [16, 17, 18, 19, 19, 18, 17, 16],
    [12, 13, 14, 15, 15, 14, 13, 12],
    [ 8,  9, 10, 11, 11, 10,  9,  8],
    [ 4,  5,  6,  7,  7,  6,  5,  4],
    [ 0,  1,  2,  3,  3,  2,  1,  0],
], dtype=np.int64)
_KING_BUCKETS_BLACK = np.array([
    [ 0,  1,  2,  3,  3,  2,  1,  0],
    [ 4,  5,  6,  7,  7,  6,  5,  4],
    [ 8,  9, 10, 11, 11, 10,  9,  8],
    [12, 13, 14, 15, 15, 14, 13, 12],
    [16, 17, 18, 19, 19, 18, 17, 16],
    [20, 21, 22, 23, 23, 22, 21, 20],
    [24, 25, 26, 27, 27, 26, 25, 24],
    [28, 29, 30, 31, 31, 30, 29, 28],
], dtype=np.int64)

KING_BUCKETS = np.zeros((2, 64), dtype=np.int64)
for r in range(8):
    for f in range(8):
        sq = r * 8 + f
        KING_BUCKETS[0, sq] = _KING_BUCKETS_WHITE[r, f] * PS_NB
        KING_BUCKETS[1, sq] = _KING_BUCKETS_BLACK[r, f] * PS_NB


# ============================================================================
# FEN parsing
# ============================================================================

_FEN_PIECE = {
    'P': W_PAWN, 'N': W_KNIGHT, 'B': W_BISHOP, 'R': W_ROOK, 'Q': W_QUEEN, 'K': W_KING,
    'p': B_PAWN, 'n': B_KNIGHT, 'b': B_BISHOP, 'r': B_ROOK, 'q': B_QUEEN, 'k': B_KING,
}


def fen_to_board(fen):
    """Returns (board_dict, stm) where board_dict maps square_idx (a1=0, h8=63) to piece code."""
    parts = fen.split()
    board_str = parts[0]
    stm = 0 if (len(parts) > 1 and parts[1] == 'w') else 1
    board = {}
    rank = 7
    file = 0
    for c in board_str:
        if c == '/':
            rank -= 1
            file = 0
        elif c.isdigit():
            file += int(c)
        else:
            sq = rank * 8 + file
            board[sq] = _FEN_PIECE[c]
            file += 1
    return board, stm


def halfka_features(fen, perspective):
    """Returns list of active feature indices for this perspective (0=white, 1=black)."""
    board, _ = fen_to_board(fen)
    king_code = W_KING if perspective == 0 else B_KING
    king_sq = None
    for sq, p in board.items():
        if p == king_code:
            king_sq = sq
            break
    if king_sq is None:
        return []
    orient = ORIENT[perspective, king_sq]
    kb = KING_BUCKETS[perspective, king_sq]
    psidx = PIECE_SQUARE_INDEX[perspective]
    feats = []
    for sq in sorted(board.keys()):
        p = board[sq]
        feat = (sq ^ int(orient)) + int(psidx[p]) + int(kb)
        feats.append(feat)
    return feats


# ============================================================================
# Self-test / verify against known outputs
# ============================================================================

def _selftest():
    """Match Gungnir's nnueverify first-feature indices."""
    print("Self-test: feature extraction vs Gungnir's nnueverify expected values")
    print()
    cases = [
        ("startpos (white)",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         22208, 22328),
        ("startpos (black)",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
         22208, 22328),
        ("KQvK (white)",
         "8/8/8/4k3/8/8/8/3QK3 w - - 0 1",
         22339, 14011),
        ("KRvK",
         "8/8/8/4k3/8/8/8/3RK3 w - - 0 1",
         22211, 13883),
    ]
    passed, failed = 0, 0
    for name, fen, exp_w0, exp_b0 in cases:
        fw = halfka_features(fen, 0)
        fb = halfka_features(fen, 1)
        ok_w = fw and fw[0] == exp_w0
        ok_b = fb and fb[0] == exp_b0
        status = "OK" if (ok_w and ok_b) else "FAIL"
        if ok_w and ok_b: passed += 1
        else: failed += 1
        print(f"  {status}  {name}:")
        print(f"           white feats={len(fw)} first={fw[0] if fw else '-'} (expected {exp_w0})")
        print(f"           black feats={len(fb)} first={fb[0] if fb else '-'} (expected {exp_b0})")
    print()
    print(f"Passed {passed}/{passed + failed} test cases.")
    return failed == 0


if __name__ == '__main__':
    ok = _selftest()
    import sys
    sys.exit(0 if ok else 1)
