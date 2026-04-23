// Gungnir — Stockfish HalfKAv2_hm NNUE implementation.
// Ported from chess.html Sessions 12–13 (SF17 small net). All integer math.
//
// SIMD (Session 35): the hot FT delta update + pairwise transform + fc_0
// matmul have AVX2 paths gated on __AVX2__. Outputs are bit-identical to
// the portable scalar code (verified by `gungnir nnueverify` — same +251 cp
// startpos eval either way).

#include "nnue.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    #define GUNGNIR_AVX2 1
    #include <immintrin.h>
#else
    #define GUNGNIR_AVX2 0
#endif

// MSVC defines __AVX2__ when /arch:AVX2 is set (since VS 2022 17.x); older
// MSVC needs a manual override. CMakeLists.txt sets /arch:AVX2 unconditionally.
#if !GUNGNIR_AVX2 && defined(_MSC_VER) && defined(_M_X64)
    #include <immintrin.h>
    #undef GUNGNIR_AVX2
    #define GUNGNIR_AVX2 1
#endif

// AVX-512 VNNI (Session 36): if available (Cascade Lake / Ice Lake / newer),
// _mm256_dpbusd_epi32 fuses maddubs+madd into one instruction (~25-40% faster
// fc_0 matmul on supporting hardware). Compile with /arch:AVX512 to enable.
// On AVX2-only CPUs this is a no-op fallback to the maddubs path above.
#if defined(__AVX512VNNI__) || defined(__AVXVNNI__)
    #define GUNGNIR_VNNI 1
#else
    #define GUNGNIR_VNNI 0
#endif

namespace gungnir {
namespace NNUE {

// ============================================================================
// Constants — small-net architecture
// ============================================================================

constexpr unsigned long EXPECTED_VERSION   = 0x7AF32F20UL;
constexpr unsigned long EXPECTED_ARCH_HASH = 0x1C103C92UL;
constexpr unsigned long EXPECTED_FT_HASH   = 0x7F234DB8UL;
constexpr int FT_INPUT_DIM   = 22528;     // HalfKAv2_hm
constexpr int FT_OUTPUT_DIM  = 128;       // TransformedFeatureDimensionsSmall
constexpr int PSQT_BUCKETS   = 8;
constexpr int LAYER_STACKS   = 8;
constexpr int L2             = 15;        // FC_0 outputs we use (16th is skip)
constexpr int FC0_OUT        = L2 + 1;    // 16
constexpr int FC1_IN_PAD     = 32;        // 30 used + 2 pad
constexpr int FC1_OUT        = 32;
constexpr int FC2_IN_PAD     = 32;
constexpr int LEB128_MAGIC_LEN = 17;
constexpr const char* LEB128_MAGIC_STR = "COMPRESSED_LEB128";

// HalfKAv2_hm piece-square index bases (from half_ka_v2_hm.h).
constexpr int PS_NB        = 11 * 64;
constexpr int PS_W_PAWN    = 0  * 64;
constexpr int PS_B_PAWN    = 1  * 64;
constexpr int PS_W_KNIGHT  = 2  * 64;
constexpr int PS_B_KNIGHT  = 3  * 64;
constexpr int PS_W_BISHOP  = 4  * 64;
constexpr int PS_B_BISHOP  = 5  * 64;
constexpr int PS_W_ROOK    = 6  * 64;
constexpr int PS_B_ROOK    = 7  * 64;
constexpr int PS_W_QUEEN   = 8  * 64;
constexpr int PS_B_QUEEN   = 9  * 64;
constexpr int PS_KING      = 10 * 64;

// PieceSquareIndex[perspective][piece_code 0..15]
constexpr int PieceSquareIndex[2][16] = {
    {  // white perspective
        0,
        PS_W_PAWN,   PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK,
        PS_W_QUEEN,  PS_KING,     0, 0,
        PS_B_PAWN,   PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK,
        PS_B_QUEEN,  PS_KING,     0,
    },
    {  // black perspective
        0,
        PS_B_PAWN,   PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK,
        PS_B_QUEEN,  PS_KING,     0, 0,
        PS_W_PAWN,   PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK,
        PS_W_QUEEN,  PS_KING,     0,
    },
};

// ============================================================================
// Orient / KingBuckets tables (filled at init)
// ============================================================================

namespace {

int g_orient[2][64];
int g_king_buckets[2][64];

void build_orient(int* dst, int base_ad, int base_eh) {
    for (int r = 0; r < 8; ++r) {
        for (int f = 0; f < 4; ++f) dst[r * 8 + f] = base_ad;
        for (int f = 4; f < 8; ++f) dst[r * 8 + f] = base_eh;
    }
}

// rows[r] is rank-r (a1=rank0). For white perspective: rank 1 is white's home.
// For black perspective: array is the same shape but in black's POV — rank 1
// (SF coords) = the far rank from black's view, which gets the small bucket
// numbers, while rank 8 (SF coords) is black's home (gets the big numbers).
constexpr int kKingBucketsWhite[8][8] = {
    {28, 29, 30, 31, 31, 30, 29, 28},  // rank 1 (white home)
    {24, 25, 26, 27, 27, 26, 25, 24},
    {20, 21, 22, 23, 23, 22, 21, 20},
    {16, 17, 18, 19, 19, 18, 17, 16},
    {12, 13, 14, 15, 15, 14, 13, 12},
    { 8,  9, 10, 11, 11, 10,  9,  8},
    { 4,  5,  6,  7,  7,  6,  5,  4},
    { 0,  1,  2,  3,  3,  2,  1,  0},
};
constexpr int kKingBucketsBlack[8][8] = {
    { 0,  1,  2,  3,  3,  2,  1,  0},
    { 4,  5,  6,  7,  7,  6,  5,  4},
    { 8,  9, 10, 11, 11, 10,  9,  8},
    {12, 13, 14, 15, 15, 14, 13, 12},
    {16, 17, 18, 19, 19, 18, 17, 16},
    {20, 21, 22, 23, 23, 22, 21, 20},
    {24, 25, 26, 27, 27, 26, 25, 24},
    {28, 29, 30, 31, 31, 30, 29, 28},
};

void init_tables() {
    static bool done = false;
    if (done) return;
    done = true;

    // White: a-d files → flip (XOR 7), e-h files → identity (XOR 0).
    build_orient(g_orient[0], 7, 0);
    // Black: a-d files → flip both (XOR 63), e-h files → flip rank only (XOR 56).
    build_orient(g_orient[1], 63, 56);

    for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) {
        g_king_buckets[0][r * 8 + f] = kKingBucketsWhite[r][f] * PS_NB;
        g_king_buckets[1][r * 8 + f] = kKingBucketsBlack[r][f] * PS_NB;
    }
}

}  // namespace

// ============================================================================
// LEB128 reader
// ============================================================================

namespace {

// Reads `count` signed-LEB128 ints (sign-extended to `bits`-wide) into out[].
// Advances `*pp`. `bits` is 16 for FT biases/weights, 32 for PSQT.
bool read_leb128(const uint8_t** pp, const uint8_t* end, int count, int bits, int32_t* out) {
    const uint8_t* p = *pp;
    if (end - p < LEB128_MAGIC_LEN + 4) return false;
    if (std::memcmp(p, LEB128_MAGIC_STR, LEB128_MAGIC_LEN) != 0) return false;
    p += LEB128_MAGIC_LEN;
    uint32_t bytes_left;
    std::memcpy(&bytes_left, p, 4);  // little-endian on x86
    p += 4;
    const uint8_t* block_end = p + bytes_left;
    if (block_end > end) return false;

    for (int i = 0; i < count; ++i) {
        int32_t result = 0;
        int shift = 0;
        while (true) {
            if (p >= block_end) return false;
            uint8_t b = *p++;
            result |= int32_t(b & 0x7F) << shift;
            shift += 7;
            if ((b & 0x80) == 0) {
                if (shift < bits && (b & 0x40) != 0) {
                    result |= ~((int32_t(1) << shift) - 1);
                }
                break;
            }
            if (shift >= bits) return false;
        }
        out[i] = result;
    }
    if (p != block_end) return false;
    *pp = p;
    return true;
}

}  // namespace

// ============================================================================
// Loaded weights
// ============================================================================

namespace {

bool g_loaded = false;
bool g_enabled = false;
unsigned long g_version = 0;
unsigned long g_arch_hash = 0;
unsigned long g_ft_hash = 0;
std::string g_desc;

// FT biases, weights, PSQT weights.
std::vector<int16_t> g_ft_biases;        // size = FT_OUTPUT_DIM
std::vector<int16_t> g_ft_weights;       // size = FT_INPUT_DIM * FT_OUTPUT_DIM
std::vector<int32_t> g_ft_psqt_weights;  // size = PSQT_BUCKETS * FT_INPUT_DIM

// 8 layer stacks — each owns small dense arrays.
struct LayerStack {
    uint32_t hash;
    int32_t  fc0_b[FC0_OUT];          // 16 int32 biases
    int8_t   fc0_w[FC0_OUT * FT_OUTPUT_DIM];  // 16 * 128
    int32_t  fc1_b[FC1_OUT];          // 32 int32 biases
    int8_t   fc1_w[FC1_OUT * FC1_IN_PAD];     // 32 * 32
    int32_t  fc2_b[1];
    int8_t   fc2_w[FC2_IN_PAD];       // 32
};
LayerStack g_stacks[LAYER_STACKS];

uint32_t read_u32(const uint8_t*& p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}
int32_t read_i32(const uint8_t*& p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}

}  // namespace

// ============================================================================
// File parsing
// ============================================================================

bool load(const std::string& path) {
    init_tables();
    g_loaded = false;

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "NNUE: cannot open " << path << std::endl;
        return false;
    }
    const std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf;
    buf.resize(static_cast<size_t>(file_size));
    if (!in.read(reinterpret_cast<char*>(&buf[0]), file_size)) {
        std::cerr << "NNUE: read failed" << std::endl;
        return false;
    }

    const uint8_t* p   = buf.data();
    const uint8_t* end = buf.data() + buf.size();

    if (end - p < 12) return false;
    g_version  = read_u32(p);
    g_arch_hash = read_u32(p);
    uint32_t desc_len = read_u32(p);
    if (uint32_t(end - p) < desc_len) return false;
    g_desc.assign(reinterpret_cast<const char*>(p), desc_len);
    p += desc_len;

    if (g_version != EXPECTED_VERSION) {
        std::cerr << "NNUE: unsupported version 0x" << std::hex << g_version
                  << " (expected 0x" << EXPECTED_VERSION << ")" << std::dec << std::endl;
        return false;
    }
    // Session 37: bigger SF nets (e.g. nn-1111cefa1111.nnue, ~133 MB) have a
    // different arch hash and L1=3072 instead of 128. Loading those requires
    // dynamic allocation of FT/layer-stack arrays sized at runtime — out of
    // scope for now. We warn and refuse to load anything other than the small net.
    constexpr unsigned long EXPECTED_BIG_ARCH_HASH = 0x3C103C92UL;  // SF main net
    if (g_arch_hash != EXPECTED_ARCH_HASH) {
        if (g_arch_hash == EXPECTED_BIG_ARCH_HASH) {
            std::cerr << "NNUE: this is the BIG net (L1=3072). The current build is "
                         "compiled for the SMALL net (L1=128); refusing to load. To "
                         "support both, layer dimensions must be made runtime-variable "
                         "(deferred refactor). Use the small net instead.\n";
        } else {
            std::cerr << "NNUE: arch hash 0x" << std::hex << g_arch_hash
                      << " is unrecognized (expected 0x" << EXPECTED_ARCH_HASH
                      << ")" << std::dec << "\n";
        }
        return false;
    }
    if (end - p < 4) return false;
    g_ft_hash = read_u32(p);

    // Feature transformer (LEB128-compressed)
    g_ft_biases.assign(FT_OUTPUT_DIM, 0);
    g_ft_weights.assign(size_t(FT_INPUT_DIM) * FT_OUTPUT_DIM, 0);
    g_ft_psqt_weights.assign(size_t(PSQT_BUCKETS) * FT_INPUT_DIM, 0);

    {
        std::vector<int32_t> tmp(FT_OUTPUT_DIM);
        if (!read_leb128(&p, end, FT_OUTPUT_DIM, 16, tmp.data())) {
            std::cerr << "NNUE: FT biases LEB128 parse failed" << std::endl;
            return false;
        }
        for (int i = 0; i < FT_OUTPUT_DIM; ++i) g_ft_biases[i] = int16_t(tmp[i]);
    }
    {
        std::vector<int32_t> tmp(size_t(FT_INPUT_DIM) * FT_OUTPUT_DIM);
        if (!read_leb128(&p, end, int(tmp.size()), 16, tmp.data())) {
            std::cerr << "NNUE: FT weights LEB128 parse failed" << std::endl;
            return false;
        }
        for (size_t i = 0; i < tmp.size(); ++i) g_ft_weights[i] = int16_t(tmp[i]);
    }
    if (!read_leb128(&p, end, int(g_ft_psqt_weights.size()), 32, g_ft_psqt_weights.data())) {
        std::cerr << "NNUE: FT PSQT weights LEB128 parse failed" << std::endl;
        return false;
    }

    // 8 layer stacks (raw little-endian, NOT compressed)
    for (int s = 0; s < LAYER_STACKS; ++s) {
        if (end - p < 4) return false;
        g_stacks[s].hash = read_u32(p);

        // fc_0
        if (end - p < int(sizeof(int32_t)) * FC0_OUT + FC0_OUT * FT_OUTPUT_DIM) return false;
        for (int i = 0; i < FC0_OUT; ++i) g_stacks[s].fc0_b[i] = read_i32(p);
        std::memcpy(g_stacks[s].fc0_w, p, FC0_OUT * FT_OUTPUT_DIM);
        p += FC0_OUT * FT_OUTPUT_DIM;

        // fc_1
        if (end - p < int(sizeof(int32_t)) * FC1_OUT + FC1_OUT * FC1_IN_PAD) return false;
        for (int i = 0; i < FC1_OUT; ++i) g_stacks[s].fc1_b[i] = read_i32(p);
        std::memcpy(g_stacks[s].fc1_w, p, FC1_OUT * FC1_IN_PAD);
        p += FC1_OUT * FC1_IN_PAD;

        // fc_2
        if (end - p < int(sizeof(int32_t)) + FC2_IN_PAD) return false;
        g_stacks[s].fc2_b[0] = read_i32(p);
        std::memcpy(g_stacks[s].fc2_w, p, FC2_IN_PAD);
        p += FC2_IN_PAD;
    }

    if (p != end) {
        std::cerr << "NNUE: warning — " << (end - p) << " trailing bytes after layer stacks" << std::endl;
    }

    g_loaded = true;
    g_enabled = true;
    std::cerr << "NNUE: loaded " << path << " (" << file_size << " bytes, "
              << g_desc.substr(0, 40)
              << (g_desc.size() > 40 ? "..." : "") << ")" << std::endl;
    return true;
}

bool is_loaded()                  { return g_loaded; }
void set_enabled(bool on)         { g_enabled = on && g_loaded; }
bool is_enabled()                 { return g_enabled; }
const std::string& description()  { return g_desc; }
unsigned long file_version()      { return g_version; }
unsigned long arch_hash()         { return g_arch_hash; }

// ============================================================================
// Feature extraction
// ============================================================================

int features(const Position& pos, int perspective, int* out) {
    init_tables();
    const Color me = (perspective == 0) ? WHITE : BLACK;
    const Square king_sq = pos.king_square(me);
    const int orient = g_orient[perspective][king_sq];
    const int kb     = g_king_buckets[perspective][king_sq];
    const int* psidx = PieceSquareIndex[perspective];
    int count = 0;
    for (Square s = SQ_A1; s < SQ_NONE; ++s) {
        const Piece p = pos.piece_on(s);
        if (p == NO_PIECE) continue;
        const int feat = (int(s) ^ orient) + psidx[p] + kb;
        out[count++] = feat;
    }
    return count;
}

// ============================================================================
// Incremental accumulator stack (Session 24)
// ============================================================================

namespace {

struct Accumulator {
    int32_t acc[2][FT_OUTPUT_DIM];     // [perspective][unit]
    int32_t psqt[2][PSQT_BUCKETS];     // [perspective][bucket]
};

// Stack of accumulator entries indexed by ply. Top = current. Push on
// on_make, pop on on_unmake. Per-thread for LazySMP.
constexpr int ACC_STACK_SIZE = 1025;
thread_local Accumulator g_acc[ACC_STACK_SIZE];
thread_local int g_acc_top = 0;

void rebuild_perspective(Accumulator& a, const Position& pos, int p) {
    int feats[40];
    const int n = features(pos, p, feats);
    int32_t* dst = a.acc[p];
#if GUNGNIR_AVX2
    // Init dst from int16 biases (sign-extended), then add feature rows.
    for (int j = 0; j < FT_OUTPUT_DIM; j += 8) {
        __m128i s = _mm_loadu_si128((const __m128i*)(g_ft_biases.data() + j));
        _mm256_storeu_si256((__m256i*)(dst + j), _mm256_cvtepi16_epi32(s));
    }
    for (int f = 0; f < n; ++f) {
        const int16_t* row = &g_ft_weights[size_t(feats[f]) * FT_OUTPUT_DIM];
        for (int j = 0; j < FT_OUTPUT_DIM; j += 8) {
            __m128i s   = _mm_loadu_si128((const __m128i*)(row + j));
            __m256i s32 = _mm256_cvtepi16_epi32(s);
            __m256i d   = _mm256_loadu_si256((__m256i*)(dst + j));
            _mm256_storeu_si256((__m256i*)(dst + j), _mm256_add_epi32(d, s32));
        }
    }
#else
    for (int i = 0; i < FT_OUTPUT_DIM; ++i) dst[i] = g_ft_biases[i];
    for (int f = 0; f < n; ++f) {
        const int16_t* row = &g_ft_weights[size_t(feats[f]) * FT_OUTPUT_DIM];
        for (int i = 0; i < FT_OUTPUT_DIM; ++i) dst[i] += row[i];
    }
#endif
    for (int k = 0; k < PSQT_BUCKETS; ++k) a.psqt[p][k] = 0;
    for (int f = 0; f < n; ++f) {
        const int32_t* row = &g_ft_psqt_weights[size_t(feats[f]) * PSQT_BUCKETS];
        for (int k = 0; k < PSQT_BUCKETS; ++k) a.psqt[p][k] += row[k];
    }
}

inline int feature_index(int p, Square king_sq, Piece piece, Square sq) {
    return (int(sq) ^ g_orient[p][king_sq]) + PieceSquareIndex[p][piece] + g_king_buckets[p][king_sq];
}

inline void apply_remove(Accumulator& a, int p, int feat) {
    const int16_t* row = &g_ft_weights[size_t(feat) * FT_OUTPUT_DIM];
    int32_t* dst = a.acc[p];
#if GUNGNIR_AVX2
    // 128 int32 -= sign-extended int16. 8 int32s per iteration.
    for (int j = 0; j < FT_OUTPUT_DIM; j += 8) {
        __m128i s   = _mm_loadu_si128((const __m128i*)(row + j));
        __m256i s32 = _mm256_cvtepi16_epi32(s);
        __m256i d   = _mm256_loadu_si256((__m256i*)(dst + j));
        d = _mm256_sub_epi32(d, s32);
        _mm256_storeu_si256((__m256i*)(dst + j), d);
    }
#else
    for (int j = 0; j < FT_OUTPUT_DIM; ++j) dst[j] -= row[j];
#endif
    const int32_t* prow = &g_ft_psqt_weights[size_t(feat) * PSQT_BUCKETS];
    for (int k = 0; k < PSQT_BUCKETS; ++k) a.psqt[p][k] -= prow[k];
}

inline void apply_add(Accumulator& a, int p, int feat) {
    const int16_t* row = &g_ft_weights[size_t(feat) * FT_OUTPUT_DIM];
    int32_t* dst = a.acc[p];
#if GUNGNIR_AVX2
    for (int j = 0; j < FT_OUTPUT_DIM; j += 8) {
        __m128i s   = _mm_loadu_si128((const __m128i*)(row + j));
        __m256i s32 = _mm256_cvtepi16_epi32(s);
        __m256i d   = _mm256_loadu_si256((__m256i*)(dst + j));
        d = _mm256_add_epi32(d, s32);
        _mm256_storeu_si256((__m256i*)(dst + j), d);
    }
#else
    for (int j = 0; j < FT_OUTPUT_DIM; ++j) dst[j] += row[j];
#endif
    const int32_t* prow = &g_ft_psqt_weights[size_t(feat) * PSQT_BUCKETS];
    for (int k = 0; k < PSQT_BUCKETS; ++k) a.psqt[p][k] += prow[k];
}

}  // namespace

void refresh(const Position& pos) {
    g_acc_top = 0;
    rebuild_perspective(g_acc[0], pos, 0);
    rebuild_perspective(g_acc[0], pos, 1);
}

void on_make(const Position& pos, Move m) {
    if (!g_loaded || !g_enabled) return;
    if (g_acc_top + 1 >= ACC_STACK_SIZE) return;  // safety

    g_acc_top++;
    Accumulator& cur = g_acc[g_acc_top];
    const Accumulator& prev = g_acc[g_acc_top - 1];

    const Color us = ~pos.stm();   // side that just moved
    const Square from = m.from();
    const Square to   = m.to();
    const MoveType mt = m.type();

    // Detect king moves. For castling, the king of `us` definitely moved.
    bool king_moved[2] = {false, false};
    if (mt == MT_CASTLING) {
        king_moved[us] = true;
    } else {
        // For non-castling: the moving piece is now at `to`. If it's a king,
        // mark this side. (Promotion can't move a king.)
        const Piece moved_now = pos.piece_on(to);
        if (mt != MT_PROMOTION && type_of(moved_now) == KING) {
            king_moved[us] = true;
        }
    }

    // For each perspective: rebuild if its king moved, else copy prev + delta.
    for (int p = 0; p < 2; ++p) {
        if (king_moved[p]) {
            rebuild_perspective(cur, pos, p);
            continue;
        }
        // Copy prev to cur, then apply delta.
        std::memcpy(cur.acc[p],  prev.acc[p],  sizeof(cur.acc[p]));
        std::memcpy(cur.psqt[p], prev.psqt[p], sizeof(cur.psqt[p]));

        const Color me = (p == 0) ? WHITE : BLACK;
        const Square king_sq = pos.king_square(me);

        if (mt == MT_EN_PASSANT) {
            const Piece our_pawn = make_piece(us, PAWN);
            const Piece their_pawn = make_piece(~us, PAWN);
            const Square cap_sq = Square(int(to) + (us == WHITE ? -8 : 8));
            apply_remove(cur, p, feature_index(p, king_sq, our_pawn, from));
            apply_remove(cur, p, feature_index(p, king_sq, their_pawn, cap_sq));
            apply_add   (cur, p, feature_index(p, king_sq, our_pawn, to));
        } else if (mt == MT_PROMOTION) {
            const Piece our_pawn = make_piece(us, PAWN);
            const Piece promoted = pos.piece_on(to);
            const Piece captured = pos.captured();
            apply_remove(cur, p, feature_index(p, king_sq, our_pawn, from));
            if (captured != NO_PIECE) {
                apply_remove(cur, p, feature_index(p, king_sq, captured, to));
            }
            apply_add(cur, p, feature_index(p, king_sq, promoted, to));
        } else if (mt == MT_CASTLING) {
            // Only opposite-side perspective reaches here (own side rebuilt above).
            const Piece king = make_piece(us, KING);
            const Piece rook = make_piece(us, ROOK);
            Square rook_from, rook_to;
            if      (to == SQ_G1) { rook_from = SQ_H1; rook_to = SQ_F1; }
            else if (to == SQ_C1) { rook_from = SQ_A1; rook_to = SQ_D1; }
            else if (to == SQ_G8) { rook_from = SQ_H8; rook_to = SQ_F8; }
            else                  { rook_from = SQ_A8; rook_to = SQ_D8; }
            apply_remove(cur, p, feature_index(p, king_sq, king, from));
            apply_remove(cur, p, feature_index(p, king_sq, rook, rook_from));
            apply_add   (cur, p, feature_index(p, king_sq, king, to));
            apply_add   (cur, p, feature_index(p, king_sq, rook, rook_to));
        } else {
            // Normal move (possibly capture).
            const Piece moved = pos.piece_on(to);
            const Piece captured = pos.captured();
            apply_remove(cur, p, feature_index(p, king_sq, moved, from));
            if (captured != NO_PIECE) {
                apply_remove(cur, p, feature_index(p, king_sq, captured, to));
            }
            apply_add(cur, p, feature_index(p, king_sq, moved, to));
        }
    }
}

void on_unmake() {
    if (g_acc_top > 0) g_acc_top--;
}

void on_null_make() {
    if (!g_loaded || !g_enabled) return;
    if (g_acc_top + 1 >= ACC_STACK_SIZE) return;
    g_acc_top++;
    g_acc[g_acc_top] = g_acc[g_acc_top - 1];  // unchanged copy
}

void on_null_unmake() {
    if (g_acc_top > 0) g_acc_top--;
}

// ============================================================================
// Forward pass — uses cached accumulator from top of stack
// ============================================================================

int evaluate(const Position& pos) {
    if (!g_loaded) return 0;
    constexpr int D = FT_OUTPUT_DIM;     // 128
    constexpr int H = D / 2;              // 64

    const Accumulator& a = g_acc[g_acc_top];
    const int32_t* acc_w = a.acc[0];
    const int32_t* acc_b = a.acc[1];
    const int32_t* psqt_w = a.psqt[0];
    const int32_t* psqt_b = a.psqt[1];

    // 3. Pairwise transform (for each perspective: ft[j] = clamp(acc[j],0,127) * clamp(acc[j+H],0,127) / 128)
    alignas(32) int32_t ft_out[D];
    const Color stm = pos.stm();
    const int32_t* perspective_acc[2] = {
        (stm == WHITE) ? acc_w : acc_b,
        (stm == WHITE) ? acc_b : acc_w,
    };
    for (int p = 0; p < 2; ++p) {
        const int32_t* acc = perspective_acc[p];
        int32_t* dst = ft_out + p * H;
#if GUNGNIR_AVX2
        const __m256i zero  = _mm256_setzero_si256();
        const __m256i v127  = _mm256_set1_epi32(127);
        for (int j = 0; j < H; j += 8) {
            __m256i s0 = _mm256_loadu_si256((const __m256i*)(acc + j));
            __m256i s1 = _mm256_loadu_si256((const __m256i*)(acc + j + H));
            s0 = _mm256_max_epi32(s0, zero);
            s0 = _mm256_min_epi32(s0, v127);
            s1 = _mm256_max_epi32(s1, zero);
            s1 = _mm256_min_epi32(s1, v127);
            __m256i prod = _mm256_mullo_epi32(s0, s1);
            _mm256_storeu_si256((__m256i*)(dst + j), _mm256_srai_epi32(prod, 7));
        }
#else
        for (int j = 0; j < H; ++j) {
            int32_t s0 = acc[j];
            int32_t s1 = acc[j + H];
            if (s0 < 0) s0 = 0; else if (s0 > 127) s0 = 127;
            if (s1 < 0) s1 = 0; else if (s1 > 127) s1 = 127;
            dst[j] = (s0 * s1) >> 7;
        }
#endif
    }

    // 4. Bucket: (piece_count - 1) / 4
    const int piece_count = popcount(pos.pieces());
    const int bucket = (piece_count - 1) / 4;

    // 5. PSQT diff
    const int32_t* p_stm = (stm == WHITE) ? psqt_w : psqt_b;
    const int32_t* p_opp = (stm == WHITE) ? psqt_b : psqt_w;
    const int32_t psqt = (p_stm[bucket] - p_opp[bucket]) / 2;

    const LayerStack& ls = g_stacks[bucket];

    // 6. fc_0: 16 int32 outputs from 128 uint8 inputs
    int32_t fc0[FC0_OUT];
#if GUNGNIR_AVX2
    // Pack ft_out (int32, all in [0,127]) into uint8 buffer for VEX maddubs.
    alignas(32) uint8_t ft_u8[FT_OUTPUT_DIM];
    for (int j = 0; j < FT_OUTPUT_DIM; j += 16) {
        // Pack 16 int32s -> 16 uint8s. We know each is in [0,127] so just take low byte.
        for (int k = 0; k < 16; ++k) ft_u8[j + k] = uint8_t(ft_out[j + k]);
    }
    for (int j = 0; j < FC0_OUT; ++j) {
        const int8_t* row = &ls.fc0_w[j * FT_OUTPUT_DIM];
        __m256i sum = _mm256_setzero_si256();
#if !GUNGNIR_VNNI
        const __m256i ones = _mm256_set1_epi16(1);
#endif
        // 128 inputs / 32 per iteration = 4 iterations
        for (int i = 0; i < FT_OUTPUT_DIM; i += 32) {
            __m256i u = _mm256_loadu_si256((const __m256i*)(ft_u8 + i));   // 32 uint8
            __m256i w = _mm256_loadu_si256((const __m256i*)(row + i));     // 32 int8
#if GUNGNIR_VNNI
            // VNNI fused: sum += dpbusd(u, w) — one instruction does it all.
            sum = _mm256_dpbusd_epi32(sum, u, w);
#else
            __m256i p16 = _mm256_maddubs_epi16(u, w);                      // 16 int16
            __m256i p32 = _mm256_madd_epi16(p16, ones);                    // 8 int32
            sum = _mm256_add_epi32(sum, p32);
#endif
        }
        // Horizontal sum of 8 int32s in `sum`.
        __m128i lo = _mm256_castsi256_si128(sum);
        __m128i hi = _mm256_extracti128_si256(sum, 1);
        __m128i s4 = _mm_add_epi32(lo, hi);                                 // 4 int32s
        s4 = _mm_hadd_epi32(s4, s4);                                        // 2 unique values in low half
        s4 = _mm_hadd_epi32(s4, s4);                                        // sum in lane 0
        fc0[j] = ls.fc0_b[j] + _mm_cvtsi128_si32(s4);
    }
#else
    for (int j = 0; j < FC0_OUT; ++j) {
        int32_t s = ls.fc0_b[j];
        const int8_t* row = &ls.fc0_w[j * FT_OUTPUT_DIM];
        for (int i = 0; i < FT_OUTPUT_DIM; ++i) s += ft_out[i] * row[i];
        fc0[j] = s;
    }
#endif

    // 7. ac_sqr_0[0..14] = clamp(min(127, (x*x) >> 19), 0, 127)
    //    ac_0    [0..14] = clamp(x >> 6, 0, 127)
    //    Concatenated into 30 uint8s, padded to 32 with zeros.
    int32_t fc1_in[FC1_IN_PAD] = {0};
    for (int j = 0; j < L2; ++j) {
        const int32_t x = fc0[j];
        int32_t sq = (x * x) >> 19;
        if (sq > 127) sq = 127;
        fc1_in[j] = sq;
    }
    for (int j = 0; j < L2; ++j) {
        int32_t v = fc0[j] >> 6;
        if (v < 0) v = 0; else if (v > 127) v = 127;
        fc1_in[L2 + j] = v;
    }
    // fc1_in[30..31] already zero

    // 8. fc_1: 32 outputs from 32 inputs
    int32_t fc1[FC1_OUT];
    for (int j = 0; j < FC1_OUT; ++j) {
        int32_t s = ls.fc1_b[j];
        const int8_t* row = &ls.fc1_w[j * FC1_IN_PAD];
        for (int i = 0; i < FC1_IN_PAD; ++i) s += fc1_in[i] * row[i];
        fc1[j] = s;
    }

    // 9. ac_1 (clipped relu)
    int32_t fc2_in[FC2_IN_PAD];
    for (int j = 0; j < FC1_OUT; ++j) {
        int32_t v = fc1[j] >> 6;
        if (v < 0) v = 0; else if (v > 127) v = 127;
        fc2_in[j] = v;
    }

    // 10. fc_2: scalar
    int32_t fc2 = ls.fc2_b[0];
    for (int i = 0; i < FC2_IN_PAD; ++i) fc2 += fc2_in[i] * ls.fc2_w[i];

    // 11. Skip connection: fc0[15] * 9600 / 8128
    const int32_t fwd_out = (fc0[15] * 9600) / 8128;
    const int32_t positional = fc2 + fwd_out;

    // 12. Final eval (SF internal cp): (psqt + positional) / 16
    return (psqt + positional) / 16;
}

}  // namespace NNUE
}  // namespace gungnir
