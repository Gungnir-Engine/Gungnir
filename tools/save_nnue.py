"""
save_nnue.py — Serialize a trained GungnirHalfKA PyTorch checkpoint to a
.nnue file byte-compatible with src/nnue.cpp's loader.

Byte layout (matches the C++ loader):
    u32  version         = 0x7AF32F20
    u32  arch_hash       = 0x1C103C92   (small-net, L1=128)
    u32  desc_len        + desc bytes
    u32  ft_hash         = 0x7F234DB8
    LEB128 block: 128         int16   FT biases
    LEB128 block: 22528*128   int16   FT weights  (row-major [feat][unit])
    LEB128 block: 8*22528     int32   PSQT weights (row-major [feat][bucket])
    8 × raw layer stacks, each:
        u32  stack_hash
        16   int32   fc_0 biases
        16*128 int8  fc_0 weights (row-major [out][in])
        32   int32   fc_1 biases
        32*32 int8   fc_1 weights
        1    int32   fc_2 bias
        32   int8    fc_2 weights

Quantization scales (Stockfish-standard for small net):
    QA = 255   (float -> int16 for FT)
    QB = 64    (float -> int8  for layer weights; 2^WeightScaleBits=6)
    layer biases: int32, scale = QA * QB = 16320
    PSQT: int32, scale = QA

Usage:
    python tools/save_nnue.py <checkpoint.pt> <output.nnue>
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gungnir_nnue import (
    FT_INPUT_DIM, FT_OUTPUT_DIM, PSQT_BUCKETS, LAYER_STACKS,
    L2, FC0_OUT, FC1_IN_PAD, FC1_OUT, FC2_IN_PAD,
)
from train_halfka import GungnirHalfKA

VERSION         = 0x7AF32F20
ARCH_HASH_SMALL = 0x1C103C92
FT_HASH         = 0x7F234DB8
LEB128_MAGIC    = b"COMPRESSED_LEB128"   # 17 bytes exactly

QA = 255     # FT scale
QB = 64      # layer weight scale (2^6)


# ============================================================================
# LEB128 signed encoder (matches the decoder in src/nnue.cpp::read_leb128)
# ============================================================================

def leb128_encode_signed(vals):
    """Encode a sequence of signed ints (two's complement per-int) as LEB128 bytes.
    Mirrors the decoder in src/nnue.cpp — sign bit is the high bit of the last
    byte's 7-bit payload (0x40). The per-integer byte count varies 1..5 depending
    on value magnitude (for int16 / int32 values)."""
    out = bytearray()
    for v in vals:
        v = int(v)
        more = True
        while more:
            byte = v & 0x7F
            v >>= 7                       # arithmetic shift in Python
            sign_bit = byte & 0x40
            # Terminate when remaining value is 0 (for positives) or -1 (for negatives)
            # AND the sign of the last emitted byte matches.
            if (v == 0 and not sign_bit) or (v == -1 and sign_bit):
                more = False
            else:
                byte |= 0x80
            out.append(byte)
    return bytes(out)


def write_leb128_block(fh, vals):
    encoded = leb128_encode_signed(vals)
    fh.write(LEB128_MAGIC)                 # 17 bytes
    fh.write(struct.pack('<I', len(encoded)))
    fh.write(encoded)


# ============================================================================
# Quantization
# ============================================================================

def quant_i16(float_vals, scale):
    """Clip-and-round floats to int16 after multiplying by scale."""
    q = np.round(float_vals * scale)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q


def quant_i8(float_vals, scale):
    q = np.round(float_vals * scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def quant_i32(float_vals, scale):
    q = np.round(float_vals * scale)
    q = np.clip(q, -2**31, 2**31 - 1).astype(np.int32)
    return q


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('output')
    ap.add_argument('--desc', default='Gungnir trained net (CC0 Lichess labels)')
    ap.add_argument('--zero', action='store_true',
                    help='Save all-zero weights (sanity-checks the file format only)')
    args = ap.parse_args()

    net = GungnirHalfKA()
    if not args.zero:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        net.load_state_dict(state)

    # --- Gather float weights into numpy ---
    ft_w     = net.ft.weight.detach().numpy()          # [22528, 128]
    ft_b     = net.ft_bias.detach().numpy()            # [128]
    psqt_w   = net.psqt.weight.detach().numpy()        # [22528, 8]

    fc0_w_all = [net.fc0[i].weight.detach().numpy() for i in range(LAYER_STACKS)]   # [16, 128]
    fc0_b_all = [net.fc0[i].bias.detach().numpy()   for i in range(LAYER_STACKS)]   # [16]
    fc1_w_all = [net.fc1[i].weight.detach().numpy() for i in range(LAYER_STACKS)]   # [32, 32]
    fc1_b_all = [net.fc1[i].bias.detach().numpy()   for i in range(LAYER_STACKS)]   # [32]
    fc2_w_all = [net.fc2[i].weight.detach().numpy() for i in range(LAYER_STACKS)]   # [1, 32]
    fc2_b_all = [net.fc2[i].bias.detach().numpy()   for i in range(LAYER_STACKS)]   # [1]

    if args.zero:
        ft_w = np.zeros_like(ft_w); ft_b = np.zeros_like(ft_b); psqt_w = np.zeros_like(psqt_w)
        fc0_w_all = [np.zeros_like(w) for w in fc0_w_all]
        fc0_b_all = [np.zeros_like(b) for b in fc0_b_all]
        fc1_w_all = [np.zeros_like(w) for w in fc1_w_all]
        fc1_b_all = [np.zeros_like(b) for b in fc1_b_all]
        fc2_w_all = [np.zeros_like(w) for w in fc2_w_all]
        fc2_b_all = [np.zeros_like(b) for b in fc2_b_all]

    # --- Quantize ---
    ft_biases_q  = quant_i16(ft_b, QA).tolist()
    # row-major flatten: weight[feat, unit] -> weight[feat*128 + unit] per C++ loader
    ft_weights_q = quant_i16(ft_w, QA).flatten(order='C').tolist()
    # PSQT scaled by QA. Lives in int32.
    psqt_flat    = psqt_w.flatten(order='C')
    psqt_q       = quant_i32(psqt_flat, QA).tolist()

    # Layer stacks — raw little-endian bytes
    stacks_bytes = bytearray()
    for i in range(LAYER_STACKS):
        stacks_bytes += struct.pack('<I', 0)    # stack hash (our loader ignores the value)

        # fc_0: 16 int32 biases, then 16*128 int8 weights
        fc0_b_q = quant_i32(fc0_b_all[i], QA * QB)
        fc0_w_q = quant_i8 (fc0_w_all[i], QB)
        for b in fc0_b_q:
            stacks_bytes += struct.pack('<i', int(b))
        stacks_bytes += fc0_w_q.tobytes()        # shape [16, 128] row-major

        # fc_1: 32 int32 biases, 32*32 int8 weights
        fc1_b_q = quant_i32(fc1_b_all[i], QA * QB)
        fc1_w_q = quant_i8 (fc1_w_all[i], QB)
        for b in fc1_b_q:
            stacks_bytes += struct.pack('<i', int(b))
        stacks_bytes += fc1_w_q.tobytes()

        # fc_2: 1 int32 bias, 32 int8 weights
        fc2_b_q = quant_i32(fc2_b_all[i], QA * QB)
        fc2_w_q = quant_i8 (fc2_w_all[i], QB)
        stacks_bytes += struct.pack('<i', int(fc2_b_q[0]))
        stacks_bytes += fc2_w_q.tobytes()

    # --- Write the file ---
    desc_bytes = args.desc.encode('ascii')[:255]
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        # Header
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', ARCH_HASH_SMALL))
        f.write(struct.pack('<I', len(desc_bytes)))
        f.write(desc_bytes)

        # FT hash
        f.write(struct.pack('<I', FT_HASH))

        # FT biases (LEB128 int16)
        write_leb128_block(f, ft_biases_q)
        # FT weights (LEB128 int16)
        write_leb128_block(f, ft_weights_q)
        # PSQT (LEB128 int32)
        write_leb128_block(f, psqt_q)

        # Layer stacks (raw)
        f.write(bytes(stacks_bytes))

    size = os.path.getsize(args.output)
    print(f"Wrote {args.output}  ({size:,} bytes)")


if __name__ == '__main__':
    main()
