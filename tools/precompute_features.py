"""
precompute_features.py — one-shot pre-computation of HalfKAv2_hm features
for a labels file. Writes .npy arrays that the trainer loads via memmap
instead of calling halfka_features() every sample every epoch.

Input:  labels.txt  (FEN|score_cp per line, WHITE-POV)
Output: <stem>.feats.npz with keys:
    feats_w  : int16 [N, 32]   (padded with -1)
    feats_b  : int16 [N, 32]
    nfeat_w  : int8  [N]       (actual feature count, max 32)
    nfeat_b  : int8  [N]
    stm      : int8  [N]       (0=white, 1=black)
    bucket   : int8  [N]       (0..7)
    score    : int32 [N]       (cp, WHITE-POV)

32 is the HalfKAv2_hm max feature count (32 pieces including both kings).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gungnir_nnue import halfka_features, fen_to_board


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('labels', help='FEN|score_cp per line')
    ap.add_argument('--out', default=None, help='Output .npz path (default: <stem>.feats.npz)')
    ap.add_argument('--max', type=int, default=-1)
    args = ap.parse_args()

    out_path = args.out or (os.path.splitext(args.labels)[0] + '.feats.npz')

    # First pass: count lines
    print(f"Counting lines in {args.labels}...", flush=True)
    n = 0
    with open(args.labels) as f:
        for line in f:
            if '|' in line:
                n += 1
                if args.max > 0 and n >= args.max:
                    break
    print(f"  {n} samples", flush=True)

    # Allocate arrays
    feats_w = np.full((n, 32), -1, dtype=np.int16)
    feats_b = np.full((n, 32), -1, dtype=np.int16)
    nfeat_w = np.zeros(n, dtype=np.int8)
    nfeat_b = np.zeros(n, dtype=np.int8)
    stm_a   = np.zeros(n, dtype=np.int8)
    bucket_a = np.zeros(n, dtype=np.int8)
    score_a = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    idx = 0
    with open(args.labels) as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            if args.max > 0 and idx >= args.max:
                break
            fen, s = line.rsplit('|', 1)
            try:
                sc = int(s)
            except ValueError:
                continue
            try:
                board, stm = fen_to_board(fen)
                fw = halfka_features(fen, 0)
                fb = halfka_features(fen, 1)
            except Exception as e:
                # Skip malformed FENs
                continue

            nfw = min(len(fw), 32)
            nfb = min(len(fb), 32)
            feats_w[idx, :nfw] = np.asarray(fw[:nfw], dtype=np.int16)
            feats_b[idx, :nfb] = np.asarray(fb[:nfb], dtype=np.int16)
            nfeat_w[idx] = nfw
            nfeat_b[idx] = nfb
            stm_a[idx] = stm
            bk = (len(board) - 1) // 4
            bucket_a[idx] = max(0, min(7, bk))
            score_a[idx] = sc

            idx += 1
            if idx % 100000 == 0:
                dt = time.time() - t0
                rate = idx / dt
                eta = (n - idx) / rate
                print(f"  {idx}/{n}  ({rate:.0f}/s, eta {eta:.0f}s)", flush=True)

    # Trim if we skipped any
    if idx < n:
        feats_w  = feats_w[:idx]
        feats_b  = feats_b[:idx]
        nfeat_w  = nfeat_w[:idx]
        nfeat_b  = nfeat_b[:idx]
        stm_a    = stm_a[:idx]
        bucket_a = bucket_a[:idx]
        score_a  = score_a[:idx]

    dt = time.time() - t0
    print(f"\nDone. {idx} samples in {dt:.1f}s ({idx/dt:.0f}/s)")
    print(f"Saving to {out_path}...")
    np.savez(out_path,
             feats_w=feats_w, feats_b=feats_b,
             nfeat_w=nfeat_w, nfeat_b=nfeat_b,
             stm=stm_a, bucket=bucket_a, score=score_a)
    sz = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({sz:.1f} MB)")


if __name__ == '__main__':
    main()
