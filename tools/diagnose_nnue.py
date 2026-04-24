"""
diagnose_nnue.py — compare NNUE eval vs classical PeSTO eval across many
positions from a labels file. Histograms the signed diff and buckets by
|label_cp| so we can see where NNUE systematically diverges.

Usage:
    python tools/diagnose_nnue.py <labels.txt> <path.nnue> [--max 10000]
"""
import argparse
import os
import subprocess
import sys


def eval_with_engine(gungnir, fen, use_nnue, nnue_path, depth=1):
    """Run a depth-1 search (essentially static eval) and parse the score."""
    cmds = ['uci']
    if nnue_path:
        cmds.append(f'setoption name NNUEFile value {nnue_path}')
    cmds.append(f'setoption name UseNNUE value {"true" if use_nnue else "false"}')
    cmds.append('isready')
    cmds.append(f'position fen {fen}')
    cmds.append(f'go depth {depth}')
    cmds.append('quit')
    p = subprocess.run([gungnir], input='\n'.join(cmds) + '\n',
                       capture_output=True, text=True, timeout=5)
    for line in p.stdout.splitlines()[::-1]:
        if 'score cp' in line:
            parts = line.split()
            try:
                i = parts.index('cp')
                return int(parts[i + 1])
            except (ValueError, IndexError):
                return None
        if 'score mate' in line:
            parts = line.split()
            try:
                i = parts.index('mate')
                mate_in = int(parts[i + 1])
                return 30000 - abs(mate_in) if mate_in > 0 else -(30000 - abs(mate_in))
            except (ValueError, IndexError):
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels')
    ap.add_argument('nnue')
    ap.add_argument('--gungnir', default='E:/Claude/Gungnir/build/Release/gungnir.exe')
    ap.add_argument('--max', type=int, default=1000)
    args = ap.parse_args()

    diffs = []       # (|label|, nnue_cp, classical_cp)
    print(f"Loading up to {args.max} positions from {args.labels}...")
    with open(args.labels) as f:
        for i, line in enumerate(f):
            if len(diffs) >= args.max:
                break
            line = line.strip()
            if '|' not in line:
                continue
            fen, s = line.rsplit('|', 1)
            try:
                label = int(s)
            except ValueError:
                continue
            nnue_cp = eval_with_engine(args.gungnir, fen, True, args.nnue)
            classical_cp = eval_with_engine(args.gungnir, fen, False, None)
            if nnue_cp is None or classical_cp is None:
                continue
            # Flip to WHITE POV (engine reports STM POV; label is already WHITE POV)
            stm = fen.split()[1]
            if stm == 'b':
                nnue_cp = -nnue_cp
                classical_cp = -classical_cp
            diffs.append((abs(label), label, nnue_cp, classical_cp))
            if len(diffs) % 50 == 0:
                print(f"  {len(diffs)}/{args.max}", flush=True)

    print(f"\nGot {len(diffs)} valid evals.")

    # Bucket by |label| magnitude
    buckets = [(0, 30), (30, 100), (100, 300), (300, 800), (800, 2000), (2000, 99999)]
    for lo, hi in buckets:
        b = [(l, nc, cc) for (al, l, nc, cc) in diffs if lo <= al < hi]
        if not b:
            continue
        nnue_errs = [abs(l - nc) for (l, nc, cc) in b]
        classical_errs = [abs(l - cc) for (l, nc, cc) in b]
        nnue_avg = sum(nnue_errs) / len(nnue_errs)
        classical_avg = sum(classical_errs) / len(classical_errs)
        # Correlation sign: does NNUE track label sign?
        nnue_sign_right = sum(1 for (l, nc, cc) in b if (l > 0) == (nc > 0)) / len(b)
        classical_sign_right = sum(1 for (l, nc, cc) in b if (l > 0) == (cc > 0)) / len(b)
        print(f"  |label|={lo:>5}..{hi:<5}  n={len(b):>5}  "
              f"NNUE err avg={nnue_avg:>6.0f} sign_ok={nnue_sign_right:.2%}  "
              f"classical err avg={classical_avg:>6.0f} sign_ok={classical_sign_right:.2%}")


if __name__ == '__main__':
    main()
