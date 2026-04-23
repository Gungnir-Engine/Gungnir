"""
Label FENs with Stockfish evaluations for NNUE distillation training.

Usage:
    python label_fens.py <fens.txt> <labels.txt> [--depth 12] [--sf <path>]

Input  (fens.txt):   one FEN per line.
Output (labels.txt): lines of "FEN|score"  where score is centipawns from WHITE's POV.
                     Mate scores are encoded as +/-30000 - mate_distance (signed, capped).

Defaults:
    --sf:    E:\\Claude\\stockfish\\stockfish\\stockfish-windows-x86-64-avx2.exe
    --depth: 12
"""
import argparse
import os
import subprocess
import sys
import time

SF_DEFAULT = r"E:\Claude\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"


def score_from_line(line):
    # Pull the last `score cp X` or `score mate X` pair out of a Stockfish info line.
    i = line.rfind(" score ")
    if i < 0:
        return None
    rest = line[i + 7:].split()
    if len(rest) < 2:
        return None
    kind, val = rest[0], rest[1]
    try:
        n = int(val)
    except ValueError:
        return None
    if kind == "cp":
        return n
    if kind == "mate":
        # Map mate-in-N to a large signed score; caller will treat |s|>=MATE_IN_MAX as skip.
        big = 30000 - min(abs(n), 500)
        return big if n > 0 else -big
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fens", help="input file: one FEN per line")
    ap.add_argument("labels", help="output file: FEN|score lines")
    ap.add_argument("--sf", default=SF_DEFAULT, help="path to stockfish binary")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=128, help="Stockfish hash MB")
    args = ap.parse_args()

    if not os.path.isfile(args.sf):
        print(f"Stockfish not found at: {args.sf}", file=sys.stderr)
        sys.exit(1)

    with open(args.fens, "r", encoding="utf-8") as f:
        fens = [ln.strip() for ln in f if ln.strip()]

    print(f"Loaded {len(fens)} FENs. Launching Stockfish (depth={args.depth}, threads={args.threads})...")

    proc = subprocess.Popen(
        [args.sf],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def read_until(marker):
        lines = []
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish exited unexpectedly")
            lines.append(line.rstrip("\n"))
            if marker in line:
                return lines

    send("uci")
    read_until("uciok")
    send(f"setoption name Threads value {args.threads}")
    send(f"setoption name Hash value {args.hash}")
    send("isready")
    read_until("readyok")

    labeled = 0
    skipped = 0
    t0 = time.time()
    with open(args.labels, "w", encoding="utf-8") as out:
        for i, fen in enumerate(fens):
            send("ucinewgame")
            send(f"position fen {fen}")
            send(f"go depth {args.depth}")
            out_lines = read_until("bestmove")

            stm_score = None
            for ln in reversed(out_lines):
                if ln.startswith("info ") and " score " in ln:
                    s = score_from_line(ln)
                    if s is not None:
                        stm_score = s
                        break

            if stm_score is None:
                skipped += 1
                continue

            # Stockfish score is from side-to-move POV. Flip to white POV.
            parts = fen.split()
            stm = parts[1] if len(parts) > 1 else "w"
            white_pov = stm_score if stm == "w" else -stm_score
            out.write(f"{fen}|{white_pov}\n")
            labeled += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(fens):
                dt = time.time() - t0
                rate = (i + 1) / dt if dt > 0 else 0
                eta = (len(fens) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(fens)}] labeled={labeled} skipped={skipped} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min")

    send("quit")
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    print(f"\nDone. Labeled {labeled} positions to {args.labels} (skipped {skipped}).")


if __name__ == "__main__":
    main()
