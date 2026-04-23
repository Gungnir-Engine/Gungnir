"""
extract_lichess_evals.py — Convert Lichess CC0 eval DB to our training format.

Input: Lichess's `lichess_db_eval.jsonl.zst` (download from
       https://database.lichess.org/#evals — published under CC0 public
       domain, no GPL entanglement).

Output: one `FEN|score_cp` line per position. `score_cp` is centipawns from
        WHITE's point of view (matches `gungnir labelfens` output). Mate
        scores encoded as ±(30000 − mate_distance_in_plies). Scores are
        capped at ±3000 cp to avoid saturating the training signal.

Usage:
    python tools/extract_lichess_evals.py data/lichess_db_eval.jsonl.zst \\
        data/labels.txt --max 1000000

Streams through the zst file — memory footprint stays small even for the
full 20 GB archive. Stop early with --max.

The Lichess JSONL format per line:
    {"fen": "...", "evals": [
        {"pvs": [{"cp": ..., "line": "..."}, ...], "knodes": ..., "depth": ...},
        ...
    ]}
FENs in this file are 4-field ("board stm castling ep") — we append "0 1"
so they're 6-field as Gungnir's parser expects.
"""
import argparse
import io
import json
import sys

try:
    import zstandard as zstd
except ImportError:
    print("ERROR: pip install zstandard", file=sys.stderr)
    sys.exit(1)

MATE_CONVENTION = 30000   # ±(30000 − plies_to_mate)


def pick_deepest_pv(evals):
    """Return the (cp, mate) tuple from the deepest eval's first pv, or (None, None)."""
    if not evals:
        return None, None
    best = max(evals, key=lambda e: e.get('depth', 0))
    pvs = best.get('pvs', [])
    if not pvs:
        return None, None
    pv0 = pvs[0]
    return pv0.get('cp'), pv0.get('mate')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Compressed .jsonl.zst file')
    ap.add_argument('output', help='Output labels .txt file')
    ap.add_argument('--max', type=int, default=-1,
                    help='Stop after this many written positions (default: unlimited)')
    ap.add_argument('--cap', type=int, default=3000,
                    help='Clamp |cp| at this magnitude (default: 3000)')
    args = ap.parse_args()

    n_read = 0
    n_written = 0
    n_skipped = 0

    with open(args.input, 'rb') as fz, open(args.output, 'w', encoding='utf-8') as out:
        dctx = zstd.ZstdDecompressor()
        raw_reader = dctx.stream_reader(fz)
        text = io.TextIOWrapper(raw_reader, encoding='utf-8', newline='\n')

        for line in text:
            n_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            fen = obj.get('fen', '')
            parts = fen.split()
            if len(parts) < 4:
                n_skipped += 1
                continue
            stm = parts[1]

            cp, mate = pick_deepest_pv(obj.get('evals', []))
            if cp is None and mate is None:
                n_skipped += 1
                continue

            # Convert to a single cp-like number (our convention matches
            # gungnir's internal score representation).
            if mate is not None:
                # Lichess mate value is positive when STM mates, negative otherwise.
                score = MATE_CONVENTION - abs(mate)
                if mate < 0:
                    score = -score
            else:
                score = cp
                if score > args.cap:  score =  args.cap
                elif score < -args.cap: score = -args.cap

            # Flip STM → WHITE POV.
            if stm == 'b':
                score = -score

            # Append halfmove + fullmove fields so the FEN is 6-field.
            if len(parts) == 4:
                fen_out = fen + ' 0 1'
            else:
                fen_out = fen

            out.write(f'{fen_out}|{score}\n')
            n_written += 1

            if n_written % 50000 == 0:
                print(f'  read={n_read}  written={n_written}  skipped={n_skipped}',
                      file=sys.stderr, flush=True)

            if args.max > 0 and n_written >= args.max:
                break

    print(f'\nDone. read={n_read}  written={n_written}  skipped={n_skipped}',
          file=sys.stderr)
    print(f'  -> {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
