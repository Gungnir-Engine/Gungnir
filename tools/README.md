# Gungnir tools — NNUE distillation pipeline (GPL-free)

End-to-end recipe for training a custom NNUE for Gungnir without any GPL
dependencies in the data or the code path.

## Licensing stance

Gungnir deliberately avoids GPL dependencies. That means:
- We do NOT ship or auto-load Stockfish's `.nnue` weight files (GPL v3).
- We do NOT use Stockfish or any GPL-derived engine to label training data.
- All training data sources listed here are either CC0 (public domain),
  permissively licensed, or produced by Gungnir itself.

Our engine code is independent of whose weights you feed it — any `.nnue`
file matching the HalfKAv2_hm small-net format (L1=128) or big-net format
(L1=3072) will load. But by default Gungnir runs with **classical PeSTO
eval** (~2300 Elo) unless you provide a compatible weights file.

## Overview

```
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ Positions (FEN list)     │ → │ Labels (FEN | score cp)  │ → │ Trained weights (.nnue)  │
│  - gungnir genfens       │   │  - gungnir labelfens     │   │  - tools/train_nnue.py   │
│  - OR Lichess CC0 DB     │   │  - OR Lichess CC0 DB     │   │  - OR nnue-pytorch repo  │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

## Data sources

### Option A — Lichess evaluation database (CC0 public domain) [RECOMMENDED]

Lichess publishes millions of human-played positions with engine evaluations,
released under CC0. Great data; no license concerns since CC0 is a deliberate
waiver. Download from:

> https://database.lichess.org/#evals

The file format is a JSONL-like archive. Each line has a FEN plus evaluations
at multiple depths. Extract the FEN and the deepest eval per line to produce
our `FEN|score_cp` format. A simple `jq` or Python script handles this.

Recommended volume for meaningful training: 1M+ positions.

### Option B — Gungnir self-labels

Weaker but fully self-contained. Run our own engine over a FEN list at high
depth:

```
gungnir labelfens fens.txt labels.txt 12
```

Ceiling for a net trained on Gungnir labels is Gungnir's own strength (student
can't exceed teacher). With classical-only Gungnir (~2300 Elo), the trained
net will plateau there. Still a useful exercise, and a fine way to smoke-test
the training pipeline end-to-end before moving to Option A for real.

### Option C — Game outcomes only (no engine-derived labels)

Use PGN game results (1/0/½) as training targets. Much weaker signal than
engine evals, slower convergence, but most aligned with the "no GPL anywhere"
stance if that matters.

## Steps

### 1. Generate or download positions

Option A: download Lichess CC0 evals (has both positions and labels — skip to step 3).

Option B: self-play:
```
gungnir genfens 500 fens.txt 4 8
```
500 self-play games at depth 4, 8 random opening plies. Produces ~30k FENs.

### 2. Label positions (if not already labeled)

```
gungnir labelfens fens.txt labels.txt 12
```

At depth 12, throughput is roughly 50-100 positions/sec on a single core.
100k positions ≈ 15-30 minutes.

### 3. Train

```
python tools/train_nnue.py labels.txt weights.bin --epochs 10
```

**Current state: this script is a SKELETON.** It trains a 768-input
PieceSquare net (NOT HalfKAv2_hm), outputs raw float32 weights that Gungnir's
loader does NOT currently consume. Use it to understand the training-loop
shape.

For weights Gungnir can load:
- You can either write a conversion script (raw float32 → `.nnue` binary
  format matching SF small-net layout), OR
- Fork and adapt `nnue-pytorch` (GPL v3 — tool only, output weights are
  yours) locally to produce HalfKAv2_hm output. Keep the trained weights
  out of your repo / distribute under your own terms.

### 4. Use

```
gungnir
> setoption name NNUEFile value path/to/your-trained.nnue
> position startpos
> go depth 10
```

Or drop the file at `Gungnir/gungnir.nnue` for auto-load on startup.

## File reference

| File                  | Purpose                                                     |
| --------------------- | ----------------------------------------------------------- |
| `train_nnue.py`       | PyTorch reference trainer (768 features, not HalfKAv2_hm).  |
| `README.md`           | This file.                                                  |

## Note on the previous `label_fens.py`

An earlier version of this directory included `label_fens.py` that invoked
Stockfish as the labeller. That's been removed to keep the pipeline
GPL-free. The `gungnir labelfens` CLI mode replaces it.
