# Gungnir tools — NNUE training pipeline

End-to-end recipe for training a custom NNUE for Gungnir.

## Overview

```
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ Positions (FEN list)     │ → │ Labels (FEN | score cp)  │ → │ Trained weights (.nnue)  │
│  - gungnir genfens       │   │  - gungnir labelfens     │   │  - tools/train_halfka.py │
│  - OR Lichess eval DB    │   │  - OR Lichess eval DB    │   │  - tools/save_nnue.py    │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

## Data sources

### Option A — Lichess evaluation database [RECOMMENDED]

Lichess publishes millions of human-played positions with engine evaluations.
Great data, already labeled. Download from:

> https://database.lichess.org/#evals

The file is a zstandard-compressed JSON-lines archive
(`lichess_db_eval.jsonl.zst`). Use `tools/extract_lichess_evals.py` to
stream-extract positions into the `FEN|score_cp` format Gungnir expects.

```
python tools/extract_lichess_evals.py \
    data/lichess_db_eval.jsonl.zst data/labels.txt --max 1000000
```

Recommended volume: 1M+ positions.

### Option B — Gungnir self-labels

Run Gungnir itself as the labeller:

```
gungnir labelfens fens.txt labels.txt 12
```

Ceiling is Gungnir's own strength; useful as a pipeline smoke-test or for
self-distillation once initial weights exist.

### Option C — Game outcomes only

Use PGN game results (1/0/½) as training targets. Much weaker signal than
engine evals, slower convergence.

## Steps

### 1. Generate or download positions

Option A: download Lichess evals (has both positions and labels — skip to step 3).

Option B: self-play:
```
gungnir genfens 500 fens.txt 4 8
```

### 2. Label positions (if not already labeled)

```
gungnir labelfens fens.txt labels.txt 12
```

### 3. Train

```
python tools/train_halfka.py data/labels.txt --out data/gungnir.pt --epochs 50
```

Quantization-aware training: weights are learned directly in int16/int8
space via fake-quantize with straight-through estimator, so the serialized
`.nnue` evaluates correctly with Gungnir's int forward pass.

Output is a PyTorch checkpoint. Convert to a loadable `.nnue`:

```
python tools/save_nnue.py data/gungnir.pt data/gungnir.nnue
```

### 4. Use

```
gungnir
> setoption name NNUEFile value data/gungnir.nnue
> position startpos
> go depth 10
```

Or drop the file at `Gungnir/gungnir.nnue` for auto-load on startup.

## File reference

| File                        | Purpose                                              |
| --------------------------- | ---------------------------------------------------- |
| `extract_lichess_evals.py`  | Extract FEN+cp pairs from Lichess zst archive.       |
| `gungnir_nnue.py`           | Feature extraction + constants (shared library).     |
| `train_halfka.py`           | PyTorch QAT trainer for HalfKAv2_hm.                 |
| `save_nnue.py`              | Serialize PyTorch checkpoint to `.nnue`.             |
| `train_nnue.py`             | Reference simple-net trainer (768 features).        |
