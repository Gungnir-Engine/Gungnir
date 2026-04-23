# Gungnir tools — NNUE distillation pipeline

End-to-end recipe for training a custom NNUE from Stockfish-labeled positions.

## Overview

```
   ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
   │  gungnir genfens   │ →  │  label_fens.py     │ →  │  train_nnue.py     │
   │  (self-play FENs)  │    │  (Stockfish labels)│    │  (Python trainer)  │
   └────────────────────┘    └────────────────────┘    └────────────────────┘
       fens.txt                  labels.txt                   weights.bin
```

## Requirements

- Stockfish binary (for labelling). Default path is hardcoded to
  `E:\Claude\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe`.
  Override with `--sf <path>` on `label_fens.py`.
- Python 3.9+ with PyTorch (`pip install torch`).
- Lots of CPU time — Stockfish at depth 12 labels ~20-40 FENs/sec.

## Step 1 — Generate self-play positions

```
gungnir genfens 500 fens.txt 4 8
```

Plays 500 self-play games at search depth 4 with 8 random opening plies for
diversity. Writes one FEN per ply visited. A 500-game run produces roughly
30,000-50,000 unique positions.

## Step 2 — Label positions with Stockfish

```
python tools/label_fens.py fens.txt labels.txt --depth 12
```

For each FEN, runs `stockfish go depth 12` and writes a `FEN|score_cp` line.
Score is in centipawns from white's POV. Mate scores encoded as ±(30000 −
mate_distance). Throughput roughly 20-40 positions per second per CPU core
at depth 12.

For a meaningful training run, target at least **100,000 labeled positions**
(~1 hour at depth 12 on a single core; faster with multiple Stockfish processes).

## Step 3 — Train

```
python tools/train_nnue.py labels.txt weights.bin --epochs 10 --lr 1e-3
```

**This script is a SKELETON / REFERENCE**, not a working SF-quality trainer.
It implements:

- **Architecture**: 768 (PieceSquare) → 256 → 32 → 1 with clipped ReLU.
  Much simpler than HalfKAv2_hm. Will NOT produce SF-grade output.
- **Loss**: Texel-style MSE on `sigmoid(score / 400)`.
- **Optimizer**: Adam.

The output `weights.bin` is raw float32, NOT a `.nnue` file Gungnir's loader
can consume directly. Wiring it into Gungnir would require either:
- Adding a "trained-by-us" weight-load path alongside the SF .nnue path, or
- Converting these weights into the SF .nnue format (non-trivial — different
  arch, quantization to int8/int16, byte layout).

## Going further — actual SF NNUE training

For a `.nnue` file Gungnir can load directly (HalfKAv2_hm, L1=128, all the
SF quantization), use the official Stockfish PyTorch trainer:

```
git clone https://github.com/official-stockfish/nnue-pytorch
cd nnue-pytorch
pip install -r requirements.txt
# Convert our labels.txt to the .binpack format expected by nnue-pytorch
# (see their README for the conversion scripts).
python train.py --features=HalfKAv2_hm ...
```

This produces an actual `.nnue` that drops in via:
```
gungnir
> setoption name NNUEFile value path/to/your-trained.nnue
```

Realistic scope: SF's own training takes billions of positions and weeks of
GPU time. A modest distillation run on 100k-1M positions can still produce
a usable net but will be substantially weaker than `nn-small.nnue`.

## File reference

| File                  | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `label_fens.py`       | Labels FEN positions with Stockfish evaluations.     |
| `train_nnue.py`       | Reference PyTorch trainer (NOT SF-format output).    |
| `README.md`           | This file.                                           |
