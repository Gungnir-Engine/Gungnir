# Gungnir

A UCI chess engine in C++, targeting **~3300–3500 Elo**.

Built from scratch as a learning project. Evaluation uses a HalfKAv2_hm
neural network — a feature-bucketed architecture pioneered by the Stockfish
project and widely adopted in computer chess — implemented here in
independent C++. Weights are trained from scratch on CC0-licensed position
data; Gungnir does **not** ship or depend on any GPL-licensed weights or
source code.

## Building

Requires:

- **MSVC** (Visual Studio Build Tools 2022+) on Windows, or clang/gcc on Linux/Mac
- **CMake** 3.20+
- A CPU with AVX2 (any Intel/AMD chip from 2013+)

```pwsh
cmake -B build
cmake --build build --config Release
```

The binary lands at `build/Release/gungnir.exe` (Windows) or `build/gungnir` (Unix).

## Running

```pwsh
./build/Release/gungnir.exe
```

Without a NNUE weights file the engine uses its classical PeSTO evaluation
(~2300 Elo). To use a compatible NNUE, either drop a `gungnir.nnue` file
beside the executable (auto-loads) or via UCI:

```
setoption name NNUEFile value path/to/your.nnue
```

Training your own weights: see [`tools/README.md`](tools/README.md) for the
CC0-licensed data pipeline (Lichess evaluation database → labels → PyTorch
trainer → serialized `.nnue`).

## Roadmap

| Version | Feature |
| --- | --- |
| v0.0.1 | ✅ Project scaffold |
| v0.1   | ✅ Bitboards + move generation (perft passes) |
| v0.2   | ✅ Make/unmake + Zobrist + threefold repetition |
| v0.3   | ✅ Alpha-beta search + qsearch + PeSTO eval + UCI |
| v0.4   | ✅ HalfKAv2_hm NNUE loader + forward pass + incremental accumulator + AVX2 SIMD |
| v0.5   | ✅ Search refinements (LMR, LMP, futility, PVS, singular extensions, cont/cap history) |
| v0.6   | ✅ LazySMP multi-threading |
| v0.7   | 🚧 Custom-trained weights (CC0 pipeline) |
| v1.0   | Tuned, SPRT-validated, competitive |

## Testing

[fastchess](https://github.com/Disservin/fastchess) (MIT licensed) is used
for engine-vs-engine matches and SPRT. It's an external tool — not bundled
with Gungnir.

## License

Copyright (c) 2026 Gungnir-Engine. All rights reserved. (License TBD —
deliberately keeping options open until weights are fully independent.)
