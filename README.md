# Gungnir

A UCI chess engine in C++, targeting **~3300–3500 Elo**.

Built from scratch as a learning project. Uses [Stockfish's NNUE architecture](https://github.com/official-stockfish/Stockfish) (HalfKAv2_hm) for evaluation, with an independent C++ implementation.

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

At v0.0.1 it just prints the starting position. Real engine behavior arrives in later milestones.

## Roadmap

| Version | Feature |
| --- | --- |
| v0.0.1 | ✅ Project scaffold |
| v0.1   | Bitboards + move generation (perft passes) |
| v0.2   | Make/unmake + Zobrist |
| v0.3   | Basic alpha-beta search + material eval + UCI |
| v0.4   | HalfKAv2_hm NNUE integration + incremental accumulator |
| v0.5   | Search refinements (LMR, NMP, LMP, singular extensions, ...) |
| v0.6   | LazySMP multi-threading |
| v1.0   | Tuned, tested, competitive |

## Testing

Uses [fastchess](https://github.com/Disservin/fastchess) for engine-vs-engine matches with SPRT.

## License

Copyright (c) 2026 Gungnir-Dev. All rights reserved. (License TBD.)
