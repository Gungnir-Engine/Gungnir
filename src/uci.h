// Gungnir — UCI protocol loop. Reads commands on stdin, writes responses on
// stdout. Supports: uci, isready, ucinewgame, position, go (depth/movetime/
// wtime/btime), quit. Skips: ponder, multipv, custom options.

#pragma once

namespace gungnir {

int uci_loop();

}  // namespace gungnir
