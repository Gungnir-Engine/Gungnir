#!/usr/bin/env bash
# ab_all_checkpoints.sh — serialize every .pt checkpoint in data/
# matching a pattern, then run a 20-game A/B gauntlet for each vs
# classical Gungnir. Prints a summary at the end.
#
# Usage: ./tools/ab_all_checkpoints.sh <pattern>
# Example: ./tools/ab_all_checkpoints.sh 'gungnir_v4*.pt'

set -e

cd "$(dirname "$0")/.."
PATTERN="${1:-gungnir_v4*.pt}"
FASTCHESS="E:/Claude/installers/fastchess/fastchess-windows-x86-64/fastchess.exe"
GUNGNIR="E:/Claude/Gungnir/build/Release/gungnir.exe"
OPENINGS="E:/Claude/installers/fastchess/fastchess-windows-x86-64/app/tests/data/openings.epd"
PYTHON="E:/Claude/.venv-gpu/Scripts/python.exe"
MATCHES_DIR="E:/Claude/Gungnir/matches"
DATA_DIR="E:/Claude/Gungnir/data"

mkdir -p "$MATCHES_DIR"

echo "=== Finding checkpoints matching $PATTERN ==="
mapfile -t CKPTS < <(ls $DATA_DIR/$PATTERN 2>/dev/null | sort)
if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "No checkpoints found."
    exit 1
fi
echo "Found ${#CKPTS[@]} checkpoints:"
printf '  %s\n' "${CKPTS[@]}"

SUMMARY_FILE="$MATCHES_DIR/gauntlet_summary_$(date +%s).txt"
echo "Gauntlet summary ($(date))" > "$SUMMARY_FILE"

for CKPT in "${CKPTS[@]}"; do
    STEM="$(basename "$CKPT" .pt)"
    NNUE="$DATA_DIR/$STEM.nnue"
    echo ""
    echo "=== Processing $STEM ==="
    echo "Serializing -> $NNUE"
    "$PYTHON" tools/save_nnue.py "$CKPT" "$NNUE" 2>&1 | tail -2

    PGN="$MATCHES_DIR/${STEM}_vs_classical.pgn"
    echo "Running 20-game A/B vs classical..."
    "$FASTCHESS" \
        -engine "cmd=$GUNGNIR" name="$STEM" "option.NNUEFile=$NNUE" option.UseNNUE=true \
        -engine "cmd=$GUNGNIR" name=classical option.UseNNUE=false \
        -each tc=5+0.05 \
        -rounds 10 -games 2 -repeat \
        -openings "file=$OPENINGS" format=epd order=random plies=8 \
        -concurrency 4 \
        -pgnout file="$PGN" notation=san \
        2>&1 | tail -5 | tee -a "$SUMMARY_FILE"
    echo "---" >> "$SUMMARY_FILE"
done

echo ""
echo "=== Summary written to $SUMMARY_FILE ==="
cat "$SUMMARY_FILE"
