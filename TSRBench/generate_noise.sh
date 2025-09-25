#!/bin/bash

# Generate corrupted benchmark datasets for TSRBench
# Prerequisites: Download datasets and place them under ./dataset/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$ROOT_DIR/dataset"

echo "=== Generating corrupted datasets ==="

cd "$SCRIPT_DIR"

# ETT datasets (spot_depth=0.01)
for name in ETTm1 ETTm2 ETTh1 ETTh2; do
    echo ""
    echo "--- Noise injection: ${name} ---"
    python -u tsrbench.py \
        --data-path "${name}.csv" \
        --root-path "$DATASET_DIR/ETT-small/" \
        --output-path "$DATASET_DIR/ETT-small/${name}_noise/" \
        --spot-type bidspot \
        --spot-n-points 8 \
        --spot-depth 0.01
done

# Electricity (spot_depth=0.02)
echo ""
echo "--- Noise injection: electricity ---"
python -u tsrbench.py \
    --data-path electricity.csv \
    --root-path "$DATASET_DIR/electricity/" \
    --output-path "$DATASET_DIR/electricity/electricity_noise/" \
    --spot-type bidspot \
    --spot-n-points 8 \
    --spot-depth 0.02

# Weather (spot_depth=0.01)
echo ""
echo "--- Noise injection: weather ---"
python -u tsrbench.py \
    --data-path weather.csv \
    --root-path "$DATASET_DIR/weather/" \
    --output-path "$DATASET_DIR/weather/weather_noise/" \
    --spot-type bidspot \
    --spot-n-points 8 \
    --spot-depth 0.01

echo ""
echo "=== Done! ==="
