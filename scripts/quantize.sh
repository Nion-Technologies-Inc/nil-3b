#!/bin/bash
# Quantization workflow for nil-3b
# Requirements: llama.cpp compiled binaries

RAW_MODEL_DIR="./models/nil-3b-raw"
F16_GGUF="./models/nil-3b-f16.gguf"
OUTPUT_GGUF="./nil-3b-q4_k_m.gguf"

echo "Step 1: Converting HF model weights to FP16 GGUF format..."
python ../llama.cpp/convert_hf_to_gguf.py $RAW_MODEL_DIR --outfile $F16_GGUF

echo "Step 2: Quantizing FP16 GGUF to 4-bit medium (q4_k_m)..."
../llama.cpp/build/bin/llama-quantize $F16_GGUF $OUTPUT_GGUF q4_k_m

echo "Quantization complete. Final model saved to $OUTPUT_GGUF"