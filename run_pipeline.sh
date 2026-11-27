#!/bin/bash
# ============================================================
# 1-CLICK PIPELINE EXECUTION SCRIPT
# Runs Stage 1 -> Stage 2 -> Stage 4 -> Stage 5 sequentially
# ============================================================

set -e

echo "============================================================"
echo "STARTING FULL LLM SCRATCH TRAINING PIPELINE"
echo "============================================================"

# Check if venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "Activating virtual environment venv..."
        source venv/bin/activate
    else
        echo "Warning: No virtual environment detected. Running with system python."
    fi
fi

# Step 1: Extract raw text corpus
echo ""
echo ">>> [1/5] Extracting raw pre-training text..."
python3 01_data/prepare_pretrain_corpus.py

# Step 2: Train custom BPE tokenizer
echo ""
echo ">>> [2/5] Training custom BPE tokenizer from scratch..."
python3 02_tokenizer/train_tokenizer.py

# Step 3: Run Stage 1 Pre-training (Causal LM)
echo ""
echo ">>> [3/5] Running Stage 1 Pre-Training (Next-Token Prediction)..."
python3 04_pretrain/train_base.py

# Step 4: Run Stage 2 Supervised Fine-Tuning (SFT)
echo ""
echo ">>> [4/5] Running Stage 2 Supervised Fine-Tuning (Chat Alignment)..."
python3 05_finetune/train_sft.py

# Step 5: Test
echo ""
echo "============================================================"
echo "TRAINING PIPELINE FINISHED SUCCESSFULLY!"
echo "Now launch interactive chat by running:"
echo "    python3 06_inference/cli_chat.py"
echo "Or start the REST API:"
echo "    uvicorn 06_inference.server_fastapi:app --host 127.0.0.1 --port 8000"
echo "Or start the Web UI:"
echo "    streamlit run 06_inference/app_streamlit.py"
echo "============================================================"

