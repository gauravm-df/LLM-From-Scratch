# 🚀 Custom LLM From Scratch (Pure Offline & Zero-Dependency)

A complete, self-contained implementation to build, train, and deploy your own **32.5-Million Parameter Large Language Model (LLM)** from absolute scratch using the proprietary datasets already included in this folder.

> **🛡️ Strict Zero-Existing Guarantee:**
> - **ZERO Pre-Trained Weights:** No LLaMA, Mistral, or Qwen base weights downloaded. All weights start as random Gaussian numbers (`torch.randn`).
> - **ZERO External APIs:** No OpenAI, Google Gemini, or Claude API calls.
> - **100% Offline & Self-Contained:** Runs completely on your local compute without requiring internet access during training or inference.

---

## 📦 Datasets Included in this Project (`01_data/`)

All training and evaluation datasets are **already placed inside `01_data/`**:

| Dataset File | Size | Records / Details | Purpose in Pipeline |
| :--- | :--- | :--- | :--- |
| **`whatsapp_chat_14lakh.csv`** | **172 MB** | **1,400,000 Messages**<br>(Columns: `message_id`, `conversation_id`, `timestamp`, `sender`, `receiver`, `message`) | **Stage 1: Pre-Training Corpus**<br>Extracted into continuous text so the model learns grammar, syntax, and next-word prediction. |
| **`train_whatsapp_700k.jsonl`** | **476 MB** | **700,000 Dialogues**<br>(Formatted in standard ChatML with `<|im_start|>` & `<|im_end|>`) | **Stage 2: Supervised Fine-Tuning (SFT)**<br>Teaches the pre-trained base model how to respond like a conversational human assistant. |
| **`val_whatsapp_20k.jsonl`** | **14 MB** | **20,000 Dialogues**<br>(100% Zero overlap with training set) | **Evaluation & Validation**<br>Used to evaluate Perplexity and validation loss to ensure the model does not overfit. |
| **`people_100k_records.csv`** | **7.2 MB** | **100,000 KYC Profile Rows**<br>(Name, father, mother, age, Aadhaar, PAN) | **Domain Reference Database**<br>Used for entity lookup queries. |
| **`train_ready_100k.jsonl`** | **59 MB** | **100,000 Entity Q&A Pairs**<br>(ChatML & strict JSON profile extraction) | **Structured Extraction Tuning**<br>Teaches the model structured JSON outputs and identity queries. |

---

## 📁 Repository Structure

```text
Custom_LLM_Project/
├── README.md                       # Complete documentation & admin setup guide
├── requirements.txt                # Locked Python dependencies
├── setup_env.sh                    # 1-click isolated venv setup (Linux / macOS)
├── setup_env.bat                   # 1-click isolated venv setup (Windows)
├── run_pipeline.sh                 # 1-click complete training pipeline runner
├── 01_data/                        # ALL DATASETS ARE PLACED HERE
│   ├── prepare_pretrain_corpus.py  # Script to extract 1.4M raw messages
│   ├── whatsapp_chat_14lakh.csv    # 14 Lakh WhatsApp message logs (172 MB)
│   ├── train_whatsapp_700k.jsonl   # 700k SFT training dialogues (476 MB)
│   ├── val_whatsapp_20k.jsonl      # 20k validation dialogues (14 MB)
│   ├── people_100k_records.csv     # 100k citizen records (7.2 MB)
│   └── train_ready_100k.jsonl      # 100k entity extraction pairs (59 MB)
├── 02_tokenizer/
│   ├── train_tokenizer.py          # Trains Byte-Level BPE tokenizer from scratch
│   └── model/                      # Output: vocab.json (16,384 tokens) & merges.txt
├── 03_model/
│   └── configuration.py            # 32.5M LLaMA/Mistral Decoder-Only Transformer
├── 04_pretrain/
│   ├── train_base.py               # Next-token prediction pre-training loop
│   └── base_model/                 # Output: Pre-trained foundation weights
├── 05_finetune/
│   ├── train_sft.py                # Supervised Fine-Tuning (SFT) with ChatML
│   └── chat_model/                 # Output: Conversational assistant weights
└── 06_inference/
    ├── cli_chat.py                 # Real-time interactive terminal chatbot
    ├── server_fastapi.py           # Production REST API (Swagger at /docs)
    └── app_streamlit.py            # ChatGPT-style Browser Web UI
```

---

## 🖥️ How to Run on Another System (Step-by-Step)

You can copy this entire `Custom_LLM_Project` folder to **ANY other computer** (Ubuntu Linux GPU, Windows PC, another Mac, or Google Colab).

All paths are dynamically resolved relative to the project root, so **no code modifications are needed**.

---

### Step 1: Set Up Isolated Environment (System Safe)

#### On Linux / macOS:
```bash
cd Custom_LLM_Project

# Run the 1-click setup script:
chmod +x setup_env.sh run_pipeline.sh
./setup_env.sh

# Or do it manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### On Windows (Command Prompt / PowerShell):
```cmd
cd Custom_LLM_Project

# Run the 1-click setup script:
setup_env.bat

# Or do it manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### On Google Colab / Cloud GPU:
```python
!pip install -r requirements.txt
```

---

### Step 2: Extract Raw Pre-Training Corpus
Extracts text from `01_data/whatsapp_chat_14lakh.csv` into continuous text lines:

```bash
python3 01_data/prepare_pretrain_corpus.py
```
*Output: `01_data/raw_pretrain_corpus.txt`*

---

### Step 3: Train Custom BPE Tokenizer from Scratch
Trains a Byte-Level Byte Pair Encoding (BPE) tokenizer directly on the text:

```bash
python3 02_tokenizer/train_tokenizer.py
```
*Output: `02_tokenizer/model/vocab.json` (16,384 tokens) and `merges.txt`.*  
*Guarantees zero `<unk>` out-of-vocabulary errors.*

---

### Step 4: Run Stage 1 Pre-Training (Language Acquisition)
Trains the 32.5M parameter blank Transformer on next-token prediction:

```bash
python3 04_pretrain/train_base.py
```
- Hardware is automatically detected: **Apple Silicon GPU (`mps`)**, **NVIDIA GPU (`cuda`)**, or **CPU fallback**.
- Evaluates Perplexity ($PPL = e^{\text{loss}}$).
- *Output: `04_pretrain/base_model/`*

---

### Step 5: Run Stage 2 Supervised Fine-Tuning (SFT)
Aligns the base model to act as a friendly conversational assistant using `01_data/train_whatsapp_700k.jsonl`:

```bash
python3 05_finetune/train_sft.py
```
- Applies response-only loss masking so the model learns specifically how to reply.
- *Output: `05_finetune/chat_model/`*

---

### Step 6: Choose Your Inference Mode

#### Option A: Interactive Terminal Chat (Instant Testing)
```bash
python3 06_inference/cli_chat.py
```

#### Option B: Production REST API Server
```bash
uvicorn 06_inference.server_fastapi:app --host 127.0.0.1 --port 8000
```
- Interactive API documentation at: `http://127.0.0.1:8000/docs`

#### Option C: Browser Web UI (ChatGPT Style)
```bash
streamlit run 06_inference/app_streamlit.py
```

---

## ⚡ 1-Click Complete Pipeline Runner
To run Steps 2 through 5 automatically in sequence:

```bash
./run_pipeline.sh
```

---

## 📊 Model Specifications

| Specification | Value | Purpose |
| :--- | :--- | :--- |
| **Architecture** | Decoder-Only Transformer | Modern autoregressive language modeling |
| **Parameters** | ~32.5 Million | Sized for fast laptop/desktop execution |
| **Vocabulary** | 16,384 tokens | Compact embedding table to conserve memory |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 | Token vector embedding size |
| **Intermediate Size ($d_{\text{ffn}}$)** | 1,376 | SwiGLU feed-forward projection |
| **Layers** | 8 Blocks | Depth for grammar, conversational flow, and reasoning |
| **Attention Heads** | 8 Heads | Multi-head self-attention with causal mask |
| **Context Window** | 512 tokens | Optimal sequence length for messaging |
| **Weight Footprint** | ~65 MB (fp16) | Runs in $<1$ GB RAM/VRAM during inference |
