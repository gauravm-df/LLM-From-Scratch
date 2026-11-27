import os
import sys
from pathlib import Path
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="My Offline Custom LLM",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My Custom LLM (Zero-Scratch)")
st.caption("A 32.5M Parameter Decoder-Only Transformer Trained on WhatsApp & Local Datasets — 100% Offline")

chat_dir = PROJECT_ROOT / "05_finetune" / "chat_model"
base_dir = PROJECT_ROOT / "04_pretrain" / "base_model"
model_dir = chat_dir if chat_dir.exists() else base_dir

@st.cache_resource
def load_llm_pipeline():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if not model_dir.exists():
        return None, None, device
    tok = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    mod = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device)
    mod.eval()
    return tok, mod, device

tokenizer, model, device = load_llm_pipeline()

if model is None:
    st.warning(f"Model weights not detected at {model_dir}. Please complete training (Phase 4 & 5) first.")
else:
    st.success(f"Model loaded and active on **{device.upper()}** acceleration.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "Hello! I am your custom offline AI assistant built from scratch. Ask me anything!"}
        ]

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_prompt := st.chat_input("Type your message here..."):
        st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        formatted = (
            "<|im_start|>system\n"
            "You are a friendly, natural English WhatsApp conversational companion.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.2,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        raw_reply = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        assistant_reply = raw_reply.split("<|im_end|>")[0].strip()

        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.write(assistant_reply)

# Run with: streamlit run 06_inference/app_streamlit.py

