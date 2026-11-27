from transformers import LlamaConfig, LlamaForCausalLM

def get_custom_llm_config(vocab_size=16384):
    """
    Returns a balanced 32.5M parameter LLaMA Decoder-Only Transformer
    specifically optimized for personal laptops (Apple Silicon MPS / NVIDIA Laptop / CPU).
    """
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,               # Token vector dimension
        intermediate_size=1376,        # SwiGLU FFN dimension (~2.7x)
        num_hidden_layers=8,           # 8 Transformer Blocks
        num_attention_heads=8,         # 8 Attention Heads (64 dims per head)
        max_position_embeddings=512,   # 512 context length
        pad_token_id=1,                # <pad>
        bos_token_id=0,                # <s>
        eos_token_id=2                 # </s>
    )
    return config

def create_blank_model(vocab_size=16384):
    """
    Instantiates the model with ZERO pre-existing weights.
    All weights are initialized randomly.
    """
    config = get_custom_llm_config(vocab_size)
    model = LlamaForCausalLM(config)
    return model

if __name__ == "__main__":
    model = create_blank_model()
    params = model.num_parameters()
    print("=" * 65)
    print("CUSTOM TRANSFORMER ARCHITECTURE (PURE SCRATCH)")
    print(f"Total Trainable Parameters: {params:,} (~{params/1e6:.1f} Million)")
    print("Zero pre-existing weights. Ready for pre-training.")
    print("=" * 65)

