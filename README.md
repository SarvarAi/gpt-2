# GPT-2 Implementation

A clean, educational implementation of GPT-2 from scratch using PyTorch. This repository contains all the necessary components to train and run inference with a transformer-based language model.

---

## 📁 Project Structure

### **config.py**
Defines the configuration and hyperparameters for the GPT-2 model.

**Key Components:**
- `GPTConfig` dataclass: Centralized configuration management
- Model architecture parameters:
  - `vocab_size` (50,257): Number of tokens in the vocabulary
  - `context_length` (1,024): Maximum sequence length
  - `emb_dim` (768): Embedding dimension (12 heads × 64 dim)
  - `n_heads` (12): Number of attention heads
  - `n_layers` (12): Number of transformer blocks
  - `drop_rate` (0.1): Dropout probability
  - `qkv_bias` (False): Whether to use bias in query/key/value projections

**Usage:**
```python
from config import GPT_CONFIG
# Access model parameters
print(GPT_CONFIG.vocab_size)
```

---

### **model.py**
Contains the core transformer architecture implementation.

**Key Components:**

1. **LayerNorm** - Custom layer normalization
   - Applies normalization with learnable scale and shift parameters
   - Essential for training stability

2. **FeedForward** - Position-wise feedforward network
   - Two linear layers with GELU activation
   - Expands to 4× embedding dimension

3. **MultiHeadAttention** - Self-attention mechanism
   - Implements scaled dot-product attention with causal masking
   - Supports multiple attention heads
   - Includes dropout for regularization
   - Projects to Q, K, V and combines head outputs

4. **TransformerBlock** - Complete transformer block
   - Combines attention and feedforward layers
   - Implements residual connections (skip connections)
   - Layer normalization before each sub-layer

5. **GPTModel** - Main model class
   - Token and position embeddings
   - Stack of transformer blocks
   - Final layer normalization and output projection
   - Generates logits for the vocabulary

**Architecture:**
```
Input Tokens
    ↓
[Token Embedding + Position Embedding]
    ↓
[Transformer Block × 12]
    ↓
[Layer Norm]
    ↓
[Linear Projection to Vocab Size]
    ↓
Output Logits (vocab_size predictions)
```

---

### **utils.py**
Utility functions for device management, text processing, and token generation.

**Key Functions:**

1. **get_device()** → `torch.device`
   - Automatically selects the best available device
   - Priority: GPU (CUDA) → Apple GPU (MPS) → CPU
   - Prints the selected device

2. **generate_tokens_greedy()** → `torch.Tensor`
   - Generates new tokens using greedy decoding
   - Takes current tokens and generates up to `max_new_tokens`
   - Respects the model's `context_size` limit
   - Returns tensor of token IDs

3. **encode_text_to_token_ids()** → `torch.Tensor`
   - Converts text string to token IDs using GPT-2 tokenizer
   - Handles special tokens
   - Returns tensor with batch dimension

4. **decode_token_ids_to_text()** → `str`
   - Converts token IDs back to readable text
   - Inverse operation of encoding
   - Removes batch dimension before decoding

**Example Usage:**
```python
tokenizer = tiktoken.get_encoding("gpt2")
tokens = encode_text_to_token_ids("Hello world", tokenizer)
text = decode_token_ids_to_text(tokens, tokenizer)
```

---

### **run_inference.py**
Main script for running inference with a trained GPT-2 model.

**Key Functions:**

1. **load_checkpoint_state_dict()** → `OrderedDict`
   - Loads the model weights from a saved checkpoint
   - Maps to the specified device (CPU/GPU)
   - Extracts the state dictionary from the checkpoint

2. **generate_response_from_prompt()** → `str`
   - Complete inference pipeline
   - Takes a text prompt and generates a continuation
   - Returns the generated text as a string

**Inference Pipeline:**
```
Prompt (str)
    ↓
[Load model weights from checkpoint]
    ↓
[Move model to device]
    ↓
[Encode prompt to tokens]
    ↓
[Generate new tokens (max 20)]
    ↓
[Decode tokens back to text]
    ↓
Response (str)
```

**Example:**
```python
prompt = "Hello, it is time"
response = generate_response_from_prompt(prompt)
print(f"Model: {response}")
```

---

## 🚀 Quick Start

### Requirements
- PyTorch
- tiktoken (GPT-2 tokenizer)

### Running Inference

```bash
python run_inference.py
```

This will:
1. Load the pre-trained model from `gpt2_model.pth`
2. Generate a response to the default prompt
3. Print the user prompt and model response

---

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 50,257 |
| Context Length | 1,024 tokens |
| Embedding Dimension | 768 |
| Attention Heads | 12 |
| Transformer Blocks | 12 |
| Total Parameters | ~124M |

---

## 🔄 Data Flow

```
run_inference.py
    ↓
[Load GPT_CONFIG from config.py]
    ↓
[Initialize GPTModel from model.py]
    ↓
[Load weights using load_checkpoint_state_dict()]
    ↓
[Encode prompt using encode_text_to_token_ids()]
    ↓
[Generate tokens using generate_tokens_greedy()]
    ↓
[Decode output using decode_token_ids_to_text()]
    ↓
Display result
```

---

## 📝 Notes

- The model uses **greedy decoding** (argmax selection) for text generation
- **Causal masking** ensures the model can only attend to previous tokens
- **Residual connections** enable efficient training of deep networks
- **Layer normalization** is applied before attention and feedforward for stability

---

## 📂 Files Reference

| File | Purpose | Key Class/Function |
|------|---------|-------------------|
| `config.py` | Configuration | `GPTConfig` |
| `model.py` | Architecture | `GPTModel`, `TransformerBlock` |
| `utils.py` | Utilities | `generate_tokens_greedy()`, `encode_text_to_token_ids()` |
| `run_inference.py` | Inference | `generate_response_from_prompt()` |
| `gpt2_model.pth` | Weights | Pre-trained model checkpoint |

---

**Created:** December 2025  
**Model:** GPT-2 (Educational Implementation)
