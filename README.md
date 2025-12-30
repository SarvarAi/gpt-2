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

### **train.py**
Comprehensive training module for GPT-2 model with full pipeline support.

**Key Classes:**

1. **GPTDatasetV1** - Custom PyTorch Dataset
   - Creates overlapping sequences using sliding window approach
   - Handles tokenization of raw text
   - Generates input-target pairs for training

**Key Functions:**

1. **create_dataloader_v1()** → `DataLoader`
   - Creates PyTorch DataLoader from raw text
   - Handles tokenization and batching
   - Supports shuffling and configurable batch size

2. **calc_loss_batch()** → `torch.Tensor`
   - Calculates cross-entropy loss for a single batch
   - Handles device placement
   - Flattens logits and targets for loss computation

3. **calc_loss_loader()** → `float`
   - Calculates average loss over entire data loader
   - Supports evaluating on subset of batches
   - Returns mean loss across batches

4. **evaluate_model()** → `Tuple[float, float]`
   - Evaluates model on both train and validation sets
   - Returns (train_loss, val_loss)
   - Switches model to eval mode during evaluation

5. **generate_and_print_sample()** → `None`
   - Generates sample text during training
   - Useful for monitoring training progress
   - Prints generated text for visual inspection

6. **train_model_simple()** → `Tuple[List[float], List[float], List[int]]`
   - Main training loop with periodic evaluation
   - Tracks training/validation losses and tokens seen
   - Generates sample text after each epoch
   - Returns loss histories and token counts

7. **save_checkpoint()** → `None`
   - Saves model state, optimizer state, and config
   - Enables resuming training or inference

8. **load_checkpoint()** → `Dict`
   - Loads checkpoint from disk
   - Returns model state, optimizer state, and config

9. **train()** → `Tuple[List[float], List[float], List[int], torch.nn.Module]`
   - Complete end-to-end training pipeline
   - Handles data loading, model initialization, training, and saving
   - Returns training history and trained model

**Training Pipeline:**
```
Raw Text Data
    ↓
[Split into train/validation]
    ↓
[Create DataLoaders with GPTDatasetV1]
    ↓
[Initialize GPTModel]
    ↓
[Training Loop]
    ├─ Forward pass
    ├─ Backward pass
    ├─ Optimizer step
    ├─ Periodic evaluation
    └─ Sample generation
    ↓
[Save Checkpoint]
    ↓
Trained Model
```

**Example Usage:**
```python
from train import train

# Load training data
with open('training_text.txt', 'r') as f:
    text_data = f.read()

# Train the model
train_losses, val_losses, tokens_seen, model = train(
    text_data,
    num_epochs=5,
    batch_size=2,
    learning_rate=0.0004,
    eval_freq=5,
    start_context="Every effort moves you"
)
```

---

## 🚀 Quick Start

### Requirements
- PyTorch
- tiktoken (GPT-2 tokenizer)

### Training

```bash
python -c "
from train import train
with open('your_text_file.txt', 'r') as f:
    text_data = f.read()
train_losses, val_losses, tokens_seen, model = train(text_data, num_epochs=5)
"
```

Or use it as a module:

```python
from train import train

with open('training_data.txt', 'r') as f:
    text_data = f.read()

train_losses, val_losses, tokens_seen, model = train(
    text_data,
    num_epochs=5,
    batch_size=2,
    learning_rate=0.0004
)
```

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
| `utils.py` | Utilities | `generate_tokens_greedy()`, `encode_text_to_token_ids()` || `train.py` | Training Pipeline | `train()`, `train_model_simple()`, `GPTDatasetV1` || `run_inference.py` | Inference | `generate_response_from_prompt()` |
| `gpt2_model.pth` | Weights | Pre-trained model checkpoint |

---

**Created:** December 2025  
**Model:** GPT-2 (Educational Implementation)
