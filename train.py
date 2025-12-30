"""
Training script for GPT-2 model.

This module contains all the necessary functions to:
1. Create data loaders from text
2. Calculate loss on batches and loaders
3. Train the model with evaluation
4. Generate and sample text during training
5. Save checkpoints
"""

from typing import Tuple, List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken

from model import GPTModel
from config import GPT_CONFIG
from utils import (
    get_device,
    generate_tokens_greedy,
    encode_text_to_token_ids, 
    decode_token_ids_to_text
)


# ============================================================================
# DATA LOADING
# ============================================================================

class GPTDatasetV1(Dataset):
    """
    Dataset for GPT-2 training.
    
    Creates overlapping sequences of tokens using a sliding window approach.
    """
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int) -> None:
        """
        Initialize the dataset.
        
        Args:
            txt (str): Raw text data
            tokenizer: Tokenizer (tiktoken GPT-2 encoding)
            max_length (int): Maximum sequence length
            stride (int): Sliding window stride
        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt: str, batch_size: int = 4, max_length: int = 256,
                         stride: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for GPT-2 training.
    
    Args:
        txt (str): Raw text data
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        stride (int): Sliding window stride
        shuffle (bool): Whether to shuffle the data
        drop_last (bool): Whether to drop the last incomplete batch
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


# ============================================================================
# LOSS CALCULATION
# ============================================================================

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Calculate cross-entropy loss for a single batch.
    
    Args:
        input_batch (torch.Tensor): Input token IDs
        target_batch (torch.Tensor): Target token IDs
        model: GPT model
        device: Device to run on (CPU/GPU)
        
    Returns:
        torch.Tensor: Loss value
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader: DataLoader, model: torch.nn.Module, device: torch.device, num_batches: Optional[int] = None) -> float:
    """
    Calculate average loss over a data loader.
    
    Args:
        data_loader: PyTorch DataLoader
        model: GPT model
        device: Device to run on (CPU/GPU)
        num_batches (int): Number of batches to evaluate on (None = all)
        
    Returns:
        float: Average loss
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, eval_iter: int) -> Tuple[float, float]:
    """
    Evaluate the model on train and validation sets.
    
    Args:
        model: GPT model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: Device to run on (CPU/GPU)
        eval_iter (int): Number of batches to evaluate on
        
    Returns:
        tuple: (train_loss, val_loss)
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ============================================================================
# TEXT GENERATION
# ============================================================================

def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """
    Convert text to token IDs.
    
    Args:
        text (str): Input text
        tokenizer: Tokenizer (tiktoken GPT-2 encoding)
        
    Returns:
        torch.Tensor: Token IDs with batch dimension
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """
    Convert token IDs back to text.
    
    Args:
        token_ids (torch.Tensor): Token IDs
        tokenizer: Tokenizer (tiktoken GPT-2 encoding)
        
    Returns:
        str: Decoded text
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model: torch.nn.Module, tokenizer: tiktoken.Encoding, device: torch.device, start_context: str) -> None:
    """
    Generate text from a starting context and print it.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer (tiktoken GPT-2 encoding)
        device: Device to run on (CPU/GPU)
        start_context (str): Starting text context
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_tokens_greedy(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


# ============================================================================
# TRAINING
# ============================================================================

def train_model_simple(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int,
                       eval_freq: int, eval_iter: int, start_context: str, tokenizer: tiktoken.Encoding) -> Tuple[List[float], List[float], List[int]]:
    """
    Main training loop for the GPT model.
    
    Args:
        model: GPT model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: PyTorch optimizer (e.g., AdamW)
        device: Device to run on (CPU/GPU)
        num_epochs (int): Number of training epochs
        eval_freq (int): Evaluation frequency (every N steps)
        eval_iter (int): Number of batches to evaluate on
        start_context (str): Starting context for sample generation
        tokenizer: Tokenizer (tiktoken GPT-2 encoding)
        
    Returns:
        tuple: (train_losses, val_losses, tokens_seen)
            - train_losses: List of training losses
            - val_losses: List of validation losses
            - tokens_seen: List of total tokens processed
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Dict, filepath: str = "gpt2_model.pth") -> None:
    """
    Save model checkpoint.
    
    Args:
        model: GPT model
        optimizer: Optimizer state
        config: Model configuration
        filepath (str): Path to save checkpoint
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, filepath)
    print(f"Model saved to {filepath}")


def load_checkpoint(filepath: str, device: torch.device) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to checkpoint
        device: Device to load on
        
    Returns:
        dict: Checkpoint containing model_state_dict, optimizer_state_dict, config
    """
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Model loaded from {filepath}")
    return checkpoint


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(text_data: str, output_path: str = "gpt2_model.pth", num_epochs: int = 5, batch_size: int = 2,
          learning_rate: float = 0.0004, eval_freq: int = 5, eval_iter: int = 5, train_ratio: float = 0.9,
          start_context: str = "Every effort moves you") -> Tuple[List[float], List[float], List[int], torch.nn.Module]:
    """
    Complete training pipeline for GPT-2.
    
    Args:
        text_data (str): Raw text data for training
        output_path (str): Path to save the model
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        eval_freq (int): Evaluation frequency (every N steps)
        eval_iter (int): Number of batches to evaluate on
        train_ratio (float): Ratio of data for training (rest for validation)
        start_context (str): Starting context for sample generation during training
        
    Returns:
        tuple: (train_losses, val_losses, tokens_seen, model)
    """
    print("=" * 60)
    print("GPT-2 Training Pipeline")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Split data
    print(f"\nPreparing data...")
    split_index = int(len(text_data) * train_ratio)
    train_data = text_data[:split_index]
    val_data = text_data[split_index:]
    print(f"Train data: {len(train_data):,} characters")
    print(f"Validation data: {len(val_data):,} characters")
    
    # Create data loaders
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=GPT_CONFIG.context_length,
        stride=GPT_CONFIG.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=GPT_CONFIG.context_length,
        stride=GPT_CONFIG.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = GPTModel(GPT_CONFIG)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # Train model
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer
    )
    
    print("=" * 60)
    print("Training completed!")
    
    # Save model
    save_checkpoint(model, optimizer, GPT_CONFIG, output_path)
    
    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
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
