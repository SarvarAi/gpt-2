from collections import OrderedDict

import torch
import tiktoken

from model import GPTModel
from config import GPT_CONFIG
from utils import (
    get_device,
    generate_tokens_greedy,
    encode_text_to_token_ids,
    decode_token_ids_to_text
)

# saved model
CHECKPOINT_PATH = "gpt2_model.pth"

# model architecture
model = GPTModel(GPT_CONFIG)


def load_checkpoint_state_dict(model_path: str, device: torch.device) -> OrderedDict:
    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint["model_state_dict"]

    return model_state


def generate_response_from_prompt(prompt: str) -> str:
    # defining device
    device = get_device()

    # loading the state
    model_state = load_checkpoint_state_dict(CHECKPOINT_PATH, device)

    # Inserting weights into skeleton and move to device
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # initializing gpt-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # predicting next tokens
    token_ids = generate_tokens_greedy(
        model=model,
        idx=encode_text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=20,
        context_size=GPT_CONFIG.context_length
    )

    # converting tokens into the text
    output_text = decode_token_ids_to_text(token_ids, tokenizer)

    return output_text


if __name__ == '__main__':
    user_prompt = "Hello, it is time"
    model_response = generate_response_from_prompt(user_prompt)

    print(f"User: {user_prompt}")
    print("=" * 30)
    print(f"Model: {model_response}")
