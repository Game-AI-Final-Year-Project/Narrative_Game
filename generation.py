import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import Tokenizer
from model import TransformerModel

def generate(model, tokenizer, prompt, max_len=200, temperature=1.0):
    
    device = next(model.parameters()).device
    model.eval()
    tokens = tokenizer.encode(prompt)

    end_token = tokenizer.vocab_encode["<END>"]

    for _ in range(max_len):
        x = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():

            logits = model(x)

        next_token_logits = logits[0, -1]

        # Temperature scaling (temperature controls the predictability)
        next_token_logits /= temperature

        probs = F.softmax(next_token_logits,dim=0)
        next_token = torch.multinomial(probs, 1).item()

        if len(tokens) >= 512:
            break

        # Token continues until the end_token comes up
        tokens.append(next_token)
        if next_token == end_token:
            break

    return tokenizer.decode(tokens)

### Loading the text

with open("vocab/vocab.json") as f:
    vocab = json.load(f)

tokenizer = Tokenizer(vocab)

### The Device will set the device to cuda. This helps the scaling.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### The Model
model = TransformerModel(vocab_size=len(tokenizer.vocab_encode)).to(device)
