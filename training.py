import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import StoryDataset
from tokenizer import Tokenizer
from model import TransformerModel
import time
import os

### Create device, runs GPU unless cuda is unavailable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Loading the text

files = [
    "story.txt",
    "55.txt",
    "1184.txt",
    "2701.txt",
    "3011.txt",
    "78717.txt"
]

text = ""

for file in files:
    with open(f"dataset/{file}", encoding="utf-8", errors="ignore") as f:
        text += f.read() + "\n"

# the vocab json and tokenizer
vocab = Tokenizer.create_vocab(text)
tokenizer = Tokenizer(vocab)

with open("vocab/vocab.json", "w") as f:
    json.dump(vocab, f)

with open("vocab/vocab.json") as f:
    vocab = json.load(f)

### Creating dataset

dataset = StoryDataset(text=text, tokenizer=tokenizer, seq_len=32)

### Creating dataloader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

### Initializing Model

model = TransformerModel(vocab_size=len(tokenizer.vocab_encode)).to(device)
if os.path.exists("model.pt"):
    try:
        model.load_state_dict(
            torch.load("model.pt", map_location=device)
        )
        print("Loading existing model")

    except RuntimeError:
        print("Checkpoint incompatible. Training from scratch.")

model.train()

### Optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4)

loss_fn = nn.CrossEntropyLoss()

### Training the model

for epoch in range(10):
    for x, y in dataloader:
        
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: {loss.item()}")


### Saving the model

torch.save(model.state_dict(), "model.pt")