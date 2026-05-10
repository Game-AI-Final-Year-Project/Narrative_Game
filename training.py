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


### Loading the text

with open("story.txt") as f:
    text = f.read()

vocab = Tokenizer.create_vocab(text)
tokenizer = Tokenizer(vocab)

with open("vocab/vocab.json", "w") as f:
    json.dump(vocab, f)

with open("vocab/vocab.json") as f:
    vocab = json.load(f)

### Creating dataset

dataset = StoryDataset(text=text, tokenizer=tokenizer, seq_len=128)

### Creating dataloader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

### Create device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Initializing Model

model = TransformerModel(vocab_size=len(tokenizer.vocab_encode)).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

### Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss()

### Training the model

for epoch in range(10):
    for x, y in dataloader:
        logits = model(x)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: {loss.item()}")

model.train()

### Saving the model

torch.save(model.state_dict(), "model.pt")