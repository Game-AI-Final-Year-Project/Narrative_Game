import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model import TransformerModel
from tokenizer import Tokenizer

from bs4 import BeautifulSoup
import requests

import kagglehub

# Download latest version
path = kagglehub.dataset_download("amarnathr/story-generation-dataset-10k")

print("Path to dataset files:", path)

with open("index.html") as fp:
    soup = BeautifulSoup(fp)

book_ids = [78717, 2701, 3011, 1184, 55]

for book_id in book_ids:
    
    page_url = f"https://www.gutenberg.org/ebooks/{book_id}"
    
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")

    txt_link = None

    for link in soup.find_all("a"):
        href = link.get("href", "")

        if "txt.utf-8" in href:
            txt_link = "https://www.gutenberg.org" + href
            break

    if txt_link:
        
        text = requests.get(txt_link).text

        with open(f"dataset/{book_id}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Downloaded {book_id}")

### Creating a dataset using class

class StoryDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        encoded = tokenizer.encode(text)
        
        self.inputs = []
        self.targets = []

        for i in range(len(encoded) - seq_len):
            self.inputs.append(encoded[i:i+seq_len])
            self.targets.append(encoded[i+1:i+seq_len+1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )
