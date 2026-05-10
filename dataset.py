import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model import TransformerModel
from tokenizer import Tokenizer

### Creating a dataset using class

class StoryDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=64):
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
