import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tokenizer import Tokenizer

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model))
        

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        
        seq_len = x.size(1)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device),diagonal=1)

        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x, mask = mask)

        return self.fc(x)


