import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class Tokenizer:

    @staticmethod
    def create_vocab(dataset):
        """
        Create a vocabulary from a dataset.

        Args:
            dataset (str): Text dataset to be used to create the character vocab.

        Returns:
            Dict[str, int]: Character vocabulary.
        """
        vocab = {
            token: index
            for index, token in enumerate(sorted(list(set(dataset))))
        }

        # Adding unknown token
        vocab["<unk>"] = len(vocab)

        return vocab

    def __init__(self, vocab):
        """
        Initialize the tokenizer.

        Args:
            vocab (Dict[str, int]): Vocabulary.
        """
        self.vocab_encode = {str(k): int(v) for k, v in vocab.items()}
        self.vocab_decode = {v: k for k, v in self.vocab_encode.items()}

    def encode(self, text):
        """
        Encode a text in level character.

        Args:
            text (str): Input text to be encoded.

        Returns:
            List[int]: List with token indices.
        """
        return [self.vocab_encode.get(char, self.vocab_encode["<unk>"]) for char in text]

    def decode(self, indices):
        """
        Decode a list of token indices.

        Args:
            indices (List[int]): List of token indices.

        Returns:
            str: The decoded text.
        """
        return "".join([self.vocab_decode.get(idx, "<unk>") for idx in indices])

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
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x)
    
dataset = StoryDataset(data_dir)

# Iterate over the dataset


### Create dataloader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)