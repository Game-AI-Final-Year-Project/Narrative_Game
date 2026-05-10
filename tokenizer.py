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
        " Create a vocabulary from a dataset. "

        words = dataset.split()
        vocab = {
            token: index
            for index, token in enumerate(sorted(list(set(words))))
        }

        # Adding unknown token
        vocab["<unk>"] = len(vocab)

        return vocab

    def __init__(self, vocab):
        " Initialize the tokenizer. "

        self.vocab_encode = vocab
        self.vocab_decode = {v: k for k, v in vocab.items()}

    def encode(self, text):
        """
        Encode a text in level character.

        Args:
            text (str): Input text to be encoded.

        Returns:
            List[int]: List with token indices.
        """
        words = text.split()
        return [self.vocab_encode.get(word, self.vocab_encode["<unk>"]) for word in words]

    def decode(self, tokens):
        """
        Decode a list of token indices.

        Args:
            indices (List[int]): List of token indices.

        Returns:
            str: The decoded text.
        """
        return "".join([self.vocab_decode.get(t, "<unk>") for t in tokens])
    
