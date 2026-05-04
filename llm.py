import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

### Creating a dataset using class

class StoryClassifier(Dataset):
    def __init__(self, data_dir, transform=None):
        pass
    