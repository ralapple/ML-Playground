'''
Simple text classifier using Pytorch
'''
import csv
import torch
import torch.nn as nn
from torchtext.datasets import IMDB
import numpy as np


# load training set
train_iter = IMDB(split='train')

# convert from iteratable to numpy matrix

# Split lines into tokens with corresponding labels
def tokenize(line):
    return line.split()

tokens = []


train_data = []

for label, line in train_iter:
    tokens += tokenize(line)
    train_data.append([label, line])


train_data = np.array(train_data)
print(train_data.shape)


# Batching

def generate_batch(batch):
    batch_indices = []
    batch_labels = []
    offsets = [0]

    for text_indices, sequence_length, label in batch:
        batch_indices.extend(text_indices)
        batch_labels.append(label)
        offsets.append(sequence_length)

    batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return batch_indices, offsets, batch_labels



class MLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
