from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessing import vocab_size
from metrics.accuracy import accuracy

vector_size = vocab_size
metric = accuracy


def train(model: nn.Sequential,
          device: torch.device,
          train_loader: DataLoader,
          optimizer,
          criterion) -> None:
    model.train()
    for data in train_loader:
        bow_vectors = data[0].view(-1, vector_size).to(device)
        labels = data[1].to(device)
        optimizer.zero_grad()
        output = model(bow_vectors)
        labels = labels.long()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()


def test(model: nn.Sequential,
         device: torch.device,
         test_loader: DataLoader,
         criterion) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for bow_vectors, labels in test_loader:
            bow_vectors = bow_vectors.view(-1, vector_size).to(device)
            labels = labels.to(device)
            output = model(bow_vectors)
            acc += accuracy(output, labels)
            loss = criterion(output, labels)
            test_loss += loss.item()
    return (test_loss / len(test_loader.dataset),
            acc / len(test_loader.dataset))
