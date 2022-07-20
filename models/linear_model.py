import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_Classifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 153739,
        num_labels: int = 13
    ) -> None:
        super(Linear_Classifier, self).__init__()
        fc1 = nn.Linear(vocab_size, vocab_size//10)
        fc2 = nn.Linear(vocab_size//10, vocab_size//100)
        fc3 = nn.Linear(vocab_size//100, vocab_size//1000)
        fc4 = nn.Linear(vocab_size//1000, num_labels)
        leaky_relu = nn.LeakyReLU(0.2)
        self.model = nn.Sequential(
            fc1, leaky_relu, fc2, leaky_relu, fc3, leaky_relu, fc4)

    def forward(self, x: torch.Tensor):
        output = self.model(x)
        return F.log_softmax(output, dim=1)
