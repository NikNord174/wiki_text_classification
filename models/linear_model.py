import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_Classifier(nn.Module):
    def __init__(
        self,
        vector_size: int = 153739,
        num_labels: int = 13,
        devided_factor: int = 20
    ) -> None:
        super(Linear_Classifier, self).__init__()
        layers = []
        # use 2*num_labels to prevent too small difference btw
        # input and output in the last linear layer
        while vector_size > 3 * num_labels:
            layers.append(
                nn.Sequential(
                    nn.Linear(vector_size, vector_size//devided_factor),
                    nn.LeakyReLU(0.2)
                )
            )
            vector_size //= devided_factor
        layers.append(
                nn.Sequential(
                    nn.Linear(vector_size, num_labels),
                    nn.LeakyReLU(0.2)
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        output = self.model(x)
        return F.log_softmax(output, dim=1)
