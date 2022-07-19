import torch


def accuracy(pred: torch.Tensor,
             ground: torch.Tensor) -> float:
    pred_class = pred.argmax(dim=1)
    return sum(pred_class == ground).item()
