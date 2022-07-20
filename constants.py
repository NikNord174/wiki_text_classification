import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


BATCH_SIZE = 500
epochs = 10
lr = 1e-3
