import time

import torch
import torch.nn as nn

import constants
from data_preprocessing import train_loader, test_loader
from models.linear_model import Linear_Classifier
from train_test_functions import train, test


if __name__ == '__main__':
    acc_list = []
    test_loss_list = []
    model = Linear_Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=constants.lr)

    t0 = time.time()
    for epoch in range(constants.epochs):
        #train(model, constants.device, train_loader, optimizer, criterion)
        #test_loss, acc = test(model, constants.device, test_loader, criterion)
        #acc_list.append(acc)
        #test_loss_list.append(test_loss)
        print('Training...')
        t1 = (time.time() - t0) / 60
        #print(
        #    'Epoch: {}, test loss: {:.3f},'.format(epoch+1, test_loss)
        #    + ' accuracy: {:.3f}, time: {:.2f} min'.format(acc, t1))
    print('Finish!')
