import os
from time import time
from datetime import datetime

import torch
import torch.nn as nn

import constants
import data_preprocessing
from data_preprocessing import train_loader, test_loader, vocab_size
from models.Linear_Model import Linear_Classifier
from models.FastText_Linear_Model import FastText_Linear_Classifier
from train_test_functions import metric, train, test


def results_recording():
    parameters['epochs:'] = epoch + 1
    parameters['losses:'] = test_loss_list
    parameters['time:'] = '{:.2f}'.format(t1)

    filename = 'experimental_results.txt'
    if not os.path.exists(filename):
        open(filename, 'a').close()
    filehandler = open(filename, 'a')
    for key, value in parameters.items():
        filehandler.write('{} {}\n'.format(str(key), str(value)))
    filehandler.write('-------------------------\n')
    print('Finish!')


if __name__ == '__main__':
    try:
        model = FastText_Linear_Classifier(vector_size=300)
        optimizer = torch.optim.Adam(model.parameters(), lr=constants.lr)
        criterion = nn.CrossEntropyLoss()
        date = datetime.now()
        comment = input('Enter a comment: ')
        parameters = {
            'Experiment:': date.strftime('%m/%d/%Y, %H:%M:%S'),
            'dataset: ': data_preprocessing.DATASET.__name__,
            'model: ': model.__class__.__name__,
            'model detail: ': model,
            'loss function: ': criterion,
            'metric: ': metric,
            'Comments: ': 'comment',
        }
        acc_list = []
        test_loss_list = []
        t0 = time()
        for epoch in range(constants.epochs):
            train(model, constants.device, train_loader, optimizer, criterion)
            test_loss, acc = test(
                model, constants.device, test_loader, criterion
            )
            acc_list.append(acc)
            test_loss_list.append(test_loss)
            t1 = (time() - t0) / 60
            print(
                'Epoch: {}, test loss: {:.3f},'.format(epoch+1, test_loss)
                + ' accuracy: {:.3f}, time: {:.2f} min'.format(acc, t1))
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Learning has been stopped manually')
    finally:
        results_recording()
