import os
import json
import pickle
import numpy as np

from argparse import ArgumentParser
from typing import List

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import constants
# choose between Wiki_Dataset_BoW, Wiki_Dataset_BoW_Vector
from datasets import Wiki_Dataset_BoW_Vector


DATASET_DIRECTORY = './.data/'
DATASET = Wiki_Dataset_BoW_Vector
DATASET_NAME = str(DATASET.__name__)


def data_import():
    """Import data from file."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", nargs="?", const='./mini_wiki_cats.jsonl',
    )
    data_path = str(parser.parse_args().input)
    with open(data_path, 'r') as json_file:
        return list(json_file)


def get_corpus(json_list: List) -> List:
    """Get corpus: list with text and label only."""
    wiki_corpus = [
        [json.loads(json_str).get('text'),
            json.loads(json_str).get('cats')[0]]
        for json_str in json_list]
    return wiki_corpus


def split_train_valid_test(corpus: Dataset, valid_ratio: float = 0.1,
                           test_ratio: float = 0.1) -> List:
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(
        corpus, lengths=[train_length, valid_length, test_length],
    )


class MyCustomUnpickler(pickle.Unpickler):
    """Helps to load data from file as a module."""
    def find_class(self, module, name):
        if module == "__main__":
            module = "data_preprocessing"
        return super().find_class(module, name)


def make_corpus() -> None:
    """Load data, make corpus and save it locally."""
    json_list = data_import()
    wiki_corpus = get_corpus(json_list)
    if not os.path.exists(DATASET_DIRECTORY):
        os.mkdir(DATASET_DIRECTORY)
    with open('.data/wiki_corpus.pickle', 'wb') as f:
        pickle.dump(wiki_corpus, f)


def create_dataset() -> None:
    """Create dataset and save it locally."""
    with open('.data/wiki_corpus.pickle', 'rb') as f:
        wiki_corpus = pickle.load(f)
    dataset = DATASET(wiki_corpus[:2000])
    if not os.path.exists(DATASET_DIRECTORY):
        os.mkdir(DATASET_DIRECTORY)
    with open(f'.data/{DATASET_NAME}.pickle', 'wb') as f:
        pickle.dump(dataset, f)


def collate(batch):
    vector = [np.ndarray(item[0]) for item in batch]
    label = [np.ndarray(item[1]) for item in batch]
    return vector, label


if not os.path.exists('.data/wiki_corpus.pickle'):
    make_corpus()
if not os.path.exists(f'.data/{DATASET_NAME}.pickle'):
    create_dataset()
with open(f'.data/{DATASET_NAME}.pickle', 'rb') as f:
    unpickler = MyCustomUnpickler(f)
    dataset = unpickler.load()
vocab_size = dataset[0][0].size()[1]
train_dataset, valid_dataset, test_dataset = split_train_valid_test(
    dataset, valid_ratio=0.1, test_ratio=0.1
)
train_loader = DataLoader(
    train_dataset,
    batch_size=constants.BATCH_SIZE,
    collate_fn=collate
    )
valid_loader = DataLoader(
    valid_dataset,
    batch_size=constants.BATCH_SIZE,
    )
test_loader = DataLoader(
    test_dataset,
    batch_size=constants.BATCH_SIZE,
   )
