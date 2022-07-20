import os
import re
import json
import pickle
import nltk
import torch

from typing import List, Set, Callable, Union
from gensim import corpora
from pymystem3 import Mystem
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import random_split
from nltk.corpus import stopwords

import constants


nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')


mystem = Mystem()
tokenizer = get_tokenizer(tokenizer=None)


DATA_PATH = '/Users/nikolai/Downloads/mini_wiki_cats.jsonl(1)'
DIRECTORY = './.data/'


def data_import(data_path: str) -> List:
    """Import data from file."""
    with open(data_path, 'r') as json_file:
        return list(json_file)


def get_corpus(json_list: List) -> List:
    """Get corpus: list with text and label only."""
    wiki_corpus = [
        [json.loads(json_str).get('text'),
            json.loads(json_str).get('cats')[0]]
        for json_str in json_list]
    return wiki_corpus


def cats_set(wiki_data: List) -> Set:
    """Get set of categories."""
    categories = []
    for article in wiki_data:
        categories.append(article[1])
    return set(categories)


def tokenize(text: str, tokenizer: Callable[[str], List] = tokenizer) -> List:
    """Tokenize docunent."""
    clean_text = re.sub(r'[^\w\s]', '', text)
    clean_text = clean_text.lower()
    tokenized_text = tokenizer(clean_text)
    lemmatized_text = [mystem.lemmatize(token)[0] for token in tokenized_text]
    clean_doc = [
        re.sub(r'\b[0-9]+\b', '<NUM>', token) for token in lemmatized_text
        ]
    clean_doc = [
        token for token in clean_doc if token not in russian_stopwords
        ]
    return clean_doc


def make_bow_vector(sentence, dict):
    """Make vector with ones and zeros like [1., 1., 0., 0., 1.]"""
    vec = torch.zeros(len(dict))
    for word in sentence:
        vec[dict.token2id[word]] += 1
    return vec.view(1, -1)


class Wiki_Dataset_BoW(Dataset):
    """Create dataset with BoW vectors and labels."""
    def __init__(self, data: List) -> None:
        tokenizer = get_tokenizer(tokenizer=None)
        self.corpus = [tokenize(article[0], tokenizer) for article in data]
        great_dictionary = corpora.Dictionary(self.corpus)
        self.bow_corpus = [
            make_bow_vector(doc, great_dictionary) for doc in self.corpus]
        # get categories and labels
        categories = cats_set(data)
        cats_dict = {cat: i for cat, i in zip(
            categories, range(len(categories)))}
        self.labels = [cats_dict.get(article[1]) for article in data]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i) -> Union[List, List, int]:
        return (
            self.bow_corpus[i],
            self.labels[i],
        )


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
    json_list = data_import(DATA_PATH)
    wiki_corpus = get_corpus(json_list)
    if not os.path.exists(DIRECTORY):
        os.mkdir(DIRECTORY)
    with open('.data/wiki_corpus.pickle', 'wb') as f:
        pickle.dump(wiki_corpus, f)


def create_dataset() -> None:
    """Create dataset and save it locally."""
    with open('.data/wiki_corpus.pickle', 'rb') as f:
        wiki_corpus = pickle.load(f)
    dataset = Wiki_Dataset_BoW(wiki_corpus[:2000])
    if not os.path.exists(DIRECTORY):
        os.mkdir(DIRECTORY)
    with open('.data/dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)


if not os.path.exists('.data/wiki_corpus.pickle'):
    make_corpus()
if not os.path.exists('.data/dataset.pickle'):
    create_dataset()
with open('.data/dataset.pickle', 'rb') as f:
    unpickler = MyCustomUnpickler(f)
    dataset = unpickler.load()
vocab_size = dataset[0][0].size()[1]
train_dataset, valid_dataset, test_dataset = split_train_valid_test(
    dataset, valid_ratio=0.1, test_ratio=0.1
)
train_loader = DataLoader(
    train_dataset,
    batch_size=constants.BATCH_SIZE,
    )
valid_loader = DataLoader(
    valid_dataset,
    batch_size=constants.BATCH_SIZE,
    )
test_loader = DataLoader(
    test_dataset,
    batch_size=constants.BATCH_SIZE,
   )
