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

nltk.download('stopwords')  # в мэин ран бефоре
russian_stopwords = stopwords.words('russian')


# set constants
BATCH_SIZE = 500
# файл конфиг или параметр ком строки
DATA_PATH = '/Users/nikolai/Downloads/mini_wiki_cats.jsonl(1)'


def data_import(data_path: str) -> List:
    """Import data from file."""
    with open(data_path, 'r') as json_file:
        return list(json_file)


def get_corpus() -> List:
    """Get corpus: list with text and label only."""
    wiki_corpus = [
        [json.loads(json_str).get('text'),
            json.loads(json_str).get('cats')[0]]
        for json_str in json_list]
    return wiki_corpus


def cats_set(json_list: List) -> Set:
    """Get set of categories."""
    categories = []
    for json_str in json_list:
        result = json.loads(json_str)
        categories.append(result['cats'][0])
    return set(categories)


mystem = Mystem()  # не в теле функции
tokenizer = get_tokenizer(tokenizer=None)


def tokenize(text: str, tokenizer: Callable[[str], List] = tokenizer) -> List:
    """Tokenize docunent."""
    clean_text = re.sub(r'[^\w\s]', '', text)
    clean_text = re.sub(r'[^\sА-Яа-я]', '', clean_text)
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
    def __init__(self, data: List) -> None:
        tokenizer = get_tokenizer(tokenizer=None)
        self.corpus = [tokenize(article[0], tokenizer) for article in data]
        great_dictionary = corpora.Dictionary(self.corpus)
        self.bow_corpus = [
            make_bow_vector(doc, great_dictionary) for doc in self.corpus]
        # get categories and labels
        categories = cats_set(json_list)
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


if __name__ == '__main__':
    # import data from file
    json_list = data_import(DATA_PATH)
    # create corpus
    wiki_corpus = get_corpus()
    with open('.data/wiki_corpus.pickle', 'wb') as f:
        pickle.dump(wiki_corpus, f)


'''
# create dataset
dataset = Wiki_Dataset_BoW(wiki_corpus)
# split dataset into three parts
train_dataset, valid_dataset, test_dataset = split_train_valid_test(
    dataset, valid_ratio=0.1, test_ratio=0.1
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=lambda x: x
    )
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=lambda x: x
    )
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=lambda x: x
    )'''
