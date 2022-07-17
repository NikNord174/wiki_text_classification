import re
import json
import nltk

from typing import List, Callable, Union
from gensim import corpora
from pymystem3 import Mystem
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import random_split
from nltk.corpus import stopwords


BATCH_SIZE = 500
DATA_PATH = '/Users/nikolai/Downloads/mini_wiki_cats.jsonl(1)'


# data import
with open(DATA_PATH, 'r') as json_file:
    json_list = list(json_file)
wiki_corpus = [
    [json.loads(json_str).get('text'),
        json.loads(json_str).get('cats')[0]]
    for json_str in json_list]


nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')
mystem = Mystem()
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


class Wiki_Dataset(Dataset):
    def __init__(self, data: List = wiki_corpus) -> None:
        tokenizer = get_tokenizer(tokenizer=None)
        self.corpus = [tokenize(article[0], tokenizer) for article in data]
        dicts = [corpora.Dictionary([doc]) for doc in self.corpus]
        counted_dicts = [dict.token2id for dict in dicts]
        self.bow_corpus = [dict.doc2bow(
            counted_dict
        ) for counted_dict, dict in zip(counted_dicts, dicts)]
        self.labels = [article[1] for article in data]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i) -> Union[List, List, int]:
        return (
            self.corpus[i],
            self.bow_corpus[i],
            self.labels[i],
        )


def split_train_valid_test(corpus: Dataset, valid_ratio: float = 0.1,
                           test_ratio: float = 0.1):
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(
        corpus, lengths=[train_length, valid_length, test_length],
    )


train_dataset, valid_dataset, test_dataset = split_train_valid_test(
    Wiki_Dataset, valid_ratio=0.1, test_ratio=0.1
)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
