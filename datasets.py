import re
import os
from typing import Callable, List, Set, Union

import nltk
from nltk.corpus import stopwords

from pymystem3 import Mystem

from gensim import corpora
from gensim.models import FastText

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


if not os.path.exists('./venv/lib/nltk_data'):
    nltk.download('stopwords', download_dir='./venv/lib/nltk_data')
russian_stopwords = stopwords.words('russian')


class Wiki_Dataset(Dataset):
    def __init__(self, data: List) -> None:
        self.mystem = Mystem()
        tokenizer = get_tokenizer(tokenizer=None)
        self.corpus = [
            self.tokenize(
                article[0],
                tokenizer,
            ) for article in data
        ]
        categories = self.cats_set(data)
        cats_dict = {
            cat: i for cat, i in zip(categories, range(len(categories)))
        }
        self.labels = [cats_dict.get(article[1]) for article in data]

    def __len__(self) -> int:
        return len(self.labels)

    def cats_set(self, wiki_data: List) -> Set:
        """Get set of categories."""
        categories = []
        for article in wiki_data:
            categories.append(article[1])
        return set(categories)

    def tokenize(self, text: str, tokenizer: Callable[[str], List]) -> List:
        """Tokenize docunent."""
        clean_text = re.sub(r'[^\w\s]', '', text)
        clean_text = clean_text.lower()
        tokenized_text = tokenizer(clean_text)
        lemmatized_text = [
            self.mystem.lemmatize(token)[0] for token in tokenized_text
        ]
        clean_doc = [
            re.sub(r'\b[0-9]+\b', '<NUM>', token) for token in lemmatized_text
            ]
        clean_doc = [
            token for token in clean_doc if token not in russian_stopwords
            ]
        return clean_doc

    def __getitem__(self, i) -> Union[torch.Tensor, int]:
        return (
            self.bow_corpus[i],
            self.labels[i],
        )


class Wiki_Dataset_BoW(Wiki_Dataset):
    def __init__(self, data: List) -> None:
        super().__init__(data)
        great_dictionary = corpora.Dictionary(self.corpus)
        self.bow_corpus = [
            self.make_bow_vector(
                doc,
                great_dictionary,
            ) for doc in self.corpus
        ]

    def make_bow_vector(self, sentence: List, dict: corpora.Dictionary):
        """Make vector with ones and zeros like [1., 1., 0., 0., 1.]"""
        vec = torch.zeros(len(dict))
        for word in sentence:
            vec[dict.token2id[word]] += 1
        return vec.view(1, -1)

    def __getitem__(self, i) -> Union[torch.Tensor, int]:
        return (
            torch.tensor(self.bow_corpus[i]),
            self.labels[i],
        )


class Wiki_Dataset_BoW_Vector(Wiki_Dataset):
    def __init__(self, data: List) -> None:
        super().__init__(data)
        self.model = FastText(
            data, min_count=5, vector_size=300
        )

    def __getitem__(self, i) -> Union[torch.Tensor, torch.Tensor]:
        mean_vector = sum(
            torch.tensor(self.model.wv[i]) for i in self.corpus[i]
        )/len(self.corpus[i])
        label = torch.tensor(self.labels[i])
        return mean_vector, label
