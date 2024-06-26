import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from torch import FloatTensor
from torch.nn.functional import normalize
from numpy import tile, array, clip

from neucube.encoder import Rate
from experiments.params import random_seed


class LSAPrep:

    def __init__(
            self, svd_components: int = 1000, iterations: int = 500, max_feat: int = None
            ):
        download('punkt')
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(
            strip_accents="ascii",
            lowercase=True,
            preprocessor=self.preprocess_and_stem,
            max_features=max_feat
            )
        self.svd = TruncatedSVD(n_components=svd_components, random_state=random_seed)
        self.encoder = Rate(iterations)

    def preprocess_and_stem(self, text: str):
        text = text.lower()
        tokens = word_tokenize(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        preprocessed_text = ' '.join(stems)
        return preprocessed_text

    def preprocess_dataset(self, sklearn_dataset, lsa=True,
                           spikes=True, lsa_normalize=False) -> tuple[FloatTensor, FloatTensor]:
        train_x = self.vectorizer.fit_transform(sklearn_dataset.data)
        if lsa:
            train_x = self.svd.fit_transform(train_x)
        else:
            train_x = train_x.toarray()
        train_x = FloatTensor(train_x)
        if lsa_normalize:
            normalize(train_x, dim=0)
        if spikes:
            train_x = self.encoder.encode_dataset(train_x)

        train_y = sklearn_dataset.target
        train_y = FloatTensor(train_y)
        return train_x, train_y


class EmbeddingPrep:
    def __init__(self, word2vec_model) -> None:
        download('punkt')
        self.word2vec = word2vec_model

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        return tokens

    # Function to convert text to a fixed-size vector using Word2Vec
    def text_to_vector(self, text, avg: bool = True, max_size: int = 100,
                       avg_normalization: bool = False):
        tokens = self.preprocess_text(text)

        # Filter tokens present in the Word2Vec model
        words = [word for word in tokens if word in self.word2vec]

        if not words:
            return None

        if avg:
            # Calculate the average vector
            average_vector = sum((self.word2vec[words])[:max_size]) / max_size
            if avg_normalization:
                average_vector = (clip(average_vector, -1., 1.) + 1) / 2
            return average_vector

        embeddings = self.word2vec[words]
        array_length = len(embeddings)
        if array_length < max_size:
            repetitions = (max_size + array_length - 1) // array_length
            embeddings = tile(embeddings, (repetitions, 1))[:max_size]
        else:
            embeddings = embeddings[:max_size]
        return embeddings

    def preprocess_dataset(self, sklearn_dataset, avg: bool = True, max_size: int = 100,
                           avg_normalization: bool = False, spikes: bool = False, threshold: float = 0.25):
        newsgroups_vectors = [self.text_to_vector(
            text, avg, max_size, avg_normalization) for text in sklearn_dataset.data]
        # Remove instances where text could not be converted to vectors
        newsgroups_vectors = [vec for vec in newsgroups_vectors if vec is not None]
        train_x = FloatTensor(array(newsgroups_vectors))
        if not avg:
            mask = train_x >= threshold
            train_x = torch.where(mask, torch.tensor(1), torch.tensor(0))
        elif spikes:
            encoder = Rate(iterations=max_size)
            train_x = encoder.encode_dataset(train_x)
        train_y = sklearn_dataset.target
        train_y = FloatTensor(train_y)
        return train_x, train_y
