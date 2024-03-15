from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from torch import FloatTensor, tensor
from numpy import resize

from neucube.encoder import Probability
from experiments.params import random_seed


class TextPrep:

    def __init__(
            self, svd_components: int = 1000, prob_iterations: int = 500, max_features: int = 5000
            ):
        download('punkt')
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(
            strip_accents="ascii",
            lowercase=True,
            preprocessor=self.preprocess_and_stem,
            max_features=max_features,
            )
        self.svd = TruncatedSVD(n_components=svd_components, random_state=random_seed)
        self.encoder = Probability(iterations=prob_iterations)

    def preprocess_and_stem(self, text: str):
        text = text.lower()
        tokens = word_tokenize(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        preprocessed_text = ' '.join(stems)
        return preprocessed_text

    def preprocess_dataset(self, sklearn_dataset, lsa=True, spikes=True) -> tuple[FloatTensor, FloatTensor]:
        train_x = self.vectorizer.fit_transform(sklearn_dataset.data)
        if lsa:
            train_x = self.svd.fit_transform(train_x)
        else:
            train_x = train_x.toarray()
        train_x = FloatTensor(train_x)
        if spikes:
            train_x = self.encoder.encode_dataset(train_x)

        train_y = sklearn_dataset.target
        train_y = FloatTensor(train_y)
        return train_x, train_y


class Word2VecPrep:
    def __init__(self, word2vec_model) -> None:
        download('punkt')
        self.word2vec = word2vec_model

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        return tokens

    # Function to convert text to a fixed-size vector using Word2Vec
    def text_to_vector(self, text, avg: bool = True):
        tokens = self.preprocess_text(text)

        # Filter tokens present in the Word2Vec model
        words = [word for word in tokens if word in self.word2vec]

        if not words:  # or (len(words) < 50):
            return None

        if avg:
            # Calculate the average vector
            return sum(self.word2vec[words]) / len(words)

        embeddings = resize(self.word2vec[words], (300, 300))
        return embeddings

    def preprocess_dataset(self, sklearn_dataset, avg: bool = True):
        # newsgroups_vectors = []
        # for text in sklearn_dataset.data:
        #     newsgroups_vectors.append(self.text_to_vector(text, avg=True))
        newsgroups_vectors = [self.text_to_vector(text, avg=avg) for text in sklearn_dataset.data]
        # Remove instances where text could not be converted to vectors
        newsgroups_vectors = [vec for vec in newsgroups_vectors if vec is not None]
        train_x = tensor(newsgroups_vectors)

        train_y = sklearn_dataset.target
        train_y = FloatTensor(train_y)
        return train_x, train_y
