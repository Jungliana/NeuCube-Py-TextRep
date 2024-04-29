import itertools
import torch
from tqdm import tqdm
import gensim.downloader as api
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from experiments.preprocess import LSAPrep, EmbeddingPrep 
from experiments.experiment import experiment


# common params
seed = 1234
spiking = False
embedding = True

# Preprocessing types:
# - TF-IDF + LSA
# - TF-IDF + LSA + SPIKES
# - WORD2VEC (AVERAGED)
# - GLOVE (AVERAGED)
# - WORD2VEC + SPIKES (RATE-CODING)
# - GLOVE + SPIKES (RATE-CODING)
# - WORD2VEC + SPIKES (THRESHOLD-CODING)
# - GLOVE + SPIKES (THRESHOLD-CODING)

# preprocessing params
categories = ['comp.graphics', 'sci.med']  # If None, load all the categories. If not None, list of category names to load
len_spikes = [50, 100, 150, 200]  # how many iterations of spike encoding are there
num_words = [50, 100]  # how many words we take from embedded sentence (probably the same as len_spikes)
lsa_components = [500, 1000, 2000, 5000]
lsa_normalization = False
embedding_models = ["glove-wiki-gigaword-300"]  # ["word2vec-google-news-300", "glove-wiki-gigaword-300"]
norm_embeddings = [False, True]

# experiment params
classfier_types = ["regression", "random_forest", "xgboost"]
param_cube_shapes = [(6, 6, 6), (8, 8, 8), (10, 10, 10), (12, 12, 12)]


# START
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Starting experiment on device: {device} ---")
dataset = fetch_20newsgroups(subset='test', categories=categories, remove=('quotes',))
all_results = []

if embedding:
    for prep_params in itertools.product(num_words, embedding_models, norm_embeddings):
        print("--- Preprocessing data... ---")
        num_word, embedding_model, norm_embedding = prep_params
        # preprocessing
        model = api.load(embedding_model)
        text_prep = EmbeddingPrep(model)
        train_x, train_y = text_prep.preprocess_dataset(dataset, avg=True, max_size=num_word,
                                                        avg_normalization=norm_embedding)
        train_x.to(device)
        train_y.to(device)

        print("--- Training... ---")
        if spiking:
            pass
        else:
            for classfier_type in classfier_types:
                results = experiment(data_x=train_x,
                                     data_y=train_y,
                                     seed=seed,
                                     clf_type=classfier_type,
                                     splits=5)
                all_results.append(list(prep_params)+[classfier_type]+results)

else:
    pass
    # experiments

df = pd.DataFrame(all_results)
df.to_csv("results/test2.csv")
