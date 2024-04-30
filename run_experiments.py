import itertools
import torch
from tqdm import tqdm
import gensim.downloader as api
import pandas as pd
import datetime
from sklearn.datasets import fetch_20newsgroups
from experiments.preprocess import LSAPrep, EmbeddingPrep 
from experiments.experiment import experiment


# common params
seed = 1234
spiking = True
embedding = False

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
len_spikes = [10, 50]  #, 100, 150, 200]  # how many iterations of spike encoding are there
num_words = [50, 100]  # how many words we take from embedded sentence (probably the same as len_spikes)
lsa_components = [500, 1000]  #, 2000, 5000]
lsa_normalizations = [False, True]
embedding_models = ["glove-wiki-gigaword-300"]  # ["word2vec-google-news-300", "glove-wiki-gigaword-300"]
norm_embeddings = [False, True]

# experiment params
classifier_types = ["regression", "random_forest", "xgboost"]
cube_shapes = [(6, 6, 6), (8, 8, 8)] #, (10, 10, 10), (12, 12, 12)]


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
            filename = f"results/test_embedding_spiking_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
            column_names = ["lsa_components", "norm_lsa", "len_spikes", "clf_type", "cube_shape",
                            "accuracy", "f1_micro", "f1_macro", "f1_weighted"]
        else:
            filename = f"results/test_embedding_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
            column_names = ["num_word", "embedding_model", "norm_embedding", "clf_type", "accuracy",
                            "f1_micro", "f1_macro", "f1_weighted"]
            for classifier_type in classifier_types:
                results = experiment(data_x=train_x,
                                     data_y=train_y,
                                     seed=seed,
                                     clf_type=classifier_type,
                                     splits=5)
                all_results.append(list(prep_params)+[classifier_type]+results)
        df = pd.DataFrame(all_results, columns=column_names)
        df.to_csv(filename)
        all_results = []

else:
    if spiking:
        column_names = ["lsa_components", "norm_lsa", "len_spikes", "clf_type", "cube_shape",
                        "accuracy", "f1_micro", "f1_macro", "f1_weighted"]
        for prep_params in itertools.product(lsa_components, lsa_normalizations, len_spikes):
            print("--- Preprocessing data... ---")
            lsa_component, lsa_normalization, len_spike = prep_params
            # preprocessing
            text_prep = LSAPrep(svd_components=lsa_component, prob_iterations=len_spike)
            train_x, train_y = text_prep.preprocess_dataset(dataset, lsa=True, spikes=True,
                                                            lsa_normalize=lsa_normalization)
            train_x.to(device)
            train_y.to(device)

            print("--- Training... ---")
            for exp_params in itertools.product(classifier_types, cube_shapes):
                classifier_type, cube_shape = exp_params
                results = experiment(data_x=train_x,
                                     data_y=train_y,
                                     seed=seed,
                                     clf_type=classifier_type,
                                     splits=5, spiking=True,
                                     shape=cube_shape)
                all_results.append(list(prep_params)+list(exp_params)+results)
            filename = f"results/test_lsa_spiking_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
            df = pd.DataFrame(all_results, columns=column_names)
            df.to_csv(filename)
            all_results = []

    else:
        filename = f"results/test_lsa_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
        column_names = ["lsa_components", "norm_lsa", "clf_type",
                        "accuracy", "f1_micro", "f1_macro", "f1_weighted"]
        for prep_params in itertools.product(lsa_components, lsa_normalizations):
            print("--- Preprocessing data... ---")
            lsa_component, lsa_normalization = prep_params
            # preprocessing
            text_prep = LSAPrep(svd_components=lsa_component)
            train_x, train_y = text_prep.preprocess_dataset(dataset, lsa=True, spikes=False,
                                                            lsa_normalize=lsa_normalization)
            train_x.to(device)
            train_y.to(device)

            print("--- Training... ---")
            for classifier_type in classifier_types:
                results = experiment(data_x=train_x,
                                     data_y=train_y,
                                     seed=seed,
                                     clf_type=classifier_type,
                                     splits=5)
                all_results.append(list(prep_params)+[classifier_type]+results)

        df = pd.DataFrame(all_results, columns=column_names)
        df.to_csv(filename)
