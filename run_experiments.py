import itertools
import torch
import gensim.downloader as api
import pandas as pd
import datetime
from sklearn.datasets import fetch_20newsgroups
from experiments.preprocess import LSAPrep, EmbeddingPrep
from experiments.experiment import experiment


# common params
seed = 4321  #1234
spiking = False
embedding = True
threshold = False

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
categories = None #['comp.graphics', 'sci.med', 'rec.motorcycles']  # If None, load all the categories. If not None, list of category names to load
len_spikes = [200]  #, 100, 150, 200]  # how many iterations of spike encoding are there
num_words = [200] #[50, 100, 150, 200]  #200  # how many words we take from embedded sentence (probably the same as len_spikes)
lsa_components = [None]  #[216, 300, 512]  #,  2000, 5000] [1776]
lsa_normalizations = [False]  #, True]
embedding_models = ["word2vec-google-news-300"]  # ["word2vec-google-news-300", "glove-wiki-gigaword-300"]
norm_embeddings = [False]
spike_train_types = ["rate", "threshold"]
mask_thresholds = [0.35] #0.35 for word2vec,  0.65 for glove

# experiment params
classifier_types = ["regression", "random_forest", "xgboost"]  #["regression",
cube_shapes = [(6, 6, 6)]  #, (10, 10, 10), (12, 12, 12)]


# START
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Starting experiment on device: {device} ---")
dataset = fetch_20newsgroups(subset='train', categories=categories, remove=('quotes',))
all_results = []
i_train = 1

if embedding:
    if spiking:
        if threshold:
            column_names = ["embedding_model", "num_word", "mask_threshold", "clf_type", "cube_shape",
                            "accuracy", "f1_micro", "f1_macro", "f1_weighted", "time"]
            num_trainings = len(num_words) * len(embedding_models) * len(mask_thresholds) * len(classifier_types) * len(cube_shapes)
            for prep_params in itertools.product(embedding_models, num_words, mask_thresholds):
                print("--- Preprocessing data... ---")
                embedding_model, num_word, mask_threshold = prep_params
                # preprocessing
                model = api.load(embedding_model)
                text_prep = EmbeddingPrep(model)
                train_x, train_y = text_prep.preprocess_dataset(dataset, avg=False, max_size=num_word,
                                                                avg_normalization=False,
                                                                spikes=False, threshold=mask_threshold)
                train_x.to(device)
                train_y.to(device)

                print("--- Training... ---")
                for exp_params in itertools.product(classifier_types, cube_shapes):
                    classifier_type, cube_shape = exp_params
                    print(f"--- [{i_train}/{num_trainings}] ---")
                    results = experiment(data_x=train_x,
                                         data_y=train_y,
                                         seed=seed,
                                         clf_type=classifier_type,
                                         splits=5, spiking=True,
                                         shape=cube_shape)
                    all_results.append(list(prep_params)+list(exp_params)+results)
                    i_train += 1
                filename = f"results/all-train_embedding_threshold_4321_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
                df = pd.DataFrame(all_results, columns=column_names)
                df.to_csv(filename, index=False)
                all_results = []
        else:
            column_names = ["embedding_model", "num_word", "clf_type", "cube_shape",
                            "accuracy", "f1_micro", "f1_macro", "f1_weighted", "time"]
            num_trainings = len(num_words) * len(embedding_models) * len(classifier_types) * len(cube_shapes)
            for prep_params in itertools.product(embedding_models, num_words):
                print("--- Preprocessing data... ---")
                embedding_model, num_word = prep_params
                # preprocessing
                model = api.load(embedding_model)
                text_prep = EmbeddingPrep(model)
                train_x, train_y = text_prep.preprocess_dataset(dataset, avg=True, max_size=num_word,
                                                                avg_normalization=False,
                                                                spikes=True)
                train_x.to(device)
                train_y.to(device)

                print("--- Training... ---")
                for exp_params in itertools.product(classifier_types, cube_shapes):
                    classifier_type, cube_shape = exp_params
                    print(f"--- [{i_train}/{num_trainings}] ---")
                    results = experiment(data_x=train_x,
                                         data_y=train_y,
                                         seed=seed,
                                         clf_type=classifier_type,
                                         splits=5, spiking=True,
                                         shape=cube_shape)
                    all_results.append(list(prep_params)+list(exp_params)+results)
                    i_train += 1
                filename = f"results/all-train_embedding_rate_4321_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
                df = pd.DataFrame(all_results, columns=column_names)
                df.to_csv(filename, index=False)
                all_results = []

    else:
        column_names = ["num_word", "embedding_model", "norm_embedding", "clf_type", "accuracy",
                        "f1_micro", "f1_macro", "f1_weighted", "time"]
        num_trainings = len(num_words) * len(embedding_models) * len(norm_embeddings) * len(classifier_types)
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
            for classifier_type in classifier_types:
                print(f"--- [{i_train}/{num_trainings}] ---")
                results = experiment(data_x=train_x,
                                     data_y=train_y,
                                     seed=seed,
                                     clf_type=classifier_type,
                                     splits=5, m_print=False)
                all_results.append(list(prep_params)+[classifier_type]+results)
                i_train += 1
            filename = f"results/all-train_embedding_4321_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
            df = pd.DataFrame(all_results, columns=column_names)
            df.to_csv(filename, index=False)
            all_results = []

else:
    if spiking:
        column_names = ["lsa_components", "len_spikes", "clf_type", "cube_shape",
                        "accuracy", "f1_micro", "f1_macro", "f1_weighted", "time"]
        num_trainings = len(lsa_components) * len(len_spikes) * len(cube_shapes) * len(classifier_types)
        for prep_params in itertools.product(lsa_components, len_spikes):
            print("--- Preprocessing data... ---")
            lsa_component, len_spike = prep_params
            # preprocessing
            text_prep = LSAPrep(svd_components=lsa_component, iterations=len_spike, max_feat=5000)
            train_x, train_y = text_prep.preprocess_dataset(dataset, lsa=False, spikes=True,
                                                            lsa_normalize=False)
            train_x.to(device)
            train_y.to(device)

            print("--- Training... ---")
            for exp_params in itertools.product(classifier_types, cube_shapes):
                print(f"--- [{i_train}/{num_trainings}] ---")
                classifier_type, cube_shape = exp_params
                results = experiment(data_x=train_x,
                                     data_y=train_y,
                                     seed=seed,
                                     clf_type=classifier_type,
                                     splits=5, spiking=True,
                                     shape=cube_shape)
                all_results.append(list(prep_params)+list(exp_params)+results)
                i_train += 1
            filename = f"results/all-train_tfidf5000_spiking_4321_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
            df = pd.DataFrame(all_results, columns=column_names)
            df.to_csv(filename, index=False)
            all_results = []

    else:
        column_names = ["lsa_components", "clf_type",
                        "accuracy", "f1_micro", "f1_macro", "f1_weighted", "time"]
        num_trainings = len(lsa_components) * len(classifier_types)
        for lsa_component in lsa_components:
            print("--- Preprocessing data... ---")
            # preprocessing
            text_prep = LSAPrep(svd_components=lsa_component, max_feat=5000)
            train_x, train_y = text_prep.preprocess_dataset(dataset, lsa=True, spikes=False,
                                                            lsa_normalize=False)
            train_x.to(device)
            train_y.to(device)

            print("--- Training... ---")
            for classifier_type in classifier_types:
                print(f"--- [{i_train}/{num_trainings}] ---")
                results = experiment(data_x=train_x,
                                     data_y=train_y,
                                     seed=seed,
                                     clf_type=classifier_type,
                                     splits=5, spiking=False)
                all_results.append([lsa_component, classifier_type]+results)
                i_train += 1
            filename = f"results/all-train_tfidf5000_lsa_4321_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
            df = pd.DataFrame(all_results, columns=column_names)
            df.to_csv(filename, index=False)
            all_results = []
