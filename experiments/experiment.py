from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tqdm import tqdm

from neucube import Reservoir
from neucube.validation import Pipeline
from neucube.sampler import SpikeCount
from experiments.params import random_seed


CLASSIFIERS = ["regression", "random_forest", "xgboost", "svc", "naive_bayes"]


def get_classifier(clf_type: str = "regression"):
    if clf_type == "regression":
        return LogisticRegression(solver='saga', multi_class='multinomial', penalty="l2")
    if clf_type == "random_forest":
        return RandomForestClassifier()
    if clf_type == "xgboost":
        return XGBClassifier()
    if clf_type == "naive_bayes":
        return MultinomialNB()
    return SVC(kernel='linear')


def snn_experiment(data_x, data_y, clf_type: str = "regression",
                   seed: int = random_seed,
                   splits: int = 5, shape: tuple[int] = (10, 10, 10),
                   res_train: bool = True):
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    y_total, pred_total = [], []

    for train_index, test_index in tqdm(kf.split(data_x)):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        res = Reservoir(inputs=data_x.shape[2], cube_shape=shape)
        sam = SpikeCount()
        clf = get_classifier(clf_type)
        pipe = Pipeline(res, sam, clf)

        pipe.fit(x_train, y_train, train=res_train)
        pred = pipe.predict(x_test)

        y_total.extend(y_test)
        pred_total.extend(pred)
    print(f'---- CLASSIFIER: {clf_type} ----')
    print(f'acc: {accuracy(y_total, pred_total)}')
    print(f'micro: {f1_score(y_total, pred_total, average="micro")}')
    print(f'macro: {f1_score(y_total, pred_total, average="macro")}')
    print(f'weighted: {f1_score(y_total, pred_total, average="weighted")}')
    print(confusion_matrix(y_total, pred_total))


def lsa_experiment(data_x, data_y,
                   clf_type: str = "regression",
                   seed: int = random_seed,
                   splits: int = 5):
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    y_total, pred_total = [], []

    for train_index, test_index in tqdm(kf.split(data_x)):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        clf = get_classifier(clf_type)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        y_total.extend(y_test)
        pred_total.extend(pred)
    print(f'\n---- CLASSIFIER: {clf_type} ----')
    print(f'acc: {accuracy(y_total, pred_total)}')
    print(f'micro: {f1_score(y_total, pred_total, average="micro")}')
    print(f'macro: {f1_score(y_total, pred_total, average="macro")}')
    print(f'weighted: {f1_score(y_total, pred_total, average="weighted")}')
    print(confusion_matrix(y_total, pred_total))


def snn_multiple_clfs(data_x, data_y, seed: int = random_seed,
                      splits: int = 10, shape: tuple[int] = (10, 10, 10)):
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    states = []
    indices = kf.split(data_x)

    for train_index, test_index in tqdm(indices):
        x_train, _ = data_x[train_index], data_x[test_index]

        res = Reservoir(inputs=data_x.shape[2], cube_shape=shape)
        sam = SpikeCount()
        pipe = Pipeline(res, sam, None)
        state = pipe.train_reservoir(x_train, train=False)
        states.append(state)

    for clf_type in CLASSIFIERS:
        y_total, pred_total = [], []
        clf = get_classifier(clf_type)
        for state, train_test_index in tqdm(zip(states, indices)):
            train_index, test_index = train_test_index
            _, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]

            pipe.classifier = clf
            clf.fit(state, y_train)
            pred = pipe.predict(x_test)
            y_total.extend(y_test)
            pred_total.extend(pred)

        print(f'---- CLASSIFIER: {clf_type} ----')
        print(f'acc: {accuracy(y_total, pred_total)}')
        print(confusion_matrix(y_total, pred_total))
