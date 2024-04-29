from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

from neucube import Reservoir
from neucube.validation import Pipeline
from neucube.sampler import SpikeCount


def get_classifier(clf_type: str = "regression"):
    if clf_type == "random_forest":
        return RandomForestClassifier()
    if clf_type == "xgboost":
        return XGBClassifier()
    return LogisticRegression(solver='saga', multi_class='multinomial', penalty="l2")


def experiment(data_x, data_y, seed,
               clf_type: str = "regression",
               splits: int = 5,
               spiking: bool = False,
               shape: tuple[int] = (6, 6, 6),
               reservoir_train: bool = True,
               m_print: bool = True
               ):

    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    y_total, pred_total = [], []

    for train_index, test_index in tqdm(kf.split(data_x)):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        clf = get_classifier(clf_type)
        if spiking:
            res = Reservoir(inputs=data_x.shape[2], cube_shape=shape)
            sam = SpikeCount()
            pipe = Pipeline(res, sam, clf)

            pipe.fit(x_train, y_train, train=reservoir_train)
            pred = pipe.predict(x_test)

        else:
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

        y_total.extend(y_test)
        pred_total.extend(pred)

    return collect_metrics(y_total, pred_total, m_print, clf_type)


def collect_metrics(y_total, pred_total, m_print: bool, clf_type: str):
    acc = accuracy(y_total, pred_total)
    f1_micro = f1_score(y_total, pred_total, average="micro")
    f1_macro = f1_score(y_total, pred_total, average="macro")
    f1_weighted = f1_score(y_total, pred_total, average="weighted")
    collected = [acc, f1_micro, f1_macro, f1_weighted]
    if m_print:
        print_metrics(clf_type, collected)
        print(confusion_matrix(y_total, pred_total))
    return collected


def print_metrics(clf_type: str, collected: list[float]):
    print(f'\n---- CLASSIFIER: {clf_type} ----')
    print(f'acc: {collected[0]}')
    print(f'micro: {collected[1]}')
    print(f'macro: {collected[2]}')
    print(f'weighted: {collected[3]}')
