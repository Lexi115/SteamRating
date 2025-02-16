from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt


def print_metrics(_test, _pred, _prob):
    print(f'Confusion Matrix:')
    print(confusion_matrix(_test, _pred, labels = [0, 1]))

    accuracy = accuracy_score(_test, _pred)
    print(f'Accuracy: {accuracy * 100:.1f}%')

    precision = precision_score(_test, _pred, pos_label = 1)
    print(f'Precision: {precision * 100:.1f}%')

    recall = recall_score(_test, _pred, pos_label = 1)
    print(f'Recall: {recall * 100:.1f}%')

    f1 = f1_score(_test, _pred, pos_label = 1)
    print(f'F1 Score: {f1 * 100:.1f}%')


def scale(_dataframe, _row, _new_row):
    scaler = MinMaxScaler()
    _dataframe[_new_row] = scaler.fit_transform(_dataframe[[_row]])


def undersample(_x_train, _y_train, _strategy = 1):
    underSampler = RandomUnderSampler(sampling_strategy=_strategy, random_state=42)
    return underSampler.fit_resample(_x_train, _y_train)


def oversample(_x_train, _y_train):
    overSampler = SMOTE()
    return overSampler.fit_resample(_x_train, _y_train)


def get_auc_record(_x, _y, smoothing_factor_option, fit_prior_option, k):
    auc_record = {}
    k_fold = StratifiedKFold(n_splits=k, shuffle=True,                                   random_state=42)

    for train_indices, test_indices in k_fold.split(_x, _y):
        X_train, X_test = _x.iloc[train_indices], _x.iloc[test_indices]
        Y_train, Y_test = _y.iloc[train_indices], _y.iloc[test_indices]

        for alpha in smoothing_factor_option:
            if alpha not in auc_record:
                auc_record[alpha] = {}
            for fit_prior in fit_prior_option:
                clf = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(X_train, Y_train)
                prediction_prob = clf.predict_proba(X_test)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(Y_test, pos_prob)
                auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)

    return auc_record


def print_auc_record(auc_record, k):
    print('smoothing    fit prior   auc')
    for smoothing, smoothing_record in auc_record.items():
        for fit_prior, auc in smoothing_record.items():
            print(f'{smoothing}            {fit_prior}       {auc/k:.5f}')


def get_roc_curve(y_test, y_prob):
    thresholds = np.arange(0.0, 1.1, 0.05)
    true_pos_rate, false_pos_rate = [], []

    P = np.sum(y_test == 1)  # Numero totale di positivi
    N = np.sum(y_test == 0)  # Numero totale di negativi

    for threshold in thresholds:
        TP = np.sum((y_prob >= threshold) & (y_test == 1))
        FP = np.sum((y_prob >= threshold) & (y_test == 0))

        TPR = TP / P if P > 0 else 0  # True Positive Rate (Sensibilità)
        FPR = FP / N if N > 0 else 0  # False Positive Rate

        true_pos_rate.append(TPR)
        false_pos_rate.append(FPR)

    return false_pos_rate, true_pos_rate  # ROC usa (FPR, TPR)

def draw_roc_curve(_false_pos_rate, _true_pos_rate):
    plt.figure()
    lw = 2

    # ROC effettiva
    plt.plot(_false_pos_rate, _true_pos_rate, color='darkorange', lw=lw, label='ROC Curve')

    # ROC ideale (va subito a 1 e poi resta costante)
    ideal_fpr = [0, 0, 1]
    ideal_tpr = [0, 1, 1]
    plt.plot(ideal_fpr, ideal_tpr, color='green', lw=lw, linestyle='--', label='Ideal ROC Curve')

    # Linea diagonale casuale
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


def k_fold(_x, _y, k):
    smoothing_factors = [1, 2, 3, 4, 5, 6]
    fit_prior_options = [True, False]

    auc_record = get_auc_record(_x, _y, smoothing_factors, fit_prior_options, k)

    # Inizializzazione dei migliori valori
    best_alpha = None
    best_fit_prior = None
    best_auc = -float('inf')  # Minimo iniziale

    # Scansiona l'AUC record per trovare la combinazione migliore
    for alpha, fit_prior_dict in auc_record.items():
        for fit_prior, auc in fit_prior_dict.items():
            if auc > best_auc:
                best_auc = auc
                best_alpha = alpha
                best_fit_prior = fit_prior

    return best_alpha, best_fit_prior