from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MinMaxScaler


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

    print('AUC with best model:', roc_auc_score(_test, _prob))


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
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

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