from pandas.core.interchange import dataframe
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler


def print_metrics(_test, _pred):
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

