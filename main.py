from pprint import pprint

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

import helper
import utils
from utils import draw_roc_curve

# Carica dataset
df = helper.get_dataframe_bernoulli()
X = df[[
    'user_reviews_bin',
    'price_final_f2p_bin',
    'price_final_over_40_bin',
    'is_multiplatform',
    'before_2019',
    'before_2010'
]]
y = df['liked']


print("Distribuzione target:")
print(y.value_counts())

# Bilanciamento dei dati
X_res, y_res = utils.undersample(X, y, 0.63)

# Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

print("Dimensioni train set:", X_train.shape, y_train.shape)
print("Distribuzione target train:")
print(y_train.value_counts())

print("Dimensioni train bilanciato:", X_train.shape, y_train.shape)
print("Distribuzione target train bilanciato:")
print(y_train.value_counts())

# Parametri per k-fold cross-validation
best_alpha, best_fit_prior = utils.k_fold(X_train, y_train, 5)

# Addestramento modello
model = BernoulliNB(alpha=best_alpha, fit_prior=best_fit_prior)
model.fit(X_train, y_train)

# Predizioni
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Valutazione modello
utils.print_metrics(y_test, y_pred, y_prob)

# Curva ROC
auc_value = roc_auc_score(y_test, y_prob)
print("AUC:", auc_value)
false_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_prob)
draw_roc_curve(false_pos_rate, true_pos_rate)

