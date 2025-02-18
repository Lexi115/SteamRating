from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

import utils

def get_dataframe_bernoulli():
    df = pd.read_csv('./resources/games.csv')

    # Pulizia outliers
    df = df[df['user_reviews'] > 20]
    df = df[df['price_final'] < 100]

    df['user_reviews_bin'] = df['user_reviews'].apply(lambda x: 1 if x >= 2000 else 0)
    df['price_final_f2p_bin'] = df['price_final'].apply(lambda x: 1 if x == 0 else 0)
    df['price_final_over_40_bin'] = df['price_final'].apply(lambda x: 1 if x >= 40 else 0)
    df['is_multiplatform'] = df[['win', 'mac', 'linux']].sum(axis=1).apply(lambda x: 1 if x > 1 else 0)
    df['before_2019'] = df['date_release'].apply(lambda x: 1 if ((datetime.strptime(x, '%Y-%m-%d').year < 2019) and (datetime.strptime(x, '%Y-%m-%d').year >= 2010)) else 0)
    df['before_2010'] = df['date_release'].apply(lambda x: 1 if (datetime.strptime(x, '%Y-%m-%d').year < 2010) else 0)
    df['cross_reviews_2019'] = df['user_reviews_bin'] * df['before_2019']
    # target
    df['liked'] = df['positive_ratio'].apply(lambda x: 1 if x >= 60 else 0)

    return df

def train():
    # Carica dataset
    df = get_dataframe_bernoulli()
    X = df[[
        'user_reviews_bin',
        'price_final_f2p_bin',
        'price_final_over_40_bin',
        'is_multiplatform',
        'before_2019',
        'before_2010',
        'cross_reviews_2019'
    ]]
    y = df['liked']

    print("Distribuzione target originale:")
    print(y.value_counts(), "\n")

    # Divisione train/test (Pareto 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("Distribuzione target train (prima del bilanciamento):")
    print(y_train.value_counts(), "\n")

    # Bilanciamento del training set
    X_train_res, y_train_res = utils.undersample(X_train, y_train, 0.63)
    #X_train_res, y_train_res = utils.oversample(X_train, y_train)

    print("Distribuzione target train (dopo bilanciamento):")
    print(y_train_res.value_counts(), "\n")

    # Parametri per k-fold cross-validation
    best_alpha, best_fit_prior = utils.k_fold(X_train_res, y_train_res, 5)
    print(f"Miglior smoothing: {best_alpha}, Miglior fit_prior: {best_fit_prior}\n")

    # Addestramento modello
    model = BernoulliNB(alpha=best_alpha, fit_prior=best_fit_prior)
    model.fit(X_train_res, y_train_res)

    # Predizioni
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Valutazione modello
    utils.print_metrics(y_test, y_pred, y_prob)

    # AUC
    auc_value = roc_auc_score(y_test, y_prob)
    print("AUC:", auc_value)
    false_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_prob)
    utils.draw_roc_curve(false_pos_rate, true_pos_rate)

