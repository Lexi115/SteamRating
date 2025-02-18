import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

import utils

def get_dataframe_randomforest():
    df = pd.read_csv('./resources/games.csv')

    # Pulizia outliers
    df = df[df['user_reviews'] > 20]
    df = df[df['price_final'] < 100]

    # Convertiamo user_reviews in categorie
    df['user_reviews_cat'] = pd.cut(df['user_reviews'],
                                    bins=[20, 1000, 2000, np.inf],
                                    labels=[0, 1, 2],
                                    include_lowest=True).astype(int)

    # Prezzo in categorie
    # 0 (Free to play)
    # 1 = (0-9.99]
    # 2 = (10-29.99]
    # 3 = (30-inf)
    df['price_final_cat'] = pd.cut(df['price_final'],
                                   bins=[-1, 0, 10, 20, np.inf],
                                   labels=[0, 1, 2, 3],
                                   include_lowest=True).astype(int)
    # Multipiattaforma
    df['is_multiplatform'] = (df[['win', 'mac', 'linux']].sum(axis=1) >= 2).astype(int)

    # Data di rilascio (prima/dopo il 2020)
    df['date_release'] = pd.to_datetime(df['date_release'], errors='coerce')
    df['release_period'] = pd.cut(df['date_release'].dt.year,
                                  bins=[-np.inf, 2009, 2019, np.inf],
                                  labels=[0, 1, 2],
                                  right=True).astype(int)

    # Target
    df['liked'] = df['positive_ratio'].apply(lambda x: 1 if x > 50 else 0)

    return df


def train():
    # Carica dataset
    df = get_dataframe_randomforest()
    X = df[[
        'user_reviews_cat',
        'price_final_cat',
        'is_multiplatform',
        'release_period',
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

    # Addestramento modello
    rf = RandomForestClassifier(random_state=42)

    # Ottimizzazione dei parametri con GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_res, y_train_res)
    model = grid_search.best_estimator_

    # Predizioni
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Distribuzione target train (dopo bilanciamento):")
    print(y_train_res.value_counts(), "\n")

    # Valutazione modello
    utils.print_metrics(y_test, y_pred, y_prob)

    # AUC
    auc_value = roc_auc_score(y_test, y_prob)
    print("AUC:", auc_value)
    false_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_prob)
    utils.draw_roc_curve(false_pos_rate, true_pos_rate)

