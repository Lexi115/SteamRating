from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

import utils


def classify_hit(row):
    if row['positive_ratio_scaled'] >= 0.6 and row['user_reviews'] >= 35:
        return 1
    else:
        return 0


# Importa dataset dal file CSV e convertilo in un DataFrame
df = pd.read_csv('./resources/games.csv')

utils.scale(df, 'positive_ratio', 'positive_ratio_scaled')
df['score'] = (df['positive_ratio_scaled'] * np.log1p(df['user_reviews']))
df['hit'] = df.apply(classify_hit, axis=1)

df['price_final_bin'] = df['price_final'].apply(lambda x: 1 if x > 0 else 0)
df['discount_bin'] = df['discount'].apply(lambda x: 1 if x > 0 else 0)
df['steam_deck_bin'] = df['steam_deck'].apply(lambda x: 1 if x == True else 0)
df['score_bin'] = df['score'].apply(lambda x: 1 if x > 4 else 0)

print(df['hit'].value_counts())

X = df[['score_bin', 'price_final_bin']]
y = df['hit']

# Dividi in dati di train e test (Pareto 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Addestra il modello classificatore
model = BernoulliNB(alpha = 1.0, fit_prior = True)
model.fit(X_train, y_train)

# Fai predizioni
y_pred = model.predict(X_test)

# Valuta modello usando le metriche di valutazione
utils.print_metrics(y_test, y_pred)