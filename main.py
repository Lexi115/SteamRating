from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

import utils

def classify_trending(row):
    release_date = datetime.strptime(row['date_release'], '%Y-%m-%d')
    today_date = datetime.now()

    #if row['score'] >= 300 and (today_date - release_date).days <= 365 * 9:
    if row['positive_ratio'] >= 75 and row['user_reviews'] >= 500:
        return 1
    else:
        return 0


# Importa dataset dal file CSV e convertilo in un DataFrame
df = pd.read_csv('./resources/games.csv')

df['score'] = (df['positive_ratio'] * np.log1p(df['user_reviews'])).astype(int)
df['trending'] = df.apply(classify_trending, axis=1)

df['release_age'] = (datetime.now() - pd.to_datetime(df['date_release'])).dt.days
df['discount_bin'] = df['discount'].apply(lambda x: 1 if x > 50 else 0)
df['is_multiplatform'] = df[['win', 'mac', 'linux']].sum(axis=1).apply(lambda x: 1 if x > 1 else 0)
df['log_user_reviews'] = np.log1p(df['user_reviews'])

X = df[['release_age', 'positive_ratio', 'log_user_reviews', 'discount_bin', 'is_multiplatform']]
y = df['trending']

print(df['trending'].value_counts())


# Dividi in dati di train e test (Pareto 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Addestra il modello classificatore
model = BernoulliNB(alpha = 1.0, fit_prior = True)
model.fit(X_train, y_train)

# Fai predizioni
y_pred = model.predict(X_test)

# Valuta modello usando le metriche di valutazione
utils.print_metrics(y_test, y_pred)