from datetime import datetime

from imblearn.over_sampling import SMOTE
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

import utils



# Importa dataset dal file CSV e convertilo in un DataFrame
df = pd.read_csv('./resources/games.csv')

df['user_reviews_bin'] = df['user_reviews'].apply(lambda x: 1 if x > 200 else 0)
df['price_original_bin'] = df['price_original'].apply(lambda x: 1 if x >= 19.99 else 0)
df['is_multiplatform'] = df[['win', 'mac', 'linux']].sum(axis=1).apply(lambda x: 1 if x > 1 else 0)
df['before_2020'] = df['date_release'].apply(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d').year < 2020 else 0)
df['is_discounted'] = df['discount'].apply(lambda x: 1 if x > 0 else 0)

# target
df['liked'] = df['positive_ratio'].apply(lambda x: 1 if x >= 70 else 0)

X = df[['user_reviews_bin', 'price_original_bin', 'is_multiplatform', 'before_2020', 'is_discounted']]
y = df['liked']

# Dividi in dati di train e test (Pareto 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

smote = SMOTE(random_state = 42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Addestra il modello classificatore
model = BernoulliNB(alpha = 1, fit_prior = True)
model.fit(X_train_resampled, y_train_resampled)

# Fai predizioni
y_pred = model.predict(X_test)

# Valuta modello usando le metriche di valutazione
utils.print_metrics(y_test, y_pred)