from datetime import datetime

from imblearn.over_sampling import SMOTE
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

import utils


def classify_hit(row):
    if row['user_reviews'] >= 1200:
        return 1
    else:
        return 0


# Importa dataset dal file CSV e convertilo in un DataFrame
df = pd.read_csv('./resources/games.csv')

utils.scale(df, 'positive_ratio', 'positive_ratio_scaled')
df['hit'] = df.apply(classify_hit, axis=1)

df['price_final_bin'] = df['price_final'].apply(lambda x: 1 if x > 0 else 0)
df['win_bin'] = df['win'].apply(lambda x: 1 if x == True else 0)
df['mac_bin'] = df['mac'].apply(lambda x: 1 if x == True else 0)
df['linux_bin'] = df['linux'].apply(lambda x: 1 if x == True else 0)
df['discount_bin'] = df['discount'].apply(lambda x: 1 if x > 0 else 0)


print(df['hit'].value_counts())

X = df[['price_final_bin', 'win_bin', 'mac_bin', 'linux_bin', 'discount_bin', 'positive_ratio_scaled']]
y = df['hit']

# Dividi in dati di train e test (Pareto 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

smote = SMOTE(random_state = 42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Addestra il modello classificatore
model = BernoulliNB(alpha = 0.5, fit_prior = True)
model.fit(X_train_resampled, y_train_resampled)

# Fai predizioni
y_pred = model.predict(X_test)

# Valuta modello usando le metriche di valutazione
utils.print_metrics(y_test, y_pred)