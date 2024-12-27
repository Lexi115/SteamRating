import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


# Riconduci a due classi (positiva = 1, negativa = 0)
def two_classes(dataframe, feature, positive_class = []):
    dataframe[feature] = dataframe[feature].apply(lambda x: 1 if x in positive_class else 0)


# Importa dataset dal file CSV e convertilo in un DataFrame
df = pd.read_csv('./resources/games.csv')

# Dividi campioni in due classi: rating positivo e rating negativo
positive_classes = [
    'Positive',
    'Mostly Positive',
    'Very Positive',
    'Overwhelmingly Positive'
]
two_classes(df, 'rating', positive_classes)

# Binarizza le features
df['positive_ratio'] = np.where(df['positive_ratio'] >= 60, 1, 0)
df['user_reviews'] = np.where(df['user_reviews'] >= 500, 1, 0)
df['price_final'] = np.where(df['price_final'] > 0, 1, 0)

# Seleziona le features binarizzate
X = df[['positive_ratio', 'user_reviews', 'price_final']]
y = df['rating']

# Dividi in dati di train e test (Pareto 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Addestra il modello classificatore
smoothing = 1.0
model = BernoulliNB(alpha = smoothing, fit_prior = True)
model.fit(X_train, y_train)

# Fai predizioni
y_pred = model.predict(X_test)

# Valuta modello usando le metriche di valutazione
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.1f}%')

print("Distribuzione delle classi nel dataset completo:")
print(y.value_counts())

print("\nDistribuzione delle classi nel training set:")
print(y_train.value_counts())

print("\nDistribuzione delle classi nel test set:")
print(y_test.value_counts())