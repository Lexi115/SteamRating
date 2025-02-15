from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
import helper
import utils
import pandas as pd

# Importa dataset e convertilo in DataFrame
df = helper.get_dataframe_multinomial()

# Selezione delle features e del target
X = df[['user_reviews_cat', 'price_original_cat', 'before_2020']]
y = df['liked']

print("Distribution", y.value_counts())

# k-fold cross-validation
k = 5
smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = utils.get_auc_record(X, y, smoothing_factor_option, fit_prior_option, k)
utils.print_auc_record(auc_record, k)

# Dividi in dati di train e test (Pareto 80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X Train: ", X_train.shape)
print("y Train: ", y_train.shape)
print("Distribution", y_train.value_counts())

# Applica undersampling parziale
X_train_resampled, y_train_resampled = utils.undersample(X_train, y_train, 1)
print("Resampled X Train: ", X_train_resampled.shape)
print("Resampled y Train: ", y_train_resampled.shape)
print("Resampled Distribution", y_train_resampled.value_counts())

# Addestra il modello classificatore
model = MultinomialNB(alpha=2.0, fit_prior=True)

# Addestra il modello sui dati di training
model.fit(X_train_resampled, y_train_resampled)

# Fai predizioni
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Valuta modello usando le metriche di valutazione
utils.print_metrics(y_test, y_pred, y_prob)

# Calcola ROC curve
true_pos, false_pos = utils.get_roc_curve(y_test, y_prob)

n_pos_test = (y_test == 1).sum()
n_neg_test = (y_test == 0).sum()

true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]

# Visualizza grafico ROC curve
utils.draw_roc_curve(true_pos_rate, false_pos_rate)