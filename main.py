from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import OneHotEncoder
import helper
import utils
import pandas as pd

# Importa dataset e convertilo in DataFrame
df = helper.get_dataframe_bernoulli()
print(df.head())
# Selezione delle features e del target
X = df[['user_reviews_bin', 'price_final_f2p_bin', 'price_final_over_50_bin', 'is_multiplatform', 'before_2019']]
#X = df[['user_reviews_cat', 'price_original_cat', 'before_2020']]
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
X_train_resampled, y_train_resampled = utils.undersample(X_train, y_train, 0.7)
X_train_resampled, y_train_resampled = utils.oversample(X_train_resampled, y_train_resampled)
print("Resampled X Train: ", X_train_resampled.shape)
print("Resampled y Train: ", y_train_resampled.shape)
print("Resampled Distribution", y_train_resampled.value_counts())

# Addestra il modello classificatore
model = BernoulliNB(alpha=2.0, fit_prior=True)

# Addestra il modello sui dati di training
model.fit(X_train_resampled, y_train_resampled)

# Fai predizioni
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Valuta modello usando le metriche di valutazione
utils.print_metrics(y_test, y_pred, y_prob)

# Calcola ROC curve
false_pos, true_pos, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(false_pos, true_pos, color='red', linestyle='-', label="Sklearn ROC")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()