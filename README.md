
<p align="center">
  <img src="https://github.com/user-attachments/assets/f53fefc0-749f-4dca-82f2-f37032738f58" alt="immagine centrata" style="width: 250px" />
  <br>
</p>

# 🎮 SteamRating - Predict Steam Game Popularity

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2.0-lightgrey)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Prevedi se un gioco su Steam piace oppure no basandoti sulle sue caratteristiche.

## 📚 Descrizione

SteamRating è un progetto di machine learning sviluppato per l'esame di Machine Learning presso l’Università degli Studi di Salerno. L'obiettivo è predire se un videogioco pubblicato su Steam sia *piaciuto* o *non piaciuto*, basandosi su alcune caratteristiche visibili nella sua pagina prodotto.

Un gioco è considerato **"piaciuto"** se ha almeno il **60% di recensioni positive**.

## 🔍 Dataset

Il dataset utilizzato è:  
📦 [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

- Campioni: **50,751** giochi
- Feature usate:
  - Titolo
  - Data di rilascio
  - Prezzo
  - Piattaforme supportate
  - Numero di recensioni
  - Percentuale di recensioni positive (target)
  - Percentuale di sconto

Tecniche utilizzate:
- Undersampling e oversampling (SMOTE)
- Grid Search e K-Fold Cross Validation per tuning
- Preprocessing con trasformazione binaria per BernoulliNB
