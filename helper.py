from datetime import datetime

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

dataset_filepath = './resources/games.csv'

def get_dataframe_bernoulli():
    df = pd.read_csv(dataset_filepath)

    df['user_reviews_bin'] = df['user_reviews'].apply(lambda x: 1 if x >= 1000 else 0)
    df['price_original_f2p_bin'] = df['price_original'].apply(lambda x: 1 if x == 0 else 0)
    df['price_original_over_50_bin'] = df['price_original'].apply(lambda x: 1 if x >= 50 else 0)
    df['is_multiplatform'] = df[['win', 'mac', 'linux']].sum(axis=1).apply(lambda x: 1 if x > 1 else 0)
    df['before_2020'] = df['date_release'].apply(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d').year < 2020 else 0)

    # target
    df['liked'] = df['positive_ratio'].apply(lambda x: 1 if x >= 60 else 0)

    return df


def get_dataframe_multinomial():
    df = pd.read_csv(dataset_filepath)

    # Convertiamo user_reviews in categorie
    df['user_reviews_cat'] = pd.cut(df['user_reviews'], bins=[-1, 0, 100, 2000, np.inf], labels=[0, 1, 2, 3])

    # Prezzo in categorie
    df['price_original_cat'] = pd.cut(df['price_original'], bins=[-1, 0, 10, 50, np.inf], labels=[0, 1, 2, 3])

    # Data di rilascio (prima/dopo il 2020)
    df['before_2020'] = df['date_release'].apply(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d').year < 2020 else 0)

    # Target
    df['liked'] = df['positive_ratio'].apply(lambda x: 1 if x >= 60 else 0)

    return df