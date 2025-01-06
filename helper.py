from datetime import datetime

import pandas as pd

dataset_filepath = './resources/games.csv'

def get_dataframe_bernoulli():
    df = pd.read_csv(dataset_filepath)

    df['user_reviews_bin'] = df['user_reviews'].apply(lambda x: 1 if x >= 100 else 0)
    df['price_original_bin'] = df['price_original'].apply(lambda x: 1 if x >= 9.99 else 0)
    df['is_multiplatform'] = df[['win', 'mac', 'linux']].sum(axis=1).apply(lambda x: 1 if x > 1 else 0)
    df['before_2020'] = df['date_release'].apply(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d').year < 2020 else 0)

    # target
    df['liked'] = df['positive_ratio'].apply(lambda x: 1 if x >= 60 else 0)

    return df
