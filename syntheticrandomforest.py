import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import utils


def get_dataframe_randomforest():
    df = pd.read_csv('resources/synthetic_pc_classification_dataset_large.csv')

    df['CPU Cores'] = pd.cut(df['CPU Cores'],
                                    bins=[2, 4, 6, 12, 16],
                                    labels=[0, 1, 2, 3],
                                    include_lowest=True).astype(int)

    df['GPU VRAM (GB)'] = pd.cut(df['GPU VRAM (GB)'],
                                   bins=[1, 2, 4, 16, 32],
                                   labels=[0, 1, 2, 3],
                                   include_lowest=True).astype(int)

    df['RAM (GB)'] = pd.cut(df['RAM (GB)'],
                                 bins=[4, 8, 16, 32, 64],
                                 labels=[0, 1, 2, 3],
                                 include_lowest=True).astype(int)

    df['SSD (GB)'] = pd.cut(df['SSD (GB)'],
                                 bins=[128, 256, 512, 2048, 8192],
                                 labels=[0, 1, 2, 3],
                                 include_lowest=True).astype(int)

    df['PSU Wattage'] = pd.cut(df['PSU Wattage'],
                            bins=[200, 300, 450, 1000, 2000],
                            labels=[0, 1, 2, 3],
                            include_lowest=True).astype(int)

    # Rimuoviamo eventuali NaN
    df = df.dropna()

    return df


def train():
    # Carica dataset
    df = get_dataframe_randomforest()
    X = df[[
        'CPU Cores',
        'GPU VRAM (GB)',
        'RAM (GB)',
        'SSD (GB)',
        'PSU Wattage'
    ]]
    y = df['Usage Type']

    print("Distribuzione target originale:")
    print(y.value_counts(), "\n")

    # Divisione train/test (Pareto 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("Distribuzione target train:")
    print(y_train.value_counts(), "\n")

    # Addestramento modello
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees in the forest
        max_depth=None,  # No limit on tree depth
        min_samples_split=2,  # Minimum samples required to split a node
        random_state=42,
        class_weight='balanced'
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predizioni
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Valutazione modello
    utils.print_metrics_multiclass(y_test, y_pred, y_prob,4)
