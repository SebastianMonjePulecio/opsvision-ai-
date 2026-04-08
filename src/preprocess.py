import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()

    # Normalizar columnas
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    return df