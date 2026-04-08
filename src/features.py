import pandas as pd

def create_features(df):
    # Ejemplo: convertir categorías
    df = df.copy()

    if "weather" in df.columns:
        df = df.join(
            pd.get_dummies(df["weather"], prefix="weather", drop_first=True)
        )

    if "traffic" in df.columns:
        df = df.join(
            pd.get_dummies(df["traffic"], prefix="traffic", drop_first=True)
        )

    df = df.drop(columns=["weather", "traffic"], errors="ignore")

    return df