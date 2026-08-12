import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


URL_DATASET = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)


def carregar_dados():
    df = pd.read_csv(URL_DATASET)

    df = df.drop_duplicates().copy()

    return df


def preparar_dados(df):

    y = df["Class"]

    X = df.drop(columns=["Class", "Time"])

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, y


def avaliar_modelo(X, y, contamination):

    modelo = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X)

    predicoes = modelo.predict(X)

    # -1 = anomalia
    #  1 = normal

    y_pred = (predicoes == -1).astype(int)

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    return precision, recall, f1


if __name__ == "__main__":

    print("=== CARREGANDO DADOS ===")

    df = carregar_dados()

    print(f"Registros: {len(df)}")

    X, y = preparar_dados(df)

    contaminations = [
        0.001,
        0.00167,
        0.002,
        0.003,
        0.005,
        0.01
    ]

    resultados = []

    print("\n=== EXPERIMENTO DE CONTAMINATION ===")

    for contamination in contaminations:

        print(
            f"\nTestando contamination = {contamination}"
        )

        precision, recall, f1 = avaliar_modelo(
            X,
            y,
            contamination
        )

        resultados.append({
            "contamination": contamination,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    resultado_df = pd.DataFrame(resultados)

    print("\n=== RESULTADOS ===")

    print(
        resultado_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )