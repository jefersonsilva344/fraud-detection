import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


URL_DATASET = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)


def carregar_dados():
    """Carrega o dataset e remove registros duplicados."""

    df = pd.read_csv(URL_DATASET)

    df = df.drop_duplicates().copy()

    return df


def preparar_dados(df):
    """Prepara as features para o modelo."""

    y = df["Class"]

    X = df.drop(columns=["Class", "Time"])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index,
    )

    return X_scaled, y


def treinar_modelo(X):
    """Treina o Isolation Forest."""

    modelo = IsolationForest(
        n_estimators=200,
        contamination=0.00167,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X)

    return modelo


def avaliar_modelo(df):
    """Avalia as previsões do modelo."""

    matriz = confusion_matrix(
        df["Class"],
        df["Fraude_Prevista"],
    )

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(matriz)

    precision = precision_score(
        df["Class"],
        df["Fraude_Prevista"],
        zero_division=0,
    )

    recall = recall_score(
        df["Class"],
        df["Fraude_Prevista"],
        zero_division=0,
    )

    f1 = f1_score(
        df["Class"],
        df["Fraude_Prevista"],
        zero_division=0,
    )

    print("\n=== MÉTRICAS DO MODELO ===")

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            df["Class"],
            df["Fraude_Prevista"],
            target_names=["Legítima", "Fraude"],
            zero_division=0,
        )
    )


if __name__ == "__main__":

    print("=== CARREGANDO DADOS ===")

    df = carregar_dados()

    print(f"Registros: {len(df)}")

    print("\n=== PREPARANDO DADOS ===")

    X, y = preparar_dados(df)

    print(f"Features: {X.shape[1]}")

    print("\n=== TREINANDO ISOLATION FOREST ===")

    modelo = treinar_modelo(X)

    print("Modelo treinado com sucesso.")

    print("\n=== GERANDO PREDIÇÕES ===")

    predicoes = modelo.predict(X)

    df["Anomalia"] = predicoes

    # Isolation Forest:
    # -1 = anomalia
    #  1 = normal
    df["Fraude_Prevista"] = (
        df["Anomalia"] == -1
    ).astype(int)

    print("\n=== DISTRIBUIÇÃO DAS PREDIÇÕES ===")

    print(
        df["Anomalia"]
        .value_counts()
        .sort_index()
    )

    print("\n=== DISTRIBUIÇÃO DAS FRAUDES PREVISTAS ===")

    print(
        df["Fraude_Prevista"]
        .value_counts()
        .sort_index()
    )

    avaliar_modelo(df)