from pathlib import Path

import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "dataset_processado.csv"
)

CONTAMINATIONS = [
    0.001,
    0.00167,
    0.002,
    0.003,
    0.005,
    0.01,
]

RANDOM_STATE = 42


# ==========================================================
# CARREGAMENTO
# ==========================================================

print("=== CARREGANDO DADOS ===")

df = pd.read_csv(DATASET_PATH)

print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")


# ==========================================================
# PREPARAÇÃO
# ==========================================================

print("\n=== PREPARANDO DADOS ===")

X = df.drop(columns=["Class"])
y = df["Class"]

print(f"Features: {X.shape[1]}")


# ==========================================================
# DIVISÃO TREINO / TESTE
# ==========================================================

print("\n=== DIVIDINDO OS DADOS ===")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Treino: {len(X_train)}")
print(f"Teste:  {len(X_test)}")


# ==========================================================
# EXPERIMENTOS
# ==========================================================

print("\n=== EXPERIMENTO DE CONTAMINATION ===")

resultados = []

for contamination in CONTAMINATIONS:

    print(
        f"\nTestando contamination = {contamination}"
    )

    modelo = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    modelo.fit(X_train)

    predicoes = modelo.predict(X_test)

    # Isolation Forest:
    # -1 = anomalia
    #  1 = normal

    y_pred = (predicoes == -1).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    resultados.append(
        {
            "contamination": contamination,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )


# ==========================================================
# RESULTADOS
# ==========================================================

resultados_df = pd.DataFrame(resultados)

print("\n=== RESULTADOS ===")

print(
    resultados_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ==========================================================
# SELEÇÃO DO MELHOR CONTAMINATION
# ==========================================================

melhor_resultado = resultados_df.loc[
    resultados_df["f1_score"].idxmax()
]

melhor_contamination = (
    melhor_resultado["contamination"]
)

print("\n=== MELHOR CONFIGURAÇÃO ===")

print(
    f"Contamination: {melhor_contamination}"
)

print(
    f"Precision: {melhor_resultado['precision']:.4f}"
)

print(
    f"Recall: {melhor_resultado['recall']:.4f}"
)

print(
    f"F1-score: {melhor_resultado['f1_score']:.4f}"
)


# ==========================================================
# MODELO FINAL
# ==========================================================

print("\n=== TREINANDO MODELO FINAL ===")

modelo_final = IsolationForest(
    n_estimators=200,
    contamination=melhor_contamination,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

modelo_final.fit(X_train)

print("Modelo final treinado com sucesso.")


# ==========================================================
# PREDIÇÕES FINAIS
# ==========================================================

predicoes_finais = modelo_final.predict(X_test)

y_pred_final = (
    predicoes_finais == -1
).astype(int)


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

print("\n=== MATRIZ DE CONFUSÃO ===")

matriz = confusion_matrix(
    y_test,
    y_pred_final,
)

print(matriz)


# ==========================================================
# MÉTRICAS
# ==========================================================

precision_final = precision_score(
    y_test,
    y_pred_final,
    zero_division=0,
)

recall_final = recall_score(
    y_test,
    y_pred_final,
    zero_division=0,
)

f1_final = f1_score(
    y_test,
    y_pred_final,
    zero_division=0,
)


print("\n=== MÉTRICAS DO MODELO FINAL ===")

print(f"Precision: {precision_final:.4f}")
print(f"Recall:    {recall_final:.4f}")
print(f"F1-score:  {f1_final:.4f}")


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n=== CLASSIFICATION REPORT ===")

print(
    classification_report(
        y_test,
        y_pred_final,
        target_names=[
            "Legítima",
            "Fraude",
        ],
        zero_division=0,
    )
)