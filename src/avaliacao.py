from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PROCESSED = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_processado.csv"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
CONTAMINATION = 0.002


# ==========================================================
# CARREGAMENTO
# ==========================================================

print("\n=== CARREGANDO DATASET PROCESSADO ===")

df = pd.read_csv(DATA_PROCESSED)

print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")


# ==========================================================
# SEPARAÇÃO FEATURES / TARGET
# ==========================================================

print("\n=== SEPARANDO FEATURES E TARGET ===")

if "Class" not in df.columns:
    raise ValueError(
        "A coluna 'Class' não existe no dataset."
    )

X = df.drop(columns=["Class"]).copy()
y = df["Class"].astype(int)


# ==========================================================
# GARANTIA CONTRA DATA LEAKAGE
# ==========================================================

if "Class" in X.columns:
    raise RuntimeError(
        "ERRO CRÍTICO: Class está presente nas features."
    )

print(f"Features utilizadas: {X.shape[1]}")
print(f"Target: Class")


# ==========================================================
# DIVISÃO TREINO / TESTE
# ==========================================================

print("\n=== DIVIDINDO OS DADOS ===")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Treino: {len(X_train)}")
print(f"Teste:  {len(X_test)}")


# ==========================================================
# TREINAMENTO DO ISOLATION FOREST
# ==========================================================

print("\n=== TREINANDO ISOLATION FOREST ===")

model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)

print("Modelo treinado com sucesso.")


# ==========================================================
# PREDIÇÃO
# ==========================================================

print("\n=== REALIZANDO PREDIÇÕES ===")

predictions = model.predict(X_test)


# Isolation Forest:
#
#  1  = normal
# -1  = anomalia
#
# Nosso dataset:
#
#  0  = legítima
#  1  = fraude

y_pred = pd.Series(predictions, index=X_test.index)

y_pred = y_pred.map({
    1: 0,
    -1: 1
})


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


print("\n=== MATRIZ DE CONFUSÃO ===")

print(cm)

print("\nTrue Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives (TP):", tp)


# ==========================================================
# MÉTRICAS
# ==========================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ==========================================================
# TAXAS ADICIONAIS
# ==========================================================

false_positive_rate = fp / (fp + tn)

false_negative_rate = fn / (fn + tp)

fraud_detection_rate = recall


# ==========================================================
# RESULTADOS
# ==========================================================

print("\n=== MÉTRICAS ===")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\n=== MÉTRICAS OPERACIONAIS ===")

print(
    f"False Positive Rate: "
    f"{false_positive_rate:.4%}"
)

print(
    f"False Negative Rate: "
    f"{false_negative_rate:.4%}"
)

print(
    f"Fraud Detection Rate: "
    f"{fraud_detection_rate:.4%}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Legítima",
        "Fraude"
    ],
    zero_division=0
)

print("\n=== CLASSIFICATION REPORT ===")

print(report)


# ==========================================================
# SALVAR RELATÓRIO TXT
# ==========================================================

report_path = (
    RESULTS_DIR
    / "evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AVALIAÇÃO - ISOLATION FOREST\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Contamination: {CONTAMINATION}\n"
    )

    file.write(
        f"Random State: {RANDOM_STATE}\n"
    )

    file.write(
        f"Features: {X.shape[1]}\n"
    )

    file.write(
        f"Treino: {len(X_train)}\n"
    )

    file.write(
        f"Teste: {len(X_test)}\n\n"
    )

    file.write(
        "MATRIZ DE CONFUSÃO\n"
    )

    file.write(
        str(cm) + "\n\n"
    )

    file.write(
        f"True Negatives: {tn}\n"
    )

    file.write(
        f"False Positives: {fp}\n"
    )

    file.write(
        f"False Negatives: {fn}\n"
    )

    file.write(
        f"True Positives: {tp}\n\n"
    )

    file.write(
        "MÉTRICAS\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1-score: {f1:.4f}\n"
    )

    file.write(
        f"False Positive Rate: "
        f"{false_positive_rate:.4%}\n"
    )

    file.write(
        f"False Negative Rate: "
        f"{false_negative_rate:.4%}\n"
    )

    file.write(
        f"Fraud Detection Rate: "
        f"{fraud_detection_rate:.4%}\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(report)


# ==========================================================
# GRÁFICO 1 - MATRIZ DE CONFUSÃO
# ==========================================================

print("\n=== GERANDO MATRIZ DE CONFUSÃO ===")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Legítima",
        "Fraude"
    ]
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Matriz de Confusão - Isolation Forest"
)

plt.tight_layout()

confusion_path = (
    RESULTS_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=150
)

plt.close()


# ==========================================================
# GRÁFICO 2 - MÉTRICAS
# ==========================================================

print("\n=== GERANDO GRÁFICO DE MÉTRICAS ===")

metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
}

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    metrics.keys(),
    metrics.values()
)

ax.set_ylim(
    0,
    1
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Métricas - Isolation Forest"
)

for index, value in enumerate(
    metrics.values()
):

    ax.text(
        index,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

metrics_path = (
    RESULTS_DIR
    / "metrics.png"
)

plt.savefig(
    metrics_path,
    dpi=150
)

plt.close()


# ==========================================================
# GRÁFICO 3 - DISTRIBUIÇÃO DAS PREDIÇÕES
# ==========================================================

print("\n=== GERANDO GRÁFICO DE PREDIÇÕES ===")

prediction_counts = pd.Series(
    y_pred
).value_counts().sort_index()

prediction_counts.index = [
    "Legítima",
    "Fraude"
]

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    prediction_counts.index,
    prediction_counts.values
)

ax.set_ylabel(
    "Quantidade"
)

ax.set_title(
    "Distribuição das Predições"
)

for index, value in enumerate(
    prediction_counts.values
):

    ax.text(
        index,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

prediction_path = (
    RESULTS_DIR
    / "prediction_distribution.png"
)

plt.savefig(
    prediction_path,
    dpi=150
)

plt.close()


# ==========================================================
# FINAL
# ==========================================================

print("\n=== AVALIAÇÃO CONCLUÍDA ===")

print("\nArquivos gerados:")

print(confusion_path)
print(metrics_path)
print(prediction_path)
print(report_path)