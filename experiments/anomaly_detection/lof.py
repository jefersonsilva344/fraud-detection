from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


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

CONTAMINATIONS = [
    0.001,
    0.00167,
    0.002,
    0.003,
    0.005,
    0.01,
]

N_NEIGHBORS = 20


# ==========================================================
# CARREGAMENTO
# ==========================================================

print("\n=== CARREGANDO DATASET PROCESSADO ===")

df = pd.read_csv(DATA_PROCESSED)

print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")


# ==========================================================
# FEATURES / TARGET
# ==========================================================

print("\n=== SEPARANDO FEATURES E TARGET ===")

if "Class" not in df.columns:
    raise ValueError(
        "A coluna 'Class' não foi encontrada."
    )

X = df.drop(columns=["Class"]).copy()

y = df["Class"].astype(int)


# ==========================================================
# PROTEÇÃO CONTRA DATA LEAKAGE
# ==========================================================

if "Class" in X.columns:
    raise RuntimeError(
        "ERRO CRÍTICO: Class está presente nas features."
    )

if X.shape[1] != 30:
    raise RuntimeError(
        f"Número inesperado de features: {X.shape[1]}. "
        "Esperado: 30."
    )

print(f"Features utilizadas: {X.shape[1]}")
print("Target: Class")


# ==========================================================
# DIVISÃO
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
# CONVERSÃO PARA NUMPY
#
# Corrige o warning:
# "X does not have valid feature names..."
# ==========================================================

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()


# ==========================================================
# EXPERIMENTO
# ==========================================================

print("\n=== EXPERIMENTO LOF ===")

results = []

for contamination in CONTAMINATIONS:

    print(
        f"\nTestando contamination = "
        f"{contamination}"
    )

    model = LocalOutlierFactor(
        n_neighbors=N_NEIGHBORS,
        contamination=contamination,
        novelty=True,
        n_jobs=-1
    )

    model.fit(X_train_np)

    predictions = model.predict(X_test_np)

    # LOF:
    #  1  = normal
    # -1  = anomalia
    #
    # Dataset:
    #  0  = legítima
    #  1  = fraude

    y_pred = np.where(
        predictions == -1,
        1,
        0
    )

    precision = precision_score(
        y_test_np,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test_np,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test_np,
        y_pred,
        zero_division=0
    )

    # score_samples:
    #
    # Quanto maior o valor, mais normal.
    #
    # Invertendo:
    #
    # quanto maior, mais anômalo.

    anomaly_scores = -model.score_samples(
        X_test_np
    )

    roc_auc = roc_auc_score(
        y_test_np,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        y_test_np,
        anomaly_scores
    )

    results.append({
        "contamination": contamination,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    })


# ==========================================================
# RESULTADOS
# ==========================================================

results_df = pd.DataFrame(results)

print("\n=== RESULTADOS LOF ===")

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================================
# MELHOR CONFIGURAÇÃO
# ==========================================================

best_result = results_df.loc[
    results_df["f1_score"].idxmax()
]

best_contamination = float(
    best_result["contamination"]
)

print("\n=== MELHOR CONFIGURAÇÃO ===")

print(
    f"Contamination: "
    f"{best_contamination}"
)

print(
    f"Precision: "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_result['recall']:.4f}"
)

print(
    f"F1-score: "
    f"{best_result['f1_score']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{best_result['roc_auc']:.4f}"
)

print(
    f"PR-AUC: "
    f"{best_result['pr_auc']:.4f}"
)


# ==========================================================
# MODELO FINAL
# ==========================================================

print("\n=== TREINANDO LOF FINAL ===")

final_model = LocalOutlierFactor(
    n_neighbors=N_NEIGHBORS,
    contamination=best_contamination,
    novelty=True,
    n_jobs=-1
)

final_model.fit(X_train_np)

print(
    "Modelo LOF treinado com sucesso."
)


# ==========================================================
# PREDIÇÕES
# ==========================================================

print("\n=== REALIZANDO PREDIÇÕES ===")

predictions = final_model.predict(
    X_test_np
)

y_pred = np.where(
    predictions == -1,
    1,
    0
)

anomaly_scores = -final_model.score_samples(
    X_test_np
)


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

cm = confusion_matrix(
    y_test_np,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n=== MATRIZ DE CONFUSÃO ===")

print(cm)

print(
    f"\nTrue Negatives: {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives: {tp}"
)


# ==========================================================
# MÉTRICAS
# ==========================================================

precision = precision_score(
    y_test_np,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test_np,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test_np,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_np,
    anomaly_scores
)

pr_auc = average_precision_score(
    y_test_np,
    anomaly_scores
)

false_positive_rate = (
    fp / (fp + tn)
)

false_negative_rate = (
    fn / (fn + tp)
)


# ==========================================================
# MÉTRICAS
# ==========================================================

print("\n=== MÉTRICAS LOF ===")

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)


# ==========================================================
# MÉTRICAS OPERACIONAIS
# ==========================================================

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
    f"{recall:.4%}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_test_np,
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
# RESULTADOS DO EXPERIMENTO
# ==========================================================

results_path = (
    RESULTS_DIR
    / "lof_experiment.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ==========================================================
# MÉTRICAS FINAIS
# ==========================================================

metrics_df = pd.DataFrame([{
    "model": "Local Outlier Factor",
    "contamination": best_contamination,
    "n_neighbors": N_NEIGHBORS,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "true_positives": tp,
}])

metrics_path = (
    RESULTS_DIR
    / "lof_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ==========================================================
# RELATÓRIO TXT
# ==========================================================

report_path = (
    RESULTS_DIR
    / "lof_evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AVALIAÇÃO - LOCAL OUTLIER FACTOR\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Contamination: {best_contamination}\n"
    )

    file.write(
        f"N Neighbors: {N_NEIGHBORS}\n"
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
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    file.write(
        f"PR-AUC: {pr_auc:.4f}\n"
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
        f"{recall:.4%}\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(report)


# ==========================================================
# GRÁFICO DE MÉTRICAS
# ==========================================================

print("\n=== GERANDO GRÁFICO DE MÉTRICAS ===")

metric_names = [
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC",
]

metric_values = [
    precision,
    recall,
    f1,
    roc_auc,
    pr_auc,
]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    metric_names,
    metric_values
)

ax.set_ylim(
    0,
    1
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Avaliação do Local Outlier Factor"
)

for index, value in enumerate(
    metric_values
):

    ax.text(
        index,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

metrics_image = (
    RESULTS_DIR
    / "lof_metrics.png"
)

plt.savefig(
    metrics_image,
    dpi=150
)

plt.close()


# ==========================================================
# FINAL
# ==========================================================

print("\n=== LOF CONCLUÍDO ===")

print("\nArquivos gerados:")

print(results_path)
print(metrics_path)
print(report_path)
print(metrics_image)