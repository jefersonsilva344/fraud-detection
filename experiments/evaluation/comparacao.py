from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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

print("\n=== CARREGANDO DATASET ===")

df = pd.read_csv(DATA_PROCESSED)

X = df.drop(
    columns=["Class"]
)

y = df["Class"].astype(int)


# ==========================================================
# VALIDAÇÃO
# ==========================================================

if "Class" in X.columns:
    raise RuntimeError(
        "Class está presente nas features."
    )

if X.shape[1] != 30:
    raise RuntimeError(
        f"Esperadas 30 features. "
        f"Encontradas: {X.shape[1]}"
    )


# ==========================================================
# DIVISÃO
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"Treino: {len(X_train)}"
)

print(
    f"Teste: {len(X_test)}"
)


# ==========================================================
# NUMPY
# ==========================================================

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()


# ==========================================================
# FUNÇÃO DE MÉTRICAS
# ==========================================================

def calculate_metrics(
    y_true,
    y_pred,
    anomaly_scores,
    model_name,
    contamination
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        y_true,
        anomaly_scores
    )

    false_positive_rate = (
        fp / (fp + tn)
    )

    false_negative_rate = (
        fn / (fn + tp)
    )

    return {
        "model": model_name,
        "contamination": contamination,
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
    }


# ==========================================================
# ISOLATION FOREST
# ==========================================================

print("\n=== AVALIANDO ISOLATION FOREST ===")

isolation_results = []

for contamination in CONTAMINATIONS:

    print(
        f"Testando contamination = "
        f"{contamination}"
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train_np
    )

    predictions = model.predict(
        X_test_np
    )

    y_pred = np.where(
        predictions == -1,
        1,
        0
    )

    # Isolation Forest:
    # score_samples maior = mais normal.
    #
    # Inversão:
    # maior = mais anômalo.

    anomaly_scores = -model.score_samples(
        X_test_np
    )

    metrics = calculate_metrics(
        y_test_np,
        y_pred,
        anomaly_scores,
        "Isolation Forest",
        contamination
    )

    isolation_results.append(
        metrics
    )


isolation_df = pd.DataFrame(
    isolation_results
)

best_isolation = isolation_df.loc[
    isolation_df["f1_score"].idxmax()
]


# ==========================================================
# LOF
# ==========================================================

print("\n=== AVALIANDO LOF ===")

lof_results = []

for contamination in CONTAMINATIONS:

    print(
        f"Testando contamination = "
        f"{contamination}"
    )

    model = LocalOutlierFactor(
        n_neighbors=N_NEIGHBORS,
        contamination=contamination,
        novelty=True,
        n_jobs=-1
    )

    model.fit(
        X_train_np
    )

    predictions = model.predict(
        X_test_np
    )

    y_pred = np.where(
        predictions == -1,
        1,
        0
    )

    anomaly_scores = -model.score_samples(
        X_test_np
    )

    metrics = calculate_metrics(
        y_test_np,
        y_pred,
        anomaly_scores,
        "Local Outlier Factor",
        contamination
    )

    lof_results.append(
        metrics
    )


lof_df = pd.DataFrame(
    lof_results
)

best_lof = lof_df.loc[
    lof_df["f1_score"].idxmax()
]


# ==========================================================
# COMPARAÇÃO
# ==========================================================

comparison = pd.DataFrame([
    best_isolation,
    best_lof
])


print("\n=== COMPARAÇÃO FINAL ===")

print(
    comparison[
        [
            "model",
            "contamination",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "false_positive_rate",
            "false_negative_rate",
        ]
    ].to_string(
        index=False
    )
)


# ==========================================================
# MELHOR MODELO
# ==========================================================

best_model = comparison.loc[
    comparison["f1_score"].idxmax()
]

print("\n=== MELHOR MODELO ===")

print(
    f"Modelo: {best_model['model']}"
)

print(
    f"Contamination: "
    f"{best_model['contamination']}"
)

print(
    f"Precision: "
    f"{best_model['precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_model['recall']:.4f}"
)

print(
    f"F1-score: "
    f"{best_model['f1_score']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{best_model['roc_auc']:.4f}"
)

print(
    f"PR-AUC: "
    f"{best_model['pr_auc']:.4f}"
)


# ==========================================================
# SALVAR
# ==========================================================

comparison_path = (
    RESULTS_DIR
    / "model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ==========================================================
# GRÁFICO
# ==========================================================

print("\n=== GERANDO GRÁFICO DE COMPARAÇÃO ===")

models = comparison["model"]

x = np.arange(
    len(models)
)

width = 0.20

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.bar(
    x - 2 * width,
    comparison["precision"],
    width,
    label="Precision"
)

ax.bar(
    x - width,
    comparison["recall"],
    width,
    label="Recall"
)

ax.bar(
    x,
    comparison["f1_score"],
    width,
    label="F1"
)

ax.bar(
    x + width,
    comparison["roc_auc"],
    width,
    label="ROC-AUC"
)

ax.bar(
    x + 2 * width,
    comparison["pr_auc"],
    width,
    label="PR-AUC"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    models
)

ax.set_ylim(
    0,
    1
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Comparação de Modelos"
)

ax.legend()

plt.tight_layout()

comparison_image = (
    RESULTS_DIR
    / "model_comparison.png"
)

plt.savefig(
    comparison_image,
    dpi=150
)

plt.close()


# ==========================================================
# RELATÓRIO
# ==========================================================

report_path = (
    RESULTS_DIR
    / "model_comparison_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "COMPARAÇÃO DE MODELOS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        comparison.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "MELHOR MODELO\n"
    )

    file.write(
        f"Modelo: {best_model['model']}\n"
    )

    file.write(
        f"Contamination: "
        f"{best_model['contamination']}\n"
    )

    file.write(
        f"Precision: "
        f"{best_model['precision']:.4f}\n"
    )

    file.write(
        f"Recall: "
        f"{best_model['recall']:.4f}\n"
    )

    file.write(
        f"F1-score: "
        f"{best_model['f1_score']:.4f}\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{best_model['roc_auc']:.4f}\n"
    )

    file.write(
        f"PR-AUC: "
        f"{best_model['pr_auc']:.4f}\n"
    )

    file.write(
        f"False Positive Rate: "
        f"{best_model['false_positive_rate']:.4%}\n"
    )

    file.write(
        f"False Negative Rate: "
        f"{best_model['false_negative_rate']:.4%}\n"
    )


# ==========================================================
# FINAL
# ==========================================================

print("\n=== COMPARAÇÃO CONCLUÍDA ===")

print(
    f"CSV: {comparison_path}"
)

print(
    f"Gráfico: {comparison_image}"
)

print(
    f"Relatório: {report_path}"
)