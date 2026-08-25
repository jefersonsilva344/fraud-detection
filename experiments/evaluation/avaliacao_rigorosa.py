from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
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

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "rigorous"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42

N_ESTIMATORS = 200

CONTAMINATIONS = [
    0.001,
    0.00167,
    0.002,
    0.003,
    0.005,
    0.01,
]


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def calculate_metrics(
    y_true,
    y_pred,
    anomaly_scores
):
    """
    Calcula todas as métricas do modelo.
    """

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

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
    )

    false_negative_rate = (
        fn / (fn + tp)
    )

    return {
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


def optimize_threshold(
    y_true,
    anomaly_scores
):
    """
    Encontra o threshold que maximiza F1
    utilizando SOMENTE o conjunto de validação.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            anomaly_scores
        )
    )

    f1_scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = np.argmax(
        f1_scores
    )

    best_threshold = thresholds[
        best_index
    ]

    return (
        best_threshold,
        precision[best_index],
        recall[best_index],
        f1_scores[best_index],
        precision,
        recall,
        thresholds,
        f1_scores,
    )


# ==========================================================
# CARREGAMENTO
# ==========================================================

print("\n=== CARREGANDO DATASET ===")

if not DATA_PROCESSED.exists():

    raise FileNotFoundError(
        f"Dataset não encontrado:\n"
        f"{DATA_PROCESSED}"
    )

df = pd.read_csv(
    DATA_PROCESSED
)

print(
    f"Registros: {len(df)}"
)

print(
    f"Colunas: {len(df.columns)}"
)


# ==========================================================
# FEATURES / TARGET
# ==========================================================

print(
    "\n=== SEPARANDO FEATURES E TARGET ==="
)

if "Class" not in df.columns:

    raise ValueError(
        "A coluna 'Class' não existe."
    )

X = df.drop(
    columns=["Class"]
).copy()

y = df["Class"].astype(int)


# ==========================================================
# VALIDAÇÃO ESTRUTURAL
# ==========================================================

if "Class" in X.columns:

    raise RuntimeError(
        "ERRO CRÍTICO: Class está nas features."
    )

if X.shape[1] != 30:

    raise RuntimeError(
        f"ERRO: esperado 30 features. "
        f"Encontradas: {X.shape[1]}"
    )

print(
    f"Features: {X.shape[1]}"
)

print(
    "Class NÃO está nas features."
)


# ==========================================================
# PRIMEIRA DIVISÃO
#
# 70% TRAIN
# 30% TEMPORÁRIO
# ==========================================================

print(
    "\n=== DIVISÃO TRAIN / TEMP ==="
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)


# ==========================================================
# SEGUNDA DIVISÃO
#
# TEMP = 30%
#
# metade -> VALIDATION = 15%
# metade -> TEST       = 15%
# ==========================================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print(
    f"Train:      {len(X_train)}"
)

print(
    f"Validation: {len(X_val)}"
)

print(
    f"Test:       {len(X_test)}"
)


# ==========================================================
# DISTRIBUIÇÃO DAS CLASSES
# ==========================================================

print(
    "\n=== DISTRIBUIÇÃO DAS CLASSES ==="
)

print(
    f"Train - fraude: "
    f"{y_train.sum()}"
)

print(
    f"Validation - fraude: "
    f"{y_val.sum()}"
)

print(
    f"Test - fraude: "
    f"{y_test.sum()}"
)


# ==========================================================
# CONVERSÃO NUMPY
# ==========================================================

X_train_np = X_train.to_numpy()

X_val_np = X_val.to_numpy()

X_test_np = X_test.to_numpy()

y_train_np = y_train.to_numpy()

y_val_np = y_val.to_numpy()

y_test_np = y_test.to_numpy()


# ==========================================================
# EXPERIMENTO NA VALIDATION
# ==========================================================

print(
    "\n=== OTIMIZAÇÃO NA VALIDATION ==="
)

validation_results = []

best_validation = None

best_validation_model = None

best_validation_scores = None

best_pr_curve = None


for contamination in CONTAMINATIONS:

    print(
        f"\nTestando contamination = "
        f"{contamination}"
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # ------------------------------------------------------
    # TREINAMENTO SOMENTE NO TRAIN
    # ------------------------------------------------------

    model.fit(
        X_train_np
    )

    # ------------------------------------------------------
    # SCORE VALIDATION
    # ------------------------------------------------------

    validation_scores = -model.score_samples(
        X_val_np
    )

    # ------------------------------------------------------
    # OTIMIZAÇÃO DO THRESHOLD
    # ------------------------------------------------------

    (
        threshold,
        precision_at_threshold,
        recall_at_threshold,
        f1_at_threshold,
        precision_curve,
        recall_curve,
        thresholds,
        f1_scores,
    ) = optimize_threshold(
        y_val_np,
        validation_scores
    )

    y_val_pred = (
        validation_scores >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_val_np,
        y_val_pred,
        validation_scores
    )

    result = {
        "contamination": contamination,
        "threshold": threshold,
        **metrics,
    }

    validation_results.append(
        result
    )

    print(
        f"Threshold: {threshold:.8f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score: "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    # ------------------------------------------------------
    # SELEÇÃO DO MELHOR MODELO
    #
    # Critério:
    # F1 na VALIDATION
    # ------------------------------------------------------

    if (
        best_validation is None
        or metrics["f1_score"]
        > best_validation["f1_score"]
    ):

        best_validation = result

        best_validation_model = model

        best_validation_scores = validation_scores

        best_pr_curve = (
            precision_curve,
            recall_curve,
            thresholds,
            f1_scores,
        )


# ==========================================================
# RESULTADOS VALIDATION
# ==========================================================

validation_df = pd.DataFrame(
    validation_results
)

print(
    "\n=== RESULTADOS DA VALIDATION ==="
)

print(
    validation_df.to_string(
        index=False
    )
)


# ==========================================================
# MELHOR CONFIGURAÇÃO
# ==========================================================

best_contamination = (
    best_validation["contamination"]
)

best_threshold = (
    best_validation["threshold"]
)


print(
    "\n=== MELHOR CONFIGURAÇÃO ==="
)

print(
    f"Contamination: "
    f"{best_contamination}"
)

print(
    f"Threshold: "
    f"{best_threshold:.8f}"
)

print(
    f"Validation Precision: "
    f"{best_validation['precision']:.4f}"
)

print(
    f"Validation Recall: "
    f"{best_validation['recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_validation['f1_score']:.4f}"
)

print(
    f"Validation ROC-AUC: "
    f"{best_validation['roc_auc']:.4f}"
)

print(
    f"Validation PR-AUC: "
    f"{best_validation['pr_auc']:.4f}"
)


# ==========================================================
# AVALIAÇÃO FINAL NO TEST
#
# IMPORTANTE:
#
# O TESTE NÃO PARTICIPOU DA ESCOLHA
# DO MODELO OU THRESHOLD.
# ==========================================================

print(
    "\n=== AVALIAÇÃO FINAL NO TEST ==="
)

test_scores = (
    -best_validation_model.score_samples(
        X_test_np
    )
)

y_test_pred = (
    test_scores >= best_threshold
).astype(int)


# ==========================================================
# MÉTRICAS TEST
# ==========================================================

test_metrics = calculate_metrics(
    y_test_np,
    y_test_pred,
    test_scores
)


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

test_cm = confusion_matrix(
    y_test_np,
    y_test_pred
)

tn, fp, fn, tp = test_cm.ravel()


print(
    "\n=== MATRIZ DE CONFUSÃO - TEST ==="
)

print(
    test_cm
)

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
# MÉTRICAS FINAIS
# ==========================================================

print(
    "\n=== MÉTRICAS FINAIS - TEST ==="
)

print(
    f"Precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall: "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1-score: "
    f"{test_metrics['f1_score']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC: "
    f"{test_metrics['pr_auc']:.4f}"
)


# ==========================================================
# MÉTRICAS OPERACIONAIS
# ==========================================================

print(
    "\n=== MÉTRICAS OPERACIONAIS - TEST ==="
)

print(
    f"False Positive Rate: "
    f"{test_metrics['false_positive_rate']:.4%}"
)

print(
    f"False Negative Rate: "
    f"{test_metrics['false_negative_rate']:.4%}"
)

print(
    f"Fraud Detection Rate: "
    f"{test_metrics['recall']:.4%}"
)


# ==========================================================
# BASELINE
# ==========================================================

fraud_rate_test = (
    y_test_np.mean()
)

print(
    "\n=== BASELINE ==="
)

print(
    f"Taxa real de fraude no Test: "
    f"{fraud_rate_test:.4%}"
)

print(
    f"Fraudes no Test: "
    f"{y_test_np.sum()}"
)

print(
    f"Fraudes detectadas: "
    f"{tp}"
)

print(
    f"Fraudes perdidas: "
    f"{fn}"
)

print(
    f"Falsos positivos: "
    f"{fp}"
)


# ==========================================================
# COMPARAÇÃO VALIDATION x TEST
# ==========================================================

comparison_df = pd.DataFrame([
    {
        "dataset": "Validation",
        "precision": best_validation["precision"],
        "recall": best_validation["recall"],
        "f1_score": best_validation["f1_score"],
        "roc_auc": best_validation["roc_auc"],
        "pr_auc": best_validation["pr_auc"],
    },
    {
        "dataset": "Test",
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1_score"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
    },
])


print(
    "\n=== VALIDATION x TEST ==="
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ==========================================================
# SALVAR RESULTADOS
# ==========================================================

validation_path = (
    RESULTS_DIR
    / "validation_results.csv"
)

validation_df.to_csv(
    validation_path,
    index=False
)


comparison_path = (
    RESULTS_DIR
    / "validation_test_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


metrics_df = pd.DataFrame([{
    "model": "Isolation Forest",
    "contamination": best_contamination,
    "threshold": best_threshold,

    "validation_precision":
        best_validation["precision"],

    "validation_recall":
        best_validation["recall"],

    "validation_f1":
        best_validation["f1_score"],

    "validation_roc_auc":
        best_validation["roc_auc"],

    "validation_pr_auc":
        best_validation["pr_auc"],

    "test_precision":
        test_metrics["precision"],

    "test_recall":
        test_metrics["recall"],

    "test_f1":
        test_metrics["f1_score"],

    "test_roc_auc":
        test_metrics["roc_auc"],

    "test_pr_auc":
        test_metrics["pr_auc"],

    "false_positive_rate":
        test_metrics["false_positive_rate"],

    "false_negative_rate":
        test_metrics["false_negative_rate"],

    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "true_positives": tp,
}])


metrics_path = (
    RESULTS_DIR
    / "rigorous_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ==========================================================
# CURVA PRECISION-RECALL
# ==========================================================

print(
    "\n=== GERANDO CURVA PRECISION-RECALL ==="
)

precision_test, recall_test, _ = (
    precision_recall_curve(
        y_test_np,
        test_scores
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    recall_test,
    precision_test
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall - Isolation Forest"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

pr_path = (
    RESULTS_DIR
    / "precision_recall_test.png"
)

plt.savefig(
    pr_path,
    dpi=150
)

plt.close()


# ==========================================================
# CURVA F1 DA VALIDATION
# ==========================================================

print(
    "\n=== GERANDO CURVA F1 ==="
)

(
    validation_precision_curve,
    validation_recall_curve,
    validation_thresholds,
    validation_f1_scores,
) = best_pr_curve


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    validation_thresholds,
    validation_f1_scores
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label="Threshold selecionado"
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "F1-score"
)

plt.title(
    "Otimização do Threshold na Validation"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

f1_path = (
    RESULTS_DIR
    / "validation_threshold_f1.png"
)

plt.savefig(
    f1_path,
    dpi=150
)

plt.close()


# ==========================================================
# GRÁFICO VALIDATION x TEST
# ==========================================================

print(
    "\n=== GERANDO COMPARAÇÃO VALIDATION x TEST ==="
)

metric_names = [
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]

validation_values = [
    best_validation["precision"],
    best_validation["recall"],
    best_validation["f1_score"],
    best_validation["roc_auc"],
    best_validation["pr_auc"],
]

test_values = [
    test_metrics["precision"],
    test_metrics["recall"],
    test_metrics["f1_score"],
    test_metrics["roc_auc"],
    test_metrics["pr_auc"],
]

x = np.arange(
    len(metric_names)
)

width = 0.35

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    x - width / 2,
    validation_values,
    width,
    label="Validation"
)

plt.bar(
    x + width / 2,
    test_values,
    width,
    label="Test"
)

plt.xticks(
    x,
    metric_names
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.title(
    "Validation x Test - Isolation Forest"
)

plt.legend()

plt.tight_layout()

comparison_image = (
    RESULTS_DIR
    / "validation_test_comparison.png"
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
    / "rigorous_evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AVALIAÇÃO RIGOROSA - ISOLATION FOREST\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "METODOLOGIA\n"
    )

    file.write(
        "Train: 70%\n"
    )

    file.write(
        "Validation: 15%\n"
    )

    file.write(
        "Test: 15%\n\n"
    )

    file.write(
        "CONFIGURAÇÃO SELECIONADA\n"
    )

    file.write(
        f"Contamination: "
        f"{best_contamination}\n"
    )

    file.write(
        f"Threshold: "
        f"{best_threshold:.8f}\n\n"
    )

    file.write(
        "VALIDATION\n"
    )

    file.write(
        f"Precision: "
        f"{best_validation['precision']:.4f}\n"
    )

    file.write(
        f"Recall: "
        f"{best_validation['recall']:.4f}\n"
    )

    file.write(
        f"F1-score: "
        f"{best_validation['f1_score']:.4f}\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{best_validation['roc_auc']:.4f}\n"
    )

    file.write(
        f"PR-AUC: "
        f"{best_validation['pr_auc']:.4f}\n\n"
    )

    file.write(
        "TEST - AVALIAÇÃO FINAL\n"
    )

    file.write(
        f"Precision: "
        f"{test_metrics['precision']:.4f}\n"
    )

    file.write(
        f"Recall: "
        f"{test_metrics['recall']:.4f}\n"
    )

    file.write(
        f"F1-score: "
        f"{test_metrics['f1_score']:.4f}\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}\n"
    )

    file.write(
        f"PR-AUC: "
        f"{test_metrics['pr_auc']:.4f}\n\n"
    )

    file.write(
        "MATRIZ DE CONFUSÃO - TEST\n"
    )

    file.write(
        str(test_cm) + "\n\n"
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
        "MÉTRICAS OPERACIONAIS\n"
    )

    file.write(
        f"False Positive Rate: "
        f"{test_metrics['false_positive_rate']:.4%}\n"
    )

    file.write(
        f"False Negative Rate: "
        f"{test_metrics['false_negative_rate']:.4%}\n"
    )

    file.write(
        f"Fraud Detection Rate: "
        f"{test_metrics['recall']:.4%}\n\n"
    )

    file.write(
        "BASELINE\n"
    )

    file.write(
        f"Fraud rate no Test: "
        f"{fraud_rate_test:.4%}\n"
    )

    file.write(
        f"Fraudes no Test: "
        f"{y_test_np.sum()}\n"
    )

    file.write(
        f"Fraudes detectadas: "
        f"{tp}\n"
    )

    file.write(
        f"Fraudes perdidas: "
        f"{fn}\n"
    )

    file.write(
        f"Falsos positivos: "
        f"{fp}\n"
    )


# ==========================================================
# FINAL
# ==========================================================

print(
    "\n=== AVALIAÇÃO RIGOROSA CONCLUÍDA ==="
)

print(
    "\nArquivos gerados:"
)

print(
    validation_path
)

print(
    comparison_path
)

print(
    metrics_path
)

print(
    pr_path
)

print(
    f1_path
)

print(
    comparison_image
)

print(
    report_path
)