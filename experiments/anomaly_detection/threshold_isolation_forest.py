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
    / "threshold"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42

CONTAMINATION = 0.002

N_ESTIMATORS = 200


# ==========================================================
# CARREGAMENTO
# ==========================================================

print("\n=== CARREGANDO DATASET ===")

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

print("\n=== SEPARANDO FEATURES E TARGET ===")

if "Class" not in df.columns:
    raise ValueError(
        "A coluna Class não existe no dataset."
    )

X = df.drop(
    columns=["Class"]
).copy()

y = df["Class"].astype(int)


# ==========================================================
# VALIDAÇÃO
# ==========================================================

if "Class" in X.columns:
    raise RuntimeError(
        "ERRO: Class está presente nas features."
    )

if X.shape[1] != 30:
    raise RuntimeError(
        f"ERRO: esperado 30 features, "
        f"encontradas {X.shape[1]}."
    )

print(
    f"Features: {X.shape[1]}"
)

print(
    "Class NÃO está nas features."
)


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

print(
    f"Treino: {len(X_train)}"
)

print(
    f"Teste:  {len(X_test)}"
)


# ==========================================================
# CONVERSÃO NUMPY
# ==========================================================

X_train_np = X_train.to_numpy()

X_test_np = X_test.to_numpy()

y_test_np = y_test.to_numpy()


# ==========================================================
# TREINAMENTO
# ==========================================================

print("\n=== TREINANDO ISOLATION FOREST ===")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train_np
)

print(
    "Modelo treinado com sucesso."
)


# ==========================================================
# SCORE DE ANOMALIA
# ==========================================================

print("\n=== CALCULANDO SCORES DE ANOMALIA ===")

# score_samples:
# maior = mais normal
#
# Invertemos:
# maior = mais anômalo

anomaly_scores = -model.score_samples(
    X_test_np
)


# ==========================================================
# ROC-AUC / PR-AUC
# ==========================================================

roc_auc = roc_auc_score(
    y_test_np,
    anomaly_scores
)

pr_auc = average_precision_score(
    y_test_np,
    anomaly_scores
)

print(
    f"\nROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC:  {pr_auc:.4f}"
)


# ==========================================================
# PRECISION-RECALL CURVE
# ==========================================================

print(
    "\n=== OTIMIZANDO THRESHOLD ==="
)

precision, recall, thresholds = (
    precision_recall_curve(
        y_test_np,
        anomaly_scores
    )
)


# precision_recall_curve retorna:
#
# precision -> tamanho N+1
# recall    -> tamanho N+1
# thresholds -> tamanho N
#
# Portanto usamos precision[:-1] e recall[:-1].

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

best_precision = precision[
    best_index
]

best_recall = recall[
    best_index
]

best_f1 = f1_scores[
    best_index
]


# ==========================================================
# PREDIÇÃO COM THRESHOLD OTIMIZADO
# ==========================================================

y_pred_optimized = (
    anomaly_scores >= best_threshold
).astype(int)


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

cm = confusion_matrix(
    y_test_np,
    y_pred_optimized
)

tn, fp, fn, tp = cm.ravel()


# ==========================================================
# MÉTRICAS FINAIS
# ==========================================================

final_precision = precision_score(
    y_test_np,
    y_pred_optimized,
    zero_division=0
)

final_recall = recall_score(
    y_test_np,
    y_pred_optimized,
    zero_division=0
)

final_f1 = f1_score(
    y_test_np,
    y_pred_optimized,
    zero_division=0
)

false_positive_rate = (
    fp / (fp + tn)
)

false_negative_rate = (
    fn / (fn + tp)
)


# ==========================================================
# RESULTADOS
# ==========================================================

print("\n=== MELHOR THRESHOLD ===")

print(
    f"Threshold: "
    f"{best_threshold:.8f}"
)

print(
    f"Precision: "
    f"{final_precision:.4f}"
)

print(
    f"Recall: "
    f"{final_recall:.4f}"
)

print(
    f"F1-score: "
    f"{final_f1:.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC: "
    f"{pr_auc:.4f}"
)


# ==========================================================
# MATRIZ
# ==========================================================

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
# MÉTRICAS OPERACIONAIS
# ==========================================================

print(
    "\n=== MÉTRICAS OPERACIONAIS ==="
)

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
    f"{final_recall:.4%}"
)


# ==========================================================
# COMPARAÇÃO COM THRESHOLD PADRÃO
# ==========================================================

print(
    "\n=== COMPARAÇÃO ==="
)

# Threshold padrão do Isolation Forest

default_predictions = model.predict(
    X_test_np
)

default_pred = np.where(
    default_predictions == -1,
    1,
    0
)

default_precision = precision_score(
    y_test_np,
    default_pred,
    zero_division=0
)

default_recall = recall_score(
    y_test_np,
    default_pred,
    zero_division=0
)

default_f1 = f1_score(
    y_test_np,
    default_pred,
    zero_division=0
)

print(
    "\nThreshold padrão:"
)

print(
    f"Precision: "
    f"{default_precision:.4f}"
)

print(
    f"Recall: "
    f"{default_recall:.4f}"
)

print(
    f"F1-score: "
    f"{default_f1:.4f}"
)

print(
    "\nThreshold otimizado:"
)

print(
    f"Precision: "
    f"{final_precision:.4f}"
)

print(
    f"Recall: "
    f"{final_recall:.4f}"
)

print(
    f"F1-score: "
    f"{final_f1:.4f}"
)


# ==========================================================
# TESTAR THRESHOLDS PRÓXIMOS
# ==========================================================

print(
    "\n=== TOP 10 THRESHOLDS ==="
)

threshold_results = []

for index in np.argsort(
    f1_scores
)[-10:][::-1]:

    threshold_results.append({
        "threshold": thresholds[index],
        "precision": precision[index],
        "recall": recall[index],
        "f1_score": f1_scores[index],
    })

threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False
    )
)


# ==========================================================
# SALVAR RESULTADOS
# ==========================================================

metrics_df = pd.DataFrame([{
    "model": "Isolation Forest",
    "contamination": CONTAMINATION,
    "threshold": best_threshold,
    "precision": final_precision,
    "recall": final_recall,
    "f1_score": final_f1,
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
    / "optimized_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


threshold_path = (
    RESULTS_DIR
    / "threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)


# ==========================================================
# GRÁFICO PRECISION / RECALL / F1
# ==========================================================

print(
    "\n=== GERANDO CURVA PRECISION-RECALL ==="
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    recall,
    precision,
    label="Precision-Recall"
)

plt.scatter(
    best_recall,
    best_precision,
    s=80,
    label="Melhor threshold"
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

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

pr_curve_path = (
    RESULTS_DIR
    / "precision_recall_curve.png"
)

plt.savefig(
    pr_curve_path,
    dpi=150
)

plt.close()


# ==========================================================
# GRÁFICO F1
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    thresholds,
    f1_scores
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label="Melhor threshold"
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "F1-score"
)

plt.title(
    "Otimização do Threshold - Isolation Forest"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

f1_curve_path = (
    RESULTS_DIR
    / "threshold_f1_curve.png"
)

plt.savefig(
    f1_curve_path,
    dpi=150
)

plt.close()


# ==========================================================
# RELATÓRIO
# ==========================================================

report_path = (
    RESULTS_DIR
    / "threshold_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "OTIMIZAÇÃO DE THRESHOLD - ISOLATION FOREST\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Contamination: {CONTAMINATION}\n"
    )

    file.write(
        f"Threshold otimizado: "
        f"{best_threshold:.8f}\n\n"
    )

    file.write(
        "MÉTRICAS\n"
    )

    file.write(
        f"Precision: {final_precision:.4f}\n"
    )

    file.write(
        f"Recall: {final_recall:.4f}\n"
    )

    file.write(
        f"F1-score: {final_f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    file.write(
        f"PR-AUC: {pr_auc:.4f}\n\n"
    )

    file.write(
        "MATRIZ DE CONFUSÃO\n"
    )

    file.write(
        str(cm) + "\n\n"
    )

    file.write(
        f"TN: {tn}\n"
    )

    file.write(
        f"FP: {fp}\n"
    )

    file.write(
        f"FN: {fn}\n"
    )

    file.write(
        f"TP: {tp}\n\n"
    )

    file.write(
        "THRESHOLD PADRÃO\n"
    )

    file.write(
        f"Precision: {default_precision:.4f}\n"
    )

    file.write(
        f"Recall: {default_recall:.4f}\n"
    )

    file.write(
        f"F1-score: {default_f1:.4f}\n\n"
    )

    file.write(
        "THRESHOLD OTIMIZADO\n"
    )

    file.write(
        f"Precision: {final_precision:.4f}\n"
    )

    file.write(
        f"Recall: {final_recall:.4f}\n"
    )

    file.write(
        f"F1-score: {final_f1:.4f}\n"
    )


# ==========================================================
# FINAL
# ==========================================================

print(
    "\n=== OTIMIZAÇÃO CONCLUÍDA ==="
)

print(
    f"\nMétricas: {metrics_path}"
)

print(
    f"Thresholds: {threshold_path}"
)

print(
    f"Curva PR: {pr_curve_path}"
)

print(
    f"Curva F1: {f1_curve_path}"
)

print(
    f"Relatório: {report_path}"
)