from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_processado.csv"
)

RESULTS_DIR = BASE_DIR / "results" / "xgboost"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def calcular_metricas(y_true, y_pred, y_prob):
    """
    Calcula métricas de classificação e métricas específicas
    para detecção de fraude.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

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
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    fraud_detection_rate = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "fraud_detection_rate": fraud_detection_rate,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


# ============================================================
# CARREGANDO DATASET
# ============================================================

print("=== CARREGANDO DATASET ===")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset não encontrado:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")


# ============================================================
# SEPARANDO FEATURES E TARGET
# ============================================================

print("\n=== SEPARANDO FEATURES E TARGET ===")

TARGET = "Class"

if TARGET not in df.columns:
    raise ValueError(
        "A coluna 'Class' não existe no dataset."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features: {X.shape[1]}")
print(f"Target: {TARGET}")


# ============================================================
# VALIDAÇÃO DE SEGURANÇA
# ============================================================

print("\n=== VALIDAÇÃO DAS FEATURES ===")

if TARGET in X.columns:
    raise ValueError(
        "ERRO CRÍTICO: Class entrou nas features!"
    )

print("Class NÃO está nas features.")

print("\nFeatures utilizadas:")

print(list(X.columns))


# ============================================================
# DIVISÃO TRAIN / TEMP
# ============================================================

print("\n=== DIVISÃO TRAIN / TEMP ===")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# DIVISÃO VALIDATION / TEST
# ============================================================

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_validation)}")
print(f"Test:       {len(X_test)}")


# ============================================================
# DISTRIBUIÇÃO DAS CLASSES
# ============================================================

print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")

print(
    f"Train - fraude: {y_train.sum()}"
)

print(
    f"Validation - fraude: {y_validation.sum()}"
)

print(
    f"Test - fraude: {y_test.sum()}"
)


# ============================================================
# SCALE POS WEIGHT
# ============================================================

print("\n=== CALCULANDO SCALE_POS_WEIGHT ===")

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"Negativas: {negative}")
print(f"Positivas: {positive}")
print(
    f"scale_pos_weight: {scale_pos_weight:.4f}"
)


# ============================================================
# TREINAMENTO XGBOOST
# ============================================================

print("\n=== TREINANDO XGBOOST ===")

model = XGBClassifier(
    objective="binary:logistic",

    n_estimators=400,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=3,

    gamma=0,

    reg_alpha=0.0,

    reg_lambda=1.0,

    scale_pos_weight=scale_pos_weight,

    eval_metric="aucpr",

    tree_method="hist",

    random_state=RANDOM_STATE,

    n_jobs=-1,
)

model.fit(
    X_train,
    y_train
)

print("Modelo XGBoost treinado com sucesso.")


# ============================================================
# PROBABILIDADES - VALIDATION
# ============================================================

print("\n=== PREDIÇÕES NA VALIDATION ===")

validation_prob = model.predict_proba(
    X_validation
)[:, 1]


validation_roc_auc = roc_auc_score(
    y_validation,
    validation_prob
)

validation_pr_auc = average_precision_score(
    y_validation,
    validation_prob
)

print(
    f"Validation ROC-AUC: {validation_roc_auc:.4f}"
)

print(
    f"Validation PR-AUC:  {validation_pr_auc:.4f}"
)


# ============================================================
# OTIMIZAÇÃO DO THRESHOLD
# ============================================================

print("\n=== OTIMIZANDO THRESHOLD ===")

precision_curve, recall_curve, thresholds = (
    precision_recall_curve(
        y_validation,
        validation_prob
    )
)

# precision_recall_curve retorna uma posição
# a mais para precision/recall.
f1_values = (
    2
    * precision_curve[:-1]
    * recall_curve[:-1]
    / (
        precision_curve[:-1]
        + recall_curve[:-1]
        + 1e-12
    )
)

best_index = np.argmax(f1_values)

best_threshold = thresholds[best_index]

best_precision = precision_curve[
    best_index
]

best_recall = recall_curve[
    best_index
]

best_f1 = f1_values[
    best_index
]


print(
    f"Melhor threshold: {best_threshold:.8f}"
)

print(
    f"Precision: {best_precision:.4f}"
)

print(
    f"Recall:    {best_recall:.4f}"
)

print(
    f"F1-score:  {best_f1:.4f}"
)


# ============================================================
# THRESHOLD CONGELADO
# ============================================================

print("\n=== THRESHOLD CONGELADO ===")

print(
    f"Threshold utilizado no Test: "
    f"{best_threshold:.8f}"
)


# ============================================================
# TEST - UMA ÚNICA VEZ
# ============================================================

print("\n=== AVALIAÇÃO FINAL NO TEST ===")

test_prob = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# MÉTRICAS
# ============================================================

metrics = calcular_metricas(
    y_test,
    test_pred,
    test_prob
)


print("\n=== MATRIZ DE CONFUSÃO ===")

print(
    confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    )
)

print(
    f"\nTrue Negatives:  {metrics['true_negatives']}"
)

print(
    f"False Positives: {metrics['false_positives']}"
)

print(
    f"False Negatives: {metrics['false_negatives']}"
)

print(
    f"True Positives:  {metrics['true_positives']}"
)


print("\n=== MÉTRICAS FINAIS ===")

print(
    f"Precision: {metrics['precision']:.4f}"
)

print(
    f"Recall:    {metrics['recall']:.4f}"
)

print(
    f"F1-score:  {metrics['f1_score']:.4f}"
)

print(
    f"ROC-AUC:   {metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC:    {metrics['pr_auc']:.4f}"
)


print("\n=== MÉTRICAS OPERACIONAIS ===")

print(
    f"False Positive Rate: "
    f"{metrics['false_positive_rate']:.4%}"
)

print(
    f"False Negative Rate: "
    f"{metrics['false_negative_rate']:.4%}"
)

print(
    f"Fraud Detection Rate: "
    f"{metrics['fraud_detection_rate']:.4%}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n=== CLASSIFICATION REPORT ===")

report = classification_report(
    y_test,
    test_pred,
    target_names=[
        "Legítima",
        "Fraude"
    ],
    zero_division=0
)

print(report)


# ============================================================
# SALVANDO MÉTRICAS
# ============================================================

metrics_output = {
    "model": "XGBoost",
    "threshold": best_threshold,
    "scale_pos_weight": scale_pos_weight,
    **metrics
}

metrics_df = pd.DataFrame(
    [metrics_output]
)

metrics_path = (
    RESULTS_DIR
    / "xgboost_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

print("\n=== GERANDO MATRIZ DE CONFUSÃO ===")

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1]
)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "XGBoost - Matriz de Confusão"
)

plt.xlabel(
    "Predição"
)

plt.ylabel(
    "Real"
)

plt.xticks(
    [0, 1],
    ["Legítima", "Fraude"]
)

plt.yticks(
    [0, 1],
    ["Legítima", "Fraude"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

confusion_path = (
    RESULTS_DIR
    / "xgboost_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=150
)

plt.close()


# ============================================================
# CURVA PRECISION-RECALL
# ============================================================

print("\n=== GERANDO CURVA PRECISION-RECALL ===")

test_precision, test_recall, _ = (
    precision_recall_curve(
        y_test,
        test_prob
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    test_recall,
    test_precision
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    f"XGBoost - Precision-Recall "
    f"(PR-AUC={metrics['pr_auc']:.4f})"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

pr_path = (
    RESULTS_DIR
    / "xgboost_precision_recall.png"
)

plt.savefig(
    pr_path,
    dpi=150
)

plt.close()


# ============================================================
# CURVA ROC
# ============================================================

print("\n=== GERANDO CURVA ROC ===")

fpr_curve, tpr_curve, _ = roc_curve(
    y_test,
    test_prob
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr_curve,
    tpr_curve
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    f"XGBoost - ROC "
    f"(ROC-AUC={metrics['roc_auc']:.4f})"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

roc_path = (
    RESULTS_DIR
    / "xgboost_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()


# ============================================================
# CURVA F1 / THRESHOLD
# ============================================================

print("\n=== GERANDO CURVA F1 ===")

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    thresholds,
    f1_values
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Threshold = {best_threshold:.4f}"
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "F1-score"
)

plt.title(
    "XGBoost - F1 por Threshold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

threshold_path = (
    RESULTS_DIR
    / "xgboost_threshold_f1.png"
)

plt.savefig(
    threshold_path,
    dpi=150
)

plt.close()


# ============================================================
# IMPORTÂNCIA DAS FEATURES
# ============================================================

print("\n=== GERANDO IMPORTÂNCIA DAS FEATURES ===")

feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False
)

importance_path = (
    RESULTS_DIR
    / "xgboost_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)


plt.figure(
    figsize=(10, 8)
)

top_features = feature_importance.head(15)

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost - Top 15 Features"
)

plt.tight_layout()

importance_plot_path = (
    RESULTS_DIR
    / "xgboost_feature_importance.png"
)

plt.savefig(
    importance_plot_path,
    dpi=150
)

plt.close()


# ============================================================
# RELATÓRIO
# ============================================================

report_path = (
    RESULTS_DIR
    / "xgboost_evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "==================================================\n"
    )

    file.write(
        "RELATÓRIO DE AVALIAÇÃO - XGBOOST\n"
    )

    file.write(
        "==================================================\n\n"
    )

    file.write(
        f"Registros totais: {len(df)}\n"
    )

    file.write(
        f"Features: {X.shape[1]}\n"
    )

    file.write(
        f"Train: {len(X_train)}\n"
    )

    file.write(
        f"Validation: {len(X_validation)}\n"
    )

    file.write(
        f"Test: {len(X_test)}\n\n"
    )

    file.write(
        f"Scale pos weight: "
        f"{scale_pos_weight:.6f}\n"
    )

    file.write(
        f"Threshold otimizado: "
        f"{best_threshold:.8f}\n\n"
    )

    file.write(
        "VALIDATION\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{validation_roc_auc:.6f}\n"
    )

    file.write(
        f"PR-AUC: "
        f"{validation_pr_auc:.6f}\n"
    )

    file.write(
        f"Precision: "
        f"{best_precision:.6f}\n"
    )

    file.write(
        f"Recall: "
        f"{best_recall:.6f}\n"
    )

    file.write(
        f"F1-score: "
        f"{best_f1:.6f}\n\n"
    )

    file.write(
        "TEST\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.6f}\n"
    )

    file.write(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.6f}\n"
    )

    file.write(
        f"Precision: "
        f"{metrics['precision']:.6f}\n"
    )

    file.write(
        f"Recall: "
        f"{metrics['recall']:.6f}\n"
    )

    file.write(
        f"F1-score: "
        f"{metrics['f1_score']:.6f}\n"
    )

    file.write(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.6f}\n"
    )

    file.write(
        f"False Negative Rate: "
        f"{metrics['false_negative_rate']:.6f}\n"
    )

    file.write(
        f"Fraud Detection Rate: "
        f"{metrics['fraud_detection_rate']:.6f}\n\n"
    )

    file.write(
        "MATRIZ DE CONFUSÃO\n"
    )

    file.write(
        f"TN: {metrics['true_negatives']}\n"
    )

    file.write(
        f"FP: {metrics['false_positives']}\n"
    )

    file.write(
        f"FN: {metrics['false_negatives']}\n"
    )

    file.write(
        f"TP: {metrics['true_positives']}\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n=== XGBOOST CONCLUÍDO ===")

print("\nArquivos gerados:")

print(metrics_path)
print(confusion_path)
print(pr_path)
print(roc_path)
print(threshold_path)
print(importance_path)
print(importance_plot_path)
print(report_path)
