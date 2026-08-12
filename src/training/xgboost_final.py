from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

# Arquivo:
# fraud-detection/src/training/xgboost_final.py
#
# parent        -> training
# parent.parent -> src
# parent.parent.parent -> fraud-detection

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ==========================================================
# CAMINHO DO DATASET
# ==========================================================

DATA_PROCESSED = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_processado.csv"
)


# ==========================================================
# CAMINHO DOS RESULTADOS
# ==========================================================

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "xgboost_final"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# CAMINHO DOS ARTIFACTS
# ==========================================================

ARTIFACTS_DIR = (
    BASE_DIR
    / "artifacts"
    / "xgboost"
)

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================

RANDOM_STATE = 42

TARGET = "Class"


# ==========================================================
# CONFIGURAÇÃO FINAL DO XGBOOST
# ==========================================================

XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 599.0242,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ==========================================================
# THRESHOLD CONGELADO
# ==========================================================

FROZEN_THRESHOLD = 0.66275984


# ==========================================================
# FUNÇÃO DE MÉTRICAS
# ==========================================================

def calcular_metricas(
    y_true,
    y_pred,
    y_scores
):
    """
    Calcula métricas de classificação
    e métricas operacionais para fraude.
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
        y_scores
    )

    pr_auc = average_precision_score(
        y_true,
        y_scores
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    fraud_detection_rate = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    return {
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "fraud_detection_rate": fraud_detection_rate,
    }


# ==========================================================
# CARREGAMENTO DO DATASET
# ==========================================================

print("=== CARREGANDO DATASET ===")


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
# SEPARANDO FEATURES E TARGET
# ==========================================================

print(
    "\n=== SEPARANDO FEATURES E TARGET ==="
)


if TARGET not in df.columns:
    raise ValueError(
        f"A coluna '{TARGET}' "
        f"não foi encontrada no dataset."
    )


X = df.drop(
    columns=[TARGET]
).copy()


y = df[TARGET].copy()


print(
    f"Features: {X.shape[1]}"
)

print(
    f"Target: {TARGET}"
)


# ==========================================================
# VALIDAÇÃO DAS FEATURES
# ==========================================================

print(
    "\n=== VALIDAÇÃO DAS FEATURES ==="
)


if TARGET in X.columns:
    raise ValueError(
        "ERRO CRÍTICO: "
        "Class entrou nas features!"
    )


print(
    "Class NÃO está nas features."
)


print(
    "\nFeatures utilizadas:"
)


print(
    list(X.columns)
)


# ==========================================================
# DIVISÃO TRAIN / TEMP
# ==========================================================

print(
    "\n=== DIVISÃO TRAIN / TEMP ==="
)


X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )
)


# ==========================================================
# DIVISÃO VALIDATION / TEST
# ==========================================================

X_validation, X_test, y_validation, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
)


print(
    f"Train:      {len(X_train)}"
)

print(
    f"Validation: {len(X_validation)}"
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
    f"Train - fraude: {y_train.sum()}"
)

print(
    f"Validation - fraude: {y_validation.sum()}"
)

print(
    f"Test - fraude: {y_test.sum()}"
)


# ==========================================================
# CONFIGURAÇÃO FINAL
# ==========================================================

print(
    "\n=== CONFIGURAÇÃO FINAL XGBOOST ==="
)


for parameter, value in XGB_PARAMS.items():
    print(
        f"{parameter}: {value}"
    )


print(
    f"threshold: {FROZEN_THRESHOLD}"
)


# ==========================================================
# TREINAMENTO XGBOOST FINAL
# ==========================================================

print(
    "\n=== TREINANDO XGBOOST FINAL ==="
)


model = XGBClassifier(
    **XGB_PARAMS
)


model.fit(
    X_train,
    y_train
)


print(
    "Modelo XGBoost final treinado com sucesso."
)


# ==========================================================
# SALVANDO MODEL.JSON
# ==========================================================

print(
    "\n=== SALVANDO ARTIFACT DO MODELO ==="
)


MODEL_PATH = (
    ARTIFACTS_DIR
    / "model.json"
)


model.save_model(
    MODEL_PATH
)


print(
    f"Modelo salvo em: {MODEL_PATH}"
)


# ==========================================================
# SALVANDO METADATA.JSON
# ==========================================================

print(
    "\n=== SALVANDO METADATA ==="
)


METADATA_PATH = (
    ARTIFACTS_DIR
    / "metadata.json"
)


metadata = {
    "model_type": "XGBClassifier",
    "model_format": "xgboost_json",
    "target": TARGET,
    "features": list(X.columns),
    "threshold": float(
        FROZEN_THRESHOLD
    ),
    "random_state": RANDOM_STATE,
    "parameters": XGB_PARAMS,
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    f"Metadata salvo em: {METADATA_PATH}"
)


# ==========================================================
# VERIFICAÇÃO NA VALIDATION
# ==========================================================

print(
    "\n=== VERIFICAÇÃO NA VALIDATION ==="
)


validation_prob = (
    model.predict_proba(
        X_validation
    )[:, 1]
)


validation_pred = (
    validation_prob >= FROZEN_THRESHOLD
).astype(int)


validation_precision = precision_score(
    y_validation,
    validation_pred,
    zero_division=0
)


validation_recall = recall_score(
    y_validation,
    validation_pred,
    zero_division=0
)


validation_f1 = f1_score(
    y_validation,
    validation_pred,
    zero_division=0
)


validation_roc_auc = roc_auc_score(
    y_validation,
    validation_prob
)


validation_pr_auc = average_precision_score(
    y_validation,
    validation_prob
)


print(
    f"Validation Precision: "
    f"{validation_precision:.4f}"
)


print(
    f"Validation Recall: "
    f"{validation_recall:.4f}"
)


print(
    f"Validation F1: "
    f"{validation_f1:.4f}"
)


print(
    f"Validation ROC-AUC: "
    f"{validation_roc_auc:.4f}"
)


print(
    f"Validation PR-AUC: "
    f"{validation_pr_auc:.4f}"
)


# ==========================================================
# THRESHOLD CONGELADO
# ==========================================================

print(
    "\n=== THRESHOLD CONGELADO ==="
)


print(
    f"Threshold utilizado no Test: "
    f"{FROZEN_THRESHOLD}"
)


print(
    "O threshold NÃO será alterado "
    "durante a avaliação do Test."
)


# ==========================================================
# TEST - UMA ÚNICA VEZ
# ==========================================================

print(
    "\n=== AVALIAÇÃO FINAL NO TEST ==="
)


print(
    "O Test será avaliado UMA ÚNICA VEZ."
)


test_prob = (
    model.predict_proba(
        X_test
    )[:, 1]
)


test_pred = (
    test_prob >= FROZEN_THRESHOLD
).astype(int)


# ==========================================================
# MÉTRICAS FINAIS
# ==========================================================

metrics = calcular_metricas(
    y_test,
    test_pred,
    test_prob
)


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

print(
    "\n=== MATRIZ DE CONFUSÃO ==="
)


print(
    f"True Negatives:  "
    f"{metrics['true_negatives']}"
)


print(
    f"False Positives: "
    f"{metrics['false_positives']}"
)


print(
    f"False Negatives: "
    f"{metrics['false_negatives']}"
)


print(
    f"True Positives:  "
    f"{metrics['true_positives']}"
)


# ==========================================================
# MÉTRICAS
# ==========================================================

print(
    "\n=== MÉTRICAS FINAIS ==="
)


print(
    f"Precision: "
    f"{metrics['precision']:.4f}"
)


print(
    f"Recall:    "
    f"{metrics['recall']:.4f}"
)


print(
    f"F1-score:  "
    f"{metrics['f1_score']:.4f}"
)


print(
    f"ROC-AUC:   "
    f"{metrics['roc_auc']:.4f}"
)


print(
    f"PR-AUC:    "
    f"{metrics['pr_auc']:.4f}"
)


# ==========================================================
# MÉTRICAS OPERACIONAIS
# ==========================================================

print(
    "\n=== MÉTRICAS OPERACIONAIS ==="
)


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


# ==========================================================
# BASELINE
# ==========================================================

print(
    "\n=== BASELINE ==="
)


fraud_rate = y_test.mean()


print(
    f"Taxa real de fraude no Test: "
    f"{fraud_rate:.4%}"
)


print(
    f"Fraudes no Test: "
    f"{y_test.sum()}"
)


print(
    f"Fraudes detectadas: "
    f"{metrics['true_positives']}"
)


print(
    f"Fraudes perdidas: "
    f"{metrics['false_negatives']}"
)


print(
    f"Falsos positivos: "
    f"{metrics['false_positives']}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print(
    "\n=== CLASSIFICATION REPORT ==="
)


report = classification_report(
    y_test,
    test_pred,
    target_names=[
        "Legítima",
        "Fraude"
    ],
    zero_division=0
)


print(
    report
)


# ==========================================================
# SALVANDO MÉTRICAS
# ==========================================================

metrics_output = {
    "model": "XGBoost",
    "threshold": FROZEN_THRESHOLD,
    "scale_pos_weight": XGB_PARAMS[
        "scale_pos_weight"
    ],
    **metrics,
}


metrics_df = pd.DataFrame(
    [metrics_output]
)


metrics_path = (
    RESULTS_DIR
    / "xgboost_final_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


# ==========================================================
# MATRIZ DE CONFUSÃO - GRÁFICO
# ==========================================================

print(
    "\n=== GERANDO MATRIZ DE CONFUSÃO ==="
)


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
    / "xgboost_final_confusion_matrix.png"
)


plt.savefig(
    confusion_path,
    dpi=150
)


plt.close()


# ==========================================================
# CURVA PRECISION-RECALL
# ==========================================================

print(
    "\n=== GERANDO CURVA PRECISION-RECALL ==="
)


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
    / "xgboost_final_precision_recall.png"
)


plt.savefig(
    pr_path,
    dpi=150
)


plt.close()


# ==========================================================
# CURVA ROC
# ==========================================================

print(
    "\n=== GERANDO CURVA ROC ==="
)


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
    / "xgboost_final_roc_curve.png"
)


plt.savefig(
    roc_path,
    dpi=150
)


plt.close()


# ==========================================================
# IMPORTÂNCIA DAS FEATURES
# ==========================================================

print(
    "\n=== GERANDO IMPORTÂNCIA DAS FEATURES ==="
)


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
    / "xgboost_final_feature_importance.csv"
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
    / "xgboost_final_feature_importance.png"
)


plt.savefig(
    importance_plot_path,
    dpi=150
)


plt.close()


# ==========================================================
# RELATÓRIO
# ==========================================================

print(
    "\n=== GERANDO RELATÓRIO ==="
)


report_path = (
    RESULTS_DIR
    / "xgboost_final_evaluation_report.txt"
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
        "RELATÓRIO DE AVALIAÇÃO - XGBOOST FINAL\n"
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
        f"{XGB_PARAMS['scale_pos_weight']:.6f}\n"
    )

    file.write(
        f"Threshold congelado: "
        f"{FROZEN_THRESHOLD:.8f}\n\n"
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
        f"{validation_precision:.6f}\n"
    )

    file.write(
        f"Recall: "
        f"{validation_recall:.6f}\n"
    )

    file.write(
        f"F1-score: "
        f"{validation_f1:.6f}\n\n"
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
        f"TP: {metrics['true_positives']}\n\n"
    )

    file.write(
        "ARTIFACTS\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Metadata: {METADATA_PATH}\n"
    )


# ==========================================================
# FINAL
# ==========================================================

print(
    "\n=== XGBOOST FINAL CONCLUÍDO ==="
)


print(
    "\nArquivos gerados:"
)


print(
    metrics_path
)

print(
    confusion_path
)

print(
    pr_path
)

print(
    roc_path
)

print(
    importance_path
)

print(
    importance_plot_path
)

print(
    report_path
)

print(
    MODEL_PATH
)

print(
    METADATA_PATH
)


print(
    "\n=== RESULTADO FINAL ==="
)


print(
    f"Precision: "
    f"{metrics['precision']:.4f}"
)


print(
    f"Recall:    "
    f"{metrics['recall']:.4f}"
)


print(
    f"F1-score:  "
    f"{metrics['f1_score']:.4f}"
)


print(
    f"ROC-AUC:   "
    f"{metrics['roc_auc']:.4f}"
)


print(
    f"PR-AUC:    "
    f"{metrics['pr_auc']:.4f}"
)


print(
    f"Threshold: "
    f"{FROZEN_THRESHOLD}"
)