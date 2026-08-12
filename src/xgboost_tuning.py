from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_processado.csv"

RESULTS_DIR = BASE_DIR / "results" / "xgboost_tuning"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42


# ==========================================================
# HIPERPARÂMETROS
# ==========================================================

PARAM_GRID = [
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.10,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.10,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.10,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
]


# ==========================================================
# FUNÇÃO PARA ENCONTRAR MELHOR THRESHOLD
# ==========================================================

def optimize_threshold(y_true, probabilities):
    """
    Encontra o threshold que maximiza o F1-score.

    O threshold é otimizado SOMENTE na Validation.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = np.argmax(f1_scores)

    return (
        thresholds[best_index],
        precision[best_index],
        recall[best_index],
        f1_scores[best_index],
    )


# ==========================================================
# CARREGAMENTO
# ==========================================================

print("=== CARREGANDO DATASET ===")

df = pd.read_csv(DATA_PATH)

print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")


# ==========================================================
# FEATURES / TARGET
# ==========================================================

print("\n=== SEPARANDO FEATURES E TARGET ===")

TARGET = "Class"

if TARGET not in df.columns:
    raise ValueError("Coluna Class não encontrada no dataset.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features: {X.shape[1]}")
print(f"Target: {TARGET}")


# ==========================================================
# VALIDAÇÃO
# ==========================================================

print("\n=== VALIDAÇÃO DAS FEATURES ===")

if TARGET in X.columns:
    raise ValueError(
        "ERRO CRÍTICO: Class está presente nas features."
    )

print("Class NÃO está nas features.")


# ==========================================================
# TRAIN / VALIDATION / TEST
# ==========================================================

print("\n=== DIVISÃO TRAIN / TEMP ===")

from sklearn.model_selection import train_test_split


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)


X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)


print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_validation)}")
print(f"Test:       {len(X_test)}")


# ==========================================================
# DISTRIBUIÇÃO
# ==========================================================

print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")

print(f"Train - fraude: {int(y_train.sum())}")
print(f"Validation - fraude: {int(y_validation.sum())}")
print(f"Test - fraude: {int(y_test.sum())}")


# ==========================================================
# SCALE POS WEIGHT
# ==========================================================

print("\n=== CALCULANDO SCALE_POS_WEIGHT ===")

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / positive

print(f"Negativas: {negative}")
print(f"Positivas: {positive}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# ==========================================================
# TUNING
# ==========================================================

print("\n" + "=" * 70)
print("=== XGBOOST HYPERPARAMETER TUNING ===")
print("=" * 70)


results = []


for index, params in enumerate(PARAM_GRID, start=1):

    print("\n" + "-" * 70)
    print(f"CONFIGURAÇÃO {index}/{len(PARAM_GRID)}")
    print("-" * 70)

    print(params)

    model = XGBClassifier(
        objective="binary:logistic",

        eval_metric="aucpr",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",

        **params,
    )

    print("\nTreinando...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # ------------------------------------------------------
    # MÉTRICAS DE RANKING
    # ------------------------------------------------------

    roc_auc = roc_auc_score(
        y_validation,
        probabilities
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities
    )

    # ------------------------------------------------------
    # THRESHOLD
    # ------------------------------------------------------

    (
        threshold,
        precision,
        recall,
        f1,
    ) = optimize_threshold(
        y_validation,
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ------------------------------------------------------
    # MATRIZ
    # ------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions
    ).ravel()

    fpr = fp / (fp + tn)

    fnr = fn / (fn + tp)

    fraud_detection_rate = tp / (tp + fn)

    # ------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------

    result = {
        "configuration": index,

        **params,

        "scale_pos_weight": scale_pos_weight,

        "threshold": threshold,

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

    results.append(result)

    print("\nValidation:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


# ==========================================================
# RESULTADOS DO TUNING
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["pr_auc", "f1_score"],
    ascending=False,
).reset_index(drop=True)


results_path = (
    RESULTS_DIR /
    "tuning_validation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 70)
print("=== RESULTADOS DO TUNING ===")
print("=" * 70)

print(
    results_df[
        [
            "configuration",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(index=False)
)


# ==========================================================
# MELHOR CONFIGURAÇÃO
# ==========================================================

best = results_df.iloc[0]


print("\n" + "=" * 70)
print("=== MELHOR CONFIGURAÇÃO ===")
print("=" * 70)

print(f"Configuração: {int(best['configuration'])}")

print(
    f"max_depth: {int(best['max_depth'])}"
)

print(
    f"learning_rate: {best['learning_rate']}"
)

print(
    f"n_estimators: {int(best['n_estimators'])}"
)

print(
    f"min_child_weight: "
    f"{int(best['min_child_weight'])}"
)

print(
    f"subsample: {best['subsample']}"
)

print(
    f"colsample_bytree: "
    f"{best['colsample_bytree']}"
)

print(
    f"Threshold: {best['threshold']:.8f}"
)

print(
    f"Validation PR-AUC: "
    f"{best['pr_auc']:.4f}"
)

print(
    f"Validation F1: "
    f"{best['f1_score']:.4f}"
)


# ==========================================================
# CONGELANDO CONFIGURAÇÃO
# ==========================================================

best_params = {
    "max_depth": int(best["max_depth"]),
    "learning_rate": float(best["learning_rate"]),
    "n_estimators": int(best["n_estimators"]),
    "min_child_weight": int(best["min_child_weight"]),
    "subsample": float(best["subsample"]),
    "colsample_bytree": float(
        best["colsample_bytree"]
    ),
}

frozen_threshold = float(
    best["threshold"]
)


# ==========================================================
# TREINAMENTO FINAL
# ==========================================================

print("\n" + "=" * 70)
print("=== TREINAMENTO FINAL ===")
print("=" * 70)

print(
    "Treinando modelo final SOMENTE com Train."
)

final_model = XGBClassifier(
    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    tree_method="hist",

    **best_params,
)


final_model.fit(
    X_train,
    y_train,
    verbose=False,
)


print("Modelo final treinado.")

print(
    f"Threshold congelado: "
    f"{frozen_threshold:.8f}"
)


# ==========================================================
# TEST
# ==========================================================

print("\n" + "=" * 70)
print("=== AVALIAÇÃO FINAL NO TEST ===")
print("=" * 70)

print(
    "O Test será avaliado UMA ÚNICA VEZ."
)


test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]


test_predictions = (
    test_probabilities >= frozen_threshold
).astype(int)


# ==========================================================
# MÉTRICAS
# ==========================================================

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)


# ==========================================================
# MATRIZ
# ==========================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions
).ravel()


fpr = fp / (fp + tn)

fnr = fn / (fn + tp)

fraud_detection_rate = tp / (tp + fn)


print("\n=== MATRIZ DE CONFUSÃO ===")

print(
    f"True Negatives:  {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives:  {tp}"
)


print("\n=== MÉTRICAS FINAIS ===")

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall:    {test_recall:.4f}"
)

print(
    f"F1-score:  {test_f1:.4f}"
)

print(
    f"ROC-AUC:   {test_roc_auc:.4f}"
)

print(
    f"PR-AUC:    {test_pr_auc:.4f}"
)


print("\n=== MÉTRICAS OPERACIONAIS ===")

print(
    f"False Positive Rate: "
    f"{fpr:.4%}"
)

print(
    f"False Negative Rate: "
    f"{fnr:.4%}"
)

print(
    f"Fraud Detection Rate: "
    f"{fraud_detection_rate:.4%}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n=== CLASSIFICATION REPORT ===")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Legítima",
            "Fraude"
        ],
        zero_division=0,
    )
)


# ==========================================================
# SALVAR MÉTRICAS
# ==========================================================

final_metrics = pd.DataFrame(
    [
        {
            "model": "XGBoost Tuned",

            "threshold": frozen_threshold,

            "precision": test_precision,

            "recall": test_recall,

            "f1_score": test_f1,

            "roc_auc": test_roc_auc,

            "pr_auc": test_pr_auc,

            "false_positive_rate": fpr,

            "false_negative_rate": fnr,

            "fraud_detection_rate": fraud_detection_rate,

            "true_negatives": tn,

            "false_positives": fp,

            "false_negatives": fn,

            "true_positives": tp,

            "validation_pr_auc": best["pr_auc"],

            "validation_f1": best["f1_score"],

            "validation_roc_auc": best["roc_auc"],
        }
    ]
)


metrics_path = (
    RESULTS_DIR /
    "xgboost_tuned_metrics.csv"
)

final_metrics.to_csv(
    metrics_path,
    index=False
)


# ==========================================================
# CURVA PRECISION-RECALL
# ==========================================================

print("\n=== GERANDO CURVA PRECISION-RECALL ===")

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_test,
        test_probabilities
    )
)


plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title(
    "XGBoost Tuned - Precision-Recall"
)

plt.grid(True)

plt.tight_layout()

pr_path = (
    RESULTS_DIR /
    "xgboost_tuned_precision_recall.png"
)

plt.savefig(pr_path)

plt.close()


# ==========================================================
# CURVA F1
# ==========================================================

print("\n=== GERANDO CURVA F1 ===")

thresholds = np.linspace(
    0.0,
    1.0,
    500,
)

f1_values = []


for threshold in thresholds:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    f1_values.append(
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )


plt.figure(figsize=(8, 6))

plt.plot(
    thresholds,
    f1_values,
)

plt.axvline(
    frozen_threshold,
    linestyle="--",
    label="Threshold congelado",
)

plt.xlabel("Threshold")

plt.ylabel("F1-score")

plt.title(
    "XGBoost Tuned - Threshold vs F1"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

f1_path = (
    RESULTS_DIR /
    "xgboost_tuned_threshold_f1.png"
)

plt.savefig(f1_path)

plt.close()


# ==========================================================
# IMPORTÂNCIA DAS FEATURES
# ==========================================================

print("\n=== GERANDO IMPORTÂNCIA DAS FEATURES ===")

importance_df = pd.DataFrame(
    {
        "feature": X_train.columns,

        "importance": final_model.feature_importances_,
    }
).sort_values(
    by="importance",
    ascending=False,
)


importance_csv = (
    RESULTS_DIR /
    "xgboost_tuned_feature_importance.csv"
)

importance_df.to_csv(
    importance_csv,
    index=False
)


plt.figure(figsize=(10, 8))

top_features = importance_df.head(15)

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1],
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "XGBoost Tuned - Top 15 Features"
)

plt.tight_layout()

importance_png = (
    RESULTS_DIR /
    "xgboost_tuned_feature_importance.png"
)

plt.savefig(importance_png)

plt.close()


# ==========================================================
# RELATÓRIO
# ==========================================================

report_path = (
    RESULTS_DIR /
    "xgboost_tuned_evaluation_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "XGBOOST TUNING - RELATÓRIO FINAL\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "CONFIGURAÇÃO SELECIONADA\n"
    )

    file.write(
        f"max_depth: {best_params['max_depth']}\n"
    )

    file.write(
        f"learning_rate: "
        f"{best_params['learning_rate']}\n"
    )

    file.write(
        f"n_estimators: "
        f"{best_params['n_estimators']}\n"
    )

    file.write(
        f"min_child_weight: "
        f"{best_params['min_child_weight']}\n"
    )

    file.write(
        f"subsample: "
        f"{best_params['subsample']}\n"
    )

    file.write(
        f"colsample_bytree: "
        f"{best_params['colsample_bytree']}\n"
    )

    file.write(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}\n"
    )

    file.write(
        f"Threshold congelado: "
        f"{frozen_threshold:.8f}\n\n"
    )

    file.write(
        "VALIDATION\n"
    )

    file.write(
        f"PR-AUC: {best['pr_auc']:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {best['roc_auc']:.4f}\n"
    )

    file.write(
        f"Precision: {best['precision']:.4f}\n"
    )

    file.write(
        f"Recall: {best['recall']:.4f}\n"
    )

    file.write(
        f"F1-score: {best['f1_score']:.4f}\n\n"
    )

    file.write(
        "TEST\n"
    )

    file.write(
        f"PR-AUC: {test_pr_auc:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {test_roc_auc:.4f}\n"
    )

    file.write(
        f"Precision: {test_precision:.4f}\n"
    )

    file.write(
        f"Recall: {test_recall:.4f}\n"
    )

    file.write(
        f"F1-score: {test_f1:.4f}\n\n"
    )

    file.write(
        "MATRIZ DE CONFUSÃO\n"
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
        "MÉTRICAS OPERACIONAIS\n"
    )

    file.write(
        f"FPR: {fpr:.4%}\n"
    )

    file.write(
        f"FNR: {fnr:.4%}\n"
    )

    file.write(
        f"Fraud Detection Rate: "
        f"{fraud_detection_rate:.4%}\n"
    )


# ==========================================================
# FINAL
# ==========================================================

print("\n" + "=" * 70)
print("=== XGBOOST TUNING CONCLUÍDO ===")
print("=" * 70)

print("\nArquivos gerados:")

print(results_path)

print(metrics_path)

print(pr_path)

print(f1_path)

print(importance_csv)

print(importance_png)

print(report_path)