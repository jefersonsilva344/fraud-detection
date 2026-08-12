from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "final_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# RESULTADOS FINAIS DO TEST
# ==========================================================

# Os valores abaixo são exclusivamente os resultados finais
# obtidos no conjunto TEST.
#
# Nenhum modelo é treinado novamente neste script.

RESULTADOS = [
    {
        "model": "Isolation Forest",
        "threshold": 0.64570164,
        "precision": 0.2216,
        "recall": 0.2606,
        "f1_score": 0.2395,
        "roc_auc": 0.9407,
        "pr_auc": 0.1083,
        "false_positive_rate": 0.1530 / 100,
        "false_negative_rate": 73.9437 / 100,
        "fraud_detection_rate": 26.0563 / 100,
        "true_negatives": 84846,
        "false_positives": 130,
        "false_negatives": 105,
        "true_positives": 37,
    },
    {
        "model": "LOF",
        "threshold": 0.0100,
        "precision": 0.001115,
        "recall": 0.007042,
        "f1_score": 0.001925,
        "roc_auc": 0.524619,
        "pr_auc": 0.002212,
        "false_positive_rate": 1.0544 / 100,
        "false_negative_rate": 99.2958 / 100,
        "fraud_detection_rate": 0.7042 / 100,
        "true_negatives": 84080,
        "false_positives": 896,
        "false_negatives": 141,
        "true_positives": 1,
    },
    {
        "model": "XGBoost",
        "threshold": 0.96644199,
        "precision": 0.9483,
        "recall": 0.7746,
        "f1_score": 0.8527,
        "roc_auc": 0.9651,
        "pr_auc": 0.8148,
        "false_positive_rate": 0.0071 / 100,
        "false_negative_rate": 22.5352 / 100,
        "fraud_detection_rate": 77.4648 / 100,
        "true_negatives": 42485,
        "false_positives": 3,
        "false_negatives": 16,
        "true_positives": 55,
    },
    {
        "model": "XGBoost Tuned",
        "threshold": 0.66275984,
        "precision": 0.8906,
        "recall": 0.8028,
        "f1_score": 0.8444,
        "roc_auc": 0.9628,
        "pr_auc": 0.8202,
        "false_positive_rate": 0.0165 / 100,
        "false_negative_rate": 19.7183 / 100,
        "fraud_detection_rate": 80.2817 / 100,
        "true_negatives": 42481,
        "false_positives": 7,
        "false_negatives": 14,
        "true_positives": 57,
    },
]


# ==========================================================
# DATAFRAME
# ==========================================================

df = pd.DataFrame(RESULTADOS)


# ==========================================================
# RANKING
# ==========================================================

# O ranking principal considera:
#
# 1. PR-AUC
# 2. F1-score
# 3. Recall
#
# PR-AUC recebe prioridade porque estamos trabalhando
# com um problema extremamente desbalanceado.

ranking = df.sort_values(
    by=["pr_auc", "f1_score", "recall"],
    ascending=False,
).reset_index(drop=True)

ranking.insert(0, "ranking", range(1, len(ranking) + 1))


# ==========================================================
# EXIBIÇÃO
# ==========================================================

print()
print("==========================================================")
print("              COMPARAÇÃO FINAL DOS MODELOS")
print("==========================================================")

print()
print("Todos os resultados abaixo pertencem EXCLUSIVAMENTE")
print("ao conjunto TEST.")
print()

print(
    ranking[
        [
            "ranking",
            "model",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "false_positive_rate",
            "false_negative_rate",
            "fraud_detection_rate",
        ]
    ].to_string(index=False)
)


# ==========================================================
# MELHOR MODELO
# ==========================================================

melhor = ranking.iloc[0]

print()
print("==========================================================")
print("                    MELHOR MODELO")
print("==========================================================")

print(f"Modelo:              {melhor['model']}")
print(f"Threshold:           {melhor['threshold']:.8f}")
print(f"Precision:           {melhor['precision']:.4f}")
print(f"Recall:              {melhor['recall']:.4f}")
print(f"F1-score:            {melhor['f1_score']:.4f}")
print(f"ROC-AUC:             {melhor['roc_auc']:.4f}")
print(f"PR-AUC:              {melhor['pr_auc']:.4f}")
print(
    f"False Positive Rate: "
    f"{melhor['false_positive_rate'] * 100:.4f}%"
)
print(
    f"False Negative Rate: "
    f"{melhor['false_negative_rate'] * 100:.4f}%"
)
print(
    f"Fraud Detection Rate: "
    f"{melhor['fraud_detection_rate'] * 100:.4f}%"
)


# ==========================================================
# MATRIZ OPERACIONAL
# ==========================================================

print()
print("==========================================================")
print("                 RESULTADO OPERACIONAL")
print("==========================================================")

for _, row in ranking.iterrows():

    print()
    print(f"{row['model']}")

    print(
        f"  TN: {int(row['true_negatives'])}"
        f" | FP: {int(row['false_positives'])}"
    )

    print(
        f"  FN: {int(row['false_negatives'])}"
        f" | TP: {int(row['true_positives'])}"
    )


# ==========================================================
# DIFERENÇA XGBOOST TUNED VS ISOLATION FOREST
# ==========================================================

isolation = df[df["model"] == "Isolation Forest"].iloc[0]
xgb_tuned = df[df["model"] == "XGBoost Tuned"].iloc[0]

print()
print("==========================================================")
print("       XGBOOST TUNED vs ISOLATION FOREST")
print("==========================================================")

print(
    f"PR-AUC: "
    f"{isolation['pr_auc']:.4f} -> {xgb_tuned['pr_auc']:.4f}"
)

print(
    f"Recall: "
    f"{isolation['recall']:.4f} -> {xgb_tuned['recall']:.4f}"
)

print(
    f"F1-score: "
    f"{isolation['f1_score']:.4f} -> {xgb_tuned['f1_score']:.4f}"
)

print(
    f"False Positive Rate: "
    f"{isolation['false_positive_rate'] * 100:.4f}% -> "
    f"{xgb_tuned['false_positive_rate'] * 100:.4f}%"
)

print(
    f"Fraud Detection Rate: "
    f"{isolation['fraud_detection_rate'] * 100:.4f}% -> "
    f"{xgb_tuned['fraud_detection_rate'] * 100:.4f}%"
)


# ==========================================================
# XGBOOST ORIGINAL VS TUNED
# ==========================================================

xgb = df[df["model"] == "XGBoost"].iloc[0]

print()
print("==========================================================")
print("             XGBOOST ORIGINAL vs TUNED")
print("==========================================================")

print(
    f"Precision: "
    f"{xgb['precision']:.4f} -> {xgb_tuned['precision']:.4f}"
)

print(
    f"Recall: "
    f"{xgb['recall']:.4f} -> {xgb_tuned['recall']:.4f}"
)

print(
    f"F1-score: "
    f"{xgb['f1_score']:.4f} -> {xgb_tuned['f1_score']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{xgb['roc_auc']:.4f} -> {xgb_tuned['roc_auc']:.4f}"
)

print(
    f"PR-AUC: "
    f"{xgb['pr_auc']:.4f} -> {xgb_tuned['pr_auc']:.4f}"
)

print(
    f"False Positive Rate: "
    f"{xgb['false_positive_rate'] * 100:.4f}% -> "
    f"{xgb_tuned['false_positive_rate'] * 100:.4f}%"
)

print(
    f"Fraud Detection Rate: "
    f"{xgb['fraud_detection_rate'] * 100:.4f}% -> "
    f"{xgb_tuned['fraud_detection_rate'] * 100:.4f}%"
)


# ==========================================================
# SALVAR CSV COMPLETO
# ==========================================================

csv_path = OUTPUT_DIR / "final_model_comparison.csv"

ranking.to_csv(
    csv_path,
    index=False,
)

print()
print(f"CSV salvo em:")
print(csv_path)


# ==========================================================
# GRÁFICO 1 — PR-AUC
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ranking["model"],
    ranking["pr_auc"],
)

plt.ylabel("PR-AUC")
plt.title("Comparação Final — PR-AUC")
plt.xticks(rotation=15)
plt.tight_layout()

pr_auc_path = OUTPUT_DIR / "final_pr_auc_comparison.png"

plt.savefig(pr_auc_path, dpi=150)
plt.close()


# ==========================================================
# GRÁFICO 2 — F1
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ranking["model"],
    ranking["f1_score"],
)

plt.ylabel("F1-score")
plt.title("Comparação Final — F1-score")
plt.xticks(rotation=15)
plt.tight_layout()

f1_path = OUTPUT_DIR / "final_f1_comparison.png"

plt.savefig(f1_path, dpi=150)
plt.close()


# ==========================================================
# GRÁFICO 3 — RECALL
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ranking["model"],
    ranking["recall"],
)

plt.ylabel("Recall")
plt.title("Comparação Final — Fraud Detection Rate")
plt.xticks(rotation=15)
plt.tight_layout()

recall_path = OUTPUT_DIR / "final_recall_comparison.png"

plt.savefig(recall_path, dpi=150)
plt.close()


# ==========================================================
# GRÁFICO 4 — FALSE POSITIVE RATE
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ranking["model"],
    ranking["false_positive_rate"] * 100,
)

plt.ylabel("False Positive Rate (%)")
plt.title("Comparação Final — False Positive Rate")
plt.xticks(rotation=15)
plt.tight_layout()

fpr_path = OUTPUT_DIR / "final_fpr_comparison.png"

plt.savefig(fpr_path, dpi=150)
plt.close()


# ==========================================================
# RELATÓRIO TXT
# ==========================================================

report_path = OUTPUT_DIR / "final_comparison_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("=" * 70 + "\n")
    file.write("RELATÓRIO FINAL — DETECÇÃO DE FRAUDES\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        "Os resultados apresentados neste relatório pertencem "
        "exclusivamente ao conjunto TEST.\n\n"
    )

    file.write("RANKING DOS MODELOS\n")
    file.write("-" * 70 + "\n\n")

    for _, row in ranking.iterrows():

        file.write(
            f"{int(row['ranking'])}. {row['model']}\n"
        )

        file.write(
            f"   Precision: {row['precision']:.4f}\n"
        )

        file.write(
            f"   Recall:    {row['recall']:.4f}\n"
        )

        file.write(
            f"   F1-score:  {row['f1_score']:.4f}\n"
        )

        file.write(
            f"   ROC-AUC:   {row['roc_auc']:.4f}\n"
        )

        file.write(
            f"   PR-AUC:    {row['pr_auc']:.4f}\n"
        )

        file.write(
            f"   FPR:       "
            f"{row['false_positive_rate'] * 100:.4f}%\n"
        )

        file.write(
            f"   FNR:       "
            f"{row['false_negative_rate'] * 100:.4f}%\n"
        )

        file.write(
            f"   Detection: "
            f"{row['fraud_detection_rate'] * 100:.4f}%\n"
        )

        file.write(
            f"   Threshold: {row['threshold']:.8f}\n\n"
        )

    file.write("=" * 70 + "\n")
    file.write("MELHOR MODELO\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        f"Modelo: {melhor['model']}\n"
    )

    file.write(
        f"PR-AUC: {melhor['pr_auc']:.4f}\n"
    )

    file.write(
        f"F1-score: {melhor['f1_score']:.4f}\n"
    )

    file.write(
        f"Recall: {melhor['recall']:.4f}\n"
    )

    file.write(
        f"Precision: {melhor['precision']:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {melhor['roc_auc']:.4f}\n"
    )

    file.write(
        f"Threshold: {melhor['threshold']:.8f}\n"
    )

    file.write("\n" + "=" * 70 + "\n")
    file.write("CONCLUSÃO\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        "O ranking foi definido prioritariamente por PR-AUC, "
        "seguido por F1-score e Recall.\n\n"
    )

    file.write(
        "O XGBoost Tuned apresenta o melhor equilíbrio entre "
        "detecção de fraudes e controle de falsos positivos "
        "entre os modelos avaliados.\n"
    )

print()
print("==========================================================")
print("             COMPARAÇÃO FINAL CONCLUÍDA")
print("==========================================================")

print()
print("Arquivos gerados:")

print(csv_path)
print(pr_auc_path)
print(f1_path)
print(recall_path)
print(fpr_path)
print(report_path)