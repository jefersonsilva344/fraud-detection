import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

URL_DATASET = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)


# ==========================================================
# 1. CARREGAMENTO DOS DADOS
# ==========================================================

df = pd.read_csv(URL_DATASET)


# ==========================================================
# 2. LIMPEZA DOS DADOS
# ==========================================================

# Remove registros completamente duplicados
df = df.drop_duplicates()


# ==========================================================
# 3. ANÁLISE DO VALOR DAS TRANSAÇÕES
# ==========================================================

print("=== VALORES DAS TRANSAÇÕES POR CLASSE ===")

print("\n--- Transações legítimas ---")
print(
    df[df["Class"] == 0]["Amount"].describe()
)

print("\n--- Transações fraudulentas ---")
print(
    df[df["Class"] == 1]["Amount"].describe()
)


# ==========================================================
# 4. ANÁLISE DO TEMPO DAS TRANSAÇÕES
# ==========================================================

print("\n=== TEMPO DAS TRANSAÇÕES POR CLASSE ===")

print("\n--- Transações legítimas ---")
print(
    df[df["Class"] == 0]["Time"].describe()
)

print("\n--- Transações fraudulentas ---")
print(
    df[df["Class"] == 1]["Time"].describe()
)


# ==========================================================
# 5. DISTRIBUIÇÃO DAS CLASSES
# ==========================================================

print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")

print(
    df["Class"].value_counts()
)

print("\n=== PERCENTUAL DAS CLASSES ===")

print(
    df["Class"].value_counts(normalize=True) * 100
)


# ==========================================================
# 6. CORRELAÇÃO COM A VARIÁVEL TARGET
# ==========================================================

correlacao_class = (
    df.corr()["Class"]
    .drop("Class")
    .sort_values()
)

print("\n=== CORRELAÇÃO DAS VARIÁVEIS COM CLASS ===")
print(correlacao_class)


# ==========================================================
# 7. GRÁFICO — DISTRIBUIÇÃO DO AMOUNT
# ==========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Distribuição do valor das transações por classe")
plt.xlabel("Classe (0 = Legítima | 1 = Fraude)")
plt.ylabel("Valor da transação")

plt.show()


# ==========================================================
# 8. GRÁFICO — DISTRIBUIÇÃO DAS CLASSES
# ==========================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Distribuição das transações por classe")
plt.xlabel("Classe (0 = Legítima | 1 = Fraude)")
plt.ylabel("Quantidade de transações")

plt.show()


# ==========================================================
# 9. HEATMAP DE CORRELAÇÃO
# ==========================================================

correlacao = df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlacao,
    cmap="coolwarm",
    center=0
)

plt.title("Matriz de correlação das variáveis")

plt.show()

# ==========================================================
# COMPARAÇÃO DAS PRINCIPAIS FEATURES
# ==========================================================

features = [
    "V17",
    "V14",
    "V12",
    "V10",
    "V16",
    "V3",
    "V7",
]

for feature in features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y=feature
    )

    plt.title(f"Distribuição de {feature} por classe")
    plt.xlabel("Classe (0 = Legítima | 1 = Fraude)")
    plt.ylabel(feature)

    plt.show()

    # ==========================================================
# DISTRIBUIÇÃO DAS PRINCIPAIS FEATURES
# ==========================================================

principais_features = ["V17", "V14", "V12"]

for feature in principais_features:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x=feature,
        hue="Class",
        bins=50,
        kde=True,
        stat="density",
        common_norm=False
    )

    plt.title(f"Distribuição de {feature} por classe")
    plt.xlabel(feature)
    plt.ylabel("Densidade")

    plt.show()