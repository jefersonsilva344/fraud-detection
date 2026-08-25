import pandas as pd


URL_DATASET = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)


def carregar_dados(url: str) -> pd.DataFrame:
    """Carrega o dataset de transações de cartão de crédito."""
    return pd.read_csv(url)


df = carregar_dados(URL_DATASET)


print("=== PRIMEIRAS 5 LINHAS ===")
print(df.head())


print("\n=== DIMENSÕES DO DATASET ===")
print(df.shape)


print("\n=== INFORMAÇÕES DO DATASET ===")
print(df.info())


print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
print(df.describe())


print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")
print(df["Class"].value_counts())


print("\n=== PERCENTUAL DAS CLASSES ===")
print(df["Class"].value_counts(normalize=True) * 100)


print("\n=== VALORES AUSENTES ===")
print(df.isnull().sum().sum())


print("\n=== DUPLICATAS ===")
print(df.duplicated().sum())


print("\n=== DUPLICATAS POR CLASSE ===")
print(df[df.duplicated(keep=False)]["Class"].value_counts())


print("\n=== DUPLICATAS ENTRE FRAUDES ===")
print(df[df["Class"] == 1].duplicated().sum())


# Remoção de registros completamente duplicados
df = df.drop_duplicates()


print("\n=== DIMENSÕES APÓS REMOÇÃO DAS DUPLICATAS ===")
print(df.shape)


print("\n=== DUPLICATAS APÓS TRATAMENTO ===")
print(df.duplicated().sum())


print("\n=== DISTRIBUIÇÃO DAS CLASSES APÓS TRATAMENTO ===")
print(df["Class"].value_counts())