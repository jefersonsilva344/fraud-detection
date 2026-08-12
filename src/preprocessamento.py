from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw" / "creditcard.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "dataset_processado.csv"


# ==========================================================
# CARREGAMENTO
# ==========================================================

print("\n=== CARREGANDO DATASET ===")

df = pd.read_csv(DATA_RAW)

print(f"Registros originais: {len(df)}")
print(f"Colunas originais: {len(df.columns)}")


# ==========================================================
# REMOÇÃO DE DUPLICATAS
# ==========================================================

print("\n=== REMOVENDO DUPLICATAS ===")

duplicatas = df.duplicated().sum()

print(f"Duplicatas encontradas: {duplicatas}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Registros após remoção: {len(df)}")


# ==========================================================
# VALIDAÇÃO DO TARGET
# ==========================================================

if "Class" not in df.columns:
    raise ValueError(
        "A coluna 'Class' não foi encontrada no dataset."
    )


# ==========================================================
# SEPARAÇÃO DEFINITIVA ENTRE FEATURES E TARGET
# ==========================================================

y = df["Class"].astype(int)

X = df.drop(columns=["Class"]).copy()


# ==========================================================
# GARANTIA ABSOLUTA CONTRA DATA LEAKAGE
# ==========================================================

if "Class" in X.columns:
    raise RuntimeError(
        "ERRO CRÍTICO: a coluna 'Class' entrou nas features."
    )


print("\n=== PREPROCESSAMENTO ===")

print(f"Features utilizadas: {X.shape[1]}")
print(f"Registros: {X.shape[0]}")

print("\nFeatures:")
print(list(X.columns))


# ==========================================================
# PADRONIZAÇÃO
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)


# ==========================================================
# NOVA VALIDAÇÃO
# ==========================================================

if "Class" in X_scaled.columns:
    raise RuntimeError(
        "ERRO CRÍTICO: 'Class' foi incluída após a padronização."
    )


if X_scaled.shape[1] != 30:
    raise RuntimeError(
        f"Número inesperado de features: {X_scaled.shape[1]}. "
        "Esperado: 30."
    )


# ==========================================================
# PRIMEIRAS LINHAS
# ==========================================================

print("\n=== PRIMEIRAS LINHAS DE X ===")

print(X_scaled.head())


# ==========================================================
# DISTRIBUIÇÃO DO TARGET
# ==========================================================

print("\n=== DISTRIBUIÇÃO DA VARIÁVEL TARGET ===")

print(y.value_counts().sort_index())


# ==========================================================
# ESTATÍSTICAS
# ==========================================================

print("\n=== MÉDIA DAS FEATURES APÓS PADRONIZAÇÃO ===")

print(X_scaled.mean().mean())


print("\n=== DESVIO PADRÃO MÉDIO APÓS PADRONIZAÇÃO ===")

print(X_scaled.std().mean())


# ==========================================================
# RECONSTRUÇÃO DO DATASET PROCESSADO
# ==========================================================

df_processado = X_scaled.copy()

# Class entra SOMENTE aqui, como target.
df_processado["Class"] = y.values


# ==========================================================
# GARANTIAS FINAIS
# ==========================================================

assert "Class" in df_processado.columns

feature_columns = [
    column
    for column in df_processado.columns
    if column != "Class"
]

assert "Class" not in feature_columns
assert len(feature_columns) == 30


# ==========================================================
# SALVAMENTO
# ==========================================================

print("\n=== SALVANDO DATASET PROCESSADO ===")

DATA_PROCESSED.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_processado.to_csv(
    DATA_PROCESSED,
    index=False
)

print(f"Arquivo salvo em:")
print(DATA_PROCESSED)

print(f"Registros salvos: {len(df_processado)}")
print(f"Colunas salvas: {len(df_processado.columns)}")

print("\nFeatures salvas:")
print(feature_columns)

print("\nTarget:")
print("Class")


# ==========================================================
# VALIDAÇÃO DO ARQUIVO FINAL
# ==========================================================

df_check = pd.read_csv(DATA_PROCESSED)

X_check = df_check.drop(columns=["Class"])

if "Class" in X_check.columns:
    raise RuntimeError(
        "ERRO FINAL: Class está presente nas features."
    )

print("\n=== VALIDAÇÃO FINAL ===")

print(f"Features: {X_check.shape[1]}")
print(f"Target: Class")
print("Class NÃO está nas features.")

print("\nPreprocessamento concluído com sucesso.")