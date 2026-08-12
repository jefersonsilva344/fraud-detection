from pydantic import BaseModel, ConfigDict, Field


class FraudPredictionRequest(BaseModel):
    """
    Dados de entrada utilizados pelo modelo XGBoost.

    As 30 features devem corresponder exatamente
    às features utilizadas durante o treinamento.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    Time: float = Field(
        ...,
        description="Tempo da transação em segundos desde a primeira transação."
    )

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(
        ...,
        ge=0,
        description="Valor da transação."
    )


class FraudPredictionResponse(BaseModel):
    """
    Resultado retornado pela API.
    """

    fraud_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probabilidade estimada de fraude."
    )

    is_fraud: bool = Field(
        ...,
        description="Indica se a transação foi classificada como fraude."
    )

    threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description="Threshold utilizado para classificação."
    )