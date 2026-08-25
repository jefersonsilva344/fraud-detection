from dataclasses import dataclass


@dataclass(frozen=True)
class RiskScore:
    """
    Representa a pontuação de risco de uma transação.
    """

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Risk score must be between 0 and 1.")

    @property
    def percentage(self) -> float:
        return self.value * 100