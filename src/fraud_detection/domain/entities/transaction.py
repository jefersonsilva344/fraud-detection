from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class Transaction:
    """
    Representa uma transação financeira no domínio.
    """

    time: float
    amount: float
    features: Mapping[str, float]
