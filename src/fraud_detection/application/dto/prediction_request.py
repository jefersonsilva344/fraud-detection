from dataclasses import dataclass
from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType


@dataclass(frozen=True)
class PredictionRequest:
    """
    Dados necessários para solicitar uma predição.
    """

    features: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.features, Mapping):
            raise TypeError("Features must be a mapping.")

        required_features = {"Time", "Amount"}
        missing_features = required_features - self.features.keys()

        if missing_features:
            raise ValueError(
                "Required transaction features are missing: "
                f"{sorted(missing_features)}"
            )

        normalized_features: dict[str, float] = {}
        for name, value in self.features.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Feature '{name}' must be a number.")

            normalized_value = float(value)
            if not isfinite(normalized_value):
                raise ValueError(f"Feature '{name}' must be finite.")

            normalized_features[name] = normalized_value

        object.__setattr__(
            self,
            "features",
            MappingProxyType(normalized_features),
        )
