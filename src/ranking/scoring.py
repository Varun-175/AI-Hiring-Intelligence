from functools import lru_cache

from src.utils.config import get_nested, load_yaml_config


class CandidateScorer:
    """
    Final candidate scoring engine.

    Combines normalized feature scores into a single
    confidence score (0-100).
    """

    DEFAULT_WEIGHTS = {
        "technical": 0.40,
        "career": 0.25,
        "behavioral": 0.20,
        "consistency": 0.15,
    }

    FEATURE_KEYS = (
        "technical",
        "career",
        "behavioral",
        "consistency",
    )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(float(value), 100.0))

    @classmethod
    @lru_cache(maxsize=1)
    def _weights(cls):
        weights = get_nested(
            load_yaml_config("configs/weights.yaml"),
            ["scoring"],
            cls.DEFAULT_WEIGHTS,
        )

        # Ensure every expected key exists
        normalized = {}

        for key in cls.FEATURE_KEYS:
            normalized[key] = float(
                weights.get(
                    key,
                    cls.DEFAULT_WEIGHTS[key],
                )
            )

        total = sum(normalized.values())

        if total <= 0:
            return cls.DEFAULT_WEIGHTS.copy()

        # Normalize weights to sum to 1.0
        return {
            key: value / total
            for key, value in normalized.items()
        }

    @classmethod
    def score(cls, features):

        weights = cls._weights()

        score = 0.0

        for feature in cls.FEATURE_KEYS:

            score += (
                cls._clamp(
                    features.get(feature, 0.0)
                )
                * weights[feature]
            )

        # Optional confidence adjustment
        confidence = cls._clamp(
            features.get("confidence", 100.0)
        )

        score *= (
            0.95 + 0.05 * (confidence / 100.0)
        )

        return round(
            cls._clamp(score),
            2,
        )