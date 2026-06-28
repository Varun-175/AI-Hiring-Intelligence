from src.feature_engineering.technical_features import TechnicalFeatures
from src.feature_engineering.career_features import CareerFeatures
from src.feature_engineering.behavioral_features import BehavioralFeatures
from src.feature_engineering.consistency_features import ConsistencyFeatures

from src.utils.config import load_yaml_config, get_nested


class FeatureEngine:
    """
    Computes all feature groups and produces the
    final normalized feature score (0-100).

    Individual feature groups:
        • Technical
        • Career
        • Behavioral
        • Consistency
    """

    def __init__(self):

        weights = load_yaml_config("configs/weights.yaml")

        self.tech_weight = get_nested(
            weights,
            ["features", "technical"],
            0.40,
        )

        self.career_weight = get_nested(
            weights,
            ["features", "career"],
            0.25,
        )

        self.behavior_weight = get_nested(
            weights,
            ["features", "behavioral"],
            0.20,
        )

        self.consistency_weight = get_nested(
            weights,
            ["features", "consistency"],
            0.15,
        )

    def compute(self, job, candidate):

        technical = TechnicalFeatures.compute(
            job,
            candidate,
        )

        career = CareerFeatures.compute(
            job,
            candidate,
        )

        behavioral = BehavioralFeatures.compute(
            candidate,
        )

        consistency = ConsistencyFeatures.compute(
            candidate,
        )

        final_score = (
            technical * self.tech_weight
            + career * self.career_weight
            + behavioral * self.behavior_weight
            + consistency * self.consistency_weight
        )

        final_score = max(
            0.0,
            min(final_score, 100.0),
        )

        return {
            "technical": round(technical, 2),
            "career": round(career, 2),
            "behavioral": round(behavioral, 2),
            "consistency": round(consistency, 2),
            "final": round(final_score, 2),
        }