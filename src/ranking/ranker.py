from src.feature_engineering.feature_engine import FeatureEngine
from src.feature_engineering.recruiter_intelligence import RecruiterIntelligence
from src.ranking.scoring import CandidateScorer
from src.utils.config import load_yaml_config, get_nested


class CandidateRanker:
    """
    Main ranking engine.

    Combines:
        • Feature Engine
        • Recruiter Intelligence
        • Candidate Scorer
    """

    def __init__(self):

        weights = load_yaml_config("configs/weights.yaml")

        self.engine = FeatureEngine()

        self.base_weight = get_nested(
            weights,
            ["feature_blend", "base"],
            0.65,
        )

        self.recruiter_weight = get_nested(
            weights,
            ["feature_blend", "recruiter_intelligence"],
            0.35,
        )

    @staticmethod
    def _clamp(value):

        return max(
            0.0,
            min(float(value), 100.0),
        )

    @classmethod
    def _blend_features(cls, base, recruiter):
        try:
            weights = load_yaml_config("configs/weights.yaml")
            base_weight = get_nested(
                weights,
                ["feature_blend", "base"],
                0.65,
            )
            recruiter_weight = get_nested(
                weights,
                ["feature_blend", "recruiter_intelligence"],
                0.35,
            )
        except Exception:
            base_weight = 0.65
            recruiter_weight = 0.35

        blended = {}

        feature_names = (
            "technical",
            "career",
            "behavioral",
            "consistency",
        )

        for feature in feature_names:

            blended[feature] = cls._clamp(
                base.get(feature, 0.0) * base_weight
                + recruiter.get(feature, 0.0) * recruiter_weight
            )

        blended["education"] = cls._clamp(
            recruiter.get("education", 0.0)
        )

        blended["confidence"] = cls._clamp(
            recruiter.get("confidence", 0.0)
        )

        blended["evidence"] = recruiter.get(
            "evidence",
            {},
        )

        return blended

    def _blend(self, base, recruiter):
        return self._blend_features(base, recruiter)

    def rank(
        self,
        job,
        candidates,
        top_k=100,
    ):

        ranked = []

        append = ranked.append

        for candidate in candidates:

            base_features = self.engine.compute(
                job,
                candidate,
            )

            recruiter_features = RecruiterIntelligence.compute(
                job,
                candidate,
            )

            features = self._blend(
                base_features,
                recruiter_features,
            )

            score = self._clamp(
                CandidateScorer.score(features)
            )

            features["final"] = score

            append(
                {
                    "candidate": candidate,
                    "score": score,
                    "features": features,
                    "confidence": features["confidence"],
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["score"],
                -item["confidence"],
                item["candidate"].candidate_id,
            )
        )

        return ranked[: min(top_k, len(ranked))]