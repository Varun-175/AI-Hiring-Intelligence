from src.feature_engineering.technical_features import TechnicalFeatures
from src.feature_engineering.career_features import CareerFeatures
from src.feature_engineering.behavioral_features import BehavioralFeatures
from src.feature_engineering.consistency_features import ConsistencyFeatures


class FeatureEngine:

    def compute(self, job, candidate):

        technical = TechnicalFeatures.compute(job, candidate)

        career = CareerFeatures.compute(job, candidate)

        behavioral = BehavioralFeatures.compute(candidate)

        consistency = ConsistencyFeatures.compute(candidate)

        final_score = (
            technical * 0.40 +
            career * 0.25 +
            behavioral * 0.20 +
            consistency * 0.15
        )

        return {
            "technical": technical,
            "career": career,
            "behavioral": behavioral,
            "consistency": consistency,
            "final": final_score
        }