from src.feature_engineering.feature_engine import FeatureEngine
from src.ranking.scoring import CandidateScorer


class CandidateRanker:

    def __init__(self):
        self.engine = FeatureEngine()

    def rank(self, job, candidates, top_k=100):

        ranked = []

        for candidate in candidates:

            features = self.engine.compute(job, candidate)

            final_score = CandidateScorer.score(features)

            ranked.append(
                {
                    "candidate": candidate,
                    "score": final_score,
                    "features": features,
                }
            )

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked[:top_k]