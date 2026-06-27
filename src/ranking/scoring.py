class CandidateScorer:
    """
    Computes the final weighted score from feature scores.
    """

    WEIGHTS = {
        "technical": 0.40,
        "career": 0.25,
        "behavioral": 0.20,
        "consistency": 0.15,
    }

    @classmethod
    def score(cls, features):

        return (
            features["technical"] * cls.WEIGHTS["technical"]
            + features["career"] * cls.WEIGHTS["career"]
            + features["behavioral"] * cls.WEIGHTS["behavioral"]
            + features["consistency"] * cls.WEIGHTS["consistency"]
        )