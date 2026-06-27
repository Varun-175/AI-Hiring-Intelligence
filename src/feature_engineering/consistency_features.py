class ConsistencyFeatures:

    @staticmethod
    def compute(candidate):

        score = 100

        if len(candidate.skills) == 0:
            score -= 30

        if candidate.years_of_experience == 0:
            score -= 30

        if len(candidate.career_history) == 0:
            score -= 40

        return max(score, 0)