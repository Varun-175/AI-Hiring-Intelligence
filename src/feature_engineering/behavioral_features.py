class BehavioralFeatures:

    @staticmethod
    def compute(candidate):

        score = 0

        signals = candidate.signals

        if not signals:
            return 50

        score += signals.get("profile_completeness", 0) * 0.3
        score += signals.get("recruiter_response_rate", 0) * 0.3
        score += signals.get("github_activity", 0) * 0.2
        score += signals.get("assessment_score", 0) * 0.2

        return min(score, 100)