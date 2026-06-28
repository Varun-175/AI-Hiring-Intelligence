class BehavioralFeatures:
    """
    Computes a behavioral/recruiter confidence score (0-100).

    Signals used (only if available):
    ---------------------------------
    ✓ Profile completeness
    ✓ Recruiter response rate
    ✓ GitHub activity
    ✓ Skill assessment scores
    ✓ Interview completion
    ✓ Offer acceptance
    ✓ Notice period
    ✓ Profile freshness
    ✓ Consistency bonus

    Missing signals do not heavily penalize the candidate.
    """

    DEFAULT_SCORE = 50.0

    WEIGHTS = {
        "profile": 0.22,
        "response": 0.18,
        "github": 0.15,
        "assessment": 0.20,
        "interview": 0.10,
        "offer": 0.05,
        "notice": 0.05,
        "freshness": 0.05,
    }

    @staticmethod
    def _normalize(value):
        """
        Normalize any numeric value to 0-100.
        """

        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if value <= 1:
            value *= 100

        return max(0.0, min(value, 100.0))

    @staticmethod
    def _average(values):

        values = [v for v in values if v is not None]

        if not values:
            return None

        return sum(values) / len(values)

    @classmethod
    def compute(cls, candidate):

        signals = getattr(candidate, "signals", {}) or {}

        if not signals:
            return cls.DEFAULT_SCORE

        # --------------------------------------------------
        # Skill Assessments
        # --------------------------------------------------

        assessments = signals.get("skill_assessment_scores") or {}

        assessment_score = (
            sum(assessments.values()) / len(assessments)
            if assessments
            else signals.get("assessment_score")
        )

        # --------------------------------------------------
        # Normalize signals
        # --------------------------------------------------

        profile = cls._normalize(
            signals.get("profile_completeness_score")
            or signals.get("profile_completeness")
        )

        recruiter = cls._normalize(
            signals.get("recruiter_response_rate")
        )

        github = cls._normalize(
            signals.get("github_activity_score")
            or signals.get("github_activity")
        )

        assessment = cls._normalize(
            assessment_score
        )

        interview = cls._normalize(
            signals.get("interview_completion_rate")
        )

        offer = cls._normalize(
            signals.get("offer_acceptance_rate")
        )

        # Lower notice period = better
        notice = signals.get("notice_period_days")

        if notice is not None:
            try:
                notice = max(
                    0,
                    min(
                        100,
                        100 - (float(notice) / 90) * 100,
                    ),
                )
            except Exception:
                notice = None

        freshness = cls._normalize(
            signals.get("profile_freshness_score")
        )

        # --------------------------------------------------
        # Weighted score
        # --------------------------------------------------

        weighted = []

        for value, weight in [
            (profile, cls.WEIGHTS["profile"]),
            (recruiter, cls.WEIGHTS["response"]),
            (github, cls.WEIGHTS["github"]),
            (assessment, cls.WEIGHTS["assessment"]),
            (interview, cls.WEIGHTS["interview"]),
            (offer, cls.WEIGHTS["offer"]),
            (notice, cls.WEIGHTS["notice"]),
            (freshness, cls.WEIGHTS["freshness"]),
        ]:

            if value is not None:
                weighted.append((value, weight))

        if not weighted:
            return cls.DEFAULT_SCORE

        total_weight = sum(weight for _, weight in weighted)

        score = sum(
            value * weight
            for value, weight in weighted
        ) / total_weight

        # --------------------------------------------------
        # Consistency bonus
        # --------------------------------------------------

        available_signals = len(weighted)

        if available_signals >= 6:
            score += 3

        elif available_signals >= 4:
            score += 1

        return round(max(0.0, min(score, 100.0)), 2)