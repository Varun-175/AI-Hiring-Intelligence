class ConsistencyFeatures:
    """
    Computes profile consistency (0-100).

    Evaluates:
    ----------
    ✓ Skills vs Experience
    ✓ Skills vs Career History
    ✓ Summary completeness
    ✓ Career completeness
    ✓ Experience consistency
    ✓ Profile quality
    """

    @staticmethod
    def compute(candidate):

        score = 100.0

        skills = getattr(candidate, "skills", []) or []
        summary = (getattr(candidate, "summary", "") or "").strip()
        history = getattr(candidate, "career_history", []) or []
        experience = getattr(candidate, "years_of_experience", 0)

        # ---------------------------------
        # Missing Skills
        # ---------------------------------

        if len(skills) == 0:
            score -= 30

        elif len(skills) < 3:
            score -= 10

        # ---------------------------------
        # Missing Summary
        # ---------------------------------

        if len(summary) < 20:
            score -= 10

        # ---------------------------------
        # Experience
        # ---------------------------------

        if experience <= 0:
            score -= 25

        elif experience > 0 and len(history) == 0:
            score -= 20

        # ---------------------------------
        # Career History
        # ---------------------------------

        if len(history) == 0:
            score -= 20

        elif len(history) == 1:
            score -= 5

        # ---------------------------------
        # Experience vs Career History
        # ---------------------------------

        if experience >= 10 and len(history) < 2:
            score -= 10

        elif experience >= 5 and len(history) == 0:
            score -= 10

        # ---------------------------------
        # Duplicate Skills
        # ---------------------------------

        normalized_skills = [
            skill.lower().strip()
            for skill in skills
        ]

        duplicate_count = len(normalized_skills) - len(set(normalized_skills))

        score -= min(duplicate_count * 2, 10)

        # ---------------------------------
        # Final Score
        # ---------------------------------

        return round(max(0.0, min(score, 100.0)), 2)