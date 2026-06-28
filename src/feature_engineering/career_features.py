class CareerFeatures:
    """
    Computes a career strength score (0-100).

    Factors:
    --------
    ✓ Relevant experience
    ✓ Seniority
    ✓ Leadership
    ✓ Career progression
    ✓ Job stability
    """

    SENIORITY = {
        "intern": 5,
        "junior": 20,
        "associate": 35,
        "engineer": 50,
        "senior": 70,
        "lead": 85,
        "staff": 90,
        "principal": 95,
        "architect": 95,
        "manager": 90,
        "director": 100,
    }

    @classmethod
    def _experience_score(cls, required, actual):

        if required <= 0:
            return 100

        ratio = actual / required

        if ratio >= 2:
            return 100

        return min(100, ratio * 100)

    @classmethod
    def _title_score(cls, title):

        title = (title or "").lower()

        score = 40

        for keyword, value in cls.SENIORITY.items():

            if keyword in title:
                score = max(score, value)

        return score

    @staticmethod
    def _career_progression(candidate):

        history = getattr(candidate, "career_history", None)

        if not history:
            return 50

        if len(history) >= 6:
            return 90

        if len(history) >= 4:
            return 75

        if len(history) >= 2:
            return 60

        return 50

    @staticmethod
    def _stability(candidate):

        history = getattr(candidate, "career_history", None)

        if not history:
            return 50

        years = max(candidate.years_of_experience, 1)

        avg_tenure = years / len(history)

        if avg_tenure >= 3:
            return 90

        if avg_tenure >= 2:
            return 75

        if avg_tenure >= 1:
            return 60

        return 40

    @classmethod
    def compute(cls, job, candidate):

        experience = cls._experience_score(
            job.experience_required,
            candidate.years_of_experience,
        )

        title = cls._title_score(
            candidate.current_title,
        )

        progression = cls._career_progression(
            candidate,
        )

        stability = cls._stability(
            candidate,
        )

        score = (
            experience * 0.45
            + title * 0.25
            + progression * 0.15
            + stability * 0.15
        )

        return round(min(score, 100), 2)