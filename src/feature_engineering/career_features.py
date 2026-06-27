class CareerFeatures:

    @staticmethod
    def compute(job, candidate):

        score = 0

        if candidate.years_of_experience >= job.experience_required:
            score += 50

        if "senior" in candidate.current_title.lower():
            score += 20

        if "lead" in candidate.current_title.lower():
            score += 15

        if "staff" in candidate.current_title.lower():
            score += 15

        return min(score, 100)