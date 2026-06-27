class TechnicalFeatures:

    @staticmethod
    def compute(job, candidate):

        job_skills = {s.lower() for s in job.required_skills}
        candidate_skills = {s.lower() for s in candidate.skills}

        overlap = len(job_skills & candidate_skills)

        if len(job_skills) == 0:
            return 0

        return overlap / len(job_skills) * 100