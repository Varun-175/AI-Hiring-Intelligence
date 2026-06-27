from heapq import nlargest


class CandidateRetriever:
    """
    Stage-1 Candidate Retriever.

    Uses lightweight lexical matching to retrieve the
    most relevant candidates before expensive ranking.
    """

    def __init__(self):
        pass

    def _tokenize(self, text: str) -> set:
        if not text:
            return set()

        return {
            token.strip().lower()
            for token in text.split()
            if token.strip()
        }

    def _score(self, job, candidate):

        score = 0

        # ----------------------------
        # Skill Match
        # ----------------------------
        job_skills = {
            s.lower()
            for s in job.required_skills
        }

        candidate_skills = {
            s.lower()
            for s in candidate.skills
        }

        skill_overlap = len(job_skills & candidate_skills)

        score += skill_overlap * 10

        # ----------------------------
        # Title Match
        # ----------------------------
        job_title = self._tokenize(job.title)
        candidate_title = self._tokenize(candidate.current_title)

        title_overlap = len(job_title & candidate_title)

        score += title_overlap * 5

        # ----------------------------
        # Summary Match
        # ----------------------------
        summary_tokens = self._tokenize(candidate.summary)

        summary_overlap = len(job_skills & summary_tokens)

        score += summary_overlap * 2

        # ----------------------------
        # Experience Bonus
        # ----------------------------
        if candidate.years_of_experience >= job.experience_required:
            score += 10

        return score

    def retrieve(self, job, candidates, top_k=5000):

        scored_candidates = []

        for candidate in candidates:

            score = self._score(job, candidate)

            scored_candidates.append(
                (score, candidate)
            )

        top = nlargest(
            top_k,
            scored_candidates,
            key=lambda x: x[0]
        )

        return top