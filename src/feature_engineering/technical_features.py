import re


class TechnicalFeatures:
    """
    Computes technical relevance between a candidate and a job.

    Scoring:
        • Required skills
        • Preferred skills
        • Skill aliases
        • Resume text evidence
    """

    ALIASES = {
        "large language models": "llm",
        "large language model": "llm",
        "llms": "llm",
        "fine tuning": "llm",
        "fine-tuning": "llm",
        "fine-tuning llms": "llm",

        "vector search": "vector database",
        "vector db": "vector database",
        "vectordb": "vector database",
        "faiss": "vector database",
        "milvus": "vector database",
        "pinecone": "vector database",

        "transformers": "llm",
        "huggingface": "transformers",

        "scikit-learn": "machine learning",
        "sklearn": "machine learning",

        "pyspark": "spark",

        "aws sagemaker": "aws",
        "amazon web services": "aws",

        "azure ml": "azure",

        "gcp ai platform": "gcp",
    }

    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9+#.]+")

    @classmethod
    def _normalize_skill(cls, skill):

        skill = (skill or "").strip().lower()

        return cls.ALIASES.get(skill, skill)

    @classmethod
    def _skill_set(cls, skills):

        normalized = set()

        for skill in skills or []:

            skill = cls._normalize_skill(skill)

            normalized.add(skill)

            for alias, canonical in cls.ALIASES.items():

                if canonical == skill:
                    normalized.add(alias)

        return normalized

    @classmethod
    def _skill_in_text(cls, skill, text):

        if skill in text:
            return True

        for alias, canonical in cls.ALIASES.items():

            if canonical == skill and alias in text:
                return True

        return False

    @classmethod
    def _coverage(cls, candidate_skills, target_skills):

        if not target_skills:
            return 100.0

        overlap = len(candidate_skills & target_skills)

        return overlap / len(target_skills) * 100

    @classmethod
    def _text_score(cls, missing_skills, text):

        if not missing_skills:
            return 100.0

        hits = sum(
            cls._skill_in_text(skill, text)
            for skill in missing_skills
        )

        return hits / len(missing_skills) * 100

    @classmethod
    def compute(cls, job, candidate):

        required = getattr(job, "_cached_required_skills_set", None)
        if required is None:
            required = cls._skill_set(getattr(job, "required_skills", []))
            job._cached_required_skills_set = required

        preferred = getattr(job, "_cached_preferred_skills_set", None)
        if preferred is None:
            preferred = cls._skill_set(getattr(job, "preferred_skills", []))
            job._cached_preferred_skills_set = preferred

        cached_candidate_skills = getattr(candidate, "_cached_skills_set", None)
        if cached_candidate_skills is None:
            cached_candidate_skills = cls._skill_set(getattr(candidate, "skills", []))
            candidate._cached_skills_set = cached_candidate_skills

        candidate_skills = set(cached_candidate_skills)

        searchable_text = getattr(candidate, "_searchable_text", None)
        if searchable_text is None:
            searchable_text = " ".join(
                filter(
                    None,
                    [
                        candidate.current_title,
                        candidate.headline,
                        candidate.summary,
                    ],
                )
            ).lower()
            candidate._searchable_text = searchable_text

        for skill in cls.SKILL_KEYWORDS:
            if cls._skill_in_text(skill, searchable_text):
                candidate_skills.update(cls._skill_set([skill]))

        required_score = cls._coverage(
            candidate_skills,
            required,
        )

        preferred_score = cls._coverage(
            candidate_skills,
            preferred,
        )

        missing_required = required - candidate_skills

        text_score = cls._text_score(
            missing_required,
            searchable_text,
        )

        final_score = (
            required_score * 0.70
            + preferred_score * 0.10
            + text_score * 0.20
        )

        return round(
            max(0.0, min(final_score, 100.0)),
            2,
        )