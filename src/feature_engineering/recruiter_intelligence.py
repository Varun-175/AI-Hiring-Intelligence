from src.feature_engineering.technical_features import TechnicalFeatures


class RecruiterIntelligence:
    AI_SKILLS = {
        "python", "machine learning", "deep learning", "nlp", "llm", "rag",
        "tensorflow", "pytorch", "transformers", "langchain",
        "vector database", "embeddings", "retrieval", "ranking",
    }
    BACKEND_SKILLS = {"python", "java", "sql", "spark", "airflow", "kafka", "docker", "kubernetes"}
    CLOUD_SKILLS = {"aws", "azure", "gcp"}
    VECTOR_SKILLS = {"vector database", "milvus", "pinecone", "faiss"}
    LLM_SKILLS = {"llm", "rag", "transformers", "langchain", "embeddings"}
    ML_SKILLS = {"machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn"}
    ALL_AI_SKILLS = AI_SKILLS | LLM_SKILLS | ML_SKILLS
    EDUCATION_KEYWORDS = {"computer science", "machine learning", "ai", "artificial intelligence", "data science"}
    SENIORITY_TERMS = {"senior", "lead", "staff", "principal", "architect", "founding"}

    @staticmethod
    def _score_100(value):
        if value is None:
            return 0.0

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if value < 0:
            return 0.0
        if value <= 1:
            return value * 100

        return min(value, 100)

    @staticmethod
    def _clamp(value):
        return max(0.0, min(float(value), 100.0))

    @staticmethod
    def _contains_any(text, terms):
        text = (text or "").lower()
        return any(term in text for term in terms)

    @classmethod
    def _candidate_text(cls, candidate):
        cached = candidate._cached_recruiter_text
        if cached is not None:
            return cached
        cached = candidate.all_text.lower()
        candidate._cached_recruiter_text = cached
        return cached

    @classmethod
    def _coverage(cls, available, required):
        if not required:
            return 0.0
        return len(available & required) / len(required) * 100

    @classmethod
    def _skill_density(cls, candidate_skills, target_skills):
        if not candidate_skills:
            return 0.0
        return len(candidate_skills & target_skills) / len(candidate_skills) * 100

    @classmethod
    def _relevant_ai_experience_years(cls, candidate):
        cached = getattr(candidate, "_cached_relevant_ai_years", None)
        if cached is not None:
            return cached

        relevant_months = 0

        for item in candidate.career_history:
            text = f"{item.get('title', '')} {item.get('description', '')}".lower()
            if cls._contains_any(text, cls.ALL_AI_SKILLS):
                relevant_months += int(item.get("duration_months") or 0)

        if relevant_months:
            result = relevant_months / 12
        else:
            text = cls._candidate_text(candidate)
            if cls._contains_any(text, cls.ALL_AI_SKILLS):
                result = candidate.years_of_experience * 0.5
            else:
                result = 0.0

        candidate._cached_relevant_ai_years = result
        return result

    @classmethod
    def _title_similarity(cls, job, candidate):
        job_tokens = getattr(job, "_cached_title_tokens", None)
        if job_tokens is None:
            job_tokens = set((job.title or "").lower().replace("-", " ").split())
            job._cached_title_tokens = job_tokens

        title_tokens = getattr(candidate, "_cached_title_tokens", None)
        if title_tokens is None:
            title_tokens = set((candidate.current_title or "").lower().replace("-", " ").split())
            candidate._cached_title_tokens = title_tokens

        if not job_tokens or not title_tokens:
            return 0.0

        return len(job_tokens & title_tokens) / len(job_tokens | title_tokens) * 100

    @classmethod
    def _career_progression_score(cls, candidate):
        cached = getattr(candidate, "_cached_career_progression", None)
        if cached is not None:
            return cached

        titles = [
            str(item.get("title", "")).lower()
            for item in candidate.career_history
            if item.get("title")
        ]

        if not titles:
            result = 40.0
        else:
            senior_roles = sum(1 for title in titles if cls._contains_any(title, cls.SENIORITY_TERMS))
            current_senior = cls._contains_any(candidate.current_title, cls.SENIORITY_TERMS)
            progression = 50 + senior_roles * 10 + (20 if current_senior else 0)
            result = cls._clamp(progression)

        candidate._cached_career_progression = result
        return result

    @classmethod
    def _job_stability_score(cls, candidate):
        cached = getattr(candidate, "_cached_job_stability", None)
        if cached is not None:
            return cached

        durations = [
            int(item.get("duration_months") or 0)
            for item in candidate.career_history
            if item.get("duration_months") is not None
        ]

        if not durations:
            result = 50.0
        else:
            average_months = sum(durations) / len(durations)
            result = cls._clamp(average_months / 36 * 100)

        candidate._cached_job_stability = result
        return result

    @classmethod
    def _education_score(cls, candidate):
        cached = getattr(candidate, "_cached_education_score", None)
        if cached is not None:
            return cached

        if not candidate.education:
            result = 40.0
        else:
            score = 45.0
            for item in candidate.education:
                field = str(item.get("field_of_study", "")).lower()
                degree = str(item.get("degree", "")).lower()
                tier = str(item.get("tier", "")).lower()

                if cls._contains_any(f"{field} {degree}", cls.EDUCATION_KEYWORDS):
                    score += 30
                if "master" in degree or "m.tech" in degree or "ph" in degree:
                    score += 10
                if tier == "tier_1":
                    score += 10
                elif tier == "tier_2":
                    score += 5
            result = cls._clamp(score)

        candidate._cached_education_score = result
        return result

    @classmethod
    def _assessment_score(cls, signals):
        assessments = signals.get("skill_assessment_scores") or {}
        if not assessments:
            return 0.0

        return sum(float(value) for value in assessments.values()) / len(assessments)

    @classmethod
    def _behavior_score(cls, signals):
        if not signals:
            return 50.0

        notice_period = float(signals.get("notice_period_days") or 90)
        availability = 100 if signals.get("open_to_work_flag") else 50
        availability += max(0, 30 - notice_period) / 30 * 20

        recruiter_interest = (
            min(float(signals.get("saved_by_recruiters_30d") or 0), 10) / 10 * 60
            + min(float(signals.get("search_appearance_30d") or 0), 300) / 300 * 40
        )

        verification = sum(
            1
            for field in ("verified_email", "verified_phone", "linkedin_connected")
            if signals.get(field)
        ) / 3 * 100

        return cls._clamp(
            cls._score_100(signals.get("profile_completeness_score")) * 0.18
            + cls._score_100(signals.get("recruiter_response_rate")) * 0.16
            + cls._score_100(signals.get("github_activity_score")) * 0.16
            + cls._assessment_score(signals) * 0.18
            + cls._score_100(signals.get("interview_completion_rate")) * 0.10
            + cls._score_100(signals.get("offer_acceptance_rate")) * 0.06
            + cls._clamp(availability) * 0.08
            + recruiter_interest * 0.05
            + verification * 0.03
        )

    @classmethod
    def _consistency_score(cls, candidate, candidate_skills):
        cached = getattr(candidate, "_cached_consistency_score", None)
        if cached is not None:
            return cached

        score = 100.0

        if not candidate.summary:
            score -= 18
        if not candidate.headline:
            score -= 8
        if not candidate.skills:
            score -= 20
        if not candidate.career_history:
            score -= 30
        if candidate.years_of_experience <= 0:
            score -= 20

        history_text = " ".join(
            f"{item.get('title', '')} {item.get('description', '')}"
            for item in candidate.career_history
        ).lower()
        supported_skills = sum(
            1 for skill in candidate_skills if TechnicalFeatures._skill_in_text(skill, history_text)
        )
        if candidate_skills:
            score += supported_skills / len(candidate_skills) * 12

        total_history_years = sum(
            int(item.get("duration_months") or 0)
            for item in candidate.career_history
        ) / 12
        if total_history_years and candidate.years_of_experience:
            gap = abs(total_history_years - candidate.years_of_experience)
            if gap > 3:
                score -= min(15, gap * 2)

        result = cls._clamp(score)
        candidate._cached_consistency_score = result
        return result

    @classmethod
    def compute(cls, job, candidate):
        candidate_skills = getattr(candidate, "_cached_skills_set", None)
        if candidate_skills is None:
            candidate_skills = TechnicalFeatures._skill_set(candidate.skills)
            candidate._cached_skills_set = candidate_skills

        required_skills = getattr(job, "_cached_required_skills_set", None)
        if required_skills is None:
            required_skills = TechnicalFeatures._skill_set(job.required_skills)
            job._cached_required_skills_set = required_skills

        preferred_skills = getattr(job, "_cached_preferred_skills_set", None)
        if preferred_skills is None:
            preferred_skills = TechnicalFeatures._skill_set(job.preferred_skills)
            job._cached_preferred_skills_set = preferred_skills

        text = cls._candidate_text(candidate)

        required_coverage = cls._coverage(candidate_skills, required_skills)
        preferred_coverage = cls._coverage(candidate_skills, preferred_skills)
        ai_density = cls._skill_density(candidate_skills, cls.AI_SKILLS)
        backend_density = cls._skill_density(candidate_skills, cls.BACKEND_SKILLS)
        cloud_coverage = cls._coverage(candidate_skills, cls.CLOUD_SKILLS)

        specialization_hits = {
            "vector": cls._contains_any(text, cls.VECTOR_SKILLS),
            "llm": cls._contains_any(text, cls.LLM_SKILLS),
            "rag": "rag" in text,
            "nlp": "nlp" in text,
            "ml": cls._contains_any(text, cls.ML_SKILLS),
        }
        specialization_score = sum(specialization_hits.values()) / len(specialization_hits) * 100

        technical = cls._clamp(
            required_coverage * 0.45
            + preferred_coverage * 0.10
            + ai_density * 0.16
            + backend_density * 0.08
            + cloud_coverage * 0.06
            + specialization_score * 0.15
        )

        relevant_ai_years = cls._relevant_ai_experience_years(candidate)
        experience_fit = (
            min(candidate.years_of_experience / job.experience_required, 1) * 100
            if job.experience_required
            else 50
        )
        relevant_experience_fit = (
            min(relevant_ai_years / job.experience_required, 1) * 100
            if job.experience_required
            else 50
        )
        leadership = 100 if cls._contains_any(candidate.current_title, {"lead", "staff", "principal", "architect"}) else 0
        career = cls._clamp(
            experience_fit * 0.28
            + relevant_experience_fit * 0.22
            + cls._title_similarity(job, candidate) * 0.18
            + cls._career_progression_score(candidate) * 0.14
            + cls._job_stability_score(candidate) * 0.10
            + leadership * 0.08
        )

        education = cls._education_score(candidate)
        behavioral = cls._behavior_score(candidate.signals)
        consistency = cls._consistency_score(candidate, candidate_skills)

        confidence = cls._clamp(
            technical * 0.40
            + career * 0.25
            + behavioral * 0.20
            + consistency * 0.10
            + education * 0.05
        )

        matched_required = sorted(candidate_skills & required_skills)

        return {
            "technical": technical,
            "career": career,
            "behavioral": behavioral,
            "consistency": consistency,
            "education": education,
            "confidence": confidence,
            "evidence": {
                "matched_required_skills": matched_required,
                "required_skill_count": len(required_skills),
                "required_coverage": required_coverage,
                "ai_skill_count": len(candidate_skills & cls.AI_SKILLS),
                "cloud_skills": sorted(candidate_skills & cls.CLOUD_SKILLS),
                "relevant_ai_years": relevant_ai_years,
                "title_similarity": cls._title_similarity(job, candidate),
                "current_title": candidate.current_title,
                "education_score": education,
                "recruiter_response_rate": candidate.signals.get("recruiter_response_rate"),
                "profile_completeness_score": candidate.signals.get("profile_completeness_score"),
                "github_activity_score": candidate.signals.get("github_activity_score"),
                "open_to_work": candidate.signals.get("open_to_work_flag"),
            },
        }
