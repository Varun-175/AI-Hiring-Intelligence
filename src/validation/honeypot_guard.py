import re
from collections import Counter

from src.feature_engineering.technical_features import TechnicalFeatures
from src.utils.console import detail, section_break, stage


class HoneypotGuard:
    """
    Lightweight post-ranking safeguard against suspicious or weakly supported profiles.

    This module does not try to predict hidden labels. It applies small,
    explainable adjustments based on profile consistency, evidence strength,
    and behavioral reliability.

    Design goals:
    - Keep penalties bounded and explainable
    - Prefer gradual penalties over hard cutoffs
    - Reward trustworthiness framing in outputs
    - Avoid over-penalizing unconventional but legitimate candidates
    """

    AI_BUZZWORDS = (
        "llm",
        "rag",
        "langchain",
        "prompt",
        "gpt",
        "ai",
        "ml",
        "nlp",
        "vector",
        "retrieval",
        "embedding",
    )

    HIGH_SENIORITY_TERMS = {
        "staff": 7,
        "principal": 8,
        "architect": 8,
        "chief": 10,
        "head": 9,
    }

    MID_SENIORITY_TERMS = {
        "lead": 5,
        "senior": 4,
    }

    RISK_BANDS = (
        ("Very Safe", 0, 20, 0.00),
        ("Low Risk", 20, 40, 0.01),
        ("Medium Risk", 40, 60, 0.03),
        ("High Risk", 60, 80, 0.05),
        ("Very High Risk", 80, 101, 0.08),
    )

    TRUST_BANDS = (
        ("Very High Trust", 80, 101),
        ("High Trust", 65, 80),
        ("Moderate Trust", 45, 65),
        ("Low Trust", 25, 45),
        ("Very Low Trust", 0, 25),
    )

    _TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b", re.IGNORECASE)

    @staticmethod
    def _clamp(value, low=0.0, high=100.0):
        return max(low, min(float(value), high))

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default=0):
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _add_reason(reasons, reason):
        if reason not in reasons:
            reasons.append(reason)

    @classmethod
    def _candidate_text(cls, candidate):
        cached = getattr(candidate, "_cached_recruiter_text", None)
        if cached is not None:
            return cached.lower()
        return (getattr(candidate, "all_text", "") or "").lower()

    @classmethod
    def _token_counts(cls, text):
        return Counter(match.group(0).lower() for match in cls._TOKEN_PATTERN.finditer(text or ""))

    @classmethod
    def _whole_word_count(cls, text, terms):
        counts = cls._token_counts(text)
        return sum(counts.get(term, 0) for term in terms)

    @classmethod
    def _risk_band(cls, score):
        for label, low, high, penalty in cls.RISK_BANDS:
            if low <= score < high:
                return label, penalty
        return "Very High Risk", 0.08

    @classmethod
    def _trust_band(cls, score):
        for label, low, high in cls.TRUST_BANDS:
            if low <= score < high:
                return label
        return "Very Low Trust"

    @staticmethod
    def _is_internship(history_item):
        title = str(history_item.get("title", "")).lower()
        return "intern" in title or "internship" in title or "trainee" in title

    @staticmethod
    def _is_current_role(history_item):
        for key in ("is_current", "current", "present"):
            value = history_item.get(key)
            if value is True:
                return True
        end_date = str(history_item.get("end_date", "")).strip().lower()
        return end_date in {"present", "current", "ongoing"}

    @classmethod
    def _consecutive_short_job_streak(cls, career_history):
        streak = 0
        best = 0

        for item in career_history or []:
            if cls._is_internship(item) or cls._is_current_role(item):
                streak = 0
                continue

            months = cls._safe_int(item.get("duration_months"), 0)
            if 0 < months < 6:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0

        return best

    @staticmethod
    def _role_family(title):
        t = (title or "").lower()

        families = {
            "engineering": ("engineer", "developer", "software", "backend", "frontend", "full stack", "platform"),
            "data_ai": ("data", "ml", "machine learning", "ai", "nlp", "scientist", "analytics"),
            "product": ("product", "pm"),
            "design": ("designer", "ux", "ui"),
            "devops": ("devops", "sre", "cloud", "infrastructure", "platform"),
            "security": ("security", "appsec"),
            "qa": ("qa", "test", "automation"),
        }

        matched = {family for family, keywords in families.items() if any(k in t for k in keywords)}
        return matched or {"other"}

    @classmethod
    def _role_family_volatility(cls, career_history):
        families = []
        for item in career_history or []:
            title = item.get("title", "")
            families.append(cls._role_family(title))

        if len(families) < 3:
            return 0

        transitions = 0
        comparisons = 0

        for i in range(1, len(families)):
            comparisons += 1
            if families[i].isdisjoint(families[i - 1]):
                transitions += 1

        if comparisons == 0:
            return 0

        return round((transitions / comparisons) * 100, 2)

    @classmethod
    def _experience_title_penalty(cls, title, years_of_experience):
        title = (title or "").lower()
        years = cls._safe_float(years_of_experience, 0)
        penalty = 0.0

        for term, minimum_years in cls.HIGH_SENIORITY_TERMS.items():
            if term in title:
                gap = max(0.0, minimum_years - years)
                penalty = min(gap * 4.0, 20.0)
                return penalty, gap > 0

        for term, minimum_years in cls.MID_SENIORITY_TERMS.items():
            if term in title:
                gap = max(0.0, minimum_years - years)
                penalty = min(gap * 3.0, 10.0)
                return penalty, gap > 0

        return 0.0, False

    @classmethod
    def _supported_ai_skills(cls, ai_skills, history_text, profile_text):
        supported = 0
        for skill in ai_skills:
            if (
                TechnicalFeatures._skill_in_text(skill, history_text)
                or TechnicalFeatures._skill_in_text(skill, profile_text)
            ):
                supported += 1
        return supported

    @classmethod
    def _freshness_penalty(cls, signals):
        """
        Optional freshness checks. Only triggers if fields exist.
        Expected normalized ranges are flexible and intentionally mild.
        """
        reasons = []
        penalty = 0.0

        last_active_days = cls._safe_float(signals.get("last_active_days"), -1)
        profile_stale_days = cls._safe_float(signals.get("days_since_profile_update"), -1)

        if last_active_days >= 120:
            penalty += 6
            reasons.append("Profile appears inactive")
        elif last_active_days >= 60:
            penalty += 3
            reasons.append("Profile activity is stale")

        if profile_stale_days >= 240:
            penalty += 4
            reasons.append("Profile has not been updated recently")
        elif profile_stale_days >= 120:
            penalty += 2
            reasons.append("Profile update recency is weak")

        return penalty, reasons[:2]

    @classmethod
    def _behavioral_risk(cls, signals, career_evidence_strength):
        reasons = []
        risk = 0.0

        profile_completeness = cls._safe_float(signals.get("profile_completeness_score"), 0)
        recruiter_response = cls._safe_float(signals.get("recruiter_response_rate"), 0)
        interview_completion = cls._safe_float(signals.get("interview_completion_rate"), 0)
        offer_acceptance = cls._safe_float(signals.get("offer_acceptance_rate"), -1)
        github_score = cls._safe_float(signals.get("github_activity_score"), -1)

        if profile_completeness < 45:
            risk += 10
            cls._add_reason(reasons, "Low profile completeness")
        elif profile_completeness < 60:
            risk += 4
            cls._add_reason(reasons, "Moderate profile completeness")

        if recruiter_response < 0.15:
            risk += 8
            cls._add_reason(reasons, "Low recruiter engagement")
        elif recruiter_response < 0.30:
            risk += 3
            cls._add_reason(reasons, "Weak recruiter engagement")

        if interview_completion < 0.45:
            risk += 8
            cls._add_reason(reasons, "Weak interview completion")
        elif interview_completion < 0.60:
            risk += 3
            cls._add_reason(reasons, "Moderate interview completion")

        if 0 <= offer_acceptance < 0.20:
            risk += 5
            cls._add_reason(reasons, "Low offer acceptance")

        # GitHub is only a weak supporting signal, not a standalone penalty.
        if github_score >= 0 and github_score < 10 and profile_completeness < 50 and career_evidence_strength < 45:
            risk += 5
            cls._add_reason(reasons, "Low public proof signals")

        freshness_penalty, freshness_reasons = cls._freshness_penalty(signals)
        risk += freshness_penalty
        for reason in freshness_reasons:
            cls._add_reason(reasons, reason)

        return risk, reasons[:4]

    @classmethod
    def compute_risk(cls, candidate, job, features):
        text = cls._candidate_text(candidate)
        career_history = getattr(candidate, "career_history", []) or []
        skills = {
            TechnicalFeatures._normalize_skill(skill)
            for skill in getattr(candidate, "skills", []) or []
            if skill
        }

        ai_skills = {
            skill for skill in skills
            if skill in TechnicalFeatures.SKILL_KEYWORDS or skill in {"llm", "rag", "nlp", "ml", "ai"}
        }

        risk = 0.0
        reasons = []

        # 1) Dense skill list and exact buzzword repetition
        buzzword_hits = cls._whole_word_count(text, cls.AI_BUZZWORDS)
        skill_count = len(getattr(candidate, "skills", []) or [])

        if skill_count >= 30:
            risk += 14
            cls._add_reason(reasons, "Extremely dense skill list")
        elif skill_count >= 20:
            risk += 8
            cls._add_reason(reasons, "Unusually dense skill list")

        if buzzword_hits >= 20:
            risk += 10
            cls._add_reason(reasons, "Heavy AI keyword repetition")
        elif buzzword_hits >= 12:
            risk += 5
            cls._add_reason(reasons, "Noticeable AI keyword repetition")

        # 2) Gradual experience-title consistency check
        title = getattr(candidate, "current_title", "") or ""
        years_of_experience = cls._safe_float(getattr(candidate, "years_of_experience", 0), 0)
        title_penalty, mismatch = cls._experience_title_penalty(title, years_of_experience)
        if mismatch and title_penalty > 0:
            risk += title_penalty
            cls._add_reason(reasons, "Experience/title mismatch")

        # 3) Consecutive short tenure check, softer and more targeted
        short_streak = cls._consecutive_short_job_streak(career_history)
        if short_streak >= 4:
            risk += 12
            cls._add_reason(reasons, "Repeated short job tenures")
        elif short_streak >= 3:
            risk += 8
            cls._add_reason(reasons, "Several consecutive short job tenures")

        # 4) Role volatility is more meaningful than industry count alone
        role_volatility = cls._role_family_volatility(career_history)
        career_score = cls._safe_float(features.get("career", 0), 0)
        if role_volatility >= 70 and career_score < 50:
            risk += 8
            cls._add_reason(reasons, "Abrupt role transitions with weak career evidence")
        elif role_volatility >= 50 and career_score < 45:
            risk += 4
            cls._add_reason(reasons, "Role continuity appears weak")

        # 5) AI skills must have some work-history support
        history_text = " ".join(
            f"{item.get('title', '')} {item.get('description', '')}"
            for item in career_history
        ).lower()

        supported_ai_skills = cls._supported_ai_skills(ai_skills, history_text, text)
        unsupported_ai_skills = max(0, len(ai_skills) - supported_ai_skills)

        if unsupported_ai_skills >= 8:
            risk += 16
            cls._add_reason(reasons, "Claimed AI skills lack work-history evidence")
        elif unsupported_ai_skills >= 5:
            risk += 10
            cls._add_reason(reasons, "Many AI skills have weak evidence")
        elif unsupported_ai_skills >= 3:
            risk += 5
            cls._add_reason(reasons, "Limited evidence for claimed AI skills")

        relevant_ai_years = cls._safe_float(features.get("evidence", {}).get("relevant_ai_years", 0), 0)
        if len(ai_skills) > 10 and relevant_ai_years < 1:
            risk += 12
            cls._add_reason(reasons, "High AI skill density with weak AI experience")
        elif len(ai_skills) >= 6 and relevant_ai_years < 0.5 and unsupported_ai_skills >= 4:
            risk += 6
            cls._add_reason(reasons, "AI specialization claims are weakly supported")

        # 6) Behavioral and profile reliability checks
        signals = getattr(candidate, "signals", {}) or {}
        evidence_strength = cls._safe_float(features.get("evidence", {}).get("required_coverage", 0), 0)
        behavioral_risk, behavioral_reasons = cls._behavioral_risk(signals, evidence_strength)
        risk += behavioral_risk
        for reason in behavioral_reasons:
            cls._add_reason(reasons, reason)

        # 7) Keyword-heavy match with weak support
        required_coverage = cls._safe_float(features.get("evidence", {}).get("required_coverage", 0), 0)
        technical_score = cls._safe_float(features.get("technical", 0), 0)
        confidence_score = cls._safe_float(features.get("confidence", 0), 0)

        if (
            required_coverage >= 60
            and technical_score < 55
            and career_score < 55
            and confidence_score < 60
        ):
            risk += 8
            cls._add_reason(reasons, "Keyword-heavy JD match with weak supporting signals")

        # 8) Ranking disagreement as uncertainty signal
        cross_score = features.get("cross_score_normalized")
        heuristic_score = features.get("final", features.get("confidence", 0))
        semantic_score = features.get("semantic_score_normalized")
        bm25_score = features.get("bm25_score_normalized")

        available_scores = [
            cls._safe_float(score, None)
            for score in (cross_score, heuristic_score, semantic_score, bm25_score)
            if score is not None
        ]

        if len(available_scores) >= 3:
            disagreement = max(available_scores) - min(available_scores)
            if disagreement >= 40:
                risk += 8
                cls._add_reason(reasons, "Ranking signals disagree strongly")
            elif disagreement >= 25:
                risk += 4
                cls._add_reason(reasons, "Ranking confidence is mixed")
        elif cross_score is not None and heuristic_score is not None:
            disagreement = abs(cls._safe_float(cross_score, 0) - cls._safe_float(heuristic_score, 0))
            if disagreement >= 30:
                risk += 5
                cls._add_reason(reasons, "Ranking signals disagree strongly")

        # 9) Extra pattern: very large skill list + unsupported AI story
        if skill_count >= 35 and unsupported_ai_skills >= 5 and relevant_ai_years < 1:
            risk += 8
            cls._add_reason(reasons, "High skill density with weak supporting evidence")

        risk = cls._clamp(risk)
        risk_level, penalty_pct = cls._risk_band(risk)

        trust_score = cls._clamp(100 - risk)
        trust_level = cls._trust_band(trust_score)

        return {
            "risk_score": round(risk, 2),
            "risk_level": risk_level,
            "risk_reasons": reasons[:3],
            "penalty_pct": penalty_pct,
            "trust_score": round(trust_score, 2),
            "trust_level": trust_level,
            "supported_ai_skills": supported_ai_skills,
            "unsupported_ai_skills": unsupported_ai_skills,
            "role_volatility": role_volatility,
            "buzzword_hits": buzzword_hits,
        }

    def apply_penalty(self, ranked_candidates, job=None):
        analyzed = []
        band_counts = Counter()
        trust_band_counts = Counter()

        highest_risk_item = None
        total_risk = 0.0
        total_penalty_pct = 0.0

        for index, item in enumerate(ranked_candidates):
            candidate = item["candidate"]
            features = item.setdefault("features", {})

            features["cross_score_normalized"] = item.get("cross_score_normalized")
            features["semantic_score_normalized"] = item.get("semantic_score_normalized")
            features["bm25_score_normalized"] = item.get("bm25_score_normalized")

            risk = self.compute_risk(candidate, job, features)

            original_score = self._safe_float(item.get("score"), 0)
            penalized_score = original_score * (1 - risk["penalty_pct"])

            item["base_score"] = round(original_score, 2)
            item["score"] = round(self._clamp(penalized_score), 2)
            item["risk_score"] = risk["risk_score"]
            item["risk_level"] = risk["risk_level"]
            item["risk_reasons"] = risk["risk_reasons"]
            item["honeypot_penalty_pct"] = risk["penalty_pct"]
            item["trust_score"] = risk["trust_score"]
            item["trust_level"] = risk["trust_level"]
            item["original_rank"] = index

            item["risk_meta"] = {
                "supported_ai_skills": risk["supported_ai_skills"],
                "unsupported_ai_skills": risk["unsupported_ai_skills"],
                "role_volatility": risk["role_volatility"],
                "buzzword_hits": risk["buzzword_hits"],
            }

            features["risk_score"] = risk["risk_score"]
            features["risk_level"] = risk["risk_level"]
            features["risk_reasons"] = risk["risk_reasons"]
            features["honeypot_penalty_pct"] = risk["penalty_pct"]
            features["trust_score"] = risk["trust_score"]
            features["trust_level"] = risk["trust_level"]

            band_counts[risk["risk_level"]] += 1
            trust_band_counts[risk["trust_level"]] += 1

            total_risk += risk["risk_score"]
            total_penalty_pct += risk["penalty_pct"]
            analyzed.append(item)

            if highest_risk_item is None or risk["risk_score"] > highest_risk_item["risk_score"]:
                highest_risk_item = item

        analyzed.sort(
            key=lambda item: (
                -item["score"],
                item["candidate"].candidate_id,
            )
        )

        avg_risk = round(total_risk / len(analyzed), 2) if analyzed else 0.0
        avg_penalty_pct = round((total_penalty_pct / len(analyzed)) * 100, 2) if analyzed else 0.0

        section_break()
        stage("Candidate Risk Validation")

        for label in ("Very Safe", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"):
            detail(label, band_counts.get(label, 0))

        detail("Average Risk", avg_risk)

        if highest_risk_item is not None:
            detail(
                "Highest Risk",
                f"{highest_risk_item['candidate'].candidate_id} ({highest_risk_item['risk_score']})"
            )

        detail("Average Penalty", f"{avg_penalty_pct}%")

        section_break()
        stage("Candidate Trust Summary")

        for label in ("Very High Trust", "High Trust", "Moderate Trust", "Low Trust", "Very Low Trust"):
            detail(label, trust_band_counts.get(label, 0))

        return analyzed