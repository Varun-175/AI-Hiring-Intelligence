from typing import Any, Dict, List

from src.models.candidate import Candidate


class CandidateNormalizer:
    """
    Converts raw candidate JSON into a normalized Candidate object.

    Features
    --------
    ✓ Null-safe
    ✓ Fast
    ✓ Handles multiple schema versions
    ✓ Cleans text
    ✓ Removes duplicate skills
    """

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _list(value: Any) -> List:
        return value if isinstance(value, list) else []

    @classmethod
    def _normalize_skills(cls, skills: List[Dict]) -> List[str]:

        seen = set()
        normalized = []

        for skill in skills:

            if not isinstance(skill, dict):
                continue

            name = cls._clean(skill.get("name"))

            if not name:
                continue

            key = name.lower()

            if key in seen:
                continue

            seen.add(key)
            normalized.append(name)

        return normalized

    @classmethod
    def normalize(cls, candidate: Dict[str, Any]) -> Candidate:

        profile = candidate.get("profile") or {}

        signals = (
            candidate.get("redrob_signals")
            or candidate.get("behavioral_signals")
            or {}
        )

        return Candidate(

            # -------------------------------------------------
            # Identity
            # -------------------------------------------------

            candidate_id=cls._clean(
                candidate.get("candidate_id")
            ),

            # -------------------------------------------------
            # Profile
            # -------------------------------------------------

            name=cls._clean(
                profile.get("anonymized_name")
            ),

            headline=cls._clean(
                profile.get("headline")
            ),

            summary=cls._clean(
                profile.get("summary")
            ),

            location=cls._clean(
                profile.get("location")
            ),

            # -------------------------------------------------
            # Employment
            # -------------------------------------------------

            current_title=cls._clean(
                profile.get("current_title")
            ),

            current_company=cls._clean(
                profile.get("current_company")
            ),

            years_of_experience=cls._float(
                profile.get("years_of_experience")
            ),

            # -------------------------------------------------
            # Skills
            # -------------------------------------------------

            skills=cls._normalize_skills(
                cls._list(candidate.get("skills"))
            ),

            # -------------------------------------------------
            # Career
            # -------------------------------------------------

            career_history=cls._list(
                candidate.get("career_history")
            ),

            # -------------------------------------------------
            # Education
            # -------------------------------------------------

            education=cls._list(
                candidate.get("education")
            ),

            certifications=cls._list(
                candidate.get("certifications")
            ),

            # -------------------------------------------------
            # Recruiter Signals
            # -------------------------------------------------

            signals=signals if isinstance(signals, dict) else {},

            # -------------------------------------------------
            # Metadata
            # -------------------------------------------------

            metadata={
                "source": candidate.get("source", ""),
                "ingested": True,
            },
        )