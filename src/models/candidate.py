from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class Candidate:
    """
    Canonical Candidate Model.

    Shared across the entire hiring pipeline.

    Features
    --------
    ✓ Memory optimized (slots=True)
    ✓ Strong typing
    ✓ Helper properties
    ✓ Serialization support
    """

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    candidate_id: str

    # ---------------------------------------------------------
    # Profile
    # ---------------------------------------------------------

    name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""

    # ---------------------------------------------------------
    # Employment
    # ---------------------------------------------------------

    current_title: str = ""
    current_company: str = ""
    years_of_experience: float = 0.0

    # ---------------------------------------------------------
    # Skills
    # ---------------------------------------------------------

    skills: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Career
    # ---------------------------------------------------------

    career_history: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------
    # Education
    # ---------------------------------------------------------

    education: List[Dict[str, Any]] = field(default_factory=list)

    certifications: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------
    # Recruiter / Behavioral Signals
    # ---------------------------------------------------------

    signals: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Optional Metadata
    # ---------------------------------------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Performance cache (populated lazily by pipeline modules)
    # ---------------------------------------------------------

    _cached_skills_set: Any = field(default=None, repr=False, compare=False)
    _cached_required_skills_set: Any = field(default=None, repr=False, compare=False)
    _cached_skills_text: Any = field(default=None, repr=False, compare=False)
    _cached_recruiter_text: Any = field(default=None, repr=False, compare=False)
    _searchable_text: Any = field(default=None, repr=False, compare=False)
    _cached_title_tokens: Any = field(default=None, repr=False, compare=False)
    _cached_career_progression: Any = field(default=None, repr=False, compare=False)
    _cached_job_stability: Any = field(default=None, repr=False, compare=False)
    _cached_education_score: Any = field(default=None, repr=False, compare=False)
    _cached_consistency_score: Any = field(default=None, repr=False, compare=False)
    _cached_relevant_ai_years: Any = field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------
    # Helper Properties
    # ---------------------------------------------------------

    @property
    def all_text(self) -> str:
        """Combined searchable text."""

        history = " ".join(
            f"{item.get('title', '')} {item.get('description', '')}"
            for item in self.career_history
        )

        education = " ".join(
            f"{item.get('degree', '')} {item.get('field_of_study', '')}"
            for item in self.education
        )

        return " ".join(
            filter(
                None,
                [
                    self.headline,
                    self.summary,
                    self.current_title,
                    self.current_company,
                    " ".join(self.skills),
                    history,
                    education,
                ],
            )
        )

    @property
    def skill_count(self) -> int:
        return len(self.skills)

    @property
    def has_behavioral_signals(self) -> bool:
        return bool(self.signals)

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "headline": self.headline,
            "summary": self.summary,
            "location": self.location,
            "current_title": self.current_title,
            "current_company": self.current_company,
            "years_of_experience": self.years_of_experience,
            "skills": self.skills,
            "career_history": self.career_history,
            "education": self.education,
            "certifications": self.certifications,
            "signals": self.signals,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"Candidate("
            f"id='{self.candidate_id}', "
            f"title='{self.current_title}', "
            f"experience={self.years_of_experience:.1f}, "
            f"skills={len(self.skills)})"
        )