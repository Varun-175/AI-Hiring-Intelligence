from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class JobProfile:
    """
    Reusable domain model representing a structured Job Description.

    Shared by parsing, retrieval, ranking, evaluation and APIs.
    """

    # ---------------------------------------------------------
    # Basic Information
    # ---------------------------------------------------------

    title: str = ""
    summary: str = ""
    location: str = ""

    # ---------------------------------------------------------
    # Skills
    # ---------------------------------------------------------

    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Experience
    # ---------------------------------------------------------

    experience_required: float = 0.0

    # ---------------------------------------------------------
    # JD Sections
    # ---------------------------------------------------------

    responsibilities: List[str] = field(default_factory=list)
    qualifications: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Optional Metadata
    # ---------------------------------------------------------

    employment_type: str = ""
    seniority_level: str = ""
    department: str = ""
    industry: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Helper Properties
    # ---------------------------------------------------------

    @property
    def all_skills(self) -> List[str]:
        """Returns required + preferred skills without duplicates."""
        return list(
            dict.fromkeys(
                self.required_skills + self.preferred_skills
            )
        )

    @property
    def skill_count(self) -> int:
        return len(self.all_skills)

    @property
    def has_preferred_skills(self) -> bool:
        return bool(self.preferred_skills)

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "location": self.location,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "experience_required": self.experience_required,
            "responsibilities": self.responsibilities,
            "qualifications": self.qualifications,
            "employment_type": self.employment_type,
            "seniority_level": self.seniority_level,
            "department": self.department,
            "industry": self.industry,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"JobProfile("
            f"title='{self.title}', "
            f"experience={self.experience_required}, "
            f"skills={self.skill_count})"
        )