from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass(slots=True)  # cache fields included below for performance
class Job:
    """
    Canonical representation of a Job Description.

    This object is shared across the entire pipeline and stores all
    structured information extracted from the JD.
    """

    # ---------------------------------------------------------
    # Core Information
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
    # Additional JD Sections
    # ---------------------------------------------------------

    responsibilities: List[str] = field(default_factory=list)
    qualifications: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Optional Metadata
    # ---------------------------------------------------------

    employment_type: str = ""
    seniority_level: str = ""
    industry: str = ""
    department: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Performance cache (populated lazily by feature modules)
    # ---------------------------------------------------------

    _cached_required_skills_set: Any = field(default=None, repr=False, compare=False)
    _cached_preferred_skills_set: Any = field(default=None, repr=False, compare=False)
    _cached_title_tokens: Any = field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------

    @property
    def all_skills(self) -> List[str]:
        """Returns required + preferred skills without duplicates."""

        return list(
            dict.fromkeys(
                self.required_skills + self.preferred_skills
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Job object."""

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
            "industry": self.industry,
            "department": self.department,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"Job("
            f"title='{self.title}', "
            f"experience={self.experience_required}, "
            f"required_skills={len(self.required_skills)}, "
            f"preferred_skills={len(self.preferred_skills)})"
        )