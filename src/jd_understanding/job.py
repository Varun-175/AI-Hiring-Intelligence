from dataclasses import dataclass, field
from typing import List


@dataclass
class Job:
    """
    Canonical representation of a Job Description.
    This object will be used throughout the pipeline instead of
    passing raw JD text.
    """

    title: str = ""
    summary: str = ""

    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)

    experience_required: float = 0.0

    location: str = ""

    responsibilities: List[str] = field(default_factory=list)
    qualifications: List[str] = field(default_factory=list)