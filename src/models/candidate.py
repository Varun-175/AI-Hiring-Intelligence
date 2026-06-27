from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Candidate:
    """
    Canonical representation of a candidate.
    Every module after preprocessing will use this object.
    """

    candidate_id: str

    # Profile
    name: str
    headline: str
    summary: str
    location: str

    # Current Role
    current_title: str
    current_company: str
    years_of_experience: float

    # Collections
    skills: List[str] = field(default_factory=list)
    career_history: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)

    # Behavioural Signals
    signals: Dict[str, Any] = field(default_factory=dict)