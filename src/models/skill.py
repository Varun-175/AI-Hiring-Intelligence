from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True, frozen=True)
class Skill:
    """
    Canonical representation of a candidate skill.

    Immutable, memory-efficient, and reusable throughout
    retrieval, ranking, and explainability.
    """

    name: str

    proficiency: str = ""

    endorsements: int = 0

    duration_months: int = 0

    category: str = ""

    verified: bool = False

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Skill":

        return cls(
            name=str(raw.get("name", "")).strip(),
            proficiency=str(raw.get("proficiency", "")).strip(),
            endorsements=max(
                0,
                int(raw.get("endorsements") or 0),
            ),
            duration_months=max(
                0,
                int(raw.get("duration_months") or 0),
            ),
            category=str(raw.get("category", "")).strip(),
            verified=bool(raw.get("verified", False)),
        )

    @property
    def years(self) -> float:
        """Experience with this skill in years."""
        return round(self.duration_months / 12.0, 2)

    @property
    def proficiency_score(self) -> int:
        """
        Converts proficiency into a numeric score.
        """

        mapping = {
            "beginner": 25,
            "intermediate": 50,
            "advanced": 75,
            "expert": 100,
        }

        return mapping.get(
            self.proficiency.lower(),
            50,
        )

    @property
    def strength(self) -> float:
        """
        Overall confidence score for this skill.
        """

        endorsement_score = min(
            self.endorsements,
            100,
        )

        experience_score = min(
            self.years * 10,
            100,
        )

        return round(
            self.proficiency_score * 0.50
            + endorsement_score * 0.20
            + experience_score * 0.30,
            2,
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "proficiency": self.proficiency,
            "endorsements": self.endorsements,
            "duration_months": self.duration_months,
            "category": self.category,
            "verified": self.verified,
        }

    def __repr__(self) -> str:

        return (
            f"Skill("
            f"name='{self.name}', "
            f"proficiency='{self.proficiency}', "
            f"years={self.years})"
        )