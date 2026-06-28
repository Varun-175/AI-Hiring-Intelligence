from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(slots=True, frozen=True)
class CareerEntry:
    """
    Normalized career history record.

    Immutable and memory-efficient representation of a
    candidate's employment history.
    """

    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: Optional[str] = None

    duration_months: int = 0

    is_current: bool = False

    industry: str = ""
    company_size: str = ""

    location: str = ""

    description: str = ""

    employment_type: str = ""

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "CareerEntry":

        return cls(
            company=str(raw.get("company", "")).strip(),
            title=str(raw.get("title", "")).strip(),
            start_date=str(raw.get("start_date", "")).strip(),
            end_date=raw.get("end_date"),
            duration_months=max(
                0,
                int(raw.get("duration_months") or 0),
            ),
            is_current=bool(raw.get("is_current", False)),
            industry=str(raw.get("industry", "")).strip(),
            company_size=str(raw.get("company_size", "")).strip(),
            location=str(raw.get("location", "")).strip(),
            description=str(raw.get("description", "")).strip(),
            employment_type=str(raw.get("employment_type", "")).strip(),
        )

    @property
    def years(self) -> float:
        return round(self.duration_months / 12.0, 2)

    @property
    def seniority(self) -> str:

        title = self.title.lower()

        if "principal" in title:
            return "Principal"

        if "staff" in title:
            return "Staff"

        if "architect" in title:
            return "Architect"

        if "lead" in title:
            return "Lead"

        if "senior" in title:
            return "Senior"

        if "manager" in title:
            return "Manager"

        return "Individual Contributor"

    @property
    def is_ai_role(self) -> bool:

        text = f"{self.title} {self.description}".lower()

        keywords = (
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "ai",
            "llm",
            "rag",
            "nlp",
            "computer vision",
            "data science",
            "ml engineer",
            "ai engineer",
        )

        return any(keyword in text for keyword in keywords)

    @property
    def is_recent(self) -> bool:

        if self.is_current:
            return True

        if not self.end_date:
            return False

        try:
            end = datetime.fromisoformat(self.end_date[:10])
            return (datetime.now() - end).days <= 365 * 3
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:

        return {
            "company": self.company,
            "title": self.title,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "duration_months": self.duration_months,
            "is_current": self.is_current,
            "industry": self.industry,
            "company_size": self.company_size,
            "location": self.location,
            "description": self.description,
            "employment_type": self.employment_type,
        }