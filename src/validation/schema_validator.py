import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


class SchemaValidator:
    """
    Lightweight schema validator for candidate records.

    Features
    --------
    ✓ Cached schema loading
    ✓ Fast required-field validation
    ✓ Type validation
    ✓ Nested object validation
    ✓ Validation statistics
    ✓ Error reporting
    """

    REQUIRED_PROFILE_FIELDS = (
        "current_title",
        "years_of_experience",
    )

    def __init__(self, schema_path: str):

        self.schema = self._load_schema(schema_path)

        self.required_fields = tuple(
            self.schema.get("required", [])
        )

        self.total_checked = 0
        self.valid = 0
        self.invalid = 0

        self.errors: List[str] = []

    # ---------------------------------------------------------
    # Schema Loading
    # ---------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_schema(path: str) -> Dict[str, Any]:

        schema_path = Path(path)

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema not found: {schema_path}"
            )

        with schema_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate(
        self,
        candidate: Dict[str, Any],
    ) -> bool:

        self.total_checked += 1

        # -------------------------
        # Top-level required fields
        # -------------------------

        for field in self.required_fields:

            if field not in candidate:

                self.invalid += 1

                self.errors.append(
                    f"Missing field: {field}"
                )

                return False

        # -------------------------
        # Profile
        # -------------------------

        profile = candidate.get("profile")

        if not isinstance(profile, dict):

            self.invalid += 1

            self.errors.append(
                "Invalid profile object"
            )

            return False

        for field in self.REQUIRED_PROFILE_FIELDS:

            if field not in profile:

                self.invalid += 1

                self.errors.append(
                    f"Missing profile.{field}"
                )

                return False

        # -------------------------
        # Skills
        # -------------------------

        skills = candidate.get("skills", [])

        if not isinstance(skills, list):

            self.invalid += 1

            self.errors.append(
                "skills must be a list"
            )

            return False

        # -------------------------
        # Career History
        # -------------------------

        career = candidate.get(
            "career_history",
            [],
        )

        if not isinstance(career, list):

            self.invalid += 1

            self.errors.append(
                "career_history must be a list"
            )

            return False

        # -------------------------
        # Experience
        # -------------------------

        try:

            float(
                profile.get(
                    "years_of_experience",
                    0,
                )
            )

        except (TypeError, ValueError):

            self.invalid += 1

            self.errors.append(
                "Invalid years_of_experience"
            )

            return False

        self.valid += 1

        return True

    # ---------------------------------------------------------
    # Dataset Validation
    # ---------------------------------------------------------

    def validate_dataset(self, loader):

        for candidate in loader.load_candidates():

            self.validate(candidate)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def get_statistics(self):

        return {
            "total_checked": self.total_checked,
            "valid": self.valid,
            "invalid": self.invalid,
            "success_rate": (
                round(
                    self.valid / self.total_checked * 100,
                    2,
                )
                if self.total_checked
                else 0.0
            ),
        }

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def reset(self):

        self.total_checked = 0
        self.valid = 0
        self.invalid = 0
        self.errors.clear()