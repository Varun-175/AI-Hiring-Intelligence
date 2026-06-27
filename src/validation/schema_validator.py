import json
from pathlib import Path


class SchemaValidator:
    """
    Validates candidate records against the official JSON schema.

    Version 1:
    - Loads schema
    - Checks top-level required fields
    - Tracks validation statistics

    We'll make it stricter in later iterations.
    """

    def __init__(self, schema_path: str):
        self.schema_path = Path(schema_path)

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {self.schema_path}")

        with self.schema_path.open("r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.required_fields = self.schema.get("required", [])

        self.total_checked = 0
        self.valid = 0
        self.invalid = 0

    def validate(self, candidate: dict) -> bool:
        """
        Validate a single candidate.
        """

        self.total_checked += 1

        for field in self.required_fields:
            if field not in candidate:
                self.invalid += 1
                return False

        self.valid += 1
        return True

    def validate_dataset(self, loader):
        """
        Validate all candidates from a CandidateLoader.
        """

        for candidate in loader.load_candidates():
            self.validate(candidate)

    def get_statistics(self):
        return {
            "total_checked": self.total_checked,
            "valid": self.valid,
            "invalid": self.invalid,
        }