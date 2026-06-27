import json
from pathlib import Path


class CandidateLoader:
    """
    Streams candidate records from a JSONL file.

    Features:
    - Memory efficient
    - Handles malformed JSON
    - Tracks loading statistics
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

        self.total_records = 0
        self.valid_records = 0
        self.invalid_records = 0

    def load_candidates(self):
        """
        Generator that yields one candidate at a time.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        with self.dataset_path.open("r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                self.total_records += 1

                line = line.strip()

                if not line:
                    continue

                try:
                    candidate = json.loads(line)

                    self.valid_records += 1

                    yield candidate

                except json.JSONDecodeError as e:

                    self.invalid_records += 1

                    print(
                        f"[WARNING] Invalid JSON at line "
                        f"{line_number}: {e}"
                    )

    def get_statistics(self):
        """
        Returns loading statistics.
        """

        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records
        }