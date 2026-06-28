from pathlib import Path

try:
    import orjson

    def loads(data: str):
        return orjson.loads(data)

except ImportError:
    import json

    def loads(data: str):
        return json.loads(data)


class CandidateLoader:
    """
    Streams candidate records from a JSONL dataset.

    Features
    --------
    ✓ Memory efficient
    ✓ Streaming generator
    ✓ Optional orjson acceleration
    ✓ Robust error handling
    ✓ Loading statistics
    """

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

        self.reset_statistics()

    # ----------------------------------------------------
    # Statistics
    # ----------------------------------------------------

    def reset_statistics(self):

        self.total_records = 0
        self.valid_records = 0
        self.invalid_records = 0

    # ----------------------------------------------------
    # Iterator
    # ----------------------------------------------------

    def __iter__(self):

        return self.load_candidates()

    # ----------------------------------------------------
    # Candidate Loader
    # ----------------------------------------------------

    def load_candidates(self):

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        with self.dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                self.total_records += 1

                line = line.strip()

                if not line:
                    continue

                try:

                    candidate = loads(line)

                    if not isinstance(candidate, dict):

                        self.invalid_records += 1
                        continue

                    self.valid_records += 1

                    yield candidate

                except Exception as exc:

                    self.invalid_records += 1

                    print(
                        f"[Loader] Invalid JSON "
                        f"(Line {line_number}): {exc}"
                    )

    # ----------------------------------------------------
    # Statistics
    # ----------------------------------------------------

    def get_statistics(self):

        success_rate = (
            (self.valid_records / self.total_records) * 100
            if self.total_records
            else 0.0
        )

        return {
            "dataset": str(self.dataset_path),
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "success_rate": round(success_rate, 2),
        }