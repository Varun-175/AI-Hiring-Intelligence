import csv
from pathlib import Path


class SubmissionGenerator:
    """
    Generates the final submission CSV.
    """

    def __init__(self, output_path="outputs/submission.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, ranked_candidates):

        with open(self.output_path, "w", newline="", encoding="utf-8") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow([
                "rank",
                "candidate_id",
                "score",
                "reason"
            ])

            for rank, item in enumerate(ranked_candidates, start=1):

                candidate = item["candidate"]

                writer.writerow([
                    rank,
                    candidate.candidate_id,
                    round(item["score"], 2),
                    ""
                ])

        return self.output_path