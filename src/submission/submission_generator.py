import csv
from pathlib import Path


class SubmissionGenerator:
    """
    Production Submission Generator.

    Generates a deterministic CSV containing:

        rank
        candidate_id
        score
        reason

    Features
    --------
    ✓ Stable output
    ✓ Evidence-based recruiter explanations
    ✓ Safe against missing fields
    ✓ Duplicate removal
    ✓ Configurable explanation length
    """

    MAX_REASON_LENGTH = 300

    def __init__(
        self,
        output_path="outputs/submission.csv",
    ):

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Explanation Builder
    # ---------------------------------------------------------

    @staticmethod
    def _add(reasons, text):

        if text and text not in reasons:
            reasons.append(text)

    def _generate_reason(self, item):

        candidate = item["candidate"]

        features = item.get("features", {})

        evidence = features.get("evidence", {})

        reasons = []

        # -------------------------------------------------
        # Skills
        # -------------------------------------------------

        matched = evidence.get(
            "matched_required_skills",
            [],
        )

        required = evidence.get(
            "required_skill_count",
            0,
        )

        if matched and required:

            shown = ", ".join(matched[:6])

            self._add(
                reasons,
                f"Matched {len(matched)}/{required} required skills ({shown})",
            )

        # -------------------------------------------------
        # Experience
        # -------------------------------------------------

        years = evidence.get(
            "relevant_ai_years",
            0,
        )

        if years >= 1:

            self._add(
                reasons,
                f"{years:.1f} years relevant AI/ML experience",
            )

        # -------------------------------------------------
        # Current Role
        # -------------------------------------------------

        if candidate.current_title:

            self._add(
                reasons,
                f"Current role: {candidate.current_title}",
            )

        # -------------------------------------------------
        # Cloud
        # -------------------------------------------------

        cloud = evidence.get(
            "cloud_skills",
            [],
        )

        if cloud:

            self._add(
                reasons,
                f"Cloud: {', '.join(cloud)}",
            )

        # -------------------------------------------------
        # Recruiter Signals
        # -------------------------------------------------

        response = evidence.get(
            "recruiter_response_rate",
        )

        if response is not None and response >= 0.5:

            self._add(
                reasons,
                f"High recruiter response ({response:.0%})",
            )

        completeness = evidence.get(
            "profile_completeness_score",
        )

        if completeness is not None and completeness >= 85:

            self._add(
                reasons,
                f"Profile completeness {completeness:.0f}%",
            )

        github = evidence.get(
            "github_activity_score",
        )

        if github is not None and github >= 60:

            self._add(
                reasons,
                f"Strong GitHub activity ({github:.0f}/100)",
            )

        if evidence.get("open_to_work"):

            self._add(
                reasons,
                "Open to work",
            )

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence = features.get(
            "confidence",
            0,
        )

        if confidence >= 90:

            self._add(
                reasons,
                "High recruiter confidence",
            )

        elif confidence >= 75:

            self._add(
                reasons,
                "Strong overall profile",
            )

        # -------------------------------------------------
        # Consistency
        # -------------------------------------------------

        if features.get(
            "consistency",
            0,
        ) >= 90:

            self._add(
                reasons,
                "Consistent career history",
            )

        # -------------------------------------------------
        # Technical
        # -------------------------------------------------

        if features.get(
            "technical",
            0,
        ) >= 85:

            self._add(
                reasons,
                "Excellent technical fit",
            )

        # -------------------------------------------------
        # Fallback
        # -------------------------------------------------

        if not reasons:

            reasons.append(
                "Relevant candidate profile"
            )

        explanation = "; ".join(reasons)

        if len(explanation) > self.MAX_REASON_LENGTH:

            explanation = (
                explanation[
                    : self.MAX_REASON_LENGTH - 3
                ]
                + "..."
            )

        return explanation

    # ---------------------------------------------------------
    # CSV Generation
    # ---------------------------------------------------------

    def generate(self, ranked_candidates):
        ranked_candidates = sorted(
            ranked_candidates,
            key=lambda item: (
                -round(float(item["score"]), 2),
                item["candidate"].candidate_id,
            ),
        )

        with self.output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:

            writer = csv.writer(
                csvfile,
                quoting=csv.QUOTE_MINIMAL,
            )

            writer.writerow(
                [
                    "candidate_id",
                    "rank",
                    "score",
                    "reasoning",
                ]
            )

            for rank, item in enumerate(
                ranked_candidates,
                start=1,
            ):

                writer.writerow(
                    [
                        item["candidate"].candidate_id,
                        rank,
                        round(
                            float(item["score"]),
                            2,
                        ),
                        self._generate_reason(item),
                    ]
                )

        return self.output_path
