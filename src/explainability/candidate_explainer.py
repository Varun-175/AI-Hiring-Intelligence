from typing import List


class CandidateExplainer:
    """
    Generates recruiter-friendly explanations for ranked candidates.

    Features
    --------
    ✓ Evidence-based explanations
    ✓ No duplicate reasons
    ✓ Score-aware
    ✓ Compact output
    ✓ Reusable across UI and CSV
    """

    MAX_REASONS = 6

    @staticmethod
    def _add(reasons: List[str], text: str):

        if text and text not in reasons:
            reasons.append(text)

    @classmethod
    def explain(cls, item):

        candidate = item["candidate"]

        features = item.get("features", {})

        evidence = features.get("evidence", {})

        reasons = []

        # ----------------------------------
        # Technical Fit
        # ----------------------------------

        if features.get("technical", 0) >= 85:
            cls._add(reasons, "Excellent technical match")

        elif features.get("technical", 0) >= 70:
            cls._add(reasons, "Strong technical alignment")

        matched = evidence.get(
            "matched_required_skills",
            [],
        )

        required = evidence.get(
            "required_skill_count",
            0,
        )

        if matched and required:

            cls._add(
                reasons,
                f"Matched {len(matched)}/{required} required skills",
            )

        # ----------------------------------
        # Career
        # ----------------------------------

        years = evidence.get(
            "relevant_ai_years",
            candidate.years_of_experience,
        )

        if years:

            cls._add(
                reasons,
                f"{years:.1f} years relevant experience",
            )

        if candidate.current_title:

            cls._add(
                reasons,
                f"Current role: {candidate.current_title}",
            )

        # ----------------------------------
        # Behaviour
        # ----------------------------------

        if features.get("behavioral", 0) >= 75:

            cls._add(
                reasons,
                "Strong recruiter engagement",
            )

        if evidence.get("open_to_work"):

            cls._add(
                reasons,
                "Open to work",
            )

        # ----------------------------------
        # Consistency
        # ----------------------------------

        if features.get("consistency", 0) >= 90:

            cls._add(
                reasons,
                "Highly consistent profile",
            )

        # ----------------------------------
        # Confidence
        # ----------------------------------

        confidence = features.get(
            "confidence",
            0,
        )

        if confidence >= 90:

            cls._add(
                reasons,
                "High recruiter confidence",
            )

        elif confidence >= 75:

            cls._add(
                reasons,
                "Strong overall profile",
            )

        # ----------------------------------
        # CrossEncoder
        # ----------------------------------

        cross = item.get(
            "cross_score_normalized",
            0,
        )

        if cross >= 85:

            cls._add(
                reasons,
                "Excellent semantic job match",
            )

        # ----------------------------------
        # Education
        # ----------------------------------

        education = features.get(
            "education",
            0,
        )

        if education >= 80:

            cls._add(
                reasons,
                "Relevant educational background",
            )

        if not reasons:

            reasons.append(
                "Relevant candidate profile"
            )

        return "; ".join(
            reasons[: cls.MAX_REASONS]
        )