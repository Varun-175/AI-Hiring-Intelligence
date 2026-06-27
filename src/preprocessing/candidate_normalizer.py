from src.models.candidate import Candidate


class CandidateNormalizer:
    """
    Converts raw JSON candidate objects into
    normalized Candidate domain objects.
    """

    @staticmethod
    def normalize(candidate: dict) -> Candidate:

        profile = candidate.get("profile", {})

        return Candidate(

            candidate_id=candidate.get("candidate_id", ""),

            name=profile.get("anonymized_name", "").strip(),

            headline=profile.get("headline", "").strip(),

            summary=profile.get("summary", "").strip(),

            location=profile.get("location", "").strip(),

            current_title=profile.get("current_title", "").strip(),

            current_company=profile.get("current_company", "").strip(),

            years_of_experience=float(
                profile.get("years_of_experience", 0)
            ),

            skills=[
                skill.get("name", "").strip()
                for skill in candidate.get("skills", [])
                if skill.get("name")
            ],

            career_history=candidate.get(
                "career_history",
                []
            ),

            education=candidate.get(
                "education",
                []
            ),

            certifications=candidate.get(
                "certifications",
                []
            ),

            signals=candidate.get(
                "behavioral_signals",
                {}
            )
        )