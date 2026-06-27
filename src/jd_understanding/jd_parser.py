import re

from src.jd_understanding.job import Job


class JDParser:
    """
    Parses raw Job Description text into a structured Job object.
    """

    def parse(self, jd_text: str) -> Job:

        job = Job()

        # Store complete JD
        job.summary = jd_text.strip()

        # -----------------------------
        # Job Title
        # -----------------------------
        title_match = re.search(
            r"Job Description:\s*(.+)",
            jd_text,
            re.IGNORECASE,
        )

        if title_match:
            job.title = title_match.group(1).strip()

        # -----------------------------
        # Experience
        # -----------------------------
        exp_match = re.search(
            r"Experience Required:\s*(\d+)\s*[-–]\s*(\d+)",
            jd_text,
            re.IGNORECASE,
        )

        if exp_match:
            low = int(exp_match.group(1))
            high = int(exp_match.group(2))

            # Average required experience
            job.experience_required = (low + high) / 2

        # -----------------------------
        # Location
        # -----------------------------
        loc_match = re.search(
            r"Location:\s*(.+)",
            jd_text,
            re.IGNORECASE,
        )

        if loc_match:
            job.location = loc_match.group(1).strip()

        # -----------------------------
        # Initial Skill Extraction
        # -----------------------------

        skill_keywords = [
            "Python",
            "Java",
            "SQL",
            "Spark",
            "Airflow",
            "Kafka",
            "Docker",
            "Kubernetes",
            "AWS",
            "Azure",
            "GCP",
            "PyTorch",
            "TensorFlow",
            "LLM",
            "RAG",
            "Machine Learning",
            "Deep Learning",
            "Retrieval",
            "Ranking",
            "Embeddings",
            "Vector Database",
            "NLP",
        ]

        jd_lower = jd_text.lower()

        for skill in skill_keywords:
            if skill.lower() in jd_lower:
                job.required_skills.append(skill)

        return job