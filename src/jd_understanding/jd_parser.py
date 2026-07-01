import re

from src.jd_understanding.job import Job


class JDParser:
    """
    Parses an unstructured Job Description into a structured Job object.

    Extracts:
        ✓ Title
        ✓ Experience
        ✓ Location
        ✓ Required Skills
        ✓ Preferred Skills
        ✓ Summary
    """

    SKILL_KEYWORDS = {
        "python",
        "java",
        "sql",
        "spark",
        "pyspark",
        "airflow",
        "kafka",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "terraform",
        "mlflow",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "ai",
        "nlp",
        "computer vision",
        "rag",
        "llm",
        "large language models",
        "langchain",
        "llamaindex",
        "transformers",
        "huggingface",
        "fine tuning",
        "fine-tuning",
        "lora",
        "qlora",
        "embeddings",
        "retrieval",
        "ranking",
        "vector database",
        "vector search",
        "faiss",
        "milvus",
        "pinecone",
        "chromadb",
        "weaviate",
        "fastapi",
        "flask",
        "bentoml",
        "ray",
        "xgboost",
        "lightgbm",
    }

    EXPERIENCE_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]

    TITLE_PATTERNS = [
        r"^\s*job\s*title\s*:?\s*(.+)$",
        r"^\s*position\s*:?\s*(.+)$",
        r"^\s*role\s*:?\s*(.+)$",
        r"^\s*job\s*description\s*:?\s*(.+)$",
    ]

    LOCATION_PATTERNS = [
        r"location\s*:?\s*(.+)",
        r"work\s*location\s*:?\s*(.+)",
    ]

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join((text or "").split())

    @classmethod
    def _extract_title(cls, text):

        for pattern in cls.TITLE_PATTERNS:

            match = re.search(
                pattern,
                text,
                flags=re.I | re.M,
            )

            if match:
                return cls._clean(match.group(1))

        first_line = text.splitlines()[0].strip()

        if len(first_line) < 120:
            return first_line

        return ""

    @classmethod
    def _extract_location(cls, text):

        for pattern in cls.LOCATION_PATTERNS:

            match = re.search(
                pattern,
                text,
                flags=re.I,
            )

            if match:
                return cls._clean(match.group(1))

        return ""

    @classmethod
    def _extract_experience(cls, text):

        for pattern in cls.EXPERIENCE_PATTERNS:

            match = re.search(
                pattern,
                text,
                flags=re.I,
            )

            if not match:
                continue

            if len(match.groups()) == 2:

                low = float(match.group(1))
                high = float(match.group(2))

                return round((low + high) / 2, 1)

            return float(match.group(1))

        return 0.0

    @classmethod
    def _extract_skills(cls, text):

        text = text.lower()

        skills = []

        for skill in sorted(
            cls.SKILL_KEYWORDS,
            key=len,
            reverse=True,
        ):

            if skill in text:
                skills.append(skill)

        seen = set()

        ordered = []

        for skill in skills:

            if skill not in seen:
                ordered.append(skill)
                seen.add(skill)

        return ordered

    @staticmethod
    def _extract_preferred(text):

        preferred = []

        lower = text.lower()

        preferred_section = re.search(
            r"(preferred qualifications|good to have|nice to have)(.*)",
            lower,
            flags=re.S,
        )

        if not preferred_section:
            return preferred

        block = preferred_section.group(2)

        for line in block.splitlines():

            line = line.strip("-•* ")

            if line:
                preferred.append(line)

        return preferred

    def parse(self, jd_text: str):

        job = Job()

        job.summary = self._clean(jd_text)

        # Extract using the original text to preserve lines/newlines
        job.title = self._extract_title(jd_text)

        job.location = self._extract_location(jd_text)

        job.experience_required = self._extract_experience(
            jd_text,
        )

        job.required_skills = self._extract_skills(
            jd_text,
        )

        job.preferred_skills = self._extract_preferred(
            jd_text,
        )

        return job
