from pathlib import Path
from docx import Document


class JDLoader:
    """
    Loads a Job Description (.docx) file and returns its text.
    """

    def __init__(self, jd_path: str):
        self.jd_path = Path(jd_path)

        if not self.jd_path.exists():
            raise FileNotFoundError(
                f"Job description not found: {self.jd_path}"
            )

    def load(self) -> str:
        """
        Read the DOCX file and return the full text.
        """

        document = Document(self.jd_path)

        paragraphs = [
            para.text.strip()
            for para in document.paragraphs
            if para.text.strip()
        ]

        return "\n".join(paragraphs)