from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """
    Central runtime configuration for the AI Hiring Intelligence Engine.

    Shared by:
        • run.py
        • Retrieval
        • Ranking
        • Submission
        • Evaluation
    """

    # ---------------------------------------------------------
    # Input Files
    # ---------------------------------------------------------

    dataset_path: Path = Path("data/raw/candidates.jsonl")
    schema_path: Path = Path("data/raw/candidate_schema.json")
    jd_path: Path = Path("data/raw/job_description.docx")

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    output_path: Path = Path("outputs/submission.csv")
    cache_dir: Path = Path("data/cache")

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    bm25_top_k: int = 2000
    semantic_top_k: int = 500
    retrieval_top_k: int = 500

    # ---------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------

    rank_top_k: int = 100
    cross_encoder_top_k: int = 100
    final_top_k: int = 100

    # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------

    embedding_batch_size: int = 512
    cross_encoder_batch_size: int = 64
    num_workers: int = 4

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    use_embedding_cache: bool = True
    use_model_cache: bool = True

    # ---------------------------------------------------------
    # Debug
    # ---------------------------------------------------------

    verbose: bool = True

    @property
    def candidate_count_hint(self) -> int:
        """Expected dataset size."""
        return 100_000

    @property
    def is_large_dataset(self) -> bool:
        return self.candidate_count_hint >= 100_000

    def ensure_directories(self) -> None:
        """Create required runtime directories."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)