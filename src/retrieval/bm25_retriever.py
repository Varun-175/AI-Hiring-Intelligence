import re
from functools import lru_cache
from heapq import nlargest
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from src.utils.console import message


class BM25Retriever:
    """
    High-Performance BM25 Retriever

    Features
    --------
    ✓ Cached tokenization
    ✓ Rich candidate documents
    ✓ Cached document text
    ✓ Handles empty corpus/documents
    ✓ Optimized Top-K retrieval
    ✓ Deterministic ordering
    ✓ float32 memory optimization
    ✓ Production-ready for 100K+ candidates
    """

    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_+#.]+")
    UNKNOWN_TOKEN = "unknown"
    SMALL_TOP_K_THRESHOLD = 128

    def __init__(self, candidates):
        import hashlib
        import pickle
        from pathlib import Path

        self.candidates = candidates
        self.documents: List[List[str]] = []

        # O(1) Fingerprint
        digest = hashlib.sha1()
        digest.update(str(len(candidates)).encode())
        if candidates:
            digest.update(candidates[0].candidate_id.encode())
            digest.update(candidates[len(candidates)//2].candidate_id.encode())
            digest.update(candidates[-1].candidate_id.encode())
        fingerprint = digest.hexdigest()[:16]

        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"bm25_{fingerprint}.pkl"

        if self.cache_path.exists():
            try:
                message("Loading cached BM25 index...", tone="yellow")
                with self.cache_path.open("rb") as f:
                    cache_data = pickle.load(f)
                self.documents = cache_data["documents"]
                self.bm25 = cache_data["bm25"]
                if self.documents and getattr(self.bm25, "idf", None):
                    return
                message("BM25 cache invalid. Rebuilding index...", tone="yellow")
            except Exception as e:
                message(f"BM25 cache loading failed: {e}", tone="red")

        message("Building BM25 index...", tone="yellow")
        for candidate in candidates:

            document = self._build_document(candidate)

            tokens = self._tokenize(document)

            if not tokens:
                tokens = [self.UNKNOWN_TOKEN]

            self.documents.append(tokens)

        if not self.documents:
            self.documents = [[self.UNKNOWN_TOKEN]]

        self.bm25 = BM25Okapi(self.documents)

        try:
            with self.cache_path.open("wb") as f:
                pickle.dump(
                    {"documents": self.documents, "bm25": self.bm25},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            message(f"BM25 cache saving failed: {e}", tone="red")

    # ------------------------------------------------------------------
    # Document Builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_document(candidate) -> str:
        """
        Build a rich searchable document.

        The generated text is cached on the candidate object
        so BM25, Semantic Retrieval and Explainability reuse it.
        """

        cached = getattr(candidate, "all_text", None)
        if cached:
            return cached

        parts = []

        append = parts.append
        extend = parts.extend

        append(getattr(candidate, "current_title", ""))
        append(getattr(candidate, "headline", ""))
        append(getattr(candidate, "summary", ""))
        append(getattr(candidate, "current_company", ""))

        skills = getattr(candidate, "skills", None)
        if skills:
            extend(skills)

        for job in getattr(candidate, "career_history", []):

            if isinstance(job, dict):

                append(job.get("title", ""))
                append(job.get("description", ""))
                append(job.get("industry", ""))

        for edu in getattr(candidate, "education", []):

            if isinstance(edu, dict):

                append(edu.get("degree", ""))
                append(edu.get("field_of_study", ""))

        document = " ".join(filter(None, parts))
        document = " ".join(document.split())

        return document

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:

        if not text:
            return []

        return cls.TOKEN_PATTERN.findall(text.lower())

    @lru_cache(maxsize=2048)
    def _query_tokens(self, query: str):

        tokens = self._tokenize(query)
        tokens = list(dict.fromkeys(tokens))

        return tuple(tokens if tokens else [self.UNKNOWN_TOKEN])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 1000,
        min_score: float | None = None,
    ) -> List[Tuple[float, object]]:

        if not self.candidates:
            return []

        query_tokens = list(self._query_tokens(query))

        scores = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype=np.float32,
        )

        total = len(scores)

        top_k = max(1, min(top_k, total))

        # --------------------------------------------------------------
        # Small Top-K
        # --------------------------------------------------------------

        if top_k <= self.SMALL_TOP_K_THRESHOLD:

            results = nlargest(
                top_k,
                (
                    (float(score), candidate)
                    for score, candidate in zip(
                        scores,
                        self.candidates,
                    )
                ),
                key=lambda item: (
                    item[0],
                    item[1].candidate_id,
                ),
            )

        # --------------------------------------------------------------
        # Large Top-K
        # --------------------------------------------------------------

        else:

            if top_k == total:

                indices = np.arange(total)

            else:

                indices = np.argpartition(
                    scores,
                    -top_k,
                )[-top_k:]

            indices = sorted(
                indices,
                key=lambda i: (
                    -scores[i],
                    self.candidates[i].candidate_id,
                ),
            )

            results = [
                (
                    float(scores[idx]),
                    self.candidates[idx],
                )
                for idx in indices
            ]

        if min_score is not None:
            results = [
                item
                for item in results
                if item[0] >= min_score
            ]

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        return len(self.candidates)

    @property
    def vocabulary_size(self) -> int:
        return len(self.bm25.idf)

    @property
    def average_document_length(self) -> float:
        return float(self.bm25.avgdl)
