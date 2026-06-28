import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


from src.utils.config import get_nested, load_yaml_config


class SemanticRetriever:
    """
    Production Semantic Retriever.

    Features
    --------
    ✓ Singleton model loading
    ✓ Memory-mapped embedding cache
    ✓ Cached query embeddings
    ✓ float32 optimization
    ✓ Normalized cosine similarity
    ✓ Candidate text reuse
    ✓ Ready for FAISS migration
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    _MODEL = None

    def __init__(self, candidates):

        config = load_yaml_config("configs/settings.yaml")

        self.candidates = candidates

        self.model_name = get_nested(
            config,
            ["models", "semantic_retriever"],
            self.DEFAULT_MODEL,
        )

        self.local_files_only = get_nested(
            config,
            ["models", "local_files_only"],
            True,
        )

        self.batch_size = get_nested(
            config,
            ["retrieval", "embedding_batch_size"],
            512,
        )

        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_path = (
            self.cache_dir /
            f"semantic_{self._fingerprint()}.npy"
        )

        self.candidate_id_to_idx = {
            candidate.candidate_id: i
            for i, candidate in enumerate(candidates)
        }

        self.model = self._load_model()

        if self.model is None:
            self.embeddings = None
            return

        self.embeddings = self._load_embeddings()

    # ---------------------------------------------------------
    # Candidate Text
    # ---------------------------------------------------------

    @staticmethod
    def _candidate_text(candidate):
        return candidate.all_text

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def _load_model(self):

        if SemanticRetriever._MODEL is not None:
            return SemanticRetriever._MODEL

        try:

            SemanticRetriever._MODEL = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
            )

            return SemanticRetriever._MODEL

        except Exception as exc:

            print(f"[Semantic] Disabled : {exc}")

            return None

    # ---------------------------------------------------------
    # Fingerprint
    # ---------------------------------------------------------

    def _fingerprint(self):

        digest = hashlib.sha1()

        digest.update(self.model_name.encode())

        digest.update(str(len(self.candidates)).encode())

        if self.candidates:
            digest.update(self.candidates[0].candidate_id.encode())
            digest.update(self.candidates[len(self.candidates)//2].candidate_id.encode())
            digest.update(self.candidates[-1].candidate_id.encode())

        return digest.hexdigest()[:16]

    # ---------------------------------------------------------
    # Encode
    # ---------------------------------------------------------

    def _encode(self):
        texts = [
            self._candidate_text(candidate)
            for candidate in self.candidates
        ]
        embeddings = self.model.encode(

            texts,

            batch_size=self.batch_size,

            normalize_embeddings=True,

            convert_to_numpy=True,

            show_progress_bar=True,

        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
            order="C",
        )

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    def _load_embeddings(self):

        if self.cache_path.exists():

            try:

                print("Loading cached embeddings...")

                embeddings = np.load(
                    self.cache_path,
                    mmap_mode=None,
                )

                if (
                    embeddings.ndim == 2
                    and embeddings.shape[0] == len(self.candidates)
                ):

                    return embeddings

                print("Embedding cache mismatch.")

            except Exception:

                print("Embedding cache corrupted.")

        print("Embedding cache not found. Will encode candidates on-the-fly during retrieve.")
        return None

    # ---------------------------------------------------------
    # Query Embedding Cache
    # ---------------------------------------------------------

    @lru_cache(maxsize=256)
    def _encode_query(self, query):

        return self.model.encode(

            query,

            normalize_embeddings=True,

            convert_to_numpy=True,

        ).astype(np.float32)

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query,
        candidates=None,
        top_k=1000,
    ) -> List[Tuple[float, object]]:

        if self.model is None:
            cands = candidates if candidates is not None else self.candidates
            return [
                (0.0, candidate)
                for candidate in cands[:top_k]
            ]

        query_embedding = self._encode_query(query)

        if self.embeddings is not None:
            if candidates is not None:
                corpus_indices = [
                    self.candidate_id_to_idx[c.candidate_id]
                    for c in candidates
                    if c.candidate_id in self.candidate_id_to_idx
                ]
                if corpus_indices:
                    candidate_embeddings = self.embeddings[corpus_indices]
                    scores = candidate_embeddings @ query_embedding
                    scored_candidates = [candidates[i] for i in range(len(corpus_indices))]
                else:
                    scores = np.array([], dtype=np.float32)
                    scored_candidates = []
            else:
                scores = self.embeddings @ query_embedding
                scored_candidates = self.candidates
        else:
            scored_candidates = candidates if candidates is not None else self.candidates
            if scored_candidates:
                texts = [self._candidate_text(c) for c in scored_candidates]
                candidate_embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                scores = candidate_embeddings @ query_embedding
            else:
                scores = np.array([], dtype=np.float32)

        if scores.size == 0:
            return []

        top_k = min(
            len(scores),
            max(1, top_k),
        )

        if top_k == len(scores):

            indices = np.argsort(scores)[::-1]

        else:

            indices = np.argpartition(
                scores,
                -top_k,
            )[-top_k:]

            indices = indices[
                np.argsort(
                    scores[indices]
                )[::-1]
            ]

        return [
            (
                float(scores[index]),
                scored_candidates[index],
            )
            for index in indices
        ]

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    @property
    def corpus_size(self):

        return len(self.candidates)

    @property
    def embedding_dimension(self):

        if self.embeddings is None:
            return 0

        return self.embeddings.shape[1]