from typing import Dict, List, Tuple

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.utils.console import message
from src.utils.config import get_nested, load_yaml_config


class HybridRetriever:
    """
    Production Hybrid Retrieval Pipeline.

    Pipeline
    --------
        Stage 1 : BM25 Retrieval
        Stage 2 : Semantic Retrieval
        Stage 3 : Reciprocal Rank Fusion (Weighted)

    Designed for datasets from 100K to 1M+ candidates.
    """

    def __init__(self, candidates):

        config = load_yaml_config("configs/settings.yaml")

        self.candidates = candidates

        self.bm25 = BM25Retriever(candidates)

        self.enable_semantic = get_nested(config, ["pipeline", "enable_semantic"], True)
        self.semantic = None

        self.bm25_top_k = get_nested(
            config,
            ["retrieval", "bm25_top_k"],
            2000,
        )

        self.semantic_top_k = get_nested(
            config,
            ["retrieval", "semantic_top_k"],
            1000,
        )

        self.rrf_k = get_nested(
            config,
            ["retrieval", "rrf_k"],
            60,
        )

        self.bm25_weight = get_nested(
            config,
            ["retrieval", "bm25_weight"],
            1.0,
        )

        self.semantic_weight = get_nested(
            config,
            ["retrieval", "semantic_weight"],
            1.2,
        )

    # ---------------------------------------------------------
    # Query Builder
    # ---------------------------------------------------------

    @staticmethod
    def _build_query(job):

        return " ".join(
            filter(
                None,
                [
                    job.title,
                    " ".join(getattr(job, "required_skills", [])),
                    " ".join(getattr(job, "preferred_skills", [])),
                ],
            )
        )

    # ---------------------------------------------------------
    # Weighted Reciprocal Rank Fusion
    # ---------------------------------------------------------

    def _rrf(
        self,
        ranked_lists: List[List[Tuple[float, object]]],
        weights: List[float] = None,
    ):

        fused: Dict[str, Dict] = {}

        if weights is None:
            weights = [1.0] * len(ranked_lists)

        for results, weight in zip(ranked_lists, weights):

            for rank, (_, candidate) in enumerate(results, start=1):

                cid = candidate.candidate_id

                if cid not in fused:

                    fused[cid] = {
                        "candidate": candidate,
                        "score": 0.0,
                        "votes": 0,
                    }

                fused[cid]["score"] += (
                    weight / (self.rrf_k + rank)
                )

                fused[cid]["votes"] += 1

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -item["score"],
                -item["votes"],
                item["candidate"].candidate_id,
            ),
        )

        return [
            (item["score"], item["candidate"])
            for item in ranked
        ]

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        job,
        top_k: int = 500,
    ):
        """
        Hybrid Retrieval Pipeline

        Stage 1 : BM25
        Stage 2 : Semantic Search on BM25 shortlist
        Stage 3 : Weighted Reciprocal Rank Fusion
        """

        query = self._build_query(job)

        # ---------------------------------------------------------
        # Stage 1 : BM25
        # ---------------------------------------------------------

        message("• Stage 1 : BM25 Retrieval", tone="yellow")

        bm25_results = self.bm25.retrieve(
            query=query,
            top_k=self.bm25_top_k,
        )

        if not bm25_results:
            return []

        shortlist = [candidate for _, candidate in bm25_results]

        message(
            f"• BM25 shortlisted {len(shortlist):,} candidates",
            tone="yellow",
        )

        # ---------------------------------------------------------
        # Stage 2 : Semantic
        # ---------------------------------------------------------

        message("• Stage 2 : Semantic Retrieval", tone="yellow")

        if self.enable_semantic:
            self.semantic = SemanticRetriever(shortlist)
            semantic_results = self.semantic.retrieve(
                query=query,
                top_k=min(
                    self.semantic_top_k,
                    len(shortlist),
                ),
            )
        else:
            semantic_results = []

        # ---------------------------------------------------------
        # Stage 3 : Fusion
        # ---------------------------------------------------------

        message("• Stage 3 : Reciprocal Rank Fusion", tone="yellow")

        fused = self._rrf(
            [bm25_results, semantic_results],
            [self.bm25_weight, self.semantic_weight],
        )

        final = [
            candidate
            for _, candidate in fused[:top_k]
        ]

        message(f"• Final Retrieved : {len(final):,}", tone="yellow")

        return final
    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    @property
    def corpus_size(self):

        return len(self.candidates)

    @property
    def bm25_index_size(self):

        return self.bm25.corpus_size

    @property
    def semantic_enabled(self):

        return self.semantic is not None
