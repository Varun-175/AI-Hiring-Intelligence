from typing import Iterable, Sequence
import math


class RankingMetrics:
    """
    Standard Information Retrieval metrics.

    Metrics
    -------
    ✓ Precision@K
    ✓ Recall@K
    ✓ F1@K
    ✓ HitRate@K
    ✓ Mean Reciprocal Rank (MRR)
    ✓ Average Precision (AP)
    ✓ Normalized Discounted Cumulative Gain (NDCG)
    """

    @staticmethod
    def precision_at_k(
        relevant_ids: Iterable[str],
        ranked_ids: Sequence[str],
        k: int,
    ) -> float:

        if k <= 0:
            return 0.0

        relevant = set(relevant_ids)

        ranked = ranked_ids[:k]

        if not ranked:
            return 0.0

        hits = sum(
            candidate in relevant
            for candidate in ranked
        )

        return hits / len(ranked)

    @staticmethod
    def recall_at_k(
        relevant_ids,
        ranked_ids,
        k,
    ):

        relevant = set(relevant_ids)

        if not relevant:
            return 0.0

        ranked = ranked_ids[:k]

        hits = sum(
            candidate in relevant
            for candidate in ranked
        )

        return hits / len(relevant)

    @classmethod
    def f1_at_k(
        cls,
        relevant_ids,
        ranked_ids,
        k,
    ):

        precision = cls.precision_at_k(
            relevant_ids,
            ranked_ids,
            k,
        )

        recall = cls.recall_at_k(
            relevant_ids,
            ranked_ids,
            k,
        )

        if precision + recall == 0:
            return 0.0

        return (
            2
            * precision
            * recall
            / (precision + recall)
        )

    @staticmethod
    def hit_rate_at_k(
        relevant_ids,
        ranked_ids,
        k,
    ):

        relevant = set(relevant_ids)

        return float(
            any(
                candidate in relevant
                for candidate in ranked_ids[:k]
            )
        )

    @staticmethod
    def mean_reciprocal_rank(
        relevant_ids,
        ranked_ids,
    ):

        relevant = set(relevant_ids)

        for rank, candidate in enumerate(
            ranked_ids,
            start=1,
        ):

            if candidate in relevant:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def average_precision(
        relevant_ids,
        ranked_ids,
    ):

        relevant = set(relevant_ids)

        if not relevant:
            return 0.0

        score = 0.0
        hits = 0

        for rank, candidate in enumerate(
            ranked_ids,
            start=1,
        ):

            if candidate in relevant:

                hits += 1

                score += hits / rank

        return score / len(relevant)

    @staticmethod
    def ndcg_at_k(
        relevant_ids,
        ranked_ids,
        k,
    ):

        relevant = set(relevant_ids)

        dcg = 0.0

        for i, candidate in enumerate(
            ranked_ids[:k],
            start=1,
        ):

            if candidate in relevant:

                dcg += 1 / math.log2(i + 1)

        ideal = min(
            len(relevant),
            k,
        )

        if ideal == 0:
            return 0.0

        idcg = sum(
            1 / math.log2(i + 1)
            for i in range(
                1,
                ideal + 1,
            )
        )

        return dcg / idcg