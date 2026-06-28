import numpy as np
from sentence_transformers import CrossEncoder

from src.utils.config import get_nested, load_yaml_config


class CrossEncoderReranker:
    """
    Stage-3 reranking using a CrossEncoder.

    Operates only on the highest-ranked candidates from the
    heuristic ranker and blends semantic relevance with the
    heuristic score.
    """

    def __init__(self):

        settings = load_yaml_config("configs/settings.yaml")
        weights = load_yaml_config("configs/weights.yaml")

        self.batch_size = get_nested(
            settings,
            ["ranking", "cross_encoder_batch_size"],
            64,
        )

        self.max_length = get_nested(
            settings,
            ["ranking", "cross_encoder_max_length"],
            256,
        )

        self.cross_top_k = get_nested(
            settings,
            ["ranking", "cross_encoder_top_k"],
            100,
        )

        self.heuristic_weight = get_nested(
            weights,
            ["reranking", "heuristic"],
            0.60,
        )

        self.cross_weight = get_nested(
            weights,
            ["reranking", "cross_encoder"],
            0.40,
        )

        model_name = get_nested(
            settings,
            ["models", "cross_encoder"],
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        local_files_only = get_nested(
            settings,
            ["models", "local_files_only"],
            True,
        )

        try:

            self.model = CrossEncoder(
                model_name,
                max_length=self.max_length,
                local_files_only=local_files_only,
            )

        except Exception as exc:

            print(f"[CrossEncoder] Disabled : {exc}")

            self.model = None

    @staticmethod
    def _normalize(scores):

        scores = np.asarray(scores, dtype=np.float32)

        if scores.size == 0:
            return scores

        low = np.percentile(scores, 5)
        high = np.percentile(scores, 95)

        if np.isclose(low, high):
            return np.full(scores.shape, 50.0, dtype=np.float32)

        scores = (scores - low) / (high - low)
        scores = np.clip(scores, 0.0, 1.0)

        return np.round(scores * 100.0, 2)

    @staticmethod
    def _minmax_100(scores):
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        low = np.min(scores)
        high = np.max(scores)
        if np.isclose(low, high):
            return np.full(scores.shape, 50.0, dtype=np.float32)
        scores = (scores - low) / (high - low)
        return np.round(scores * 100.0, 2)

    @staticmethod
    def _candidate_text(candidate):
        return candidate.all_text

    @staticmethod
    def _job_text(job):

        return " ".join(
            filter(
                None,
                [
                    job.title,
                    job.summary,
                    " ".join(job.required_skills),
                    " ".join(job.preferred_skills),
                ],
            )
        )

    def rerank(
        self,
        job,
        ranked_candidates,
        top_k=None,
    ):

        if self.model is None:
            return ranked_candidates

        top_k = top_k or self.cross_top_k

        ranked_candidates = ranked_candidates[:top_k]

        if not ranked_candidates:
            return ranked_candidates

        print(
            f"\nRunning CrossEncoder on {len(ranked_candidates)} candidates..."
        )

        job_text = self._job_text(job)

        pairs = [
            (
                job_text,
                self._candidate_text(item["candidate"]),
            )
            for item in ranked_candidates
        ]

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        normalized = self._normalize(raw_scores)

        for item, raw, norm in zip(
            ranked_candidates,
            raw_scores,
            normalized,
        ):

            item["cross_score"] = round(float(raw), 4)

            item["cross_score_normalized"] = float(norm)

            blended = (
                item["score"] * self.heuristic_weight
                + norm * self.cross_weight
            )

            item["score"] = round(
                float(np.clip(blended, 0.0, 100.0)),
                2,
            )

        ranked_candidates.sort(
            key=lambda item: (
                -item["score"],
                -item["cross_score_normalized"],
                item["candidate"].candidate_id,
            )
        )

        return ranked_candidates