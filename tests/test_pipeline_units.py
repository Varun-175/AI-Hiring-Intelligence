import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.feature_engineering.behavioral_features import BehavioralFeatures
from src.feature_engineering.technical_features import TechnicalFeatures
from src.jd_understanding.jd_parser import JDParser
from src.models.candidate import Candidate
from src.preprocessing.candidate_normalizer import CandidateNormalizer
from src.ranking.ranker import CandidateRanker
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.submission.submission_generator import SubmissionGenerator
from data.raw.validate_submission import validate_submission


class PipelineUnitTests(unittest.TestCase):
    def test_normalizer_uses_redrob_signals(self):
        raw = {
            "candidate_id": "CAND_0000001",
            "profile": {"years_of_experience": "6.5"},
            "skills": [{"name": "Python"}],
            "redrob_signals": {"profile_completeness_score": 90},
        }

        candidate = CandidateNormalizer.normalize(raw)

        self.assertEqual(candidate.signals["profile_completeness_score"], 90)
        self.assertEqual(candidate.years_of_experience, 6.5)

    def test_jd_parser_handles_unicode_dash_experience(self):
        job = JDParser().parse(
            "Job Description: Senior AI Engineer\n"
            "Experience Required: 5–9 years\n"
            "Location: Pune\n"
            "Build RAG systems with Vector Search, Python and LLMs."
        )

        self.assertEqual(job.title, "Senior AI Engineer")
        self.assertEqual(job.experience_required, 7.0)
        self.assertIn("Python", job.required_skills)
        self.assertIn("Vector Search", job.required_skills)

    def test_behavioral_features_use_dataset_signal_names(self):
        candidate = Candidate(
            candidate_id="CAND_0000001",
            name="",
            headline="",
            summary="",
            location="",
            current_title="",
            current_company="",
            years_of_experience=1,
            signals={
                "profile_completeness_score": 80,
                "recruiter_response_rate": 0.5,
                "github_activity_score": 70,
                "skill_assessment_scores": {"Python": 60, "NLP": 80},
                "interview_completion_rate": 0.75,
                "offer_acceptance_rate": 0.25,
            },
        )

        self.assertGreater(BehavioralFeatures.compute(candidate), 60)

    def test_technical_features_match_aliases_and_profile_text(self):
        job = JDParser().parse(
            "Job Description: AI Engineer\n"
            "Experience Required: 5-9 years\n"
            "Python, LLM, Vector Database, Retrieval"
        )
        candidate = Candidate(
            candidate_id="CAND_0000001",
            name="",
            headline="RAG engineer with vector search systems",
            summary="Built retrieval systems with FAISS and Milvus.",
            location="",
            current_title="Senior AI Engineer",
            current_company="",
            years_of_experience=7,
            skills=["Python", "Fine-tuning LLMs"],
        )

        self.assertGreaterEqual(TechnicalFeatures.compute(job, candidate), 75)

    def test_ranker_blends_without_overwriting_base_features(self):
        features = CandidateRanker._blend_features(
            {
                "technical": 100,
                "career": 80,
                "behavioral": 60,
                "consistency": 40,
            },
            {
                "technical": 0,
                "career": 0,
                "behavioral": 0,
                "consistency": 0,
                "confidence": 10,
            },
        )

        self.assertEqual(features["technical"], 65)
        self.assertEqual(features["confidence"], 10)

    def test_cross_encoder_scores_are_normalized_to_100(self):
        normalized = CrossEncoderReranker._minmax_100([-2, 0, 2])

        self.assertEqual(list(normalized), [0, 50, 100])

    def test_ranker_orders_higher_quality_candidate_first(self):
        job = JDParser().parse(
            "Job Description: Senior AI Engineer\n"
            "Experience Required: 5-9 years\n"
            "Python LLM RAG NLP Vector Database"
        )
        strong = Candidate(
            candidate_id="CAND_0000001",
            name="",
            headline="Senior AI engineer",
            summary="Built RAG and NLP systems with vector databases.",
            location="",
            current_title="Senior AI Engineer",
            current_company="",
            years_of_experience=8,
            skills=["Python", "LLM", "RAG", "NLP", "Milvus"],
            career_history=[
                {
                    "title": "Senior AI Engineer",
                    "duration_months": 48,
                    "description": "Built production RAG, LLM and NLP systems.",
                }
            ],
            signals={"profile_completeness_score": 95},
        )
        weak = Candidate(
            candidate_id="CAND_0000002",
            name="",
            headline="Backend engineer",
            summary="Maintains APIs.",
            location="",
            current_title="Backend Engineer",
            current_company="",
            years_of_experience=4,
            skills=["Java"],
            career_history=[],
        )

        ranked = CandidateRanker().rank(job, [weak, strong], top_k=2)

        self.assertEqual(ranked[0]["candidate"].candidate_id, "CAND_0000001")
        self.assertGreaterEqual(ranked[0]["score"], ranked[1]["score"])

    def test_submission_rejects_duplicate_candidate_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            path.write_text(
                "rank,candidate_id,score,reason\n"
                "1,CAND_0000001,90.0,Good\n"
                "2,CAND_0000001,80.0,Duplicate\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                validate_submission(path)

    def test_submission_reason_uses_evidence(self):
        candidate = Candidate(
            candidate_id="CAND_0000001",
            name="",
            headline="",
            summary="",
            location="",
            current_title="Staff Machine Learning Engineer",
            current_company="",
            years_of_experience=8,
        )
        item = {
            "candidate": candidate,
            "score": 90,
            "features": {
                "consistency": 95,
                "evidence": {
                    "matched_required_skills": ["llm", "nlp", "python", "rag"],
                    "required_skill_count": 5,
                    "relevant_ai_years": 6.5,
                    "github_activity_score": 70,
                    "open_to_work": True,
                },
            },
        }

        reason = SubmissionGenerator()._generate_reason(item)

        self.assertIn("Matched 4/5 required skills", reason)
        self.assertIn("6.5 years relevant AI/ML experience", reason)
        self.assertIn("Currently Staff Machine Learning Engineer", reason)

    def test_rrf_keeps_unique_candidates_and_rewards_consensus(self):
        candidates = [
            Candidate(
                candidate_id=f"CAND_000000{i}",
                name="",
                headline="",
                summary="",
                location="",
                current_title="",
                current_company="",
                years_of_experience=0,
            )
            for i in range(1, 4)
        ]
        retriever = HybridRetriever(candidates)
        fused = retriever._rrf(
            [
                [(1.0, candidates[0]), (0.9, candidates[1])],
                [(1.0, candidates[1]), (0.9, candidates[0]), (0.8, candidates[2])],
            ]
        )

        self.assertEqual(len({candidate.candidate_id for _, candidate in fused}), 3)
        self.assertEqual(fused[0][1].candidate_id, "CAND_0000001")


if __name__ == "__main__":
    unittest.main()
