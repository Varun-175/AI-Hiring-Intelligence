import time

from src.ingestion.candidate_loader import CandidateLoader
from src.validation.schema_validator import SchemaValidator
from src.preprocessing.candidate_normalizer import CandidateNormalizer

from src.jd_understanding.jd_loader import JDLoader
from src.jd_understanding.jd_parser import JDParser

from src.retrieval.hybrid_retriever import HybridRetriever
from src.ranking.ranker import CandidateRanker
from src.submission.submission_generator import SubmissionGenerator

from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.validation.honeypot_guard import HoneypotGuard
from src.utils.console import detail, header, section_break, separator, stage, status


# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = "data/raw/candidates.jsonl"
SCHEMA_PATH = "data/raw/candidate_schema.json"
JD_PATH = "data/raw/job_description.docx"

RETRIEVAL_TOP_K = 500
FINAL_TOP_K = 100
OUTPUT_WIDTH = 80


# ==========================================================
# Candidate Pipeline
# ==========================================================

def load_candidates():
    stage("[1/5] Loading Candidates")

    loader = CandidateLoader(DATASET_PATH)
    validator = SchemaValidator(SCHEMA_PATH)

    candidates = []

    for raw in loader.load_candidates():

        if validator.validate(raw):
            candidates.append(
                CandidateNormalizer.normalize(raw)
            )

    stats = loader.get_statistics()

    status("Loaded Candidates", f"{len(candidates):,}")
    status("Invalid Records", stats["invalid_records"])

    return candidates


# ==========================================================
# Job Pipeline
# ==========================================================

def load_job():

    section_break(width=OUTPUT_WIDTH)
    stage("[2/5] Parsing Job Description")

    loader = JDLoader(JD_PATH)

    parser = JDParser()

    job = parser.parse(loader.load())

    detail("Title", job.title)
    detail("Experience", f"{job.experience_required} Years")
    detail("Skills", len(job.required_skills))

    return job


# ==========================================================
# Retrieval
# ==========================================================

def retrieve_candidates(job, candidates):

    section_break(width=OUTPUT_WIDTH)
    stage("[3/5] Hybrid Retrieval")

    retriever = HybridRetriever(candidates)

    retrieved = retriever.retrieve(
        job,
        top_k=RETRIEVAL_TOP_K
    )

    status("Retrieved Candidates", len(retrieved))

    return retrieved


# ==========================================================
# Ranking
# ==========================================================

def rank_candidates(job, candidates):

    section_break(width=OUTPUT_WIDTH)
    stage("[4/5] Ranking Candidates")

    ranker = CandidateRanker()

    ranked = ranker.rank(
        job,
        candidates,
        top_k=FINAL_TOP_K
    )

    status("Feature Engineering Complete")
    status("Recruiter Intelligence Applied")
    status("Top Candidates Selected", FINAL_TOP_K)

    return ranked


# ==========================================================
# Submission
# ==========================================================

def generate_submission(ranked):

    section_break(width=OUTPUT_WIDTH)
    stage("[5/5] Generating Submission")

    generator = SubmissionGenerator()

    output = generator.generate(ranked)

    status("Submission Saved", output)


# ==========================================================
# Preview
# ==========================================================

def preview(ranked):

    section_break(width=OUTPUT_WIDTH)
    print()
    print("TOP 10 CANDIDATES")
    print()

    print(
        f"{'Rank':<6}"
        f"{'Candidate ID':<18}"
        f"{'Score':<9}"
        f"{'Current Role'}"
    )

    print("-" * OUTPUT_WIDTH)

    for rank, item in enumerate(ranked[:10], start=1):

        candidate = item["candidate"]

        print(
            f"{rank:<6}"
            f"{candidate.candidate_id:<18}"
            f"{item['score']:>7.2f}  "
            f"{candidate.current_title}"
        )


# ==========================================================
# Main
# ==========================================================

def main():

    start = time.time()

    print()
    header("AI HIRING INTELLIGENCE ENGINE", width=OUTPUT_WIDTH)

    # ---------------------------------------------------
    # Step 1 : Load Candidates
    # ---------------------------------------------------
    candidates = load_candidates()

    # ---------------------------------------------------
    # Step 2 : Load Job Description
    # ---------------------------------------------------
    job = load_job()

    # ---------------------------------------------------
    # Step 3 : Hybrid Retrieval
    # ---------------------------------------------------
    retrieved = retrieve_candidates(job, candidates)

    # ---------------------------------------------------
    # Step 4 : Feature Ranking
    # ---------------------------------------------------
    ranked = rank_candidates(
        job,
        retrieved
    )

    # ---------------------------------------------------
    # Step 5 : Cross Encoder Re-ranking
    # ---------------------------------------------------
    print()
    stage("[4.5/5] Cross-Encoder Re-ranking")

    reranker = CrossEncoderReranker()

    ranked = reranker.rerank(
        job,
        ranked
    )
    status("Re-ranked Top Candidates", len(ranked[:getattr(reranker, 'cross_top_k', len(ranked))]))

    guard = HoneypotGuard()
    ranked = guard.apply_penalty(ranked, job)

    # ---------------------------------------------------
    # Step 6 : Preview
    # ---------------------------------------------------
    preview(ranked)

    # ---------------------------------------------------
    # Step 7 : Submission
    # ---------------------------------------------------
    generate_submission(ranked)

    end = time.time()

    execution_time = end - start

    print()
    header("PIPELINE COMPLETED SUCCESSFULLY", width=OUTPUT_WIDTH)
    detail("Execution Time", f"{execution_time:.2f} seconds", indent=0)
    detail("Candidates/sec", f"{len(candidates) / execution_time:.2f}", indent=0)
    separator("=", width=OUTPUT_WIDTH, tone="cyan")


if __name__ == "__main__":
    main()
