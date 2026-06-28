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


# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = "data/raw/candidates.jsonl"
SCHEMA_PATH = "data/raw/candidate_schema.json"
JD_PATH = "data/raw/job_description.docx"

RETRIEVAL_TOP_K = 500
FINAL_TOP_K = 100


# ==========================================================
# Candidate Pipeline
# ==========================================================

def load_candidates():
    print("\n[1/5] Loading candidates...")

    loader = CandidateLoader(DATASET_PATH)
    validator = SchemaValidator(SCHEMA_PATH)

    candidates = []

    for raw in loader.load_candidates():

        if validator.validate(raw):
            candidates.append(
                CandidateNormalizer.normalize(raw)
            )

    stats = loader.get_statistics()

    print(f"Loaded Candidates : {len(candidates):,}")
    print(f"Invalid Records   : {stats['invalid_records']}")

    return candidates


# ==========================================================
# Job Pipeline
# ==========================================================

def load_job():

    print("\n[2/5] Loading Job Description...")

    loader = JDLoader(JD_PATH)

    parser = JDParser()

    job = parser.parse(loader.load())

    print(f"Job Title : {job.title}")
    print(f"Experience: {job.experience_required} Years")
    print(f"Skills    : {len(job.required_skills)}")

    return job


# ==========================================================
# Retrieval
# ==========================================================

def retrieve_candidates(job, candidates):

    print("\n[3/5] Hybrid Retrieval...")

    retriever = HybridRetriever(candidates)

    retrieved = retriever.retrieve(
        job,
        top_k=RETRIEVAL_TOP_K
    )

    print(f"Retrieved {len(retrieved)} candidates")

    return retrieved


# ==========================================================
# Ranking
# ==========================================================

def rank_candidates(job, candidates):

    print("\n[4/5] Ranking candidates...")

    ranker = CandidateRanker()

    ranked = ranker.rank(
        job,
        candidates,
        top_k=FINAL_TOP_K
    )

    print(f"Top {FINAL_TOP_K} candidates selected")

    return ranked


# ==========================================================
# Submission
# ==========================================================

def generate_submission(ranked):

    print("\n[5/5] Generating Submission...")

    generator = SubmissionGenerator()

    output = generator.generate(ranked)

    print(f"Submission Saved : {output}")


# ==========================================================
# Preview
# ==========================================================

def preview(ranked):

    print("\n")
    print("=" * 90)
    print("TOP 10 CANDIDATES")
    print("=" * 90)

    print(
        f"{'Rank':<6}"
        f"{'Candidate ID':<18}"
        f"{'Score':<10}"
        f"{'Current Role'}"
    )

    print("-" * 90)

    for rank, item in enumerate(ranked[:10], start=1):

        candidate = item["candidate"]

        print(
            f"{rank:<6}"
            f"{candidate.candidate_id:<18}"
            f"{item['score']:>8.2f}   "
            f"{candidate.current_title}"
        )


# ==========================================================
# Main
# ==========================================================

def main():

    start = time.time()

    print("=" * 90)
    print("          AI HIRING INTELLIGENCE ENGINE")
    print("=" * 90)

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
    print("\n[4.5/5] CrossEncoder Re-ranking...")

    reranker = CrossEncoderReranker()

    ranked = reranker.rerank(
        job,
        ranked
    )

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

    print("\n" + "=" * 90)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(f"Execution Time : {execution_time:.2f} seconds")
    print(f"Candidates/sec : {len(candidates) / execution_time:.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()