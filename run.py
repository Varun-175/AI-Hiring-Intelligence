from src.ingestion.candidate_loader import CandidateLoader
from src.validation.schema_validator import SchemaValidator
from src.preprocessing.candidate_normalizer import CandidateNormalizer
from src.jd_understanding.jd_loader import JDLoader
from src.jd_understanding.jd_parser import JDParser
from src.retrieval.retriever import CandidateRetriever
from src.feature_engineering.feature_engine import FeatureEngine
from src.ranking.ranker import CandidateRanker
from src.submission.submission_generator import SubmissionGenerator


def main():
    print("=" * 50)
    print("AI Hiring Intelligence Engine")
    print("=" * 50)

    # Initialize modules
    loader = CandidateLoader("data/raw/candidates.jsonl")
    validator = SchemaValidator("data/raw/candidate_schema.json")

    # Load JD
    jd_loader = JDLoader("data/raw/job_description.docx")
    jd_text = jd_loader.load()

    # Parse JD
    parser = JDParser()
    job = parser.parse(jd_text)

    normalized_count = 0
    normalized_candidates = []

    print("\nFirst 3 Normalized Candidates")
    print("-" * 50)

    # Complete Pipeline
    for raw_candidate in loader.load_candidates():
        is_valid = validator.validate(raw_candidate)

        if not is_valid:
            continue

        candidate = CandidateNormalizer.normalize(raw_candidate)
        normalized_candidates.append(candidate)
        normalized_count += 1

        if normalized_count <= 3:
            print(candidate)
            print("-" * 50)

    # Loader Statistics
    print("\nLoader Statistics")
    print("-" * 30)
    for key, value in loader.get_statistics().items():
        print(f"{key}: {value}")

    # Schema Validation Statistics
    print("\nSchema Validation")
    print("-" * 30)
    for key, value in validator.get_statistics().items():
        print(f"{key}: {value}")

    # Normalization Statistics
    print("\nNormalization")
    print("-" * 30)
    print(f"Normalized Candidates : {normalized_count}")

    print("\nPipeline Status")
    print("-" * 30)
    print("✓ Candidate Loader")
    print("✓ Schema Validator")
    print("✓ Candidate Normalizer")
    print("✓ Candidate Retriever")
    print("✓ Feature Engine")
    print("✓ Candidate Ranker")
    print("\nReady for Ranking 🚀")

    print("\nJob Description")
    print("-" * 50)
    print(jd_text[:1000])

    print("\nParsed Job")
    print("-" * 40)
    print(f"Title      : {job.title}")
    print(f"Experience : {job.experience_required}")
    print(f"Location   : {job.location}")
    print(f"Skills     : {job.required_skills}")

    # Retrieval step
    retriever = CandidateRetriever()
    top_candidates = retriever.retrieve(
        job,
        normalized_candidates,
        top_k=10
    )

    print("\nTop Retrieved Candidates")
    print("-" * 50)

    for rank, (score, candidate) in enumerate(top_candidates, start=1):
        print(
            f"{rank:02d}. "
            f"{candidate.candidate_id} | "
            f"{candidate.current_title} | "
            f"Score = {score}"
        )

    # Feature Engineering step
    engine = FeatureEngine()

    print("\nTop Candidate Scores")
    print("-" * 60)

    for score, candidate in top_candidates:
        features = engine.compute(job, candidate)

        print(
            f"{candidate.candidate_id} | "
            f"{candidate.current_title} | "
            f"Final={features['final']:.2f}"
        )

    # Ranking step
    ranker = CandidateRanker()

    ranked_candidates = ranker.rank(
        job,
        [candidate for _, candidate in top_candidates],
        top_k=10,
    )

    print("\nFinal Ranked Candidates")
    print("-" * 80)

    for idx, item in enumerate(ranked_candidates, start=1):
        candidate = item["candidate"]

        print(
            f"{idx:02d}. "
            f"{candidate.candidate_id} | "
            f"{candidate.current_title} | "
            f"{item['score']:.2f}"
        )
    
    generator = SubmissionGenerator()

    output_file = generator.generate(ranked_candidates)
    print(f"\nSubmission generated at: {output_file}")


if __name__ == "__main__":
    main()