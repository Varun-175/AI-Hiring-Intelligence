from src.ingestion.candidate_loader import CandidateLoader
from src.validation.schema_validator import SchemaValidator
from src.preprocessing.candidate_normalizer import CandidateNormalizer
from src.jd_understanding.jd_loader import JDLoader
from src.jd_understanding.jd_loader import JDLoader
from src.jd_understanding.jd_parser import JDParser

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

    print("\nFirst 3 Normalized Candidates")
    print("-" * 50)

    # Complete Pipeline
    for raw_candidate in loader.load_candidates():

        # Validate candidate
        is_valid = validator.validate(raw_candidate)

        if not is_valid:
            continue

        # Normalize candidate
        candidate = CandidateNormalizer.normalize(raw_candidate)

        normalized_count += 1

        # Print only first 3 candidates
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
    print("\nReady for Feature Engineering 🚀")

    jd_loader = JDLoader("data/raw/job_description.docx")

    jd_text = jd_loader.load()

    print("\nJob Description")
    print("-" * 50)
    print(jd_text[:1000])   # Print first 1000 characters

    print("\nParsed Job")
    print("-" * 40)
    print(f"Title      : {job.title}")
    print(f"Experience : {job.experience_required}")
    print(f"Location   : {job.location}")
    print(f"Skills     : {job.required_skills}")


if __name__ == "__main__":
    main()