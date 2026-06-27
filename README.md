# AI Hiring Intelligence Engine

An AI-powered candidate retrieval and ranking system built for the Redrob AI Hiring Challenge.

The system processes over **100,000 candidate profiles**, intelligently retrieves the most relevant candidates for a given job description, ranks them using multiple recruiter-inspired signals, and generates a submission-ready output.

---

# Overview

Traditional recruitment systems rely heavily on keyword matching, often resulting in poor candidate recommendations.

This project adopts a **multi-stage AI pipeline** that mimics how experienced recruiters evaluate talent by considering:

- Technical expertise
- Career progression
- Behavioral signals
- Profile consistency
- Job relevance

---

# Current Architecture

```
                    Job Description
                           │
                           ▼
                     JD Loader
                           │
                           ▼
                     JD Parser
                           │
                           ▼
                     Job Object
                           │
                           ▼

Candidates.jsonl
       │
       ▼
Candidate Loader
       │
       ▼
Schema Validator
       │
       ▼
Candidate Normalizer
       │
       ▼
Candidate Objects
       │
       ▼
Candidate Retriever
       │
       ▼
Feature Engineering
       │
       ▼
Ranking Engine
       │
       ▼
Submission Generator
       │
       ▼
submission.csv
```

---

# Project Structure

```
AI-Hiring-Intelligence/

├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── outputs/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── validation/
│   ├── jd_understanding/
│   ├── retrieval/
│   ├── feature_engineering/
│   ├── ranking/
│   ├── submission/
│   └── models/
│
├── tests/
│
├── requirements.txt
├── run.py
└── README.md
```

---

# Features Implemented

## Data Pipeline

- Candidate Loader
- Schema Validation
- Candidate Normalization

---

## Job Understanding

- Job Description Loader
- Job Parser
- Structured Job Object

---

## Retrieval

Current Version

- Keyword-based retrieval
- Skill overlap
- Title overlap
- Summary overlap

Returns Top-K candidates.

---

## Feature Engineering

Current feature groups

- Technical Features
- Career Features
- Behavioral Features
- Consistency Features

---

## Ranking

Weighted scoring model

```
Technical      40%
Career         25%
Behavioral     20%
Consistency    15%
```

Produces ranked candidates.

---

## Submission

Automatically generates

```
outputs/submission.csv
```

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Language | Python 3.11 |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Machine Learning | Scikit-learn |
| NLP | Sentence Transformers *(planned)* |
| Retrieval | BM25 *(planned)* |
| Similarity | RapidFuzz |
| Documents | python-docx |

---

# Current Pipeline

```
Load Dataset
      ↓
Validate Dataset
      ↓
Normalize Candidates
      ↓
Load Job Description
      ↓
Parse Job Description
      ↓
Retrieve Candidates
      ↓
Compute Features
      ↓
Rank Candidates
      ↓
Generate Submission
```

---

# Output

The system currently generates

```
outputs/submission.csv
```

Example

```
rank,candidate_id,score,reason
1,CAND_0088025,56.25,
2,CAND_0006567,52.50,
...
```

---

# Roadmap

## Completed

- Project setup
- Candidate Loader
- Schema Validator
- Candidate Normalizer
- Job Loader
- Job Parser
- Candidate Retriever
- Feature Engineering
- Ranking Engine
- Submission Generator

---

## In Progress

- Recruiter Intelligence
- Better Feature Engineering

---

## Planned

- Hybrid Retrieval (BM25 + Semantic Search)
- Cross Encoder Re-ranking
- Explanation Generation
- Runtime Optimization
- Evaluation Metrics
- Final Competition Submission

---

# Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python run.py
```

Output

```
outputs/submission.csv
```

---

# Team

Developed as part of the **Redrob AI Hiring Challenge**.

---

# Vision

The goal of this project is to move beyond traditional resume matching and build a recruiter-inspired AI system capable of identifying the most relevant candidates using technical expertise, career progression, behavioral intelligence, and explainable ranking.