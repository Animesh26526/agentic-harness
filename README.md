# Agentic Harness: Reliability, Evaluation & Monitoring Framework for AI Agents

**Agentic Harness** is a modular framework designed to wrap stochastic LLM Agent operations in a deterministic validation and evaluation wrapper to guarantee production-grade consistency, instruction compliance, and factual alignment.

---

## Current V1 Implementation Status

This repository contains the foundation layer of the Agentic Harness framework:
* **Configuration Management**: A centralized, extensible `Config` class supporting environment variable loading.
* **Agent wrappers**: `GeminiAgent` implementing Google's Gemini 1.5 Flash API with built-in rate-limit capturing and exception recovery.
* **Semantic Evaluator**: CPU-optimized local embedding matching utilizing the `SentenceTransformers` model (`all-MiniLM-L6-v2`).
* **Rule Validator**: Deterministic Python-based regex validator verifying lengths, forbidden keywords, JSON syntax structure, and missing keys.
* **Benchmark Schema**: Initial data models for structured testing across 4 core task categories.
* **Unit Testing**: Complete `pytest` test suites validating rules and similarity behaviors.

---

## Project Directory Structure

```text
agentic-harness/
│
├── requirements.txt            # Package dependencies
├── .env.example                # Template for API credentials
├── README.md                   # Setup instructions and documentation
│
├── harness/                    # Core Library Namespace
│   ├── __init__.py
│   ├── config.py               # Environmental threshold configs
│   │
│   ├── agent/                  # LLM Agent Wrappers
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Abstract Base Agent Interface
│   │   └── gemini_agent.py     # Gemini 1.5 Flash Wrapper
│   │
│   └── evaluators/             # Evaluator Modules
│       ├── __init__.py
│       ├── base_evaluator.py   # Abstract Base Evaluator
│       ├── semantic.py         # SentenceTransformers cosine similarity
│       └── rule_based.py       # Syntax, schema, regex & constraint validators
│
├── data/                       # Dataset Store
│   └── benchmark_dataset.json  # 12-sample test dataset (3 per category)
│
└── tests/                      # Unit Test Suite
    ├── test_semantic.py        # Semantic evaluation tests
    └── test_rules.py           # Syntactic rule validation tests
```

---

## Setup & Installation Instructions

### Local Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the root directory and configure:
```text
GEMINI_API_KEY=your_actual_api_key_here
DEFAULT_GEMINI_MODEL=gemini-2.5-flash
```

---

## Running & Verification

### Run Tests
```bash
pytest
```

### Run Streamlit Dashboard
```bash
    streamlit run app/main.py
```

### Run Benchmark
```bash
python run_benchmark.py
```

### List Models
```bash
python scripts/list_available_models.py
```