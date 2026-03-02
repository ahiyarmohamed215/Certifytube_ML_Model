<p align="center">
  <h1 align="center">CertifyTube</h1>
  <p align="center">
    Dual-verification machine learning service for certifying youtube informal learning engagement.
    <br />
    <strong>Layer 1:</strong> Engagement Scoring &nbsp;·&nbsp;
    <strong>Layer 2:</strong> Quiz Verification
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-2.1-orange?logo=xgboost" alt="XGBoost">
  <img src="https://img.shields.io/badge/EBM-InterpretML-purple" alt="EBM">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20R1-green" alt="DeepSeek">
  <img src="https://img.shields.io/badge/MySQL-Transcript%20Cache-4479A1?logo=mysql&logoColor=white" alt="MySQL">
</p>

---

## Overview

CertifyTube is a dual-verification system that determines whether a learner genuinely engaged with an online video before issuing a certificate. It uses two independent verification layers:

| Layer | Purpose | How It Works |
|-------|---------|-------------|
| **Layer 1 — Engagement Analysis** | Score how actively the learner watched | XGBoost or EBM model analyzes 49 behavioral features (pauses, seeks, speed, coverage, etc.) and returns a score (0–1) with explainable contributor breakdown |
| **Layer 2 — Quiz Verification** | Verify content comprehension | LLM generates transcript-grounded quiz questions from the YouTube video; the backend grades the learner's answers |

Only learners who pass **both** layers receive a certificate.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/engagement/analyze/xgboost` | Engagement scoring with SHAP explanations |
| `POST` | `/engagement/analyze/ebm` | Engagement scoring with EBM native explanations |
| `POST` | `/quiz/generate` | Generate quiz from YouTube video ID or transcript |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Project Structure

```
certifytube_ml_model/
├── app/
│   ├── main.py                          # FastAPI application entry point
│   ├── transripts.py                    # YouTube transcript fetcher + MySQL cache
│   ├── api/
│   │   ├── engagement_routes.py         # Engagement analysis endpoints
│   │   ├── engagement_schemas.py        # Request/response models
│   │   ├── quiz_routes.py              # Quiz generation endpoint
│   │   └── quiz_schemas.py             # Quiz request/response models
│   └── core/
│       ├── settings.py                  # Environment configuration
│       ├── database.py                  # MySQL connection pool
│       └── logging.py                   # Logging setup
├── verification/
│   ├── engagement/
│   │   ├── common/
│   │   │   ├── behavior_map.py          # Feature → behavior category mapping
│   │   │   ├── text_explainer.py        # Human-readable explanation generator
│   │   │   └── validate.py              # Feature validation
│   │   ├── contracts/                   # Feature contract versioning
│   │   ├── xgboost/
│   │   │   ├── artifacts/               # Trained model + scaler + feature list
│   │   │   ├── inference/predict.py     # XGBoost prediction pipeline
│   │   │   └── explain/shap_explain.py  # SHAP-based explanations
│   │   └── ebm/
│   │       ├── artifacts/               # Trained EBM model
│   │       ├── inference/predict.py     # EBM prediction pipeline
│   │       └── explain/ebm_explain.py   # Native glass-box explanations
│   └── quiz/
│       ├── generator/
│       │   ├── quiz_gen.py              # LLM-powered quiz generation
│       │   └── prompts.py               # Prompt templates
│       ├── transcript/processor.py      # Transcript cleaning and chunking
│       └── validator/groundedness.py    # Quiz groundedness validation
├── tests/                               # Unit tests
├── data/                                # Training data
├── .env.example                         # Environment template
├── requirements.txt                     # Python dependencies
├── BACKEND_INTEGRATION_GUIDE.md         # Full API docs for Spring Boot backend
└── README.md
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **MySQL 8.0+** (for transcript caching)
- **OpenRouter API key** (for LLM-powered quiz generation)

### 1. Clone & setup

```bash
git clone https://github.com/your-username/certifytube_ml_model.git
cd certifytube_ml_model

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### 3. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open the interactive API docs at **http://127.0.0.1:8000/docs**

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | API key for LLM quiz generation | *(required)* |
| `QUIZ_MODEL` | LLM model name | `deepseek/deepseek-r1` |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `LLM_TIMEOUT_SECONDS` | Timeout per LLM call | `40` |
| `QUIZ_MAX_QUESTIONS` | Maximum quiz questions | `20` |
| `QUIZ_GENERATION_TIMEOUT_SECONDS` | Total quiz generation timeout | `300` |
| `DB_HOST` | MySQL host | `localhost` |
| `DB_PORT` | MySQL port | `3306` |
| `DB_USER` | MySQL username | `root` |
| `DB_PASSWORD` | MySQL password | *(required)* |
| `DB_NAME` | MySQL database name | `certifytube` |

---

## How It Works

### Layer 1 — Engagement Analysis

```
Backend computes 49 features    →    POST /engagement/analyze/xgboost
from learner events                  (or /ebm)
                                          │
                                          ▼
                                    ML returns:
                                    • engagement_score (0–1)
                                    • explanation (friendly message)
                                    • top positive/negative contributors
```

The engagement model analyzes behavioral signals like:

| Category | Features |
|----------|----------|
| **Coverage** | watch_time_ratio, completion_ratio, completed_flag |
| **Attention** | attention_index, engagement_velocity, play_pause_ratio |
| **Skipping** | skip_time_ratio, num_seek_forward, early_skip_flag |
| **Rewatching** | rewatch_time_ratio, num_seek_backward, deep_flag |
| **Pacing** | fast_ratio, slow_ratio, playback_speed_variance |
| **Pausing** | pause_freq_per_min, avg_pause_duration_sec, long_pause_ratio |

### Layer 2 — Quiz Verification

```
Backend sends video_id    →    POST /quiz/generate
                                    │
                                    ▼
                              ML service:
                              1. Checks MySQL transcript cache
                              2. Fetches from YouTube if not cached
                              3. Saves transcript to MySQL
                              4. LLM generates grounded questions
                              5. Returns quiz with answers
```

Quiz question types: **MCQ**, **True/False**, **Fill-in-the-Blank**, **Short Answer**, **Coding**

Each question includes: difficulty level, Bloom's taxonomy level, source segment from transcript, correct answer, and explanation.

---

## Testing

```bash
# Unit tests
python -m pytest tests/ -v

# Quick health check
curl http://localhost:8000/health

# Scripted endpoint tests (PowerShell)
.\TEST_ENDPOINTS.ps1
```

---

## Backend Integration

See **[BACKEND_INTEGRATION_GUIDE.md](BACKEND_INTEGRATION_GUIDE.md)** for complete API documentation including:

- Full request/response schemas for all endpoints
- All 49 feature definitions with computation formulas
- Java (Spring Boot) code examples and DTOs
- Database schema recommendations
- Decision flow diagram
- Error handling reference

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **API Framework** | FastAPI 0.115 |
| **Engagement Models** | XGBoost 2.1 + SHAP, InterpretML EBM |
| **Quiz LLM** | DeepSeek R1 via OpenRouter |
| **Transcript Source** | YouTube Transcript API |
| **Transcript Cache** | MySQL 8.0+ |
| **Explainability** | SHAP (XGBoost), Native term scores (EBM) |

---

## License

This project is part of a Final Year Project (FYP) at IIT.
