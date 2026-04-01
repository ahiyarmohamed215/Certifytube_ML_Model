# CertifyTube ML Backend

Production-style backend service for the machine learning layer of the CertifyTube final year project at IIT.

This repository powers two verification stages used before certificate issuance:

1. engagement verification as a continuous session score from learner interaction behavior
2. transcript-grounded quiz generation for content validation

> Academic project notice: this repository is source-available for review only. It is not open-source software. Copying, reuse, redistribution, or submission as another person's work is not permitted without prior written permission from the author. See [LICENSE](LICENSE).

## What This Service Does

CertifyTube is designed for a larger application stack where a separate backend collects video-session events and calls this ML service for scoring and quiz generation.

The service provides:

- FastAPI endpoints for engagement scoring and quiz generation
- XGBoost regression inference with SHAP-based local explanations
- EBM regression inference with native term-level explanations
- strict feature-contract validation for engagement payloads
- transcript retrieval, processing, and MySQL-backed caching
- MySQL-backed persistence for engagement events, computed features, quiz state, and grading
- LLM-based quiz generation through an OpenRouter-compatible API

## API Summary

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/engagement/analyze/xgboost` | Continuous engagement score with XGBoost |
| `POST` | `/engagement/analyze/ebm` | Continuous engagement score with EBM |
| `POST` | `/quiz/generate` | Transcript-backed quiz generation |
| `POST` | `/quiz/grade` | Grade answers against the stored quiz answer key |
| `GET` | `/docs` | Swagger UI |

## Service Architecture

### Engagement Verification

The engagement layer estimates a continuous engagement score from learner interaction behavior using a strict engineered feature contract. The models are trained as regressors against a bounded target in `[0, 1]`.

The payload is validated before inference, then scored by either:

- XGBoost regressor with SHAP explanations
- EBM regressor with native explainability

Typical output includes:

- `engagement_score` in `[0, 1]`
- `engagement_status` as `engaged` or `not_engaged`
- natural-language explanation
- top positive contributors
- top negative contributors

The service can now return a binary `engagement_status` together with the score. The backend may provide either an explicit `engagement_status` override or an `engagement_threshold`. The explanation returned by the service stays neutral and reason-focused so the frontend or backend can wrap it with its own learner-facing copy.

### Quiz Verification

The quiz layer verifies content understanding using transcript-grounded question generation.

The flow is:

1. receive `video_id` and session context
2. read transcript cache from MySQL when available
3. fetch transcript when cache is missing
4. clean and process transcript content
5. generate quiz questions through the configured LLM provider
6. persist quiz questions, options, answers, and explanations in MySQL
7. return the public quiz questions to the backend
8. grade submitted answers later against the stored answer key

## Integration Contract

### Engagement Request Shape

All engagement endpoints use the same request envelope. The preferred format is raw player events for one session:

```json
{
  "session_id": "session-001",
  "feature_version": "v1.0",
  "events": [
    {
      "event_id": "evt-001",
      "session_id": "session-001",
      "event_type": "play",
      "player_state": 1,
      "playback_rate": 1.0,
      "current_time_sec": 12.4,
      "video_duration_sec": 600.0,
      "created_at_utc": "2026-04-01T10:00:00Z"
    }
  ]
}
```

Important contract rules:

- `session_id` is required
- `feature_version` is required
- provide exactly one of `events` or `features`
- `events` is the preferred format because the service computes the canonical feature row itself
- `features` is still accepted for backward compatibility, but every feature must be numeric and match the versioned contract exactly
- missing or extra feature keys are rejected with `400`

The exact feature contract is versioned under `verification/engagement/contracts/`.

### Engagement Response Shape

Both engagement models return:

- model identifier
- `session_id`
- `feature_version`
- `engagement_score`
- `engagement_status`
- explanation text
- top positive contributors
- top negative contributors

XGBoost uses SHAP contributor objects. EBM uses native contribution objects.

`engagement_score` is still the direct regression output exposed by the API. `engagement_status` is the binary label paired with that score for frontend use, based on the backend-provided status or threshold when available.

### Quiz Request Shape

```json
{
  "session_id": "session-001",
  "video_id": "dQw4w9WgXcQ",
  "video_duration_sec": 600
}
```

Quiz responses return:

- `session_id`
- `video_id`
- `questions`
- `total_questions`

Each question includes:

- `question_id`
- `type`
- `question`
- `options` when applicable
- `correct_answer`
- `explanation`
- `difficulty`

## Error Behavior

The service is explicit about request and upstream failures.

### Engagement Endpoints

- `400` invalid feature contract or invalid feature values
- `500` missing model artifacts or unexpected processing failure

### Quiz Endpoint

- `400` invalid video input
- `404` transcript genuinely does not exist
- `503` transcript upstream unavailable or blocked
- `502` quiz provider failure
- `500` unexpected internal failure

## Repository Layout

```text
certifytube_ml_service/
|-- app/
|   |-- main.py
|   |-- api/
|   |   |-- engagement_routes.py
|   |   |-- engagement_schemas.py
|   |   |-- quiz_routes.py
|   |   `-- quiz_schemas.py
|   `-- core/
|       |-- database.py
|       |-- logging.py
|       `-- settings.py
|-- verification/
|   |-- engagement/
|   |   |-- common/
|   |   |-- contracts/
|   |   |-- ebm/
|   |   `-- xgboost/
|   `-- quiz/
|       |-- generator/
|       |-- transcript/
|       `-- validator/
|-- data/
|-- reports/
|-- .env.example
|-- LICENSE
|-- requirements.txt
`-- README.md
```

## Technology Stack

| Area | Technology |
| --- | --- |
| API | FastAPI |
| Serving | Uvicorn |
| ML | XGBoost regressor, Explainable Boosting Regressor |
| Explainability | SHAP, native EBM term explanations |
| Data tooling | pandas, NumPy, scikit-learn |
| Quiz generation | OpenRouter-compatible LLM API |
| Transcript source | `youtube-transcript-api` |
| Cache | MySQL |

## Local Setup

### Prerequisites

- Python 3.13 recommended
- MySQL 8+
- OpenRouter API key

### Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then start the service:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Local URLs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Configuration

Core environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | API key for quiz generation |
| `QUIZ_MODEL` | LLM model identifier |
| `OPENROUTER_BASE_URL` | OpenRouter-compatible base URL |
| `OPENROUTER_SITE_URL` | Site URL sent with OpenRouter headers |
| `OPENROUTER_APP_NAME` | Application name sent with OpenRouter headers |
| `LLM_TIMEOUT_SECONDS` | Per-request LLM timeout |
| `QUIZ_MAX_QUESTIONS` | Max generated questions |
| `QUIZ_MAX_ATTEMPTS_PER_QUESTION` | Retry limit per question |
| `QUIZ_GENERATION_TIMEOUT_SECONDS` | Total quiz generation timeout |
| `DB_HOST` | MySQL host |
| `DB_PORT` | MySQL port |
| `DB_USER` | MySQL user |
| `DB_PASSWORD` | MySQL password |
| `DB_NAME` | MySQL database name |

## Model Artifacts

- XGBoost artifacts are stored under `verification/engagement/xgboost/artifacts/`
- EBM artifacts are stored under `verification/engagement/ebm/artifacts/`
- shared preprocessing and validation utilities are under `verification/engagement/common/`
- the end-to-end training notebook is `verification/engagement/engagement_model_pipeline.ipynb`
- generated reports are stored under `reports/`

## Project Positioning

This repository is intentionally structured like a backend service repository rather than a notebook dump or coursework archive. It contains:

- serving code
- training and evaluation code
- model artifacts
- validation logic
- integration-ready request and response contracts

## License

This project is protected by a custom academic and proprietary license.

- it is an IIT final year project
- it is not released under an open-source license
- copying, redistribution, reuse, or submission as another person's work is prohibited without written permission

See [LICENSE](LICENSE) for the full terms.
