# Transparency-Driven Engagement Analysis for Informal YouTube Learning

This project implements an auditable and explainable machine learning framework
to assess learner engagement in informal video-based learning environments.

## Features
- Behavioral feature extraction
- **XGBoost** engagement classifier with SHAP-based explainability
- **EBM (Explainable Boosting Machine)** glass-box classifier with native explanations
- Dual-model API: choose `model_type` = `"xgboost"` or `"ebm"` at inference time
- Calibration and threshold analysis
- REST API for backend integration

## Note
Model evaluation uses synthetically generated labels for demonstration purposes.


This service scores learner engagement from session-level features and returns:
- Engagement score (0–1)
- ENGAGED / NOT_ENGAGED decision (threshold = 0.85)
- SHAP-based (XGBoost) or native (EBM) local explanations
- Evidence-driven explanation text

- Feature contract enforcement (v1.0)

## Quick Start

### 1) Create venv + install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
