# CertifyTube Engagement ML Service (XGBoost + SHAP + Counterfactuals)

This service scores learner engagement from session-level features and returns:
- Engagement score (0–1)
- ENGAGED / NOT_ENGAGED decision (threshold = 0.85)
- SHAP-based local explanations (top positive/negative contributors)
- Evidence-driven explanation text
- Counterfactual guidance (only when NOT_ENGAGED)
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
