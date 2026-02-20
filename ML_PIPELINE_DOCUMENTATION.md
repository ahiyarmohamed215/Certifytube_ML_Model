# CertifyTube — End-to-End ML Pipeline Documentation

> **Two models, one mission:** Predict whether a learner is genuinely engaged with a video lecture,
> and explain *why* — using XGBoost (black-box + SHAP) and EBM (glass-box, inherently interpretable).

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Source & Feature Engineering](#2-data-source--feature-engineering)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Train / Test Splitting Strategy](#4-train--test-splitting-strategy)
5. [Model 1 — XGBoost](#5-model-1--xgboost)
6. [Model 2 — EBM (Explainable Boosting Machine)](#6-model-2--ebm-explainable-boosting-machine)
7. [Model Comparison (Head-to-Head)](#7-model-comparison-head-to-head)
8. [Explainability Pipeline](#8-explainability-pipeline)
9. [Inference & API Serving](#9-inference--api-serving)
10. [Project Structure](#10-project-structure)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       CertifyTube ML Service                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐     ┌──────────────┐     ┌───────────────────────┐ │
│  │   Raw Data   │────▶│ Preprocessing│────▶│  Stratified-Grouped   │ │
│  │  (CSV)       │     │  (clean +    │     │  Train/Test Split     │ │
│  │              │     │   impute)    │     │  (80/20)              │ │
│  └─────────────┘     └──────────────┘     └──────────┬────────────┘ │
│                                                       │              │
│                                    ┌──────────────────┼────────┐     │
│                                    ▼                  ▼        │     │
│                          ┌──────────────┐   ┌──────────────┐   │     │
│                          │   XGBoost    │   │     EBM      │   │     │
│                          │  Training    │   │   Training   │   │     │
│                          │  (40-trial   │   │  (40-trial   │   │     │
│                          │  random      │   │   random     │   │     │
│                          │  search +    │   │  search +    │   │     │
│                          │  xgb.cv)     │   │   5-fold CV) │   │     │
│                          └──────┬───────┘   └──────┬───────┘   │     │
│                                 │                  │           │     │
│                                 ▼                  ▼           │     │
│                          ┌──────────────┐   ┌──────────────┐   │     │
│                          │  Evaluation  │   │  Evaluation  │   │     │
│                          │  + Reports   │   │  + Reports   │   │     │
│                          └──────┬───────┘   └──────┬───────┘   │     │
│                                 │                  │           │     │
│                                 ▼                  ▼           │     │
│                          ┌──────────────┐   ┌──────────────┐   │     │
│                          │  model.joblib│   │  ebm_model   │   │     │
│                          │  + SHAP      │   │  .joblib     │   │     │
│                          └──────┬───────┘   └──────┬───────┘   │     │
│                                 │                  │           │     │
│                                 ▼                  ▼           │     │
│                          ┌─────────────────────────────────┐   │     │
│                          │       FastAPI Service            │   │     │
│                          │  POST /engagement/analyze/xgboost│  │     │
│                          │  POST /engagement/analyze/ebm    │   │     │
│                          └─────────────────────────────────┘   │     │
│                                                                │     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Source & Feature Engineering

| Item | Detail |
|------|--------|
| **Source file** | `data/processed/sessions_features.csv` |
| **Size** | ~748 KB |
| **Unit of observation** | One row = one learner–video viewing session |
| **Label column** | `engagement_label` (binary: 0 = NOT_ENGAGED, 1 = ENGAGED) |
| **Group column** | `user_id` (ensures no learner data leaks between train and test) |
| **Number of features** | **49 numeric features** |

### Feature Categories (7 behavioral domains)

| Category | Example Features | What It Captures |
|----------|------------------|------------------|
| **Coverage** | `watch_time_ratio`, `completion_ratio`, `watch_time_sec`, `completed_flag` | How much of the video was actually watched |
| **Skipping** | `num_seek_forward`, `skip_time_ratio`, `early_skip_flag`, `skim_flag` | Forward-seeking behavior (content skipping) |
| **Rewatching** | `num_seek_backward`, `rewatch_time_ratio`, `deep_flag` | Backward-seeking (reviewing content) |
| **Reflective Pausing** | `num_pause`, `avg_pause_duration_sec`, `long_pause_ratio`, `pause_freq_per_min` | Pause behavior (thinking/note-taking) |
| **Speed Watching** | `avg_playback_rate_when_playing`, `fast_ratio`, `slow_ratio`, `playback_speed_variance` | Playback speed patterns |
| **Attention Consistency** | `attention_index`, `engagement_velocity`, `seek_density_per_min`, `play_pause_ratio` | Overall attention signals |
| **Playback Quality** | `num_buffering_events`, `buffering_time_sec`, `buffering_freq_per_min` | Technical playback interruptions |

---

## 3. Preprocessing Pipeline

Both models use the **identical** preprocessing function (`_prepare_features()`):

```
Raw CSV
  │
  ▼
1. Drop non-feature columns:
   - engagement_label (target)
   - user_id (grouping key)
   - session_id, video_id, video_title (identifiers)
  │
  ▼
2. Select only numeric columns
   (drops any accidental string/object columns)
  │
  ▼
3. Handle infinite values:
   - Replace ±inf with NaN
  │
  ▼
4. Median imputation:
   - Fill NaN with column median
   (not 0, because 0 can be a real signal — e.g.,
    0 pauses is meaningful)
  │
  ▼
Result: 49 clean numeric features ready for training
```

**Key design decision:** Median imputation was chosen over zero-fill because features like `num_pause = 0` or `completion_ratio = 0` are valid signals. Using the median avoids injecting false signal.

---

## 4. Train / Test Splitting Strategy

| Aspect | Detail |
|--------|--------|
| **Method** | `StratifiedGroupKFold` from scikit-learn |
| **Split ratio** | 80% train / 20% test |
| **Stratification** | Preserves label distribution (ENGAGED vs NOT_ENGAGED ratio) in both splits |
| **Grouping** | By `user_id` — all sessions from the same learner stay in the same split |
| **Random state** | 42 (deterministic, reproducible) |

**Why grouped split?** If the same learner's sessions appear in both train and test, the model could memorize learner-specific patterns instead of learning general engagement signals. Grouping by `user_id` prevents this **data leakage**.

The train/test session IDs are saved to `ml/artifacts/split.json` (XGBoost) and `ml/artifacts/ebm_split.json` (EBM) for full reproducibility.

---

## 5. Model 1 — XGBoost

### 5.1 What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is a **gradient-boosted decision tree** ensemble. It builds trees sequentially, where each new tree corrects the errors of the previous ones. It is a **black-box** model — powerful but not inherently interpretable.

### 5.2 Hyperparameter Tuning

| Aspect | Detail |
|--------|--------|
| **Strategy** | 40-trial random search |
| **CV method** | `xgb.cv()` with 5-fold Stratified Grouped K-Fold |
| **Scoring metric** | Mean AUC (Area Under ROC Curve) |
| **Early stopping** | 50 rounds (stops if validation AUC doesn't improve) |
| **Max boosting rounds** | 5,000 (capped by early stopping) |

### 5.3 Hyperparameter Search Space

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `max_depth` | 3–7 | Tree complexity (deeper = more complex) |
| `min_child_weight` | 1–10 | Minimum samples per leaf (regularization) |
| `eta` (learning rate) | 0.01–0.2 | Step size (lower = more cautious) |
| `subsample` | 0.6–1.0 | Row sampling per tree (prevents overfitting) |
| `colsample_bytree` | 0.6–1.0 | Feature sampling per tree |
| `gamma` | 0–5 | Minimum loss reduction for split |
| `lambda` (L2 reg) | 0.1–10 | L2 regularization strength |
| `alpha` (L1 reg) | 0–5 | L1 regularization strength |

### 5.4 Best Hyperparameters Found

| Parameter | Value |
|-----------|-------|
| `max_depth` | 7 |
| `min_child_weight` | 1.224 |
| `eta` | 0.115 |
| `subsample` | 0.854 |
| `colsample_bytree` | 0.642 |
| `gamma` | 0.702 |
| `lambda` | 4.249 |
| `alpha` | 4.831 |
| **Best round** | **291** (out of 5,000 max) |
| **Best CV AUC** | **0.9860** |

### 5.5 Final Training

After finding the best hyperparameters via CV, the final model is trained on the **entire training split** (80%) with:
- The best parameters from the random search
- `num_boost_round = 291` (the CV-determined optimal)
- Evaluation tracked on both train and test for learning curve plots

### 5.6 Test Set Results (XGBoost)

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9917** |
| **AUC-PR** | **0.9917** |
| Precision @ 0.5 | 0.9574 |
| Recall @ 0.5 | 0.9184 |
| F1 @ 0.5 | 0.9375 |
| **Precision @ 0.85** | **1.0000** |
| **Recall @ 0.85** | **0.8163** |
| **F1 @ 0.85** | **0.8989** |

**Confusion Matrix (threshold = 0.85):**

|  | Predicted NOT_ENGAGED | Predicted ENGAGED |
|--|----------------------|-------------------|
| **Actual NOT_ENGAGED** | 103 | 0 |
| **Actual ENGAGED** | 18 | 80 |

> **Interpretation:** At threshold 0.85, XGBoost has **zero false positives** (100% precision) — it never falsely certifies someone as engaged. The 18 false negatives mean some genuinely engaged learners are missed, which is the safer failure mode.

### 5.7 Artifacts Saved

| File | Purpose |
|------|---------|
| `ml/artifacts/model.joblib` | Serialized XGBoost Booster |
| `ml/artifacts/feature_columns.json` | Ordered list of 49 feature names |
| `ml/artifacts/split.json` | Train/test session IDs for reproducibility |
| `ml/artifacts/metadata.json` | Full audit trail (params, metrics, timestamps) |
| `reports/metrics.json` | Training metrics |
| `reports/metrics_test.json` | Test set metrics |
| `reports/learning_curve_auc.png` | Train vs Test AUC over boosting rounds |
| `reports/learning_curve_logloss.png` | Train vs Test logloss (overfitting check) |
| `reports/roc_curve.png` | ROC curve |
| `reports/pr_curve.png` | Precision–Recall curve |
| `reports/calibration.png` | Calibration plot |
| `reports/confusion_matrix_085.png` | Confusion matrix at threshold 0.85 |
| `reports/feature_importance_gain.png` | Top-20 features by gain |
| `reports/threshold_sweep.csv` | Accuracy/F1 across thresholds |

---

## 6. Model 2 — EBM (Explainable Boosting Machine)

### 6.1 What is EBM?

EBM (Explainable Boosting Machine) from Microsoft's InterpretML is a **Generalized Additive Model (GAM)** with interactions:

```
prediction = intercept + f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ) + f_ij(xᵢ, xⱼ) + ...
```

Each `fᵢ(xᵢ)` is learned via boosting but is a **univariate function** — meaning the model's decisions can be fully explained by showing the individual contribution curves. This makes EBM a **glass-box** model: the explanation IS the model, not an approximation.

### 6.2 Why EBM as a Second Model?

| Aspect | XGBoost | EBM |
|--------|---------|-----|
| **Model type** | Black-box ensemble | Glass-box GAM |
| **Explainability** | Post-hoc (SHAP approximation) | Built-in (exact contributions) |
| **Explanations** | Approximate Shapley values | Exact term scores |
| **Verifiability** | Cannot verify explanation matches prediction | `intercept + Σ contributions = logit` (provable) |
| **Interactions** | Implicit (hidden in tree structure) | Explicit (named interaction terms) |

### 6.3 Hyperparameter Tuning

| Aspect | Detail |
|--------|--------|
| **Strategy** | 40-trial random search (matching XGBoost approach) |
| **CV method** | 5-fold Stratified Grouped K-Fold (manual loop) |
| **Scoring metric** | Mean AUC |
| **CV max rounds** | 500 (speed optimization for CV) |
| **Final max rounds** | 5,000 (with early stopping, `validation_size=0.15`) |
| **Outer bags (CV)** | 1 (no bagging during CV — faster) |

### 6.4 Hyperparameter Search Space

| Parameter | Values Tried | Purpose |
|-----------|-------------|---------|
| `max_bins` | 128, 256, 512 | Binning resolution for continuous features |
| `learning_rate` | 0.005, 0.01, 0.02, 0.03, 0.05, 0.08 | Step size per boosting round |
| `max_leaves` | 2, 3, 4, 5, 6 | Maximum leaves per tree (controls complexity) |
| `min_samples_leaf` | 2, 4, 5, 8, 10, 15 | Regularization (minimum samples per leaf) |
| `interactions` | 0, 3, 5, 8, 10 | Number of automatic pairwise interaction terms |
| `max_interaction_bins` | 16, 32, 64 | Bin resolution for interaction terms |

### 6.5 Best Hyperparameters Found

| Parameter | Value |
|-----------|-------|
| `max_bins` | 256 |
| `learning_rate` | 0.01 |
| `max_leaves` | 2 |
| `min_samples_leaf` | 8 |
| `interactions` | 10 |
| `max_interaction_bins` | 16 |
| **Best CV AUC** | **0.9891** |

### 6.6 Final Training

The final EBM is trained on the full training split with:
- Best parameters from random search
- `max_rounds = 5,000` (with early stopping at 50 rounds)
- `validation_size = 0.15` (inner split for early stopping)
- Full bagging restored (default outer_bags)

### 6.7 Test Set Results (EBM)

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9925** |
| **AUC-PR** | **0.9921** |
| Precision @ 0.5 | 0.9684 |
| Recall @ 0.5 | 0.9388 |
| F1 @ 0.5 | 0.9534 |
| **Precision @ 0.85** | **0.9659** |
| **Recall @ 0.85** | **0.8673** |
| **F1 @ 0.85** | **0.9140** |

**Confusion Matrix (threshold = 0.85):**

|  | Predicted NOT_ENGAGED | Predicted ENGAGED |
|--|----------------------|-------------------|
| **Actual NOT_ENGAGED** | 100 | 3 |
| **Actual ENGAGED** | 13 | 85 |

> **Interpretation:** EBM has 3 false positives but catches 5 more truly engaged learners than XGBoost. It has a more balanced precision/recall tradeoff.

### 6.8 EBM Model Structure

The trained EBM has **59 terms**:
- **49 main-effect terms** (one per feature — univariate shape functions)
- **10 interaction terms** (automatically discovered pairwise interactions)

Example interaction terms discovered:
- `session_duration_sec & deep_flag`
- `watch_time_ratio & fast_ratio`
- `num_pause & attention_index`

### 6.9 Artifacts Saved

| File | Purpose |
|------|---------|
| `ml/artifacts/ebm_model.joblib` | Serialized EBM model (~1.5 MB) |
| `ml/artifacts/ebm_feature_columns.json` | Ordered feature names |
| `ml/artifacts/ebm_split.json` | Train/test session IDs |
| `ml/artifacts/ebm_metadata.json` | Full audit trail + top-10 CV trials |
| `reports/ebm_metrics.json` | Training metrics |
| `reports/ebm_metrics_test.json` | Test set metrics |
| `reports/ebm_roc_curve.png` | ROC curve |
| `reports/ebm_pr_curve.png` | Precision–Recall curve |
| `reports/ebm_calibration.png` | Calibration plot |
| `reports/ebm_confusion_matrix_085.png` | Confusion matrix at 0.85 |
| `reports/ebm_feature_importance.png` | Top-20 features by importance |
| `reports/ebm_term_importances.png` | Term importance bar chart |
| `reports/ebm_shape_functions.png` | Per-feature shape function curves |
| `reports/ebm_threshold_sweep.csv` | Accuracy/F1 across thresholds |

---

## 7. Model Comparison (Head-to-Head)

### 7.1 Performance Metrics

| Metric | XGBoost | EBM | Winner |
|--------|---------|-----|--------|
| **AUC-ROC** | 0.9917 | **0.9925** | EBM (marginal) |
| **AUC-PR** | 0.9917 | **0.9921** | EBM (marginal) |
| **CV AUC** | 0.9860 | **0.9891** | EBM |
| Precision @ 0.85 | **1.0000** | 0.9659 | XGBoost |
| Recall @ 0.85 | 0.8163 | **0.8673** | EBM |
| **F1 @ 0.85** | 0.8989 | **0.9140** | EBM |
| False Positives @ 0.85 | **0** | 3 | XGBoost |
| False Negatives @ 0.85 | 18 | **13** | EBM |

### 7.2 Trade-off Summary

| Aspect | XGBoost | EBM |
|--------|---------|-----|
| **Prediction accuracy** | Excellent | Excellent (slightly better AUC) |
| **Precision focus** | 100% precision — never false certifies | 96.6% precision — very rare false certifies |
| **Recall focus** | Misses 18/98 engaged learners | Misses 13/98 engaged learners |
| **Explainability** | SHAP (post-hoc approximation) | Exact, verifiable contributions |
| **Auditability** | Cannot prove explanation matches prediction | Can mathematically prove it |
| **Model size** | 226 KB | 1.5 MB |
| **Inference speed** | Faster | Slightly slower |

### 7.3 When to Use Each

- **XGBoost**: Best when you want **zero false positives** (never giving an engagement certificate to someone who isn't engaged). Conservative, precision-oriented.
- **EBM**: Best when you want **higher recall** (catching more truly engaged learners) with **fully transparent explanations** that can be audited and explained to stakeholders.

---

## 8. Explainability Pipeline

### 8.1 XGBoost + SHAP

SHAP (SHapley Additive exPlanations) uses game-theoretic Shapley values to explain each prediction:

```
Step 1: Load trained XGBoost Booster
Step 2: Create SHAP TreeExplainer (cached singleton)
Step 3: For each prediction request:
        a. Build feature vector (49 features)
        b. Compute SHAP values for each feature
        c. Sort by SHAP value
        d. Return top-3 positive contributors (pushing towards ENGAGED)
        e. Return top-3 negative contributors (pushing towards NOT_ENGAGED)
```

**Output:** Each contributor has `shap_value` (approximate contribution in log-odds space).

**Key limitation:** SHAP values are **approximations** of the true Shapley values. They don't perfectly decompose the prediction.

### 8.2 EBM Native Explanations

EBM's explanation IS the model — no approximation needed:

```
Step 1: Load trained EBM model
Step 2: For each prediction request:
        a. Call ebm.explain_local(x)
        b. Extract exact term scores for each feature
        c. Handle interaction terms: split contribution 50/50 to component features
        d. Sanity check: intercept + Σ contributions ≡ predicted logit (must match!)
        e. Sort by contribution value
        f. Return top-3 positive contributors (pushing towards ENGAGED)
        g. Return top-3 negative contributors (pushing towards NOT_ENGAGED)
```

**Output:** Each contributor has `contribution` (exact term score in log-odds space).

**Key advantage:** We can PROVE: `intercept + Σ all_contributions = model_logit`. If this check fails, we log a warning.

### 8.3 Behavioral Mapping & Text Explanation

Both models share the same downstream pipeline:

```
Feature contributors
        │
        ▼
Behavior Map (49 features → 7 categories)
  e.g., watch_time_ratio → "coverage"
        │
        ▼
Reason Code Generation
  e.g., "coverage" + positive → "HIGH_COVERAGE"
  e.g., "coverage" + negative → "LOW_COVERAGE"
        │
        ▼
Human-Readable Explanation
  e.g., "Engagement was confirmed for this session
         due to sustained attention and strong coverage."
```

---

## 9. Inference & API Serving

### 9.1 Architecture

The ML service is built with **FastAPI** and serves two independent endpoints:

| Endpoint | Model | Response Schema |
|----------|-------|-----------------|
| `POST /engagement/analyze/xgboost` | XGBoost Booster | `XGBoostAnalyzeResponse` |
| `POST /engagement/analyze/ebm` | EBM Classifier | `EBMAnalyzeResponse` |

### 9.2 Request Format (Same for Both)

```json
{
  "session_id": "abc-123",
  "feature_version": "v1.0",
  "features": {
    "watch_time_ratio": 0.95,
    "completion_ratio": 0.88,
    "num_pause": 3,
    ... (all 49 features)
  }
}
```

### 9.3 XGBoost Response Format

```json
{
  "model": "xgboost",
  "session_id": "abc-123",
  "feature_version": "v1.0",
  "engagement_score": 0.92,
  "threshold": 0.85,
  "status": "ENGAGED",
  "explanation": "Engagement was confirmed due to sustained attention and strong coverage.",
  "reason_codes": ["HIGH_ATTENTION", "HIGH_COVERAGE"],
  "shap_top_positive": [
    {
      "feature": "watch_time_ratio",
      "shap_value": 0.45,
      "feature_value": 0.95,
      "behavior_category": "coverage"
    }
  ],
  "shap_top_negative": [
    {
      "feature": "num_buffering_events",
      "shap_value": -0.03,
      "feature_value": 2.0,
      "behavior_category": "playback_quality"
    }
  ]
}
```

### 9.4 EBM Response Format

```json
{
  "model": "ebm",
  "session_id": "abc-123",
  "feature_version": "v1.0",
  "engagement_score": 0.89,
  "threshold": 0.85,
  "status": "ENGAGED",
  "explanation": "Engagement was confirmed due to sustained attention and strong coverage.",
  "reason_codes": ["HIGH_ATTENTION", "HIGH_COVERAGE"],
  "ebm_top_positive": [
    {
      "feature": "watch_time_ratio",
      "contribution": 1.87,
      "feature_value": 0.95,
      "behavior_category": "coverage"
    }
  ],
  "ebm_top_negative": [
    {
      "feature": "num_buffering_events",
      "contribution": -0.12,
      "feature_value": 2.0,
      "behavior_category": "playback_quality"
    }
  ]
}
```

**Key differences between the two responses:**
- `model` field: hard literal (`"xgboost"` vs `"ebm"`)
- Score field name: `shap_value` (SHAP approximation) vs `contribution` (exact EBM term)
- List field prefix: `shap_top_*` vs `ebm_top_*`

### 9.5 Request Processing Flow

```
Backend sends POST request
        │
        ▼
1. Contract Enforcement
   - Load feature_contract_v1.json (49 expected features)
   - Validate: no missing features, no extra features, all numeric
        │
        ▼
2. Feature Validation
   - Double-check types and presence
        │
        ▼
3. Model Prediction
   - XGBoost: Build DMatrix → booster.predict() → probability
   - EBM: Build numpy array → ebm.predict_proba()[:, 1] → probability
        │
        ▼
4. Threshold Decision
   - score >= 0.85 → ENGAGED
   - score < 0.85 → NOT_ENGAGED
        │
        ▼
5. Explanation Generation
   - XGBoost: SHAP TreeExplainer → shap_values
   - EBM: explain_local() → exact term scores
        │
        ▼
6. Text & Reason Codes
   - Map features → behavioral categories → reason codes
   - Generate human-readable explanation text
        │
        ▼
7. Return structured JSON response
```

### 9.6 Feature Contract System

The **Feature Contract** (`ml/contracts/feature_contract_v1.json`) enforces a strict API agreement:

- Lists exactly 49 features the backend **must** send
- Version-tagged (`v1.0`) to support future feature evolution
- Both directions enforced: missing features **AND** extra features are rejected
- Ensures the model always receives exactly the features it was trained on

---

## 10. Project Structure

```
certifytube_ml_model/
├── app/                                # FastAPI application
│   ├── main.py                         # App entry point, router registration
│   ├── api/
│   │   ├── schemas.py                  # Pydantic request/response models
│   │   ├── routes.py                   # /engagement/analyze/* endpoints
│   │   └── quizz_routes.py            # Quiz generation endpoints
│   └── core/
│       └── logging.py                  # Logging configuration
│
├── ml/                                 # Machine learning modules
│   ├── training/
│   │   ├── split.py                    # Stratified grouped train/test split
│   │   ├── train.py                    # XGBoost training pipeline
│   │   ├── train_ebm.py               # EBM training pipeline
│   │   ├── evaluate.py                 # XGBoost evaluation + reports
│   │   └── evaluate_ebm.py            # EBM evaluation + reports
│   │
│   ├── inference/
│   │   ├── load.py                     # Model loading (cached singletons)
│   │   ├── predict.py                  # XGBoost prediction + routing
│   │   ├── predict_ebm.py             # EBM prediction
│   │   └── validate.py                # Feature validation
│   │
│   ├── explain/
│   │   ├── shap_explain.py            # SHAP explanations for XGBoost
│   │   ├── ebm_explain.py             # Native explanations for EBM
│   │   ├── behavior_map.py            # Feature → behavioral category mapping
│   │   └── text_explainer.py          # Human-readable explanation generator
│   │
│   ├── contracts/
│   │   ├── contract.py                # Contract loading & validation logic
│   │   └── feature_contract_v1.json   # 49-feature contract definition
│   │
│   └── artifacts/                      # Trained model files
│       ├── model.joblib               # XGBoost model (226 KB)
│       ├── feature_columns.json       # XGBoost feature order
│       ├── metadata.json              # XGBoost training metadata
│       ├── split.json                 # XGBoost train/test split
│       ├── ebm_model.joblib           # EBM model (1.5 MB)
│       ├── ebm_feature_columns.json   # EBM feature order
│       ├── ebm_metadata.json          # EBM training metadata
│       └── ebm_split.json             # EBM train/test split
│
├── data/processed/
│   └── sessions_features.csv          # Preprocessed feature dataset
│
├── reports/                            # Evaluation reports & charts
│   ├── metrics_test.json              # XGBoost test metrics
│   ├── ebm_metrics_test.json          # EBM test metrics
│   ├── roc_curve.png                  # XGBoost ROC
│   ├── ebm_roc_curve.png             # EBM ROC
│   ├── confusion_matrix_085.png       # XGBoost confusion matrix
│   ├── ebm_confusion_matrix_085.png   # EBM confusion matrix
│   └── ...                            # (more charts and CSVs)
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Train XGBoost
```bash
python -m ml.training.train
```

### Train EBM
```bash
python -m ml.training.train_ebm
```

### Evaluate (generate reports)
```bash
python -m ml.training.evaluate        # XGBoost
python -m ml.training.evaluate_ebm    # EBM
```

### Start the API server
```bash
uvicorn app.main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI to test both endpoints.
