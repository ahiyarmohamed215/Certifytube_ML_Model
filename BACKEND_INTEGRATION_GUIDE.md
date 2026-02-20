# Backend Integration Guide — CertifyTube ML API

> Everything your backend needs to call the ML endpoints.  
> Base URL: `http://<ml-server>:8000`

---

## Quick Start

```
ML Server:   uvicorn app.main:app --host 0.0.0.0 --port 8000
Health:      GET  /health                         → {"status": "ok"}
XGBoost:     POST /engagement/analyze/xgboost     → XGBoostAnalyzeResponse
EBM:         POST /engagement/analyze/ebm         → EBMAnalyzeResponse
Swagger UI:  GET  /docs                           → Interactive API docs
```

---

## 1. Request Format (Same for Both Endpoints)

**Method:** `POST`  
**Content-Type:** `application/json`

```json
{
  "session_id": "string (required) — your unique session identifier",
  "feature_version": "v1.0",
  "features": {
    // ALL 49 features below — EVERY key is REQUIRED, ALL values must be numeric (float/int)
    // Missing or extra keys → 400 error
    // Non-numeric values → 400 error
  }
}
```

---

## 2. The 49 Required Features

Your backend must compute and send **exactly** these 49 features. The order doesn't matter (it's a JSON object), but every key must be present.

```json
{
  "session_duration_sec": 322.95,
  "video_duration_sec": 600.0,
  "last_position_sec": 393.14,
  "completed_flag": 0,
  "watch_time_sec": 502.01,
  "watch_time_ratio": 0.837,
  "completion_ratio": 0.655,
  "engagement_velocity": 1.554,
  "num_pause": 8,
  "total_pause_duration_sec": 49.64,
  "avg_pause_duration_sec": 6.21,
  "median_pause_duration_sec": 4.98,
  "pause_freq_per_min": 0.80,
  "long_pause_count": 2,
  "long_pause_ratio": 0.25,
  "num_seek": 2,
  "num_seek_forward": 1,
  "num_seek_backward": 1,
  "total_seek_forward_sec": 80.92,
  "total_seek_backward_sec": 57.41,
  "avg_seek_forward_sec": 80.92,
  "avg_seek_backward_sec": 57.41,
  "largest_forward_seek_sec": 144.57,
  "largest_backward_seek_sec": 107.17,
  "seek_jump_std_sec": 83.29,
  "seek_forward_ratio": 0.50,
  "seek_backward_ratio": 0.50,
  "skip_time_ratio": 0.135,
  "rewatch_time_ratio": 0.096,
  "rewatch_to_skip_ratio": 0.709,
  "seek_density_per_min": 0.20,
  "first_seek_time_sec": 91.61,
  "early_skip_flag": 0,
  "num_ratechange": 0,
  "time_at_speed_lt1x_sec": 28.30,
  "time_at_speed_1x_sec": 269.88,
  "time_at_speed_gt1x_sec": 24.77,
  "fast_ratio": 0.041,
  "slow_ratio": 0.047,
  "playback_speed_variance": 0.088,
  "avg_playback_rate_when_playing": 1.0,
  "unique_speed_levels": 3,
  "num_buffering_events": 3,
  "buffering_time_sec": 2.94,
  "buffering_freq_per_min": 0.30,
  "play_pause_ratio": 0.75,
  "attention_index": 0.556,
  "skim_flag": 0,
  "deep_flag": 1
}
```

### How to Compute Each Feature

| # | Feature | Formula / Source |
|---|---------|-----------------|
| 1 | `session_duration_sec` | `last_event_timestamp - first_event_timestamp` |
| 2 | `video_duration_sec` | Video metadata (total duration) |
| 3 | `last_position_sec` | Last known playback position |
| 4 | `completed_flag` | `1` if `last_position_sec >= 0.95 * video_duration_sec`, else `0` |
| 5 | `watch_time_sec` | Sum of all play intervals (exclude pauses, buffering) |
| 6 | `watch_time_ratio` | `watch_time_sec / session_duration_sec` |
| 7 | `completion_ratio` | `last_position_sec / video_duration_sec` |
| 8 | `engagement_velocity` | `watch_time_sec / session_duration_sec` |
| 9 | `num_pause` | Count of pause events |
| 10 | `total_pause_duration_sec` | Sum of all pause durations |
| 11 | `avg_pause_duration_sec` | `total_pause_duration_sec / num_pause` (0 if no pauses) |
| 12 | `median_pause_duration_sec` | Median of individual pause durations (0 if no pauses) |
| 13 | `pause_freq_per_min` | `num_pause / (session_duration_sec / 60)` |
| 14 | `long_pause_count` | Pauses > 10 seconds |
| 15 | `long_pause_ratio` | `long_pause_count / num_pause` (0 if no pauses) |
| 16 | `num_seek` | Total seek events (forward + backward) |
| 17 | `num_seek_forward` | Seeks where `new_position > old_position` |
| 18 | `num_seek_backward` | Seeks where `new_position < old_position` |
| 19 | `total_seek_forward_sec` | Sum of `(new_pos - old_pos)` for forward seeks |
| 20 | `total_seek_backward_sec` | Sum of `(old_pos - new_pos)` for backward seeks |
| 21 | `avg_seek_forward_sec` | `total_seek_forward_sec / num_seek_forward` (0 if none) |
| 22 | `avg_seek_backward_sec` | `total_seek_backward_sec / num_seek_backward` (0 if none) |
| 23 | `largest_forward_seek_sec` | Max single forward seek distance (0 if none) |
| 24 | `largest_backward_seek_sec` | Max single backward seek distance (0 if none) |
| 25 | `seek_jump_std_sec` | Std deviation of all seek distances (0 if < 2 seeks) |
| 26 | `seek_forward_ratio` | `num_seek_forward / num_seek` (0 if no seeks) |
| 27 | `seek_backward_ratio` | `num_seek_backward / num_seek` (0 if no seeks) |
| 28 | `skip_time_ratio` | `total_seek_forward_sec / video_duration_sec` |
| 29 | `rewatch_time_ratio` | `total_seek_backward_sec / video_duration_sec` |
| 30 | `rewatch_to_skip_ratio` | `total_seek_backward_sec / total_seek_forward_sec` (0 if no fwd seeks) |
| 31 | `seek_density_per_min` | `num_seek / (watch_time_sec / 60)` |
| 32 | `first_seek_time_sec` | Timestamp of first seek event (0 if no seeks) |
| 33 | `early_skip_flag` | `1` if first seek is forward AND within first 10% of video, else `0` |
| 34 | `num_ratechange` | Count of speed-change events |
| 35 | `time_at_speed_lt1x_sec` | Time spent at playback rate < 1.0 |
| 36 | `time_at_speed_1x_sec` | Time spent at playback rate = 1.0 |
| 37 | `time_at_speed_gt1x_sec` | Time spent at playback rate > 1.0 |
| 38 | `fast_ratio` | `time_at_speed_gt1x_sec / watch_time_sec` |
| 39 | `slow_ratio` | `time_at_speed_lt1x_sec / watch_time_sec` |
| 40 | `playback_speed_variance` | Variance of playback rate samples |
| 41 | `avg_playback_rate_when_playing` | Average playback speed during play (not pause/buffer) |
| 42 | `unique_speed_levels` | Count of distinct speed values used |
| 43 | `num_buffering_events` | Count of buffer events |
| 44 | `buffering_time_sec` | Total time spent buffering |
| 45 | `buffering_freq_per_min` | `num_buffering_events / (session_duration_sec / 60)` |
| 46 | `play_pause_ratio` | `watch_time_sec / total_pause_duration_sec` (handle 0 pause) |
| 47 | `attention_index` | Composite: suggest `watch_time_ratio * (1 - skip_time_ratio)` |
| 48 | `skim_flag` | `1` if `skip_time_ratio > 0.3`, else `0` |
| 49 | `deep_flag` | `1` if `rewatch_time_ratio > 0.05`, else `0` |

### ⚠️ Edge Case Rules

| Situation | What to Do |
|-----------|-----------|
| Division by zero (e.g., `avg_pause / 0 pauses`) | Send `0.0` |
| Feature results in infinity | Send `0.0` |
| Feature results in NaN | Send `0.0` |
| No events at all | Send `0.0` for all features except `video_duration_sec` |

---

## 3. XGBoost Endpoint

### Request

```
POST /engagement/analyze/xgboost
Content-Type: application/json
```

```json
{
  "session_id": "ffacb945-5277-49e6-b173-beb98ddbd91f",
  "feature_version": "v1.0",
  "features": {
    "session_duration_sec": 322.95,
    "video_duration_sec": 600.0,
    "last_position_sec": 393.14,
    "completed_flag": 0,
    "watch_time_sec": 502.01,
    "watch_time_ratio": 0.837,
    "completion_ratio": 0.655,
    "engagement_velocity": 1.554,
    "num_pause": 8,
    "total_pause_duration_sec": 49.64,
    "avg_pause_duration_sec": 6.21,
    "median_pause_duration_sec": 4.98,
    "pause_freq_per_min": 0.80,
    "long_pause_count": 2,
    "long_pause_ratio": 0.25,
    "num_seek": 2,
    "num_seek_forward": 1,
    "num_seek_backward": 1,
    "total_seek_forward_sec": 80.92,
    "total_seek_backward_sec": 57.41,
    "avg_seek_forward_sec": 80.92,
    "avg_seek_backward_sec": 57.41,
    "largest_forward_seek_sec": 144.57,
    "largest_backward_seek_sec": 107.17,
    "seek_jump_std_sec": 83.29,
    "seek_forward_ratio": 0.50,
    "seek_backward_ratio": 0.50,
    "skip_time_ratio": 0.135,
    "rewatch_time_ratio": 0.096,
    "rewatch_to_skip_ratio": 0.709,
    "seek_density_per_min": 0.20,
    "first_seek_time_sec": 91.61,
    "early_skip_flag": 0,
    "num_ratechange": 0,
    "time_at_speed_lt1x_sec": 28.30,
    "time_at_speed_1x_sec": 269.88,
    "time_at_speed_gt1x_sec": 24.77,
    "fast_ratio": 0.041,
    "slow_ratio": 0.047,
    "playback_speed_variance": 0.088,
    "avg_playback_rate_when_playing": 1.0,
    "unique_speed_levels": 3,
    "num_buffering_events": 3,
    "buffering_time_sec": 2.94,
    "buffering_freq_per_min": 0.30,
    "play_pause_ratio": 0.75,
    "attention_index": 0.556,
    "skim_flag": 0,
    "deep_flag": 1
  }
}
```

### Response (200 OK)

```json
{
  "model": "xgboost",
  "session_id": "ffacb945-5277-49e6-b173-beb98ddbd91f",
  "feature_version": "v1.0",
  "engagement_score": 0.92,
  "explanation": "The primary factors influencing this score were attention consistency and content coverage.",
  "shap_top_positive": [
    {
      "feature": "watch_time_ratio",
      "shap_value": 0.45,
      "feature_value": 0.837,
      "behavior_category": "coverage"
    },
    {
      "feature": "completion_ratio",
      "shap_value": 0.32,
      "feature_value": 0.655,
      "behavior_category": "coverage"
    },
    {
      "feature": "attention_index",
      "shap_value": 0.15,
      "feature_value": 0.556,
      "behavior_category": "attention_consistency"
    }
  ],
  "shap_top_negative": [
    {
      "feature": "num_buffering_events",
      "shap_value": -0.03,
      "feature_value": 3.0,
      "behavior_category": "playback_quality"
    },
    {
      "feature": "skip_time_ratio",
      "shap_value": -0.01,
      "feature_value": 0.135,
      "behavior_category": "skipping"
    }
  ]
}
```

---

## 4. EBM Endpoint

### Request

```
POST /engagement/analyze/ebm
Content-Type: application/json
```

Same request body as XGBoost — identical JSON structure.

### Response (200 OK)

```json
{
  "model": "ebm",
  "session_id": "ffacb945-5277-49e6-b173-beb98ddbd91f",
  "feature_version": "v1.0",
  "engagement_score": 0.89,
  "explanation": "The primary factors influencing this score were attention consistency and content coverage.",
  "ebm_top_positive": [
    {
      "feature": "watch_time_ratio",
      "contribution": 1.87,
      "feature_value": 0.837,
      "behavior_category": "coverage"
    },
    {
      "feature": "completion_ratio",
      "contribution": 1.23,
      "feature_value": 0.655,
      "behavior_category": "coverage"
    },
    {
      "feature": "attention_index",
      "contribution": 0.45,
      "feature_value": 0.556,
      "behavior_category": "attention_consistency"
    }
  ],
  "ebm_top_negative": [
    {
      "feature": "num_buffering_events",
      "contribution": -0.12,
      "feature_value": 3.0,
      "behavior_category": "playback_quality"
    },
    {
      "feature": "skip_time_ratio",
      "contribution": -0.05,
      "feature_value": 0.135,
      "behavior_category": "skipping"
    }
  ]
}
```

---

## 5. Response Field Reference

### Fields Common to Both Endpoints

| Field | Type | Description |
|-------|------|-------------|
| `model` | `"xgboost"` or `"ebm"` | Hard literal — use this to distinguish responses |
| `session_id` | string | Echo of request session_id |
| `feature_version` | string | Echo of request version |
| `engagement_score` | float (0.0–1.0) | Model's predicted probability of engagement |
| `explanation` | string | Human-readable text explaining the factors driving the score |

### XGBoost-Only Fields

| Field | Type | Description |
|-------|------|-------------|
| `shap_top_positive` | ShapContributor[] | Top 3 features pushing **toward** engagement |
| `shap_top_negative` | ShapContributor[] | Top 3 features pushing **against** engagement |

**ShapContributor:**

| Field | Type | Description |
|-------|------|-------------|
| `feature` | string | Feature name (from the 49) |
| `shap_value` | float | SHAP value (positive = toward engagement) |
| `feature_value` | float | The raw value sent for this feature |
| `behavior_category` | string | Behavioral category (see category list below) |

### EBM-Only Fields

| Field | Type | Description |
|-------|------|-------------|
| `ebm_top_positive` | EBMContributor[] | Top 3 features pushing **toward** engagement |
| `ebm_top_negative` | EBMContributor[] | Top 3 features pushing **against** engagement |

**EBMContributor:**

| Field | Type | Description |
|-------|------|-------------|
| `feature` | string | Feature name (from the 49) |
| `contribution` | float | EBM term score (positive = toward engagement) |
| `feature_value` | float | The raw value sent for this feature |
| `behavior_category` | string | Behavioral category (see category list below) |


> **Note:** The ML service returns only the raw `engagement_score` and `explanation`. The ENGAGED / NOT_ENGAGED decision and threshold comparison should be handled by your backend.

---

## 7. Error Responses

### 400 — Bad Request

**Missing features:**
```json
{
  "detail": "Missing features: ['attention_index', 'deep_flag']"
}
```

**Extra features:**
```json
{
  "detail": "Unexpected features: ['extra_feature']"
}
```

**Non-numeric value:**
```json
{
  "detail": "Feature 'watch_time_ratio' must be numeric"
}
```

**Invalid feature_version:**
```json
{
  "detail": "Unknown feature contract version: v2.0"
}
```

### 422 — Validation Error (Pydantic)

```json
{
  "detail": [
    {
      "loc": ["body", "session_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 — Internal Server Error

```json
{
  "detail": "Model artifacts missing. Train the model first."
}
```

---

## 8. Code Examples

### Java (Spring Boot — RestTemplate)

```java
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import java.util.*;

public class MLClient {
    private static final String ML_BASE_URL = "http://ml-server:8000";
    private final RestTemplate rest = new RestTemplate();

    public Map<String, Object> analyzeEngagement(
            String sessionId,
            Map<String, Double> features,
            String model  // "xgboost" or "ebm"
    ) {
        String url = ML_BASE_URL + "/engagement/analyze/" + model;

        Map<String, Object> body = new HashMap<>();
        body.put("session_id", sessionId);
        body.put("feature_version", "v1.0");
        body.put("features", features);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        ResponseEntity<Map> response = rest.postForEntity(url, request, Map.class);

        return response.getBody();
    }
}
```

### JavaScript / TypeScript (fetch)

```typescript
const ML_BASE_URL = "http://ml-server:8000";

async function analyzeEngagement(
  sessionId: string,
  features: Record<string, number>,
  model: "xgboost" | "ebm"
) {
  const response = await fetch(
    `${ML_BASE_URL}/engagement/analyze/${model}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        feature_version: "v1.0",
        features: features,
      }),
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  return await response.json();
}

// Usage:
const result = await analyzeEngagement("session-123", computedFeatures, "xgboost");
console.log(result.engagement_score);  // 0.92
console.log(result.explanation);       // "The primary factors influencing this score were..."
```

### cURL (Quick Test)

```bash
curl -X POST http://localhost:8000/engagement/analyze/xgboost \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-001",
    "feature_version": "v1.0",
    "features": {
      "session_duration_sec": 300.0,
      "video_duration_sec": 600.0,
      "last_position_sec": 580.0,
      "completed_flag": 1,
      "watch_time_sec": 290.0,
      "watch_time_ratio": 0.97,
      "completion_ratio": 0.97,
      "engagement_velocity": 0.97,
      "num_pause": 3,
      "total_pause_duration_sec": 10.0,
      "avg_pause_duration_sec": 3.33,
      "median_pause_duration_sec": 3.0,
      "pause_freq_per_min": 0.6,
      "long_pause_count": 0,
      "long_pause_ratio": 0.0,
      "num_seek": 1,
      "num_seek_forward": 0,
      "num_seek_backward": 1,
      "total_seek_forward_sec": 0.0,
      "total_seek_backward_sec": 15.0,
      "avg_seek_forward_sec": 0.0,
      "avg_seek_backward_sec": 15.0,
      "largest_forward_seek_sec": 0.0,
      "largest_backward_seek_sec": 15.0,
      "seek_jump_std_sec": 0.0,
      "seek_forward_ratio": 0.0,
      "seek_backward_ratio": 1.0,
      "skip_time_ratio": 0.0,
      "rewatch_time_ratio": 0.025,
      "rewatch_to_skip_ratio": 0.0,
      "seek_density_per_min": 0.2,
      "first_seek_time_sec": 120.0,
      "early_skip_flag": 0,
      "num_ratechange": 0,
      "time_at_speed_lt1x_sec": 0.0,
      "time_at_speed_1x_sec": 290.0,
      "time_at_speed_gt1x_sec": 0.0,
      "fast_ratio": 0.0,
      "slow_ratio": 0.0,
      "playback_speed_variance": 0.0,
      "avg_playback_rate_when_playing": 1.0,
      "unique_speed_levels": 1,
      "num_buffering_events": 0,
      "buffering_time_sec": 0.0,
      "buffering_freq_per_min": 0.0,
      "play_pause_ratio": 29.0,
      "attention_index": 0.97,
      "skim_flag": 0,
      "deep_flag": 0
    }
  }'
```

---

## 9. Decision Logic for Backend

```
                    ┌─────────────────────────────┐
                    │  Learner finishes watching   │
                    │  a video session             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Backend computes 49         │
                    │  features from raw events    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  POST /engagement/analyze/   │
                    │  xgboost (or ebm)            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  ML returns:                 │
                    │  - engagement_score (0–1)    │
                    │  - explanation (text)        │
                    │  - top contributors          │
                    └──────────────┬──────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ Backend decides:   │
                         │ score >= threshold?│
                         └────┬─────────┬────┘
                              │         │
                         YES  │         │  NO
                              │         │
                    ┌─────────▼──┐  ┌───▼────────────┐
                    │ Grant      │  │ Don't grant     │
                    │ certificate│  │ certificate     │
                    │            │  │                 │
                    │ Store:     │  │ Store:          │
                    │ - score    │  │ - score         │
                    │ - explain  │  │ - explain       │
                    └────────────┘  └─────────────────┘
```

### What to Store in Your Database

| Column | Source | Type |
|--------|--------|------|
| `session_id` | Request | VARCHAR |
| `model_used` | Response → `model` | VARCHAR |
| `engagement_score` | Response → `engagement_score` | FLOAT |
| `status` | Backend decision (score ≥ threshold) | ENUM |
| `explanation` | Response → `explanation` | TEXT |
| `top_positive` | Response → `shap_top_positive` or `ebm_top_positive` | JSON |
| `top_negative` | Response → `shap_top_negative` or `ebm_top_negative` | JSON |
| `analyzed_at` | Your timestamp | TIMESTAMP |

---

## 10. Which Model Should I Call?

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Default / production** | `xgboost` | 100% precision @ 0.85 — zero false positives |
| **Need auditable explanations** | `ebm` | Exact, verifiable term-by-term breakdown |
| **User is appealing a decision** | `ebm` | Can show exactly which behaviors led to denial |
| **Both (comparison / A/B test)** | Call both endpoints | Same request body works for both |
