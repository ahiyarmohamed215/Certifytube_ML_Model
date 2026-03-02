from verification.engagement.common.text_explainer import build_user_explanation


def test_engaged_explanation_is_friendly_and_congratulatory():
    explanation = build_user_explanation(
        shap_top_negative=[{"feature": "num_seek", "shap": -0.2, "value": 12}],
        shap_top_positive=[
            {"feature": "completion_ratio", "shap": 0.4, "value": 0.98},
            {"feature": "rewatch_time_ratio", "shap": 0.2, "value": 0.2},
        ],
        engagement_score=0.91,
    )

    assert "Congratulations" in explanation
    assert "91%" in explanation
    assert "Keep this up" in explanation


def test_not_engaged_explanation_is_empathetic_and_actionable():
    explanation = build_user_explanation(
        shap_top_negative=[
            {"feature": "num_seek", "shap": -0.3, "value": 20},
            {"feature": "fast_ratio", "shap": -0.2, "value": 0.6},
        ],
        shap_top_positive=[],
        engagement_score=0.58,
    )

    assert "58%" in explanation
    assert "Congratulations" not in explanation
    # Should contain an empathetic/actionable message, not formal "Status:"
    assert "Status:" not in explanation


def test_not_engaged_skipping_gives_specific_advice():
    explanation = build_user_explanation(
        shap_top_negative=[
            {"feature": "skip_time_ratio", "shap": -0.5, "value": 0.4},
        ],
        shap_top_positive=[],
        engagement_score=0.45,
    )

    assert "skipped" in explanation.lower()
    assert "45%" in explanation
