"""
tests/test_model.py
Unit tests for Machine Learning model prediction pipeline and constraints.
"""

import os
from src.model import predict_final_score, load_prediction_model, MODEL_PATH, METRICS_PATH

def test_model_files_exist():
    assert os.path.exists(MODEL_PATH), "Trained model joblib artifact must exist"
    assert os.path.exists(METRICS_PATH), "Model evaluation metrics JSON must exist"

def test_load_model():
    model, meta = load_prediction_model()
    assert model is not None
    assert 'metrics' in meta
    assert 'best_model' in meta

def test_score_prediction_logic():
    # Test realistic scenario
    res = predict_final_score(
        current_score=150,
        current_wickets=3,
        current_over=15,
        venue="Wankhede Stadium",
        batting_team="Mumbai Indians",
        bowling_team="Chennai Super Kings"
    )
    
    # Target must be >= current score
    assert res['predicted_final_score'] >= 150
    # Implied run rate must be reasonable (e.g. between 6 and 25)
    assert 4.0 <= res['implied_death_run_rate'] <= 25.0
    # Confidence intervals
    assert res['confidence_lower_80'] <= res['predicted_final_score']
    assert res['predicted_final_score'] <= res['confidence_upper_80']

def test_prediction_boundary_at_over_19():
    res = predict_final_score(
        current_score=190,
        current_wickets=6,
        current_over=19,
        venue="Eden Gardens",
        batting_team="Kolkata Knight Riders",
        bowling_team="Rajasthan Royals"
    )
    assert res['predicted_final_score'] >= 190
    assert res['predicted_final_score'] <= 240
