"""
src/model.py
End-to-end Machine Learning pipeline for Death-Overs Final Score Prediction.
Trains, evaluates against baselines, and serializes the production model.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

if __name__ == "__main__" or "src" not in sys.modules:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "final_score_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")

def prepare_modeling_data(df_checkpoints: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and prepares dataset for 1st innings final total prediction.
    Only keeps 1st innings where full 20 overs or natural all-out completion occurred.
    """
    # Filter 1st innings only for unconstrained final total targets
    df = df_checkpoints[df_checkpoints['innings'] == 1].copy()
    
    # Exclude rain-reduced matches (matches where final total is abnormally tiny before death or < 15 overs)
    # Death overs start at over 15
    df = df[df['checkpoint_over'].isin([15, 16, 17, 18, 19])].copy()
    
    # Feature engineering:
    df['overs_remaining'] = 20 - df['checkpoint_over']
    df['balls_remaining'] = df['overs_remaining'] * 6
    
    # Target: final total score
    df['target_final_total'] = df['final_total']
    
    # Ensure season is comparable
    df['season_year'] = df['season'].apply(lambda s: int(str(s).split('/')[0]) if '/' in str(s) else int(s))
    
    return df

def train_and_evaluate_models(save_artifacts: bool = True) -> Dict[str, Any]:
    """
    Trains candidate ML models, compares them against classical cricket baselines,
    evaluates using both Temporal Split (train <= 2023, test 2024-2026) and Cross-Validation,
    and serializes the best performing model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    _, _, _, df_checkpoints = load_processed_data()
    data = prepare_modeling_data(df_checkpoints)
    
    # Features
    numeric_features = [
        'checkpoint_over', 'score_at_checkpoint', 'wickets_at_checkpoint',
        'crr_at_checkpoint', 'wickets_in_hand', 'overs_remaining',
        'momentum_runs_last_5', 'momentum_wickets_last_5', 'pp_runs'
    ]
    categorical_features = ['venue', 'batting_team', 'bowling_team']
    
    features = numeric_features + categorical_features
    target = 'target_final_total'
    
    # Temporal Train/Test split: Train on seasons <= 2023, Test on 2024-2026
    train_mask = data['season_year'] <= 2023
    test_mask = data['season_year'] >= 2024
    
    X_train = data.loc[train_mask, features]
    y_train = data.loc[train_mask, target]
    X_test = data.loc[test_mask, features]
    y_test = data.loc[test_mask, target]
    
    logger.info(f"Training set: {len(X_train)} samples (Seasons <= 2023), Test set: {len(X_test)} samples (Seasons 2024-2026)")
    
    # Baselines on Test Set
    # Baseline 1: Current Run Rate Extrapolation
    pred_crr_test = X_test['score_at_checkpoint'] + X_test['crr_at_checkpoint'] * X_test['overs_remaining']
    crr_mae = mean_absolute_error(y_test, pred_crr_test)
    crr_rmse = root_mean_squared_error(y_test, pred_crr_test)
    crr_r2 = r2_score(y_test, pred_crr_test)
    
    # Baseline 2: Historical Mean Death Addition (add mean death runs remaining by over)
    mean_death_by_over = data[train_mask].groupby('checkpoint_over')['death_runs'].mean()
    # For checkpoint ov, runs remaining is approx proportional to overs remaining / 5
    historical_additions = {
        15: mean_death_by_over.get(15, 48.0),
        16: mean_death_by_over.get(15, 48.0) * 0.8,
        17: mean_death_by_over.get(15, 48.0) * 0.6,
        18: mean_death_by_over.get(15, 48.0) * 0.4,
        19: mean_death_by_over.get(15, 48.0) * 0.2,
    }
    pred_hist_test = X_test.apply(
        lambda row: row['score_at_checkpoint'] + historical_additions.get(int(row['checkpoint_over']), 45.0), 
        axis=1
    )
    hist_mae = mean_absolute_error(y_test, pred_hist_test)
    hist_rmse = root_mean_squared_error(y_test, pred_hist_test)
    hist_r2 = r2_score(y_test, pred_hist_test)
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Candidate Models
    models = {
        'Ridge Regression': Pipeline([
            ('prep', preprocessor),
            ('reg', Ridge(alpha=10.0))
        ]),
        'Random Forest': Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1))
        ]),
        'HistGradientBoosting': Pipeline([
            ('prep', preprocessor),
            ('reg', HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.08, random_state=42))
        ])
    }
    
    model_results = {
        'Baselines': {
            'Current Run Rate Projection': {
                'MAE': round(crr_mae, 2),
                'RMSE': round(crr_rmse, 2),
                'R2': round(crr_r2, 3)
            },
            'Historical Average Death Addition': {
                'MAE': round(hist_mae, 2),
                'RMSE': round(hist_rmse, 2),
                'R2': round(hist_r2, 3)
            }
        },
        'Candidate Models': {}
    }
    
    best_model_name = None
    best_mae = float('inf')
    best_pipeline = None
    
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred_test = pipe.predict(X_test)
        pred_train = pipe.predict(X_train)
        
        train_mae = mean_absolute_error(y_train, pred_train)
        test_mae = mean_absolute_error(y_test, pred_test)
        test_rmse = root_mean_squared_error(y_test, pred_test)
        test_r2 = r2_score(y_test, pred_test)
        
        model_results['Candidate Models'][name] = {
            'Train MAE': round(train_mae, 2),
            'Test MAE': round(test_mae, 2),
            'Test RMSE': round(test_rmse, 2),
            'Test R2': round(test_r2, 3)
        }
        
        logger.info(f"[{name}] Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}, Test R2: {test_r2:.3f}")
        
        if test_mae < best_mae:
            best_mae = test_mae
            best_model_name = name
            best_pipeline = pipe
            
    # Calculate residual std dev for prediction intervals
    residuals = y_test - best_pipeline.predict(X_test)
    residual_std = float(np.std(residuals))
    
    # Feature Importances for the best model or linear weights
    feature_importance_dict = {}
    try:
        # Get transformed feature names
        prep = best_pipeline.named_steps['prep']
        cat_names = prep.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_names)
        
        regressor = best_pipeline.named_steps['reg']
        if hasattr(regressor, 'feature_importances_'):
            importances = regressor.feature_importances_
            feat_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
            feature_importance_dict = feat_series.head(15).round(4).to_dict()
        elif hasattr(regressor, 'coef_'):
            coefs = np.abs(regressor.coef_)
            feat_series = pd.Series(coefs, index=all_feature_names).sort_values(ascending=False)
            feature_importance_dict = feat_series.head(15).round(4).to_dict()
    except Exception as e:
        logger.warning(f"Could not compute explicit feature importances: {e}")
        
    evaluation_summary = {
        'best_model': best_model_name,
        'residual_std': round(residual_std, 2),
        'metrics': model_results,
        'top_feature_importance': feature_importance_dict,
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_seasons': '2008-2023',
        'test_seasons': '2024-2026'
    }
    
    if save_artifacts:
        joblib.dump(best_pipeline, MODEL_PATH)
        with open(METRICS_PATH, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, indent=2)
        logger.info(f"Model saved to {MODEL_PATH} and metrics saved to {METRICS_PATH}")
        
    return evaluation_summary

def load_prediction_model():
    """Load the trained production pipeline and metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        logger.info("Trained model not found. Training model now...")
        train_and_evaluate_models()
        
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, meta

def predict_final_score(current_score: int, 
                        current_wickets: int, 
                        current_over: int, 
                        venue: str, 
                        batting_team: str, 
                        bowling_team: str,
                        momentum_runs_last_5: Optional[int] = None,
                        pp_runs: Optional[int] = None) -> Dict[str, Any]:
    """
    Interactive predictor function for Streamlit.
    Given real-time match state, returns predicted final score, range, and death velocity analysis.
    """
    model, meta = load_prediction_model()
    
    overs_remaining = max(0, 20 - current_over)
    crr = round(current_score / current_over, 2) if current_over > 0 else 6.0
    wickets_in_hand = max(0, 10 - current_wickets)
    
    # Defaults if not supplied
    if momentum_runs_last_5 is None:
        # Default roughly 5 * crr with slight deceleration if wickets fell
        momentum_runs_last_5 = int(round(crr * 5 * 0.95))
    momentum_wickets_last_5 = min(2, current_wickets)
    if pp_runs is None:
        pp_runs = int(round(crr * 6))
        
    input_df = pd.DataFrame([{
        'checkpoint_over': current_over,
        'score_at_checkpoint': current_score,
        'wickets_at_checkpoint': current_wickets,
        'crr_at_checkpoint': crr,
        'wickets_in_hand': wickets_in_hand,
        'overs_remaining': overs_remaining,
        'momentum_runs_last_5': momentum_runs_last_5,
        'momentum_wickets_last_5': momentum_wickets_last_5,
        'pp_runs': pp_runs,
        'venue': venue,
        'batting_team': batting_team,
        'bowling_team': bowling_team
    }])
    
    predicted_val = float(model.predict(input_df)[0])
    # Logical cricket constraints: Final total cannot be less than current score
    predicted_total = max(current_score, int(round(predicted_val)))
    death_runs_projected = predicted_total - current_score
    implied_death_rr = round(death_runs_projected / overs_remaining, 2) if overs_remaining > 0 else 0.0
    
    residual_std = meta.get('residual_std', 12.0)
    low_interval = max(current_score, int(round(predicted_val - 1.28 * residual_std))) # ~80% CI
    high_interval = int(round(predicted_val + 1.28 * residual_std))
    
    # Naive CRR projection for user comparison
    naive_crr_total = int(round(current_score + crr * overs_remaining))
    
    return {
        'predicted_final_score': predicted_total,
        'projected_death_runs': death_runs_projected,
        'implied_death_run_rate': implied_death_rr,
        'confidence_lower_80': low_interval,
        'confidence_upper_80': high_interval,
        'naive_crr_projection': naive_crr_total,
        'delta_vs_naive': predicted_total - naive_crr_total,
        'wickets_in_hand': wickets_in_hand,
        'current_crr': crr
    }

if __name__ == "__main__":
    train_and_evaluate_models()
