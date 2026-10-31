"""
tests/test_advisor.py
Unit tests for the T20 Strategy Advisor Engine.
"""

from src.data_loader import load_processed_data
from src.advisor import evaluate_death_bowling_options, simulate_death_chase, get_clutch_archetypes

def test_evaluate_death_bowling_options():
    _, _, df_death, _ = load_processed_data()
    candidates = ["JJ Bumrah", "SL Malinga", "B Kumar"]
    res = evaluate_death_bowling_options(df_death, "MS Dhoni", "RA Jadeja", candidates, target_over=19)
    assert not res.empty
    assert len(res) == len(candidates)
    assert 'matchup_edge_score' in res.columns
    assert 'recommendation' in res.columns
    assert 'target_over_econ' in res.columns
    # Scores must be within bounded range [-100, 100]
    assert (res['matchup_edge_score'] >= -100).all()
    assert (res['matchup_edge_score'] <= 100).all()

def test_evaluate_bowler_without_death_overs():
    """Verify advisor gracefully handles bowler with minimal/no historical death overs."""
    _, _, df_death, _ = load_processed_data()
    candidates = ["Unknown Bowler XYZ"]
    res = evaluate_death_bowling_options(df_death, "MS Dhoni", "RA Jadeja", candidates, target_over=19)
    assert not res.empty
    assert len(res) == 1
    assert 'matchup_edge_score' in res.columns

def test_simulate_death_chase_favorable():
    _, _, _, df_checkpoints = load_processed_data()
    # Scenario: 150/3 chasing 190 at over 15 (needs 40 off 30, RRR = 8.0, 7 wkts in hand)
    sim = simulate_death_chase(df_checkpoints, current_score=150, wickets_lost=3, target=190, current_over=15)
    assert 'win_probability_pct' in sim
    assert 0.0 <= sim['win_probability_pct'] <= 100.0
    assert sim['runs_needed'] == 40
    assert sim['required_run_rate'] == 8.0
    assert sim['wickets_in_hand'] == 7
    # High win rate expected for 40 off 30 with 7 wkts in hand
    assert sim['win_probability_pct'] > 50.0

def test_simulate_death_chase_extreme_unlikely():
    _, _, _, df_checkpoints = load_processed_data()
    # Scenario: 110/8 chasing 210 at over 15 (needs 100 off 30, RRR = 20.0, 2 wkts in hand)
    sim = simulate_death_chase(df_checkpoints, current_score=110, wickets_lost=8, target=210, current_over=15)
    assert sim['win_probability_pct'] < 15.0

def test_simulate_death_chase_target_already_reached():
    _, _, _, df_checkpoints = load_processed_data()
    # Target reached already (current_score >= target)
    sim = simulate_death_chase(df_checkpoints, current_score=185, wickets_lost=2, target=180, current_over=16)
    assert sim['runs_needed'] == 0
    assert sim['win_probability_pct'] == 100.0

def test_get_clutch_archetypes():
    _, _, df_death, _ = load_processed_data()
    df_b, df_bw = get_clutch_archetypes(df_death, min_balls=50)
    assert not df_b.empty
    assert not df_bw.empty
    assert 'archetype' in df_b.columns
    assert 'archetype' in df_bw.columns
    assert len(df_b['archetype'].unique()) >= 3
    assert len(df_bw['archetype'].unique()) >= 3
