"""
tests/test_metrics.py
Unit tests for death overs statistical calculation engines.
"""

from src.data_loader import load_processed_data
from src.metrics import (
    get_batter_death_stats,
    get_batter_over_breakdown,
    get_bowler_death_stats,
    get_bowler_over_breakdown,
    get_team_death_profile,
    get_venue_death_profile,
    get_h2h_death_matchup
)

def test_batter_death_stats():
    _, _, df_death, _ = load_processed_data()
    stats = get_batter_death_stats(df_death, min_balls=50)
    assert not stats.empty
    assert 'death_iq_score' in stats.columns
    assert 'strike_rate' in stats.columns
    assert 'boundary_pct' in stats.columns
    # All strike rates should be positive numbers
    assert (stats['strike_rate'] >= 0).all()
    # Boundaries <= balls
    assert (stats['boundaries'] <= stats['balls']).all()

def test_batter_over_breakdown():
    _, _, df_death, _ = load_processed_data()
    # MS Dhoni is known for death overs
    breakdown = get_batter_over_breakdown(df_death, "MS Dhoni")
    assert len(breakdown) == 5
    assert set(breakdown['over_index']) == {16, 17, 18, 19, 20}
    assert breakdown['runs'].sum() > 0

def test_bowler_death_stats():
    _, _, df_death, _ = load_processed_data()
    stats = get_bowler_death_stats(df_death, min_balls=50)
    assert not stats.empty
    assert 'economy_rate' in stats.columns
    assert 'dot_pct' in stats.columns
    assert 'wickets' in stats.columns
    assert (stats['economy_rate'] >= 0).all()

def test_team_and_venue_profiles():
    _, _, df_death, df_checkpoints = load_processed_data()
    team_profile = get_team_death_profile(df_death)
    assert not team_profile.empty
    assert 'net_death_differential' in team_profile.columns
    
    venue_profile = get_venue_death_profile(df_death, df_checkpoints, min_matches=5)
    assert not venue_profile.empty
    assert 'death_rpo' in venue_profile.columns

def test_h2h_matchup():
    _, _, df_death, _ = load_processed_data()
    # Test matchup between AB de Villiers and Jasprit Bumrah
    h2h = get_h2h_death_matchup(df_death, "AB de Villiers", "JJ Bumrah")
    assert 'balls' in h2h
    assert 'runs' in h2h
    assert 'dismissals' in h2h
    assert h2h['balls'] >= 0
