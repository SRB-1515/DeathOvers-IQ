"""
tests/test_data_loader.py
Unit tests for Cricsheet match parsing, cricket rule compliance, and data pipeline.
"""

import os
import glob
from src.data_loader import parse_match_file, load_processed_data

def test_parse_single_match():
    files = glob.glob("data/raw/*.json")
    assert len(files) > 0, "Raw JSON files must exist in data/raw"
    sample_file = files[0]
    
    summary, deliveries, checkpoints = parse_match_file(sample_file)
    
    assert summary is not None
    assert 'match_id' in summary
    assert 'venue' in summary
    assert 'season' in summary
    assert len(deliveries) > 0
    
    # Check delivery structure
    first_deliv = deliveries[0]
    assert 'phase' in first_deliv
    assert first_deliv['phase'] in ['Powerplay', 'Middle', 'Death']
    assert 'batter' in first_deliv
    assert 'bowler' in first_deliv
    assert 'total_runs' in first_deliv
    assert 'is_ball_faced' in first_deliv
    
    # Check death phase over definition (overs 16-20 correspond to indices 15-19)
    death_delivs = [d for d in deliveries if d['phase'] == 'Death']
    for d in death_delivs:
        assert d['over'] >= 15 and d['over'] <= 19

def test_cricket_rules_accounting():
    """Verify cricket accounting: legal balls, extras, dismissals."""
    _, df_deliveries, df_death, _ = load_processed_data()
    
    # Wide balls must NOT count as legal balls
    wides = df_deliveries[df_deliveries['is_wide'] == 1]
    assert (wides['is_legal'] == 0).all()
    assert (wides['is_ball_faced'] == 0).all()
    
    # No balls must NOT count as legal balls, but ARE balls faced
    noballs = df_deliveries[df_deliveries['is_noball'] == 1]
    assert (noballs['is_legal'] == 0).all()
    assert (noballs['is_ball_faced'] == 1).all()
    
    # Bowler wickets must NEVER include run outs
    run_outs = df_deliveries[df_deliveries['wicket_kind'].str.lower() == 'run out']
    assert (run_outs['is_bowler_wicket'] == 0).all()
    
    # Phase boundaries:
    # Over index 5 -> Powerplay (Over 6)
    # Over index 6 -> Middle (Over 7)
    # Over index 14 -> Middle (Over 15)
    # Over index 15 -> Death (Over 16)
    assert (df_deliveries[df_deliveries['over'] == 5]['phase'] == 'Powerplay').all()
    assert (df_deliveries[df_deliveries['over'] == 6]['phase'] == 'Middle').all()
    assert (df_deliveries[df_deliveries['over'] == 14]['phase'] == 'Middle').all()
    assert (df_deliveries[df_deliveries['over'] == 15]['phase'] == 'Death').all()

def test_load_processed_data():
    df_matches, df_deliveries, df_death, df_checkpoints = load_processed_data()
    assert len(df_matches) == 1243
    assert len(df_deliveries) > 200000
    assert len(df_death) > 50000
    assert len(df_checkpoints) > 10000
    assert (df_death['phase'] == 'Death').all()
