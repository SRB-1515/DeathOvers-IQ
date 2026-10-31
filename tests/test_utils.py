"""
tests/test_utils.py
Unit tests for data normalization utilities.
"""

from src.utils import normalize_team, normalize_venue, TEAM_COLORS

def test_normalize_team():
    assert normalize_team("Royal Challengers Bangalore") == "Royal Challengers Bengaluru"
    assert normalize_team("Kings XI Punjab") == "Punjab Kings"
    assert normalize_team("Delhi Daredevils") == "Delhi Capitals"
    assert normalize_team("Mumbai Indians") == "Mumbai Indians"
    assert normalize_team("") == "Unknown Team"
    assert normalize_team(None) == "Unknown Team"

def test_normalize_venue():
    assert normalize_venue("Eden Gardens, Kolkata") == "Eden Gardens"
    assert normalize_venue("Wankhede Stadium, Mumbai") == "Wankhede Stadium"
    assert normalize_venue("M Chinnaswamy Stadium, Bengaluru") == "M Chinnaswamy Stadium"
    assert normalize_venue("Feroz Shah Kotla") == "Arun Jaitley Stadium"
    assert normalize_venue("") == "Unknown Venue"

def test_team_colors():
    assert "Chennai Super Kings" in TEAM_COLORS
    assert "Mumbai Indians" in TEAM_COLORS
    assert TEAM_COLORS["Chennai Super Kings"].startswith("#")
