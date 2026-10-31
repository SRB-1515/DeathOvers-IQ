"""
src/metrics.py
Analytical engine for calculating death-over specific cricket performance metrics
for Batters, Bowlers, Teams, and Venues across IPL history.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

def get_batter_death_stats(df_death: pd.DataFrame, 
                           min_balls: int = 30,
                           season_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """
    Computes comprehensive death overs batting statistics.
    Filters:
        min_balls: Minimum legal balls faced in death overs
        season_range: Optional tuple (start_season, end_season)
    """
    df = df_death.copy()
    if season_range:
        df = df[(df['season'] >= season_range[0]) & (df['season'] <= season_range[1])]

    # Legal deliveries faced by batter (wides don't count as balls faced)
    faced = df[df['is_wide'] == 0]
    
    # Aggregations
    agg_runs = df.groupby('batter')['batter_runs'].sum()
    agg_balls = faced.groupby('batter')['is_legal'].count()
    agg_fours = faced.groupby('batter')['is_boundary_4'].sum()
    agg_sixes = faced.groupby('batter')['is_boundary_6'].sum()
    agg_dots = faced[faced['batter_runs'] == 0].groupby('batter')['is_legal'].count()
    agg_inns = df.groupby('batter')['match_id'].nunique()
    
    # Dismissals: where player_out == batter
    dismissals = df[df['player_out'] == df['batter']].groupby('batter')['is_wicket'].count()
    
    # Assemble DataFrame
    stats = pd.DataFrame({
        'innings': agg_inns,
        'runs': agg_runs,
        'balls': agg_balls,
        'fours': agg_fours,
        'sixes': agg_sixes,
        'dots': agg_dots,
        'dismissals': dismissals
    }).fillna(0)
    
    # Filter by min_balls
    stats = stats[stats['balls'] >= min_balls].copy()
    if stats.empty:
        return stats
        
    stats['fours'] = stats['fours'].astype(int)
    stats['sixes'] = stats['sixes'].astype(int)
    stats['dots'] = stats['dots'].astype(int)
    stats['dismissals'] = stats['dismissals'].astype(int)
    stats['boundaries'] = stats['fours'] + stats['sixes']
    
    # Rate metrics
    stats['strike_rate'] = ((stats['runs'] / stats['balls']) * 100).round(2)
    stats['boundary_pct'] = ((stats['boundaries'] / stats['balls']) * 100).round(2)
    stats['dot_pct'] = ((stats['dots'] / stats['balls']) * 100).round(2)
    stats['average'] = np.where(stats['dismissals'] > 0, 
                                (stats['runs'] / stats['dismissals']).round(2), 
                                stats['runs'])
    stats['balls_per_boundary'] = np.where(stats['boundaries'] > 0, 
                                          (stats['balls'] / stats['boundaries']).round(2), 
                                          np.nan)
    
    # Clutch Impact Score (Weighted composite of Strike Rate, Boundary %, Volume)
    # Higher strike rate and high boundaries in pressure situations
    stats['death_iq_score'] = (
        (stats['strike_rate'] * 0.45) + 
        (stats['boundary_pct'] * 1.5) + 
        (np.log1p(stats['runs']) * 10) - 
        (stats['dot_pct'] * 0.5)
    ).round(1)
    
    stats = stats.reset_index()
    return stats.sort_values(by=['death_iq_score', 'runs'], ascending=[False, False])

def get_batter_over_breakdown(df_death: pd.DataFrame, batter_name: str) -> pd.DataFrame:
    """
    Breakdown of batter performance by each death over (16, 17, 18, 19, 20).
    """
    df = df_death[df_death['batter'] == batter_name].copy()
    if df.empty:
        return pd.DataFrame()
        
    records = []
    # over index 15 = Over 16, 16 = Over 17, ..., 19 = Over 20
    for ov_idx in range(15, 20):
        actual_over = ov_idx + 1
        sub = df[df['over'] == ov_idx]
        sub_faced = sub[sub['is_wide'] == 0]
        
        runs = sub['batter_runs'].sum()
        balls = len(sub_faced)
        fours = sub_faced['is_boundary_4'].sum()
        sixes = sub_faced['is_boundary_6'].sum()
        dots = len(sub_faced[sub_faced['batter_runs'] == 0])
        sr = round((runs / balls * 100), 1) if balls > 0 else 0.0
        
        records.append({
            'over_number': f"Over {actual_over}",
            'over_index': actual_over,
            'runs': runs,
            'balls': balls,
            'fours': int(fours),
            'sixes': int(sixes),
            'dots': dots,
            'strike_rate': sr
        })
    return pd.DataFrame(records)

def get_bowler_death_stats(df_death: pd.DataFrame, 
                           min_balls: int = 30,
                           season_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """
    Computes comprehensive death overs bowling statistics.
    Filters:
        min_balls: Minimum legal balls bowled in death overs
        season_range: Optional tuple (start_season, end_season)
    """
    df = df_death.copy()
    if season_range:
        df = df[(df['season'] >= season_range[0]) & (df['season'] <= season_range[1])]

    # Bowler runs conceded = batter_runs + wides + noballs (legbyes and byes are not bowler's fault)
    df['bowler_runs_conceded'] = df['batter_runs'] + df['is_wide'] * df['extra_runs'] + df['is_noball'] * df['extra_runs']
    
    legal = df[df['is_legal'] == 1]
    
    agg_runs = df.groupby('bowler')['bowler_runs_conceded'].sum()
    agg_balls = legal.groupby('bowler')['is_legal'].sum()
    agg_wkts = df.groupby('bowler')['is_bowler_wicket'].sum()
    agg_dots = legal[legal['total_runs'] == 0].groupby('bowler')['is_legal'].count()
    agg_fours = df.groupby('bowler')['is_boundary_4'].sum()
    agg_sixes = df.groupby('bowler')['is_boundary_6'].sum()
    agg_inns = df.groupby('bowler')['match_id'].nunique()

    stats = pd.DataFrame({
        'innings': agg_inns,
        'balls': agg_balls,
        'runs_conceded': agg_runs,
        'wickets': agg_wkts,
        'dot_balls': agg_dots,
        'fours_conceded': agg_fours,
        'sixes_conceded': agg_sixes
    }).fillna(0)

    stats = stats[stats['balls'] >= min_balls].copy()
    if stats.empty:
        return stats
        
    stats['wickets'] = stats['wickets'].astype(int)
    stats['dot_balls'] = stats['dot_balls'].astype(int)
    stats['fours_conceded'] = stats['fours_conceded'].astype(int)
    stats['sixes_conceded'] = stats['sixes_conceded'].astype(int)
    stats['boundaries_conceded'] = stats['fours_conceded'] + stats['sixes_conceded']
    stats['overs'] = (stats['balls'] // 6) + (stats['balls'] % 6) / 10.0
    
    # Rate metrics
    stats['economy_rate'] = ((stats['runs_conceded'] / stats['balls']) * 6.0).round(2)
    stats['dot_pct'] = ((stats['dot_balls'] / stats['balls']) * 100.0).round(2)
    stats['boundary_pct'] = ((stats['boundaries_conceded'] / stats['balls']) * 100.0).round(2)
    stats['strike_rate'] = np.where(stats['wickets'] > 0, 
                                    (stats['balls'] / stats['wickets']).round(2), 
                                    np.nan)
    stats['bowling_average'] = np.where(stats['wickets'] > 0, 
                                        (stats['runs_conceded'] / stats['wickets']).round(2), 
                                        np.nan)
    
    # Death Bowling Mastery IQ Score
    # Higher dot%, higher wicket rate, lower economy rate
    stats['death_iq_score'] = (
        (stats['dot_pct'] * 1.3) + 
        (stats['wickets'] / (stats['balls'] / 6.0) * 18.0) - 
        (stats['economy_rate'] * 3.5) +
        (np.log1p(stats['balls']) * 4.0)
    ).round(1)
    
    stats = stats.reset_index()
    return stats.sort_values(by=['death_iq_score', 'wickets'], ascending=[False, False])

def get_bowler_over_breakdown(df_death: pd.DataFrame, bowler_name: str) -> pd.DataFrame:
    """
    Breakdown of bowler performance by each death over (16, 17, 18, 19, 20).
    """
    df = df_death[df_death['bowler'] == bowler_name].copy()
    if df.empty:
        return pd.DataFrame()
        
    df['bowler_runs_conceded'] = df['batter_runs'] + df['is_wide'] * df['extra_runs'] + df['is_noball'] * df['extra_runs']
    records = []
    
    for ov_idx in range(15, 20):
        actual_over = ov_idx + 1
        sub = df[df['over'] == ov_idx]
        legal_sub = sub[sub['is_legal'] == 1]
        
        runs = sub['bowler_runs_conceded'].sum()
        balls = len(legal_sub)
        wkts = sub['is_bowler_wicket'].sum()
        dots = len(legal_sub[legal_sub['total_runs'] == 0])
        econ = round((runs / balls * 6.0), 2) if balls > 0 else 0.0
        
        records.append({
            'over_number': f"Over {actual_over}",
            'over_index': actual_over,
            'balls': balls,
            'overs': round(balls / 6.0, 1),
            'runs': runs,
            'wickets': int(wkts),
            'dots': dots,
            'economy': econ
        })
    return pd.DataFrame(records)

def get_team_death_profile(df_death: pd.DataFrame) -> pd.DataFrame:
    """
    Computes death overs batting and bowling profiles for all franchises.
    """
    df = df_death.copy()
    legal = df[df['is_legal'] == 1]
    
    # Batting
    bat_runs = df.groupby('batting_team')['total_runs'].sum()
    bat_balls = legal.groupby('batting_team')['is_legal'].count()
    bat_wkts = df.groupby('batting_team')['is_wicket'].sum()
    bat_fours = df.groupby('batting_team')['is_boundary_4'].sum()
    bat_sixes = df.groupby('batting_team')['is_boundary_6'].sum()
    bat_inns = df.groupby('batting_team')['match_id'].nunique()
    
    # Bowling
    df['bowl_runs'] = df['batter_runs'] + df['is_wide'] * df['extra_runs'] + df['is_noball'] * df['extra_runs']
    bowl_runs = df.groupby('bowling_team')['bowl_runs'].sum()
    bowl_balls = legal.groupby('bowling_team')['is_legal'].count()
    bowl_wkts = df.groupby('bowling_team')['is_bowler_wicket'].sum()
    bowl_inns = df.groupby('bowling_team')['match_id'].nunique()
    
    teams = list(set(bat_runs.index).union(set(bowl_runs.index)))
    
    records = []
    for t in teams:
        b_runs = bat_runs.get(t, 0)
        b_balls = bat_balls.get(t, 1)
        b_inns = bat_inns.get(t, 1)
        bw_runs = bowl_runs.get(t, 0)
        bw_balls = bowl_balls.get(t, 1)
        bw_inns = bowl_inns.get(t, 1)
        
        bat_rr = round((b_runs / b_balls) * 6.0, 2) if b_balls > 0 else 0
        bowl_econ = round((bw_runs / bw_balls) * 6.0, 2) if bw_balls > 0 else 0
        net_death_diff = round(bat_rr - bowl_econ, 2)
        
        records.append({
            'team': t,
            'batting_innings': b_inns,
            'batting_death_rpo': bat_rr,
            'batting_avg_runs_per_death': round(b_runs / max(1, b_inns), 1),
            'batting_boundary_pct': round(((bat_fours.get(t, 0) + bat_sixes.get(t, 0)) / max(1, b_balls)) * 100, 1),
            'bowling_innings': bw_inns,
            'bowling_death_econ': bowl_econ,
            'bowling_wkts_per_inns': round(bowl_wkts.get(t, 0) / max(1, bw_inns), 2),
            'net_death_differential': net_death_diff
        })
        
    df_teams = pd.DataFrame(records)
    # Only keep teams with at least 15 death innings
    df_teams = df_teams[df_teams['batting_innings'] >= 15].sort_values(by='net_death_differential', ascending=False)
    return df_teams

def get_venue_death_profile(df_death: pd.DataFrame, df_checkpoints: pd.DataFrame, min_matches: int = 10) -> pd.DataFrame:
    """
    Computes venue-level death scoring benchmarks and typical death totals.
    """
    df = df_death.copy()
    legal = df[df['is_legal'] == 1]
    
    v_runs = df.groupby('venue')['total_runs'].sum()
    v_balls = legal.groupby('venue')['is_legal'].count()
    v_matches = df.groupby('venue')['match_id'].nunique()
    v_sixes = df.groupby('venue')['is_boundary_6'].sum()
    v_fours = df.groupby('venue')['is_boundary_4'].sum()
    
    # 1st innings final totals and death totals from checkpoints
    cp15 = df_checkpoints[(df_checkpoints['checkpoint_over'] == 15) & (df_checkpoints['innings'] == 1)]
    cp_venue = cp15.groupby('venue').agg({
        'final_total': 'mean',
        'death_runs': 'mean'
    }).rename(columns={'final_total': 'avg_1st_innings_score', 'death_runs': 'avg_death_runs'})
    
    records = []
    for v in v_matches.index:
        matches = v_matches[v]
        if matches < min_matches:
            continue
        runs = v_runs.get(v, 0)
        balls = v_balls.get(v, 1)
        rpo = round((runs / balls) * 6.0, 2)
        b_pct = round(((v_fours.get(v, 0) + v_sixes.get(v, 0)) / balls) * 100.0, 1)
        six_per_match = round(v_sixes.get(v, 0) / matches, 1)
        
        avg_score = round(cp_venue.loc[v, 'avg_1st_innings_score'], 1) if v in cp_venue.index else 165.0
        avg_death = round(cp_venue.loc[v, 'avg_death_runs'], 1) if v in cp_venue.index else 50.0
        
        records.append({
            'venue': v,
            'matches': matches,
            'avg_1st_innings_total': avg_score,
            'avg_death_runs_1st_inn': avg_death,
            'death_rpo': rpo,
            'boundary_pct': b_pct,
            'sixes_per_match_at_death': six_per_match
        })
        
    df_venues = pd.DataFrame(records).sort_values(by='death_rpo', ascending=False)
    return df_venues

def get_h2h_death_matchup(df_death: pd.DataFrame, batter: str, bowler: str) -> Dict:
    """
    Detailed head-to-head matchup between a batter and bowler in death overs.
    """
    sub = df_death[(df_death['batter'] == batter) & (df_death['bowler'] == bowler)]
    if sub.empty:
        return {'balls': 0, 'runs': 0, 'dismissals': 0, 'strike_rate': 0.0, 'dots': 0, 'fours': 0, 'sixes': 0}
        
    legal = sub[sub['is_legal'] == 1]
    runs = sub['batter_runs'].sum()
    balls = len(legal)
    fours = int(legal['is_boundary_4'].sum())
    sixes = int(legal['is_boundary_6'].sum())
    dots = int(len(legal[legal['batter_runs'] == 0]))
    dismissals = int(sub[sub['player_out'] == batter]['is_wicket'].sum())
    sr = round((runs / balls * 100.0), 1) if balls > 0 else 0.0
    
    return {
        'balls': balls,
        'runs': runs,
        'dismissals': dismissals,
        'strike_rate': sr,
        'dots': dots,
        'fours': fours,
        'sixes': sixes
    }
