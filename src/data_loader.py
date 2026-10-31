"""
src/data_loader.py
High-performance parser for Cricsheet IPL JSON files.
Converts raw ball-by-ball match data into structured Parquet tables.
"""

import os
import sys
import glob
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import pandas as pd

if __name__ == "__main__" or "src" not in sys.modules:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import normalize_team, normalize_venue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NON_BOWLER_DISMISSALS = {
    'run out', 'retired hurt', 'retired out', 
    'obstructing the field', 'hit the ball twice'
}

def parse_match_file(filepath: str) -> Tuple[Optional[Dict], List[Dict], List[Dict]]:
    """
    Parse a single Cricsheet IPL JSON match file.
    Returns:
        (match_info_dict, deliveries_list, innings_checkpoints_list)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None, [], []

    info = data.get('info', {})
    match_id = os.path.splitext(os.path.basename(filepath))[0]
    
    # Metadata
    season = str(info.get('season', 'Unknown'))
    dates = info.get('dates', [])
    date_val = dates[0] if dates else 'Unknown'
    venue_raw = info.get('venue', 'Unknown')
    venue = normalize_venue(venue_raw)
    city = info.get('city', venue.split(',')[0].strip())
    teams = [normalize_team(t) for t in info.get('teams', [])]
    
    toss = info.get('toss', {})
    toss_winner = normalize_team(toss.get('winner', ''))
    toss_decision = toss.get('decision', '')
    
    outcome = info.get('outcome', {})
    winner = normalize_team(outcome.get('winner', ''))
    result_type = outcome.get('result', '')
    margin = 0
    margin_type = ''
    if 'by' in outcome:
        if 'runs' in outcome['by']:
            margin = outcome['by']['runs']
            margin_type = 'runs'
        elif 'wickets' in outcome['by']:
            margin = outcome['by']['wickets']
            margin_type = 'wickets'
            
    pom = info.get('player_of_match', [])
    player_of_match = pom[0] if pom else ''

    match_summary = {
        'match_id': match_id,
        'season': season,
        'date': date_val,
        'venue_raw': venue_raw,
        'venue': venue,
        'city': city,
        'team1': teams[0] if len(teams) > 0 else '',
        'team2': teams[1] if len(teams) > 1 else '',
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'winner': winner,
        'result_type': result_type,
        'margin': margin,
        'margin_type': margin_type,
        'player_of_match': player_of_match
    }

    deliveries = []
    innings_checkpoints = []
    
    innings_list = data.get('innings', [])
    for inn_idx, inn_data in enumerate(innings_list, start=1):
        # Only analyze standard 1st and 2nd innings (skip super overs for core totals)
        if inn_idx > 2:
            continue
            
        batting_team = normalize_team(inn_data.get('team', ''))
        # Bowling team is the other team
        bowling_team = ''
        for t in teams:
            if t != batting_team:
                bowling_team = t
                break
        if not bowling_team and len(teams) == 2:
            bowling_team = teams[1] if teams[0] == batting_team else teams[0]
            
        overs_list = inn_data.get('overs', [])
        
        cumulative_runs = 0
        cumulative_wickets = 0
        cumulative_legal_balls = 0
        
        # Track overs 0-14 runs for momentum
        runs_at_overs = {}
        wickets_at_overs = {}
        
        pp_runs = 0
        pp_wickets = 0
        middle_runs = 0
        middle_wickets = 0
        death_runs = 0
        death_wickets = 0

        for over_data in overs_list:
            over_num = over_data.get('over', 0)  # 0-indexed (0 to 19)
            delivs = over_data.get('deliveries', [])
            
            # Phase classification
            if over_num < 6:
                phase = 'Powerplay'
            elif over_num < 15:
                phase = 'Middle'
            else:
                phase = 'Death'
                
            for ball_idx, deliv in enumerate(delivs, start=1):
                batter = deliv.get('batter', '')
                bowler = deliv.get('bowler', '')
                non_striker = deliv.get('non_striker', '')
                
                runs_dict = deliv.get('runs', {})
                b_runs = runs_dict.get('batter', 0)
                e_runs = runs_dict.get('extras', 0)
                tot_runs = runs_dict.get('total', 0)
                
                extras_dict = deliv.get('extras', {})
                is_wide = 1 if 'wides' in extras_dict else 0
                is_noball = 1 if 'noballs' in extras_dict else 0
                is_bye = 1 if 'byes' in extras_dict else 0
                is_legbye = 1 if 'legbyes' in extras_dict else 0
                is_penalty = 1 if 'penalty' in extras_dict else 0
                
                is_legal = 0 if (is_wide or is_noball) else 1
                is_ball_faced = 1 if is_wide == 0 else 0
                
                # Boundaries
                is_four = 1 if b_runs == 4 else 0
                is_six = 1 if b_runs == 6 else 0
                
                # Dot ball (legal ball with 0 total runs scored)
                is_dot = 1 if (is_legal and tot_runs == 0) else 0
                
                # Wickets
                wickets_list = deliv.get('wickets', [])
                is_wkt = 1 if len(wickets_list) > 0 else 0
                player_out = ''
                wkt_kind = ''
                is_bowler_wkt = 0
                
                if is_wkt:
                    first_w = wickets_list[0]
                    player_out = first_w.get('player_out', '')
                    wkt_kind = first_w.get('kind', '')
                    if wkt_kind.lower() not in NON_BOWLER_DISMISSALS:
                        is_bowler_wkt = 1
                
                cumulative_runs += tot_runs
                if is_wkt:
                    cumulative_wickets += 1
                if is_legal:
                    cumulative_legal_balls += 1
                    
                if phase == 'Powerplay':
                    pp_runs += tot_runs
                    if is_wkt:
                        pp_wickets += 1
                elif phase == 'Middle':
                    middle_runs += tot_runs
                    if is_wkt:
                        middle_wickets += 1
                else:
                    death_runs += tot_runs
                    if is_wkt:
                        death_wickets += 1

                deliveries.append({
                    'match_id': match_id,
                    'season': season,
                    'date': date_val,
                    'venue': venue,
                    'innings': inn_idx,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'over': over_num,
                    'ball_in_over': ball_idx,
                    'phase': phase,
                    'batter': batter,
                    'bowler': bowler,
                    'non_striker': non_striker,
                    'batter_runs': b_runs,
                    'extra_runs': e_runs,
                    'total_runs': tot_runs,
                    'is_legal': is_legal,
                    'is_ball_faced': is_ball_faced,
                    'is_wide': is_wide,
                    'is_noball': is_noball,
                    'is_bye': is_bye,
                    'is_legbye': is_legbye,
                    'is_penalty': is_penalty,
                    'is_dot': is_dot,
                    'is_boundary_4': is_four,
                    'is_boundary_6': is_six,
                    'is_wicket': is_wkt,
                    'player_out': player_out,
                    'wicket_kind': wkt_kind,
                    'is_bowler_wicket': is_bowler_wkt,
                    'cumulative_runs': cumulative_runs,
                    'cumulative_wickets': cumulative_wickets
                })

            # Record end of over stats (after over index `over_num`, i.e., over_num + 1 overs completed)
            runs_at_overs[over_num + 1] = cumulative_runs
            wickets_at_overs[over_num + 1] = cumulative_wickets

        final_total = cumulative_runs
        final_wickets = cumulative_wickets
        
        # Checkpoints for ML prediction at overs 15, 16, 17, 18, 19
        # Checkpoint at Over 15 = after 15 overs (indices 0..14)
        for cp_over in [15, 16, 17, 18, 19]:
            if cp_over in runs_at_overs:
                cp_runs = runs_at_overs[cp_over]
                cp_wickets = wickets_at_overs[cp_over]
                # Momentum: runs in last 5 overs
                prior_over = cp_over - 5
                prior_runs = runs_at_overs.get(prior_over, 0)
                prior_wickets = wickets_at_overs.get(prior_over, 0)
                momentum_runs = cp_runs - prior_runs
                momentum_wickets = cp_wickets - prior_wickets
                
                innings_checkpoints.append({
                    'match_id': match_id,
                    'season': season,
                    'venue': venue,
                    'innings': inn_idx,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'checkpoint_over': cp_over,
                    'score_at_checkpoint': cp_runs,
                    'wickets_at_checkpoint': cp_wickets,
                    'crr_at_checkpoint': round(cp_runs / cp_over, 2),
                    'wickets_in_hand': max(0, 10 - cp_wickets),
                    'momentum_runs_last_5': momentum_runs,
                    'momentum_wickets_last_5': momentum_wickets,
                    'pp_runs': pp_runs,
                    'pp_wickets': pp_wickets,
                    'final_total': final_total,
                    'final_wickets': final_wickets,
                    'death_runs': death_runs,
                    'death_wickets': death_wickets
                })

    return match_summary, deliveries, innings_checkpoints

def process_all_raw_matches(raw_dir: str = "data/raw", 
                            processed_dir: str = "data/processed", 
                            max_workers: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process all raw JSON match files in raw_dir and save processed Parquet files.
    """
    os.makedirs(processed_dir, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    logger.info(f"Starting processing of {len(raw_files)} raw match files...")
    
    all_matches = []
    all_deliveries = []
    all_checkpoints = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(parse_match_file, raw_files))
        
    for match_summary, delivs, checkpoints in results:
        if match_summary:
            all_matches.append(match_summary)
        if delivs:
            all_deliveries.extend(delivs)
        if checkpoints:
            all_checkpoints.extend(checkpoints)

    df_matches = pd.DataFrame(all_matches)
    df_deliveries = pd.DataFrame(all_deliveries)
    df_checkpoints = pd.DataFrame(all_checkpoints)

    # Filter death deliveries for fast query execution
    df_death = df_deliveries[df_deliveries['phase'] == 'Death'].copy()

    # Save to parquet
    matches_path = os.path.join(processed_dir, "matches.parquet")
    deliveries_path = os.path.join(processed_dir, "ball_by_ball.parquet")
    death_path = os.path.join(processed_dir, "death_deliveries.parquet")
    checkpoints_path = os.path.join(processed_dir, "innings_checkpoints.parquet")

    df_matches.to_parquet(matches_path, index=False)
    df_deliveries.to_parquet(deliveries_path, index=False)
    df_death.to_parquet(death_path, index=False)
    df_checkpoints.to_parquet(checkpoints_path, index=False)

    logger.info(f"Processed successfully: {len(df_matches)} matches, {len(df_deliveries)} deliveries, {len(df_death)} death deliveries, {len(df_checkpoints)} checkpoints.")
    return df_matches, df_deliveries, df_death, df_checkpoints

def load_processed_data(processed_dir: str = "data/processed", force_reload: bool = False):
    """
    Load processed Parquet datasets from disk. If not found or force_reload is True, processes raw data.
    """
    matches_path = os.path.join(processed_dir, "matches.parquet")
    deliveries_path = os.path.join(processed_dir, "ball_by_ball.parquet")
    death_path = os.path.join(processed_dir, "death_deliveries.parquet")
    checkpoints_path = os.path.join(processed_dir, "innings_checkpoints.parquet")

    if force_reload or not (os.path.exists(matches_path) and os.path.exists(deliveries_path) and os.path.exists(death_path) and os.path.exists(checkpoints_path)):
        logger.info("Processed parquet files not found or reload requested. Building from raw JSON...")
        return process_all_raw_matches(processed_dir=processed_dir)

    df_matches = pd.read_parquet(matches_path)
    df_deliveries = pd.read_parquet(deliveries_path)
    df_death = pd.read_parquet(death_path)
    df_checkpoints = pd.read_parquet(checkpoints_path)

    return df_matches, df_deliveries, df_death, df_checkpoints

if __name__ == "__main__":
    process_all_raw_matches()
