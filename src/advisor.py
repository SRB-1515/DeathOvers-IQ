"""
src/advisor.py
Strategic Decision Advisor Engine for T20 Death-Overs Cricket.
Provides prescriptive recommendations for:
  1. Tactical Bowler Allocation (e.g. Over 19 / Over 20 matchup optimization).
  2. 2nd Innings Death Chase & Target Defense Probabilistic Simulation.
  3. Clutch Player Archetype Classification.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from src.metrics import get_h2h_death_matchup, get_bowler_death_stats, get_batter_death_stats

def evaluate_death_bowling_options(df_death: pd.DataFrame,
                                   batter_1: str,
                                   batter_2: str,
                                   candidate_bowlers: List[str],
                                   target_over: int = 19) -> pd.DataFrame:
    """
    Evaluates candidate bowlers to bowl a specific critical death over (e.g. Over 19)
    against the two active batters at the crease.
    Computes an empirical Tactical Matchup Edge Score and ranked recommendation.
    """
    df = df_death.copy()
    bw_stats_all = get_bowler_death_stats(df, min_balls=20).set_index('bowler')
    
    # Over index: Over 19 is over index 18 (0-indexed)
    over_idx = target_over - 1
    sub_target_over = df[df['over'] == over_idx].copy()
    sub_target_over['bowler_runs_conceded'] = (
        sub_target_over['batter_runs'] + 
        sub_target_over['is_wide'] * sub_target_over['extra_runs'] + 
        sub_target_over['is_noball'] * sub_target_over['extra_runs']
    )
    legal_target = sub_target_over[sub_target_over['is_legal'] == 1]
    
    # Group by bowler for the specific over
    over_runs = sub_target_over.groupby('bowler')['bowler_runs_conceded'].sum()
    over_balls = legal_target.groupby('bowler')['is_legal'].count()
    over_wkts = sub_target_over.groupby('bowler')['is_bowler_wicket'].sum()
    over_dots = legal_target[legal_target['total_runs'] == 0].groupby('bowler')['is_legal'].count()
    
    results = []
    
    for bowler in candidate_bowlers:
        # 1. Overall Death profile
        overall_econ = 9.8
        overall_dot_pct = 28.0
        overall_balls = 0
        if bowler in bw_stats_all.index:
            row = bw_stats_all.loc[bowler]
            overall_econ = float(row['economy_rate'])
            overall_dot_pct = float(row['dot_pct'])
            overall_balls = int(row['balls'])
            
        # 2. Over-specific record
        ov_balls = int(over_balls.get(bowler, 0))
        ov_runs = float(over_runs.get(bowler, 0))
        ov_wkts = int(over_wkts.get(bowler, 0))
        ov_dots = int(over_dots.get(bowler, 0))
        
        if ov_balls >= 12:
            ov_econ = round((ov_runs / ov_balls) * 6.0, 2)
            ov_dot_pct = round((ov_dots / ov_balls) * 100.0, 1)
        else:
            ov_econ = overall_econ
            ov_dot_pct = overall_dot_pct
            
        # 3. Head-to-Head against Batter 1
        h2h_1 = get_h2h_death_matchup(df, batter_1, bowler) if batter_1 else {'balls': 0, 'runs': 0, 'dismissals': 0, 'strike_rate': 0.0}
        # 4. Head-to-Head against Batter 2
        h2h_2 = get_h2h_death_matchup(df, batter_2, bowler) if batter_2 else {'balls': 0, 'runs': 0, 'dismissals': 0, 'strike_rate': 0.0}
        
        # Calculate Blended Expected Strike Rate from batters
        h2h_sample = h2h_1['balls'] + h2h_2['balls']
        if h2h_sample >= 12:
            blended_batter_sr = ((h2h_1['runs'] + h2h_2['runs']) / h2h_sample) * 100.0
            h2h_weight = min(0.65, h2h_sample / 30.0)
        else:
            blended_batter_sr = 165.0  # League average death strike rate
            h2h_weight = (h2h_sample / 30.0) * 0.4
            
        # Matchup Edge Score calculation:
        # Scale: -100 (Batter Dominance) to +100 (Bowler Dominance)
        # Factor A: Bowler Economy efficiency (baseline ~ 9.5 RPO)
        econ_component = (9.5 - ov_econ) * 8.0
        # Factor B: Dot ball rate (baseline ~ 30%)
        dot_component = (ov_dot_pct - 30.0) * 1.5
        # Factor C: H2H suppression
        h2h_component = (165.0 - blended_batter_sr) * 0.4 * h2h_weight
        # Factor D: Wicket threat bonus
        wkt_component = (ov_wkts / max(1, ov_balls / 6.0)) * 10.0
        
        edge_score = round(econ_component + dot_component + h2h_component + wkt_component, 1)
        # Clamp between -100 and +100
        edge_score = max(-100.0, min(100.0, edge_score))
        
        # Recommendation Category
        if edge_score >= 25.0:
            rec_tag = "⭐ Top Recommendation (Lock)"
        elif edge_score >= 5.0:
            rec_tag = "✅ Reliable Option"
        elif edge_score >= -15.0:
            rec_tag = "⚖️ Neutral / Par Matchup"
        else:
            rec_tag = "⚠️ High Risk / Unfavorable"

        results.append({
            'bowler': bowler,
            'matchup_edge_score': edge_score,
            'recommendation': rec_tag,
            'overall_death_econ': overall_econ,
            'target_over_econ': ov_econ,
            'target_over_dot_pct': ov_dot_pct,
            'h2h_balls_vs_b1': h2h_1['balls'],
            'h2h_sr_vs_b1': h2h_1['strike_rate'],
            'h2h_outs_vs_b1': h2h_1['dismissals'],
            'h2h_balls_vs_b2': h2h_2['balls'],
            'h2h_sr_vs_b2': h2h_2['strike_rate'],
            'h2h_outs_vs_b2': h2h_2['dismissals'],
            'total_death_balls': overall_balls
        })
        
    df_results = pd.DataFrame(results).sort_values(by='matchup_edge_score', ascending=False)
    return df_results

def simulate_death_chase(df_checkpoints: pd.DataFrame,
                         current_score: int,
                         wickets_lost: int,
                         target: int,
                         current_over: int = 15,
                         venue: Optional[str] = None) -> Dict[str, Any]:
    """
    Probabilistic simulation and strategic blueprint for a 2nd innings death chase.
    Uses historical chase outcomes across 1,150+ IPL matches entering the death overs.
    """
    runs_needed = max(0, target - current_score)
    overs_remaining = max(1, 20 - current_over)
    balls_remaining = overs_remaining * 6
    rrr = round((runs_needed / overs_remaining), 2)
    wickets_in_hand = max(0, 10 - wickets_lost)
    
    # Extract 2nd innings death chases from historical checkpoints
    inn1 = df_checkpoints[df_checkpoints['innings'] == 1][['match_id', 'checkpoint_over', 'final_total']].drop_duplicates()
    inn1_targets = inn1[inn1['checkpoint_over'] == 15][['match_id', 'final_total']].rename(columns={'final_total': 'inn1_total'})
    inn1_targets['target_score'] = inn1_targets['inn1_total'] + 1
    
    inn2 = df_checkpoints[(df_checkpoints['innings'] == 2) & (df_checkpoints['checkpoint_over'] == current_over)].merge(
        inn1_targets, on='match_id'
    )
    inn2['hist_runs_needed'] = inn2['target_score'] - inn2['score_at_checkpoint']
    inn2['hist_rrr'] = inn2['hist_runs_needed'] / float(overs_remaining)
    inn2['chase_won'] = (inn2['final_total'] >= inn2['target_score']).astype(int)
    
    # Filter matches in similar state
    # Wickets tolerance: +/- 1 wicket
    w_min = max(0, wickets_in_hand - 1)
    w_max = min(10, wickets_in_hand + 1)
    # RRR tolerance: +/- 2.0 RPO
    r_min = max(0, rrr - 2.0)
    r_max = rrr + 2.0
    
    similar_matches = inn2[
        (inn2['wickets_in_hand'] >= w_min) & 
        (inn2['wickets_in_hand'] <= w_max) & 
        (inn2['hist_rrr'] >= r_min) & 
        (inn2['hist_rrr'] <= r_max)
    ]
    
    if len(similar_matches) >= 10:
        win_prob = round(float(similar_matches['chase_won'].mean()) * 100.0, 1)
        sample_size = len(similar_matches)
    else:
        # Fallback to broader statistical logistic curve
        # Formula calibrated to IPL death chase empirical boundaries
        log_odds = (wickets_in_hand * 0.75) - (rrr * 0.62) + 1.2
        p = 1.0 / (1.0 + np.exp(-log_odds))
        win_prob = round(float(p * 100.0), 1)
        sample_size = len(similar_matches)
        
    # Pressure category
    if win_prob >= 70.0:
        pressure_label = "🟢 Highly Favorable Chase"
        tactics = "Batting team in command. Minimize dot-ball clusters and focus on 1-2 boundary balls per over to cruise."
    elif win_prob >= 45.0:
        pressure_label = "🟡 Evenly Poised Contest"
        tactics = "Target one specific bowler in overs 16-17 to bring RRR under 10.0 before the 19th over."
    elif win_prob >= 20.0:
        pressure_label = "🟠 High Pressure Chase"
        tactics = "Requires maximum risk-taking. Must hit 2+ boundaries every over; cannot afford maiden or 2-run overs."
    else:
        pressure_label = "🔴 Extreme Miracle Required"
        tactics = "All-out boundary assault needed. Strike rotation is secondary to immediate six-hitting."
        
    # Boundary frequency required
    boundaries_needed_approx = int(np.ceil((runs_needed * 0.70) / 5.0))
    
    return {
        'runs_needed': runs_needed,
        'overs_remaining': overs_remaining,
        'balls_remaining': balls_remaining,
        'required_run_rate': rrr,
        'wickets_in_hand': wickets_in_hand,
        'win_probability_pct': win_prob,
        'similar_matches_sample': sample_size,
        'pressure_assessment': pressure_label,
        'tactical_blueprint': tactics,
        'boundaries_needed_approx': boundaries_needed_approx
    }

def get_clutch_archetypes(df_death: pd.DataFrame, min_balls: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Segments batters and bowlers into distinct strategic tactical archetypes.
    """
    # Batters
    df_b = get_batter_death_stats(df_death, min_balls=min_balls)
    b_archetypes = []
    for _, row in df_b.iterrows():
        sr = row['strike_rate']
        bpct = row['boundary_pct']
        dpct = row['dot_pct']
        avg = row['average']
        
        if sr >= 195.0 and bpct >= 24.0:
            arch = "⚡ Nuclear Finisher"
        elif sr >= 165.0 and dpct <= 32.0:
            arch = "🎯 Master Closer / Anchor"
        elif sr >= 170.0 and dpct > 35.0:
            arch = "💥 High-Risk Power Hitter"
        elif avg >= 28.0 and sr >= 145.0:
            arch = "🛡️ Stable Accelerator"
        else:
            arch = "🔄 Strike Rotator"
            
        b_archetypes.append(arch)
    df_b['archetype'] = b_archetypes
    
    # Bowlers
    df_bw = get_bowler_death_stats(df_death, min_balls=min_balls)
    bw_archetypes = []
    for _, row in df_bw.iterrows():
        econ = row['economy_rate']
        dpct = row['dot_pct']
        sr = row['strike_rate'] if pd.notnull(row['strike_rate']) else 30.0
        
        if econ <= 8.3 and dpct >= 35.0:
            arch = "🔒 Lockdown Enforcer"
        elif sr <= 13.0 and econ <= 9.6:
            arch = "⚡ Strike Wicket Hunter"
        elif econ <= 9.3:
            arch = "⚖️ Control Specialist"
        else:
            arch = "⚠️ High-Variance Closer"
            
        bw_archetypes.append(arch)
    df_bw['archetype'] = bw_archetypes
    
    return df_b, df_bw
