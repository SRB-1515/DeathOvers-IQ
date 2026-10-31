# ⚡ DeathOvers IQ — A Data-Driven T20 Cricket Strategy Advisor
### Official Submission for AQX Sports Analytics Data Bowl 4.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-21%20passed-success.svg)](file:///c:/Users/Admin/Desktop/Deathovers/tests)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20Source-Cricsheet-informational.svg)](https://cricsheet.org/)

---

## 📌 Executive Summary

In T20 cricket, the **Death Overs (Overs 16–20)** represent the highest-leverage phase of the match. A single 22-run 19th over or a clutch 4-wicket collapse routinely flips win probability curves. Despite its critical importance, teams often rely on crude heuristics (e.g. current run rate extrapolation or rigid bowler rotations) that fail to account for wickets in hand, venue dimensions, and matchup suppression.

**DeathOvers IQ** is a competition-ready, end-to-end cricket analytics platform and tactical strategy advisor built on **19 IPL seasons (2008–2026)** comprising **1,243 matches**, **295,557 deliveries**, and **67,173 death-over balls**.

The application combines:
1. **Prescriptive Tactical Strategy Advisor:** Solves the captain's "19th & 20th Over Bowler Allocation" dilemma and provides empirical 2nd innings death chase win probabilities based on 1,150+ historical match states.
2. **Machine Learning Final Score Predictor:** Predicts final 20-over totals at any death checkpoint (completed Overs 15 through 19) with 80% confidence intervals and trajectory progression.
3. **Deep Macro & Micro Analytics:** Interactive batter clutch ratings, bowler mastery metrics, franchise net death differentials, and over-by-over (16th vs 20th) velocity curves.

---

## 🏛️ System Architecture

```
Deathovers/
├── data/
│   ├── raw/                         # 1,243 Cricsheet IPL JSON match files (untouched, preserved)
│   └── processed/                   # High-speed Parquet columnar cache (< 3 MB total, < 50ms load)
│       ├── ball_by_ball.parquet     # 295,557 delivery records
│       ├── death_deliveries.parquet # 67,173 death-over deliveries (Overs 16-20)
│       ├── innings_checkpoints.parquet # Checkpoint dataset (Overs 15-19)
│       └── matches.parquet          # Match-level summaries & results
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Franchise & venue normalization, IPL color scheme
│   ├── data_loader.py               # Multithreaded parser (parses 1,243 files in ~7.5s)
│   ├── metrics.py                   # Batter, bowler, team, and venue death analytics engines
│   ├── advisor.py                   # Prescriptive strategy engine (bowler selection & chase sim)
│   └── model.py                     # ML pipeline, out-of-time evaluation & inference
├── models/
│   ├── final_score_model.joblib     # Production serialized pipeline
│   └── evaluation_metrics.json      # Out-of-time evaluation metrics & feature weights
├── tests/
│   ├── test_utils.py                # Normalization unit tests
│   ├── test_data_loader.py          # Cricket rules, extras & checkpoint tests
│   ├── test_metrics.py              # Batting/bowling/team metric tests
│   ├── test_advisor.py              # Strategy advisor & chase simulator tests
│   └── test_model.py                # Inference boundary and confidence interval tests
├── app.py                           # 7-Tab Streamlit Interactive Application
├── requirements.txt                 # Frozen dependencies
└── README.md                        # Documentation & competition report
```

---

## 🏏 Rigorous Cricket Rule Compliance

`DeathOvers IQ` strictly implements official MCC Laws of Cricket and standard sports analytics accounting:
- **Death Phase Definition:** Exactly Overs 16.1 through 20.0 (0-indexed over numbers `15, 16, 17, 18, 19`).
- **Legal vs. Illegal Balls:** Wides and No-balls are marked as illegal deliveries and do not increment the bowler's legal ball count.
- **Balls Faced Accounting:** Wides do **not** increment a batter's balls faced; No-balls **do** increment balls faced.
- **Bowler Wickets vs. Fielding Wickets:** Run-outs, retired hurts/outs, and obstructing field dismissals are strictly excluded from bowler wickets and bowling strike rate.
- **Bowler Runs Conceded:** Byes and Leg-byes are team extras and are **not** charged to the bowler's runs conceded or economy rate.

---

## 🧠 Strategic Decision Support Features

### 1. Tactical Bowler Selector (19th / 20th Over Dilemma)
- Ingests the two active batters at the crease (e.g. MS Dhoni & Ravindra Jadeja, or Andre Russell & Rinku Singh).
- Filters squad bowling options and evaluates bowler overall death economy, target over-specific economy (e.g. historical Over 19 performance), dot-ball frequency, and Head-to-Head suppression.
- Outputs a **Tactical Matchup Edge Score** (-100 to +100) and explicit decision tags:
  - `⭐ Top Recommendation (Lock)`
  - `✅ Reliable Option`
  - `⚖️ Neutral / Par Matchup`
  - `⚠️ High Risk / Unfavorable`

### 2. 2nd Innings Death Chase & Target Defense Simulator
- Given target, current score entering the death overs, overs completed, and wickets lost.
- Computes:
  - Required Run Rate (RRR)
  - Empirical Historical Win Probability (trained on 1,157 actual IPL 2nd innings death chases)
  - Required boundary frequency (e.g. ~2.3 boundaries per over needed)
  - Interactive empirical Heatmap showing why **Wickets in Hand** is the single greatest determinant of chase victory:
    - *At RRR 12–14 RPO, teams with 7 wickets in hand win 40% of the time, whereas teams with 4 wickets in hand win only 7%!*

### 3. Clutch Tactical Archetypes
- **Batters:** Segmented into *⚡ Nuclear Finishers* (SR > 195, Boundary% > 24%), *🎯 Master Closers / Anchors* (SR 165–185, Dot% < 32%), *💥 High-Risk Power Hitters*, and *🛡️ Stable Accelerators*.
- **Bowlers:** Segmented into *🔒 Lockdown Enforcers* (Econ <= 8.3, Dot% >= 35%), *⚡ Strike Wicket Hunters* (Balls/Wicket <= 13), *⚖️ Control Specialists*, and *⚠️ High-Variance Closers*.

---

## 🔬 Machine Learning Evaluation & Benchmarks

The model predicts the **final 20-over total** from any death checkpoint (completed Overs 15, 16, 17, 18, 19). It was trained on **5,025 checkpoints (Seasons 2008–2023)** and tested on **1,060 unseen future checkpoints (Seasons 2024–2026)**:

| Model / Baseline | Test MAE (Runs) | Test RMSE | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: |
| 📌 **Current Run Rate (CRR) Extrapolation** *(Baseline)* | 10.20 | 13.46 | 0.858 | Baseline |
| 📌 **Historical Average Death Addition** *(Baseline)* | 9.70 | 12.88 | 0.870 | Baseline |
| 🔹 **HistGradientBoosting Regressor** | 10.14 | 13.84 | 0.850 | Candidate |
| 🔹 **Random Forest Regressor** | 9.33 | 12.61 | 0.875 | Candidate |
| ⭐ **Ridge Regression (Regularized Pipeline)** | **8.68** | **11.37** | **0.899** | **Selected Production Model** |

### Key Model Insights:
- **Ridge Regression** achieved the best generalization score (**MAE 8.68 runs**, $R^2 = 0.899$), successfully resisting overfitting to recent extreme high-scoring matches while adjusting for wickets in hand and venue elevation.
- **Wickets in Hand Multiplier:** Holding 7+ wickets in hand at Over 15 adds an average of +14.2 to +18.5 runs in the death overs compared to entering with 4 or fewer wickets down.
- **Venue Lift:** Wankhede Stadium (+12.2 RPO) and M. Chinnaswamy Stadium (+12.8 RPO) exhibit marked death acceleration compared to venues like Ekana Cricket Stadium or Chepauk.

---

## ⚡ Quickstart & Local Execution

The application is completely self-contained. It requires **no API keys, no paid subscriptions, no external database server, and no live internet connection** after dependency installation.

### 1. Prerequisites
- Python 3.10+ (tested and verified on Python 3.14)
- Pip

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Pytest Suite
Run all 21 unit and integration tests:
```bash
python -m pytest tests/
```
Output:
```text
============================= 21 passed in 11.01s =============================
```

### 4. Launch the Streamlit Application
Run locally with:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Analytical Highlights from the Data

- **AB de Villiers** is the greatest death-overs batter in IPL history: **1,868 runs** in overs 16–20 at a **222.91 strike rate** with **140 sixes** and a 214.4 Death IQ Clutch Score.
- **Sunil Narine** and **Lasith Malinga** form the all-time death bowling benchmark: Narine maintains an extraordinary **7.25 career death economy**, while Malinga took **108 death wickets** with a 31.3% dot-ball rate.
- **Chennai Super Kings** (+0.84) and **Mumbai Indians** (+0.83) lead the historical Franchise Death Differential matrix, explaining their combined 10 IPL titles.

---

## 📜 Submission Details
- **Competition:** AQX Sports Analytics Data Bowl 4.0
- **Author:** DeathOvers IQ Analytics Team
- **Data Source:** [Cricsheet](https://cricsheet.org/) ball-by-ball IPL dataset (licensed under CC BY 4.0).
