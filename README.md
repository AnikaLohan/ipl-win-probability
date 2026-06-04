# IPL Win Probability Engine
### Ball-by-ball win prediction using Machine Learning


## What This Project Does

This project predicts the probability of a team winning an IPL match — updated after **every single ball bowled**.

For example: Mumbai Indians are chasing 180. It's over 15. They need 52 runs off 30 balls with 4 wickets in hand. What's their exact probability of winning? This model answers that — and updates the answer on every delivery.

---

## Why This Project Is Unique

Most IPL projects on the internet do basic analysis — top scorers, team win rates, average scores by venue. This project is different:

- It works at the **ball-by-ball level**, not match level
- It models **game state** (wickets, run rate, phase of play) — not just historical averages
- It uses **LightGBM** which captures non-linear relationships (e.g. losing a wicket in over 18 matters far more than in over 2)
- The model is **calibrated** — a prediction of 70% actually means the team wins 70% of the time
- It is deployed as a **live Streamlit app** where you can input any match state and get an instant prediction

---

## Dataset

**Source:** [Cricsheet.org](https://cricsheet.org/downloads/) — free, open, ball-by-ball data for every IPL match from 2008 to 2024.

**Format:** One JSON file per match (~1000 files total). Each file contains:
- Match metadata (teams, venue, date, toss, result)
- Full ball-by-ball delivery data for both innings
- Extras breakdown (wides, no-balls, leg-byes, byes)
- Wicket details (kind of dismissal, fielders)

**Size after parsing:** ~250,000 rows (one row per delivery across all matches)

---

## Project Structure

```
IplWinProbability/
├── app/
│   └── streamlit_app.py        ← Live match simulator (web app)
├── data/
│   ├── raw/                    ← Original Cricsheet JSON files (1 per match)
│   ├── processed/              ← Cleaned and engineered CSVs
│   └── models/                 ← Saved trained model files
├── notebooks/
│   ├── 01_parse_data.ipynb     ← Step 1: JSON → DataFrame
│   ├── 02_feature_engineering.ipynb  ← Step 2: Build ML features
│   ├── 03_model_training.ipynb ← Step 3: Train and evaluate model
│   └── 04_analysis.ipynb       ← Step 4: Insights and visualizations
├── src/
│   ├── parser.py               ← Function to convert JSON to rows
│   ├── features.py             ← Feature engineering functions
│   └── model.py                ← Model training and inference
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## How It Works — Step by Step

### Step 1: Data Parsing (notebook 01)

Each IPL match is stored as a JSON file. The parser reads every file and converts it into a table where **each row = one delivery**.

**What the parser does:**
- Opens the JSON file and reads match metadata (teams, venue, toss, result)
- Loops through every over and every ball in both innings
- For each ball, it records: runs scored, whether it was a wide or no-ball, whether a wicket fell, and running totals (score so far, wickets so far)
- Wides and no-balls are recorded but do NOT count as legal balls (so `legal_ball` counter only goes up on valid deliveries)
- The 2nd innings target is stored inside the 2nd innings object in the JSON and is extracted correctly

**Output columns:**

| Column | What it means |
|--------|---------------|
| `match_id` | Unique ID for the match |
| `date` | Date the match was played |
| `venue` | Stadium name |
| `inning` | 1 = first innings, 2 = second innings (chase) |
| `over` | Over number (0 to 19) |
| `legal_ball` | Count of legal balls bowled so far in the innings |
| `runs_this_ball` | Runs scored on this specific delivery |
| `runs_so_far` | Total runs scored in the innings up to this ball |
| `wickets_so_far` | Total wickets fallen up to this ball |
| `is_wicket` | 1 if a wicket fell on this ball, 0 otherwise |
| `is_wide` | 1 if this was a wide, 0 otherwise |
| `is_noball` | 1 if this was a no-ball, 0 otherwise |
| `target` | Runs the 2nd innings team needs to win (NaN for 1st innings) |
| `winner` | Which team won the match |
| `toss_winner` | Which team won the toss |
| `toss_decision` | Whether the toss winner chose to bat or field |

**Example output (first match ever — KKR vs RCB, April 18 2008):**
```
(225, 22)
   inning  over  legal_ball  runs_so_far  wickets_so_far  is_wicket  target
0       1     0           1            1               0          0     NaN
1       1     0           2            1               0          0     NaN   ← wide, ball count stayed same
2       1     0           2            2               0          0     NaN
...
```
KKR scored 222 runs. RCB were bowled out for 82 chasing 223. McCullum scored 158*.

---

### Step 2: Feature Engineering (notebook 02)

Raw delivery data is not enough for a model. We engineer features that capture the **actual game situation** at each moment.

**Features built:**

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| `balls_remaining` | 120 − legal_ball | Core pressure metric |
| `wickets_remaining` | 10 − wickets_so_far | Batting resources left |
| `current_run_rate` | runs_so_far ÷ (balls bowled ÷ 6) | How fast batting team is scoring |
| `runs_required` | target − runs_so_far | Runs still needed (2nd innings only) |
| `required_run_rate` | runs_required ÷ (balls_remaining ÷ 6) | How fast they need to score |
| `run_rate_diff` | current_run_rate − required_run_rate | Positive = batting team ahead |
| `phase` | Powerplay (0-5), Middle (6-14), Death (15-19) | Game phase changes dynamics |
| `toss_advantage` | 1 if batting team won toss, else 0 | Toss impact on win rate |

**Target variable:** `batting_team_won` — did the team currently batting end up winning the match? This is 1 or 0 for every single delivery, turning 250,000 rows into 250,000 labeled training examples.

---

### Step 3: Model Training (notebook 03)

**Algorithm: LightGBM (Gradient Boosting)**

Why LightGBM and not simpler models:
- Logistic Regression assumes all features affect the outcome equally and linearly — not true in cricket
- A wicket in over 18 is catastrophic; a wicket in over 2 is recoverable. LightGBM learns this automatically
- LightGBM trains in seconds even on 250,000 rows

**Train/Validation/Test Split — by season, not randomly:**

| Split | Seasons | Why |
|-------|---------|-----|
| Training | 2008–2021 | Historical data to learn from |
| Validation | 2022 | Tune the model |
| Test | 2023–2024 | Final evaluation on unseen data |

Random splitting is wrong here because it leaks future match patterns into training data.

**Calibration:**
Raw LightGBM outputs are scores, not true probabilities. We apply Isotonic Calibration so that a prediction of 0.70 genuinely means the team wins 70% of the time — critical for a win probability display.

**Evaluation metrics:**
- **Brier Score** (lower is better) — measures how accurate probability estimates are
- **Log Loss** (lower is better) — penalizes confident wrong predictions more heavily

---

### Step 4: Streamlit App (app/)

A live web app with two modes:

**Live Input Mode:** Enter current match state (over, score, wickets, target) → get win probability instantly with a gauge chart.

**Historical Replay Mode:** Pick any past IPL match → watch the win probability chart update ball by ball.

**To run the app:**
```bash
streamlit run app/streamlit_app.py
```

---

## How to Run This Project Locally

**Requirements:** Python 3.10+, Windows/Mac/Linux

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/IplWinProbability
cd IplWinProbability

# 2. Create and activate virtual environment
python -m venv ipl_env
ipl_env\Scripts\activate          # Windows
source ipl_env/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn plotly streamlit joblib tqdm jupyter

# 4. Download data from Cricsheet
# Go to https://cricsheet.org/downloads → IPL (JSON) → download ZIP
# Unzip all .json files into data/raw/

# 5. Run notebooks in order
# Open notebooks/01_parse_data.ipynb → run all cells
# Open notebooks/02_feature_engineering.ipynb → run all cells
# Open notebooks/03_model_training.ipynb → run all cells

# 6. Launch the app
streamlit run app/streamlit_app.py
```

---

## Dependencies

```
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.0
lightgbm==4.1.0
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.17.0
streamlit==1.28.0
joblib==1.3.2
tqdm==4.66.1
```

---

## Key Findings

- [x] Model Brier Score: 0.1971
- [x] Model Log Loss: 0.5802
- [x] Model Accuracy: 68.22%
- [x] Most important feature: current_run_rate
- [x] Top 4 features are all chase-related (2nd innings)
- [x] Toss advantage has moderate impact — less than run rate but more than match phase
- [x] Phase and balls_remaining have near zero importance — model gets this from required_run_rate already

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data parsing and manipulation |
| LightGBM | Win probability model |
| Scikit-learn | Calibration and evaluation metrics |
| Plotly | Interactive charts |
| Streamlit | Web app deployment |
| Cricsheet | Ball-by-ball IPL data source |

---

## Author

**Anika**
Built as part of a 6-project ML/AI portfolio targeting Data Science and AI/ML roles.
Other projects in the portfolio: Delhi AQI Forecaster, NCRB Crime Atlas, RBI Sentiment Analyzer, India Disease Early Warning System, Hinglish NLP Analyzer.

---

*Data source: [Cricsheet.org](https://cricsheet.org) — free ball-by-ball cricket data*