"""
config.py
---------
Central configuration for the LoL Match Analytics capstone project.
All thresholds, constants, and paths live here — nothing hardcoded in notebooks.
"""

from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
PLOTS_DIR   = ROOT_DIR / "plots"
REPORTS_DIR = ROOT_DIR / "reports"

# ── Dataset ────────────────────────────────────────────────────────────────────
GAMES_CSV = DATA_DIR / "games.csv" if (DATA_DIR / "games.csv").exists() else DATA_DIR / "games_sample.csv"
CHAMPION_INFO_JSON  = DATA_DIR / "champion_info.json"
CHAMPION_INFO_2_JSON= DATA_DIR / "champion_info_2.json"
SPELL_INFO_JSON     = DATA_DIR / "summoner_spell_info.json"

# ── Statistical thresholds ─────────────────────────────────────────────────────
ALPHA               = 0.05          # Significance level for all hypothesis tests
BONFERRONI_N        = 138           # Number of champions (for multiple testing correction)
ALPHA_BONFERRONI    = ALPHA / BONFERRONI_N  # Bonferroni-corrected alpha

MIN_GAMES_CHAMPION  = 100           # Minimum games for champion to be included in analysis
MIN_GAMES_COMBO     = 100           # Minimum games for objective combo analysis
MIN_GAMES_BUCKET    = 50            # Minimum games for duration bucket analysis

# ── Balance thresholds ─────────────────────────────────────────────────────────
WIN_RATE_OP         = 53.0          # Win rate above this = potentially overpowered
WIN_RATE_UP         = 47.0          # Win rate below this = potentially underpowered
BAN_RATE_HIGH       = 30.0          # Ban rate above this = high community concern
BAN_RATE_EXTREME    = 50.0          # Ban rate above this = extreme community concern

# ── Model parameters ──────────────────────────────────────────────────────────
CV_FOLDS            = 5             # K-fold cross validation folds
RF_N_ESTIMATORS     = 200           # Random forest trees
RF_MAX_DEPTH        = 10            # Max depth to avoid overfitting
RANDOM_STATE        = 42            # Reproducibility seed
TEST_SIZE           = 0.2           # Train/test split ratio

# ── Game constants ─────────────────────────────────────────────────────────────
TEAM_SIZE           = 5             # Players per team
TEAMS               = [1, 2]        # Team identifiers in dataset
CHAMP_SLOTS         = range(1, 6)   # Champion slots per team (1-5)
BAN_SLOTS           = range(1, 6)   # Ban slots per team (1-5)
TOTAL_PLAYERS       = 10            # Per match

EARLY_GAME_MINS     = 15            # Definition of early game
MID_GAME_MINS       = 25            # Definition of mid game
LATE_GAME_MINS      = 35            # Definition of late game

DURATION_BINS       = [0, 20, 25, 30, 35, 40, 999]
DURATION_LABELS     = ["<20 min", "20-25", "25-30", "30-35", "35-40", "40+ min"]

# ── Objective columns ─────────────────────────────────────────────────────────
FIRST_OBJECTIVES = {
    "First Blood":        "firstBlood",
    "First Tower":        "firstTower",
    "First Inhibitor":    "firstInhibitor",
    "First Baron":        "firstBaron",
    "First Dragon":       "firstDragon",
    "First Rift Herald":  "firstRiftHerald",
}

KILL_OBJECTIVES = {
    "Tower Kills":      ("t1_towerKills",     "t2_towerKills"),
    "Dragon Kills":     ("t1_dragonKills",    "t2_dragonKills"),
    "Baron Kills":      ("t1_baronKills",     "t2_baronKills"),
    "Inhibitor Kills":  ("t1_inhibitorKills", "t2_inhibitorKills"),
    "Rift Herald":      ("t1_riftHeraldKills","t2_riftHeraldKills"),
}

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "blue":   "#4A90D9",
    "red":    "#E05C5C",
    "green":  "#5BBF6E",
    "purple": "#9B7FD4",
    "orange": "#F0A862",
    "gray":   "#9E9E9E",
    "gold":   "#C9A84C",
}
