"""
data_loader.py
--------------
All data loading, cleaning, and feature engineering for the LoL capstone project.
Clean separation between raw data and analysis — notebooks import from here.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


# ── Raw loaders ───────────────────────────────────────────────────────────────

def load_raw_matches() -> pd.DataFrame:
    """Load the raw games CSV with no transformations."""
    return pd.read_csv(GAMES_CSV)


def load_champion_map() -> dict:
    """Return {champion_id (int): champion_name (str)} mapping."""
    with open(CHAMPION_INFO_JSON) as f:
        data = json.load(f)
    return {v["id"]: v["name"] for v in data["data"].values()}


def load_spell_map() -> dict:
    """Return {spell_id (int): spell_name (str)} mapping."""
    with open(SPELL_INFO_JSON) as f:
        data = json.load(f)
    return {v["id"]: v["name"] for v in data["data"].values()}


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to the raw DataFrame.
    Returns a new DataFrame — never mutates the input.

    Features added:
    - match_date: datetime from epoch ms
    - game_duration_min: duration in minutes
    - duration_bucket: categorical bin
    - t1_won / t2_won: binary win flags
    - t1_first_* / t2_first_*: binary objective flags per team
    - objective_advantage columns: t1 - t2 per objective type
    - total_objectives_t1 / t2: sum of all objectives secured
    - gold_proxy: tower kills * 300 + baron kills * 900 + inhibitor kills * 1500
    """
    df = df.copy()

    # ── Temporal features
    df["match_date"]        = pd.to_datetime(df["creationTime"], unit="ms")
    df["match_week"]        = df["match_date"].dt.to_period("W").astype(str)
    df["match_month"]       = df["match_date"].dt.to_period("M").astype(str)
    df["game_duration_min"] = df["gameDuration"] / 60

    df["duration_bucket"]   = pd.cut(
        df["game_duration_min"],
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        right=False
    )

    # ── Win flags
    df["t1_won"] = (df["winner"] == 1).astype(int)
    df["t2_won"] = (df["winner"] == 2).astype(int)

    # ── Per-team first objective flags (1 = this team got it, 0 = they didn't)
    for label, col in FIRST_OBJECTIVES.items():
        clean = col.replace("first", "").lower()
        df[f"t1_first_{clean}"] = (df[col] == 1).astype(int)
        df[f"t2_first_{clean}"] = (df[col] == 2).astype(int)

    # ── Objective advantage (positive = team 1 ahead)
    for label, (t1_col, t2_col) in KILL_OBJECTIVES.items():
        key = label.lower().replace(" ", "_")
        df[f"{key}_advantage"] = df[t1_col] - df[t2_col]

    # ── Total objectives per team
    df["t1_total_objectives"] = (
        df["t1_towerKills"] + df["t1_dragonKills"] +
        df["t1_baronKills"] + df["t1_inhibitorKills"] +
        df["t1_riftHeraldKills"]
    )
    df["t2_total_objectives"] = (
        df["t2_towerKills"] + df["t2_dragonKills"] +
        df["t2_baronKills"] + df["t2_inhibitorKills"] +
        df["t2_riftHeraldKills"]
    )
    df["objectives_advantage"] = df["t1_total_objectives"] - df["t2_total_objectives"]

    # ── Gold proxy (weighted sum of structural objectives)
    # Weights based on approximate in-game gold value
    df["t1_gold_proxy"] = (
        df["t1_towerKills"]     * 300 +
        df["t1_baronKills"]     * 900 +
        df["t1_inhibitorKills"] * 1500 +
        df["t1_dragonKills"]    * 200
    )
    df["t2_gold_proxy"] = (
        df["t2_towerKills"]     * 300 +
        df["t2_baronKills"]     * 900 +
        df["t2_inhibitorKills"] * 1500 +
        df["t2_dragonKills"]    * 200
    )
    df["gold_proxy_advantage"] = df["t1_gold_proxy"] - df["t2_gold_proxy"]

    return df


def load_matches() -> pd.DataFrame:
    """Load and fully engineer the matches dataset. Main entry point for notebooks."""
    raw = load_raw_matches()
    return engineer_features(raw)


# ── Champion stats builder ────────────────────────────────────────────────────

def build_champion_stats(df: pd.DataFrame, champ_map: dict,
                         min_games: int = MIN_GAMES_CHAMPION) -> pd.DataFrame:
    """
    Build per-champion win rate, pick rate, and ban rate statistics.

    Parameters
    ----------
    df : engineered matches DataFrame
    champ_map : {id: name} mapping from load_champion_map()
    min_games : minimum games threshold for inclusion

    Returns
    -------
    DataFrame with columns: champ_id, champion, games, wins, win_rate,
                            pick_rate, ban_count, ban_rate
    """
    # Pick stats
    records = []
    for team in TEAMS:
        for slot in CHAMP_SLOTS:
            col = f"t{team}_champ{slot}id"
            tmp = df[[col, "t1_won"]].copy()
            tmp.columns = ["champ_id", "t1_won"]
            tmp["won"] = tmp["t1_won"] if team == 1 else 1 - tmp["t1_won"]
            records.append(tmp[["champ_id", "won"]])

    all_picks = pd.concat(records, ignore_index=True)
    stats = all_picks.groupby("champ_id").agg(
        games=("won", "count"),
        wins=("won", "sum")
    ).reset_index()
    stats["win_rate"]  = (stats["wins"] / stats["games"] * 100).round(2)
    stats["pick_rate"] = (stats["games"] / (len(df) * TOTAL_PLAYERS) * 100).round(3)

    # Ban stats
    bans = []
    for team in TEAMS:
        for slot in BAN_SLOTS:
            bans.extend(df[f"t{team}_ban{slot}"].tolist())
    ban_counts = pd.Series(bans)
    ban_counts = ban_counts[ban_counts != 0].value_counts().reset_index()
    ban_counts.columns = ["champ_id", "ban_count"]
    ban_counts["ban_rate"] = (ban_counts["ban_count"] / len(df) * 100).round(2)

    # Merge
    stats = stats.merge(ban_counts, on="champ_id", how="left")
    stats["ban_count"] = stats["ban_count"].fillna(0).astype(int)
    stats["ban_rate"]  = stats["ban_rate"].fillna(0.0)

    # Map names
    stats["champion"] = stats["champ_id"].map(champ_map).fillna("Unknown")

    # Filter minimum games
    stats = stats[stats["games"] >= min_games].reset_index(drop=True)

    return stats


# ── Model feature builder ─────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "t1_first_blood",   "t1_first_tower",    "t1_first_inhibitor",
    "t1_first_baron",   "t1_first_dragon",   "t1_first_riftherald",
    "t1_towerKills",    "t1_inhibitorKills", "t1_baronKills",
    "t1_dragonKills",   "t1_riftHeraldKills",
    "tower_kills_advantage", "baron_kills_advantage",
    "dragon_kills_advantage", "inhibitor_kills_advantage",
]

FEATURE_LABELS = [
    "First Blood",   "First Tower",    "First Inhibitor",
    "First Baron",   "First Dragon",   "First Rift Herald",
    "Tower Kills",   "Inhibitor Kills","Baron Kills",
    "Dragon Kills",  "Rift Herald",
    "Tower Advantage", "Baron Advantage",
    "Dragon Advantage", "Inhibitor Advantage",
]

def build_model_features(df: pd.DataFrame):
    """Return X (features) and y (target) for the win prediction model."""
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["t1_won"]
    return X, y
