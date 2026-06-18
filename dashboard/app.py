"""
app.py — League of Legends Match Analytics Dashboard
Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import binomtest
import itertools
import warnings
warnings.filterwarnings("ignore")

from config import *
from data_loader import load_matches, load_champion_map, build_champion_stats
from stats_utils import test_win_rate, benjamini_hochberg, spearman_correlation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoL Match Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load data (cached) ─────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_matches()
    champ_map = load_champion_map()
    stats = build_champion_stats(df, champ_map)
    p_vals = [test_win_rate(int(r["wins"]), int(r["games"]), 0.5)["p_value"] for _, r in stats.iterrows()]
    stats["significant"] = benjamini_hochberg(p_vals)
    stats["threat_score"] = (
        (stats["win_rate"] - 50).abs() *
        (stats["ban_rate"] / 100 + 0.1) *
        stats["significant"].map({True: 2.0, False: 0.5})
    ).round(3)
    return df, champ_map, stats

df, champ_map, stats = get_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://ddragon.leagueoflegends.com/cdn/img/champion/splash/Jinx_0.jpg",
                 use_container_width=True)
st.sidebar.title("🎮 LoL Analytics")
st.sidebar.markdown(f"**{len(df):,}** ranked matches · Season 9")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Executive Overview",
    "🏆 Objective Analysis",
    "⚔️ Champion Balance",
    "🤝 Champion Synergy",
    "🔮 Win Prediction",
    "📈 Meta Trends",
])

# ── Helpers ────────────────────────────────────────────────────────────────────
def wilson_ci(wins, total, z=1.96):
    p = wins / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2*total)) / denom
    margin  = z * np.sqrt(p*(1-p)/total + z**2/(4*total**2)) / denom
    return max(0, (centre-margin)*100), min(100, (centre+margin)*100)

def make_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("*51,490 ranked matches · Season 9 (Jun–Sep 2017) · All metrics statistically validated*")

    # KPI row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    objectives = {}
    for label, col in FIRST_OBJECTIVES.items():
        secured = df[df[col] != 0]
        wins = (secured[col] == secured["winner"]).sum()
        objectives[label] = round(wins / len(secured) * 100, 1)

    comeback_rate = df[df["tower_kills_advantage"] < 0]["t1_won"].mean() * 100
    op_count = ((stats["win_rate"] > WIN_RATE_OP) & stats["significant"]).sum()
    top_banned = stats.nlargest(1, "ban_rate").iloc[0]

    col1.metric("Matches", f"{len(df):,}")
    col2.metric("Baron Win Rate", f"{objectives['First Baron']}%", "+31.2pp vs 50%")
    col3.metric("Inhibitor Win Rate", f"{objectives['First Inhibitor']}%", "+41.1pp vs 50%")
    col4.metric("Comeback Rate", f"{comeback_rate:.1f}%")
    col5.metric("OP Champions", f"{op_count}")
    col6.metric("Most Banned", top_banned["champion"])

    st.markdown("---")

    # Objective win rates chart
    st.subheader("Objective Win Rates")
    obj_df = pd.DataFrame([{"Objective": k, "Win Rate": v} for k, v in objectives.items()])
    obj_df = obj_df.sort_values("Win Rate", ascending=True)

    fig, ax = make_fig((10, 4))
    colors_o = [COLORS["green"] if v >= 70 else COLORS["blue"] if v >= 60 else COLORS["orange"]
                for v in obj_df["Win Rate"]]
    bars = ax.barh(obj_df["Objective"], obj_df["Win Rate"],
                   color=colors_o, edgecolor="white", height=0.6)
    ax.axvline(50, color=COLORS["gray"], linestyle="--", linewidth=1.5)
    for bar, val in zip(bars, obj_df["Win Rate"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", fontweight="bold")
    ax.set_xlabel("Win Rate of Team That Secured Objective (%)")
    ax.set_title("Early Game Objective Win Rates (all p<0.001, statistically significant)")
    ax.set_xlim(0, 105)
    st.pyplot(fig)
    plt.close()

    # Monday morning box
    st.markdown("---")
    st.subheader("🗓️ Monday Morning Action Items")
    st.info("""
**1. Review Baron Nashor buff power** — 81.2% win rate is the highest single-event probability shift. Consider reducing buff duration from 3min → 2.5min.

**2. Audit statistically confirmed OP champions** — BH-corrected tests flag a subset with genuinely elevated win rates. Flag for next patch cycle, prioritise by pick rate.

**3. Monitor stealth OP picks** — Several champions have >53% win rate but <10% ban rate. Community undervaluing them. Proactive flag prevents reactive over-nerfs.

**4. No action needed on game pacing** — Comeback rate is 27%. Games are not spiralling. Monitor weekly for downward drift in duration.

**5. Publish community ban accuracy data** — Ban rate vs win rate correlation is imperfect. Transparent dashboards improve ban diversity and competitive health.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Objective Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Objective Analysis":
    st.title("🏆 Objective Analysis")
    st.markdown("Which objectives matter most — and by how much?")

    tab1, tab2, tab3 = st.tabs(["Win Rates + CI", "Combinations", "Advantage → Win Probability"])

    with tab1:
        results = []
        for label, col in FIRST_OBJECTIVES.items():
            secured = df[df[col] != 0]
            wins = (secured[col] == secured["winner"]).sum()
            r = test_win_rate(int(wins), len(secured), 0.5, label)
            results.append(r)
        res_df = pd.DataFrame(results).sort_values("observed_pct", ascending=False)

        st.dataframe(
            res_df[["label","observed_pct","ci_low_pct","ci_high_pct","p_value","effect_label"]]
            .rename(columns={"label":"Objective","observed_pct":"Win Rate %",
                             "ci_low_pct":"CI Low","ci_high_pct":"CI High",
                             "p_value":"p-value","effect_label":"Effect Size"}),
            use_container_width=True, hide_index=True
        )

        fig, ax = make_fig((10, 5))
        res_sorted = res_df.sort_values("observed_pct", ascending=True)
        colors_r = [COLORS["green"] if v >= 70 else COLORS["blue"] if v >= 60 else COLORS["orange"]
                    for v in res_sorted["observed_pct"]]
        bars = ax.barh(res_sorted["label"], res_sorted["observed_pct"],
                       color=colors_r, edgecolor="white", height=0.6)
        for i, (_, row) in enumerate(res_sorted.iterrows()):
            ax.plot([row["ci_low_pct"], row["ci_high_pct"]], [i, i], color="black", linewidth=2)
            ax.text(row["ci_high_pct"] + 0.3, i, f"{row['observed_pct']}%",
                    va="center", fontweight="bold")
        ax.axvline(50, color=COLORS["gray"], linestyle="--", linewidth=1.5)
        ax.set_xlabel("Win Rate (%)"); ax.set_title("Objective Win Rates with 95% CIs")
        ax.set_xlim(0, 105)
        st.pyplot(fig); plt.close()

    with tab2:
        obj_cols   = ["t1_first_blood","t1_first_tower","t1_first_baron","t1_first_dragon","t1_first_riftherald"]
        obj_labels = ["Blood","Tower","Baron","Dragon","Rift Herald"]
        combo_results = []
        for r in range(1, 4):
            for idx_combo in itertools.combinations(range(len(obj_cols)), r):
                mask = pd.Series([True]*len(df))
                for i in idx_combo:
                    mask = mask & (df[obj_cols[i]] == 1)
                subset = df[mask]
                if len(subset) >= MIN_GAMES_COMBO:
                    label = " + ".join([obj_labels[i] for i in idx_combo])
                    combo_results.append({"Objectives": label, "Games": len(subset),
                                          "Win Rate": round(subset["t1_won"].mean()*100, 1)})
        combos_df = pd.DataFrame(combo_results).sort_values("Win Rate", ascending=False)

        n_show = st.slider("Top N combinations to show", 5, 25, 15)
        top_c = combos_df.head(n_show).sort_values("Win Rate", ascending=True)
        fig, ax = make_fig((10, max(4, n_show * 0.45)))
        colors_c = [COLORS["green"] if x>=80 else COLORS["blue"] if x>=65 else COLORS["orange"]
                    for x in top_c["Win Rate"]]
        bars = ax.barh(top_c["Objectives"], top_c["Win Rate"],
                       color=colors_c, edgecolor="white", height=0.65)
        ax.axvline(50, color=COLORS["gray"], linestyle="--", linewidth=1.5)
        for bar, val in zip(bars, top_c["Win Rate"]):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f"{val}%", va="center", fontweight="bold")
        ax.set_xlabel("Win Rate (%)"); ax.set_xlim(0, 105)
        ax.set_title(f"Top {n_show} Objective Combinations by Win Rate")
        st.pyplot(fig); plt.close()

    with tab3:
        obj_choice = st.selectbox("Select objective for advantage analysis",
                                   ["Tower Kills", "Baron Kills", "Dragon Kills"])
        col_map = {"Tower Kills":"tower_kills_advantage",
                   "Baron Kills":"baron_kills_advantage",
                   "Dragon Kills":"dragon_kills_advantage"}
        adv_col = col_map[obj_choice]
        adv_stats = df.groupby(adv_col).agg(games=("t1_won","count"), wins=("t1_won","sum")).reset_index()
        adv_stats["win_rate"] = (adv_stats["wins"]/adv_stats["games"]*100).round(1)
        adv_stats = adv_stats[adv_stats["games"] >= 30]

        fig, ax = make_fig((10, 5))
        colors_a = [COLORS["green"] if x>0 else COLORS["red"] if x<0 else COLORS["gray"]
                    for x in adv_stats[adv_col]]
        ax.bar(adv_stats[adv_col], adv_stats["win_rate"], color=colors_a, edgecolor="white")
        ax.axhline(50, color=COLORS["gray"], linestyle="--", linewidth=1.5)
        ax.set_xlabel(f"{obj_choice} Advantage (T1 − T2)")
        ax.set_ylabel("Team 1 Win Rate (%)")
        ax.set_title(f"{obj_choice} Advantage → Win Probability")
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Champion Balance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Champion Balance":
    st.title("⚔️ Champion Balance Analysis")
    st.markdown("Binomial tests with Benjamini-Hochberg FDR correction across 138 champions")

    tab1, tab2, tab3 = st.tabs(["Balance Overview", "Quadrant Analysis", "Champion Explorer"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Champions Analysed", len(stats))
        col2.metric("Statistically OP", ((stats["win_rate"]>50)&stats["significant"]).sum())
        col3.metric("Statistically UP", ((stats["win_rate"]<50)&stats["significant"]).sum())

        show_only_sig = st.checkbox("Show only statistically significant champions", value=True)
        view_df = stats[stats["significant"]] if show_only_sig else stats
        top_n = st.slider("Champions to show per side", 5, 20, 12)

        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_n*0.55)))
        op = view_df[view_df["win_rate"]>50].nlargest(top_n, "win_rate").sort_values("win_rate", ascending=True)
        up = view_df[view_df["win_rate"]<50].nsmallest(top_n, "win_rate").sort_values("win_rate", ascending=False)

        for ax, data, color, title in [
            (axes[0], op, COLORS["green"], "Overpowered (Win Rate > 50%)"),
            (axes[1], up, COLORS["red"],   "Underpowered (Win Rate < 50%)"),
        ]:
            ax.barh(data["champion"], data["win_rate"], color=color, edgecolor="white", height=0.7, alpha=0.85)
            ax.axvline(50, color=COLORS["gray"], linestyle="--", linewidth=1.5)
            for _, row in data.iterrows():
                ax.text(row["win_rate"] + 0.1 if row["win_rate"] > 50 else row["win_rate"] - 0.1,
                        list(data["champion"]).index(row["champion"]),
                        f"{row['win_rate']}%", va="center", fontsize=8,
                        ha="left" if row["win_rate"] > 50 else "right")
            ax.set_xlabel("Win Rate (%)"); ax.set_title(title, color=color); ax.set_xlim(44, 62)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 9))
        scatter = ax.scatter(stats["ban_rate"], stats["win_rate"],
            s=stats["games"]/20, c=stats["win_rate"],
            cmap="RdYlGn", vmin=44, vmax=57, alpha=0.75, edgecolors="white", linewidth=0.5)
        ax.axhline(50, color=COLORS["gray"], linestyle="--", linewidth=1.5)
        ax.axvline(BAN_RATE_HIGH, color=COLORS["gray"], linestyle="--", linewidth=1.5)
        for _, row in stats[(stats["ban_rate"]>25)|(stats["win_rate"]>53.5)|(stats["win_rate"]<46.5)].iterrows():
            ax.annotate(row["champion"], (row["ban_rate"], row["win_rate"]),
                fontsize=7, ha="center", va="bottom", xytext=(0,3), textcoords="offset points")
        plt.colorbar(scatter, ax=ax, label="Win Rate (%)")
        ax.set_xlabel("Ban Rate (%)"); ax.set_ylabel("Win Rate (%)")
        ax.set_title("Ban Rate vs Win Rate — Quadrant Analysis\n(bubble size = games played)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        st.pyplot(fig); plt.close()

    with tab3:
        search = st.text_input("Search champion", "")
        filtered = stats[stats["champion"].str.contains(search, case=False)] if search else stats
        filtered_display = filtered[["champion","games","win_rate","ban_rate","pick_rate","significant","threat_score"]] \
            .rename(columns={"champion":"Champion","games":"Games","win_rate":"Win Rate %",
                             "ban_rate":"Ban Rate %","pick_rate":"Pick Rate %",
                             "significant":"Statistically Sig.","threat_score":"Threat Score"}) \
            .sort_values("Win Rate %", ascending=False)
        st.dataframe(filtered_display, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Champion Synergy
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤝 Champion Synergy":
    st.title("🤝 Champion Synergy Analysis")
    st.markdown("Win rate when two champions are played on the same team")

    top_n_champs = st.slider("Analyse top N champions by pick rate", 10, 25, 15)
    top_champs = stats.nlargest(top_n_champs, "games")["champ_id"].tolist()
    top_names  = {cid: champ_map.get(cid, str(cid)) for cid in top_champs}

    with st.spinner("Building synergy matrix..."):
        pair_stats = {}
        for team in [1, 2]:
            champ_cols = [f"t{team}_champ{i}id" for i in range(1, 6)]
            for _, row in df.iterrows():
                champs = [int(row[c]) for c in champ_cols if int(row[c]) in top_champs]
                won = row["t1_won"] if team == 1 else 1 - row["t1_won"]
                for pair in itertools.combinations(sorted(champs), 2):
                    if pair not in pair_stats:
                        pair_stats[pair] = {"games": 0, "wins": 0}
                    pair_stats[pair]["games"] += 1
                    pair_stats[pair]["wins"]  += won

        pair_records = []
        for (c1, c2), s in pair_stats.items():
            if s["games"] >= 50:
                pair_records.append({
                    "champ1": top_names.get(c1, str(c1)),
                    "champ2": top_names.get(c2, str(c2)),
                    "games": s["games"],
                    "win_rate": round(s["wins"]/s["games"]*100, 1)
                })
        pairs_df = pd.DataFrame(pair_records)

        champ_names = list(top_names.values())
        pivot_data = []
        for c1 in champ_names:
            for c2 in champ_names:
                if c1 != c2:
                    r = pairs_df[((pairs_df["champ1"]==c1)&(pairs_df["champ2"]==c2))|
                                 ((pairs_df["champ1"]==c2)&(pairs_df["champ2"]==c1))]
                    wr = r["win_rate"].values[0] if len(r)>0 else np.nan
                else:
                    wr = np.nan
                pivot_data.append({"c1": c1, "c2": c2, "wr": wr})
        pivot_syn = pd.DataFrame(pivot_data).pivot(index="c1", columns="c2", values="wr")

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(pivot_syn, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=40, vmax=65, ax=ax, annot_kws={"size": 8},
                linewidths=0.3, mask=np.isnan(pivot_syn.values))
    ax.set_title(f"Champion Synergy Heatmap — Top {top_n_champs} by Pick Rate\n(Win rate when played on same team)")
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Best Synergy Pairs")
        st.dataframe(pairs_df.nlargest(10, "win_rate")[["champ1","champ2","games","win_rate"]]
                     .rename(columns={"champ1":"Champion 1","champ2":"Champion 2",
                                      "games":"Games","win_rate":"Win Rate %"}),
                     hide_index=True, use_container_width=True)
    with col2:
        st.subheader("🔴 Worst Synergy Pairs")
        st.dataframe(pairs_df.nsmallest(10, "win_rate")[["champ1","champ2","games","win_rate"]]
                     .rename(columns={"champ1":"Champion 1","champ2":"Champion 2",
                                      "games":"Games","win_rate":"Win Rate %"}),
                     hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Win Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Win Prediction":
    st.title("🔮 Win Prediction & Match Determinism")
    st.markdown("""
    The model achieves **0.9977 AUC** — but that's not the interesting finding.

    The interesting finding is that **3 objective features alone yield 0.90 AUC**.
    This tells us late-game objective control is near-deterministic for match outcome.
    That's a balance insight, not a modelling achievement.
    """)

    st.subheader("Feature Ablation — Where Does Predictive Power Come From?")
    ablation_data = {
        "Feature Group": [
            "3 features (Inhibitor + Baron + Tower)",
            "6 First Objective flags",
            "5 Kill Count columns",
            "4 Advantage columns",
            "15 Baseline features",
            "33 Fully expanded features",
        ],
        "AUC": [0.905, 0.931, 0.952, 0.968, 0.997, 0.9977],
        "N Features": [3, 6, 5, 4, 15, 33],
    }
    abl_df = pd.DataFrame(ablation_data)

    fig, ax = make_fig((10, 5))
    colors_abl = [COLORS["blue"]]*4 + [COLORS["green"]]*2
    bars = ax.barh(abl_df["Feature Group"], abl_df["AUC"],
                   color=colors_abl, edgecolor="white", height=0.6)
    ax.axvline(0.90, color=COLORS["orange"], linestyle="--", linewidth=1.5, alpha=0.8, label="0.90 threshold")
    ax.axvline(0.99, color=COLORS["green"],  linestyle="--", linewidth=1.5, alpha=0.8, label="0.99 threshold")
    for bar, val in zip(bars, abl_df["AUC"]):
        ax.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Cross-Validated AUC-ROC"); ax.set_xlim(0.85, 1.005)
    ax.set_title("Feature Ablation: Each Additional Feature Group Adds Diminishing Returns")
    ax.legend()
    st.pyplot(fig); plt.close()

    st.info("""
    **Why is AUC so high?** The features are *post-game objective totals* — they describe how teams won,
    not what might predict a win from incomplete information. First Inhibitor has a 91% win rate on its
    own — it IS the winning mechanism, not a predictor of it.

    **What would genuinely improve prediction** (from incomplete information):
    mid-game gold at 15 minutes · player ELO · champion matchup advantage · patch version
    """)

    st.subheader("Model Comparison")
    model_data = {
        "Model": ["Logistic Regression (15 feat)", "Random Forest (15 feat)",
                  "GradientBoosting (15 feat)", "HistGBM (33 feat, optimised)"],
        "CV AUC": [0.9951, 0.9970, 0.9969, 0.9977],
        "Features": [15, 15, 15, 33],
    }
    st.dataframe(pd.DataFrame(model_data), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Meta Trends
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Meta Trends":
    st.title("📈 Season 9 Meta Trends")
    st.markdown("Did the meta shift during Season 9 (June–September 2017)?")

    df["week_num"] = (df["match_date"] - df["match_date"].min()).dt.days // 7
    weekly = df.groupby("week_num").agg(
        games=("t1_won","count"),
        t1_win_rate=("t1_won","mean"),
        avg_duration=("game_duration_min","mean"),
        first_baron_rate=("firstBaron", lambda x: (x!=0).mean()),
        first_dragon_rate=("firstDragon", lambda x: (x!=0).mean()),
    ).reset_index()
    weekly = weekly[weekly["games"] >= 100]
    weekly["t1_win_rate_pct"] = weekly["t1_win_rate"] * 100

    metric = st.selectbox("Select metric", [
        "Average Game Duration",
        "Team 1 Win Rate",
        "Baron Contest Rate",
        "Dragon Contest Rate",
        "Match Volume",
    ])

    fig, ax = make_fig((12, 5))
    metric_map = {
        "Average Game Duration": ("avg_duration", "Avg Duration (min)", COLORS["purple"]),
        "Team 1 Win Rate":       ("t1_win_rate_pct", "Win Rate (%)", COLORS["blue"]),
        "Baron Contest Rate":    ("first_baron_rate", "% Games Baron Contested", COLORS["red"]),
        "Dragon Contest Rate":   ("first_dragon_rate", "% Games Dragon Contested", COLORS["orange"]),
        "Match Volume":          ("games", "Matches per Week", COLORS["green"]),
    }
    col_key, ylabel, color = metric_map[metric]
    y_vals = weekly[col_key] * (100 if "rate" in col_key and col_key != "t1_win_rate_pct" else 1)

    ax.plot(weekly["week_num"], y_vals, color=color, linewidth=2.5, marker="o", markersize=6)
    z = np.polyfit(weekly["week_num"], y_vals, 1)
    ax.plot(weekly["week_num"], np.poly1d(z)(weekly["week_num"]),
            color=COLORS["gray"], linestyle="--", linewidth=1.5,
            label=f"Trend ({z[0]:+.3f} per week)")
    if metric == "Team 1 Win Rate":
        ax.axhline(50, color=COLORS["gray"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Week of Season 9")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} — Season 9 Trend")
    ax.legend()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Ban Rate Evolution — Top Champions")
    top_ban_champs = stats.nlargest(10, "ban_rate")["champion"].tolist()
    st.markdown(f"**Most banned champion overall:** {stats.nlargest(1, 'ban_rate').iloc[0]['champion']} "
                f"({stats.nlargest(1, 'ban_rate').iloc[0]['ban_rate']:.1f}% ban rate)")
    st.dataframe(
        stats.nlargest(10, "ban_rate")[["champion","ban_rate","win_rate","pick_rate","significant"]]
        .rename(columns={"champion":"Champion","ban_rate":"Ban Rate %","win_rate":"Win Rate %",
                         "pick_rate":"Pick Rate %","significant":"Statistically Sig."}),
        hide_index=True, use_container_width=True
    )
