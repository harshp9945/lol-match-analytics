# LoL Match Analytics — source package
from .config import *
from .data_loader import load_matches, load_champion_map, load_spell_map, build_champion_stats, build_model_features, FEATURE_COLUMNS, FEATURE_LABELS
from .stats_utils import test_win_rate, test_objective_impact, test_duration_groups, test_normality, spearman_correlation, bonferroni_correction, benjamini_hochberg, cohens_h, cohens_d, odds_ratio_ci
from .plot_utils import set_style, save_plot, add_significance_stars, add_bar_labels, add_confidence_interval, plot_win_rates_with_ci, plot_correlation_heatmap, plot_roc_curve, plot_feature_importance
