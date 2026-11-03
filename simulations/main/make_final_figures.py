"""
Generate the final 2-panel MSE Ratio figure, 2-panel Raw MSE figure, and a
2x2 composite figure that stacks the MSE Ratio row on top of the Raw MSE row.

For each learner (Ridge, Ridgeless), curves from the "Fixed Angle, Changing
Norm" and "Fixed Norm, Changing Angle" CSVs are pooled and plotted together.
Each curve is identified by its (angle, ||mu_2||) pair in the legend.

Reads:
  raw_ridgeless_fixed_angle.csv
  raw_ridge_fixed_angle.csv
  raw_ridgeless_fixed_norm.csv
  raw_ridge_fixed_norm.csv

Writes:
  fig_mse_ratio_final.{png,pdf}
  fig_raw_mse_final.{png,pdf}
  fig_combined_final.{png,pdf}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito colorblind-safe palette: 5 well-separated colors for 5 curves.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]

XLABEL = "Number of Features (p)"

GROUP_COLS = ["angle_deg", "mu2_norm"]


def aggregate_ratio(df):
    """For each (angle_deg, mu2_norm, p), compute mean ratio and 95% CI."""
    rows = []
    for (a, n, p), sub in df.groupby(GROUP_COLS + ["p"]):
        ratios = sub["ratio"].values
        mean_r = np.mean(ratios)
        sem = stats.sem(ratios)
        ci = stats.t.interval(0.95, len(ratios) - 1, loc=mean_r, scale=sem)
        rows.append({
            "angle_deg": a, "mu2_norm": n, "p": p,
            "mean_ratio": mean_r,
            "ci_lower": ci[0], "ci_upper": ci[1],
            "n_reps": len(ratios),
        })
    return pd.DataFrame(rows).sort_values(GROUP_COLS + ["p"]).reset_index(drop=True)


def aggregate_raw(df):
    """For each (angle_deg, mu2_norm, p), compute mean and 95% CI of MSEs."""
    rows = []
    for (a, n, p), sub in df.groupby(GROUP_COLS + ["p"]):
        m = len(sub)
        out = {"angle_deg": a, "mu2_norm": n, "p": p, "n_reps": m}
        for col in ("mse_ensemble", "mse_merged"):
            vals = sub[col].values
            mean_v = np.mean(vals)
            sem = stats.sem(vals)
            ci = stats.t.interval(0.95, m - 1, loc=mean_v, scale=sem)
            out[f"{col}_mean"] = mean_v
            out[f"{col}_lo"] = ci[0]
            out[f"{col}_hi"] = ci[1]
        rows.append(out)
    return pd.DataFrame(rows).sort_values(GROUP_COLS + ["p"]).reset_index(drop=True)


def pool(df_fixed_angle, df_fixed_norm):
    """Concatenate both scenarios; shared (angle, norm) groups are pooled."""
    return pd.concat([df_fixed_angle, df_fixed_norm], ignore_index=True)


def curve_label(angle_deg, mu2_norm):
    return (rf"angle = {int(round(angle_deg))}$^\circ$, "
            rf"$\|\mu_2\| = {int(round(mu2_norm))}$")


def ordered_groups(df_agg):
    """
    Return list of (angle_deg, mu2_norm) tuples in a sensible plotting order:
    fixed-angle group first (angle=60, norm ascending), then the remaining
    fixed-norm angles ascending. The (60, 1) row sits in the fixed-angle block.
    """
    pairs = sorted(set(zip(df_agg["angle_deg"], df_agg["mu2_norm"])))
    fixed_angle = sorted([p for p in pairs if p[0] == 60.0],
                         key=lambda x: x[1])
    fixed_norm_other = sorted([p for p in pairs
                               if p[0] != 60.0 and p[1] == 1.0],
                              key=lambda x: x[0])
    return fixed_angle + fixed_norm_other


def draw_ratio_panel(ax, df_agg, title, panel_letter=None):
    groups = ordered_groups(df_agg)
    for (a, n), color in zip(groups, COLORS):
        sub = df_agg[(df_agg["angle_deg"] == a) & (df_agg["mu2_norm"] == n)] \
            .sort_values("p")
        p_vals = sub["p"].values
        mean = sub["mean_ratio"].values
        lo = sub["ci_lower"].values
        hi = sub["ci_upper"].values
        ax.plot(p_vals, mean, marker="o", linewidth=2.2, markersize=7,
                color=color, label=curve_label(a, n), zorder=3)
        ax.fill_between(p_vals, lo, hi, color=color, alpha=0.20, zorder=1)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
               zorder=0)
    ax.set_xlabel(XLABEL, fontsize=13)
    ax.set_ylabel("MSE Ratio (Ensemble / Pooled)", fontsize=13)
    ax.set_title(title, fontsize=19, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11, labelbottom=True, labelleft=True)
    if panel_letter is not None:
        ax.text(-0.10, 1.05, f"({panel_letter})", transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="left")


def draw_raw_panel(ax, df_agg, title, panel_letter=None):
    groups = ordered_groups(df_agg)
    color_handles = []
    for (a, n), color in zip(groups, COLORS):
        sub = df_agg[(df_agg["angle_deg"] == a) & (df_agg["mu2_norm"] == n)] \
            .sort_values("p")
        p_vals = sub["p"].values

        ens_mean = sub["mse_ensemble_mean"].values
        ens_lo = sub["mse_ensemble_lo"].values
        ens_hi = sub["mse_ensemble_hi"].values
        ax.plot(p_vals, ens_mean, marker="o", linewidth=2.0,
                markersize=6, color=color, linestyle="-", zorder=3)
        ax.fill_between(p_vals, ens_lo, ens_hi, color=color, alpha=0.18,
                        zorder=1)
        color_handles.append(
            Line2D([0], [0], color=color, linewidth=2.0,
                   label=curve_label(a, n))
        )

        mrg_mean = sub["mse_merged_mean"].values
        mrg_lo = sub["mse_merged_lo"].values
        mrg_hi = sub["mse_merged_hi"].values
        ax.plot(p_vals, mrg_mean, marker="s", linewidth=2.0,
                markersize=6, color=color, linestyle="--", zorder=3)
        ax.fill_between(p_vals, mrg_lo, mrg_hi, color=color, alpha=0.10,
                        zorder=1)

    ax.set_xlabel(XLABEL, fontsize=13)
    ax.set_ylabel("Test MSE", fontsize=13)
    ax.set_title(title, fontsize=19, fontweight="bold")

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o",
               markersize=6, label="Ensemble"),
        Line2D([0], [0], color="black", linestyle="--", marker="s",
               markersize=6, label="Pooled"),
    ]
    ax.legend(handles=color_handles + style_handles,
              fontsize=9, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11, labelbottom=True, labelleft=True)
    if panel_letter is not None:
        ax.text(-0.10, 1.05, f"({panel_letter})", transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="left")


# Fixed y-axis limits and ticks per learner. The Ridge plots are zoomed in to
# [0, 2] (matching ticks across both Ridge panels). The Ridgeless plots use a
# wider range to fit the ratio peak; both ridgeless panels share the same
# axis/ticks so the Raw MSE plot lines up with the MSE Ratio plot.
RIDGE_YLIM = (0.0, 2.0)
RIDGE_YTICKS = [0.0, 0.5, 1.0, 1.5, 2.0]
RIDGELESS_YLIM = (0.0, 10.0)
RIDGELESS_YTICKS = [0, 2, 4, 6, 8, 10]


def apply_ridge_axis(ax):
    ax.set_ylim(*RIDGE_YLIM)
    ax.set_yticks(RIDGE_YTICKS)


def apply_ridgeless_axis(ax):
    ax.set_ylim(*RIDGELESS_YLIM)
    ax.set_yticks(RIDGELESS_YTICKS)


def make_two_panel_ratio_figure(ridge_agg, ridgeless_agg, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    draw_ratio_panel(axes[0], ridge_agg,
                     "Ridge Regression: MSE Ratio", panel_letter="a")
    draw_ratio_panel(axes[1], ridgeless_agg,
                     "Ridgeless Regression: MSE Ratio", panel_letter="b")
    apply_ridge_axis(axes[0])
    apply_ridgeless_axis(axes[1])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    plt.savefig(savepath + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(savepath + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {savepath}.png and .pdf")


def make_two_panel_raw_figure(ridge_agg, ridgeless_agg, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    draw_raw_panel(axes[0], ridge_agg,
                   "Ridge Regression: Raw MSE", panel_letter="a")
    draw_raw_panel(axes[1], ridgeless_agg,
                   "Ridgeless Regression: Raw MSE", panel_letter="b")
    apply_ridge_axis(axes[0])
    apply_ridgeless_axis(axes[1])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    plt.savefig(savepath + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(savepath + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {savepath}.png and .pdf")


def make_composite_figure(ratio_ridge, ratio_ridgeless,
                          raw_ridge, raw_ridgeless, savepath):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    draw_ratio_panel(axes[0, 0], ratio_ridge,
                     "Ridge Regression: MSE Ratio", panel_letter="a")
    draw_ratio_panel(axes[0, 1], ratio_ridgeless,
                     "Ridgeless Regression: MSE Ratio", panel_letter="b")
    draw_raw_panel(axes[1, 0], raw_ridge,
                   "Ridge Regression: Raw MSE", panel_letter="c")
    draw_raw_panel(axes[1, 1], raw_ridgeless,
                   "Ridgeless Regression: Raw MSE", panel_letter="d")

    apply_ridge_axis(axes[0, 0])
    apply_ridgeless_axis(axes[0, 1])
    apply_ridge_axis(axes[1, 0])
    apply_ridgeless_axis(axes[1, 1])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.14, hspace=0.30)
    plt.savefig(savepath + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(savepath + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {savepath}.png and .pdf")


if __name__ == "__main__":
    df_ridge_fa = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_ridge_fixed_angle.csv"))
    df_ridge_fn = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_ridge_fixed_norm.csv"))
    df_rl_fa = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_ridgeless_fixed_angle.csv"))
    df_rl_fn = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_ridgeless_fixed_norm.csv"))

    ridge_pool = pool(df_ridge_fa, df_ridge_fn)
    rl_pool = pool(df_rl_fa, df_rl_fn)

    ratio_ridge = aggregate_ratio(ridge_pool)
    ratio_ridgeless = aggregate_ratio(rl_pool)
    raw_ridge = aggregate_raw(ridge_pool)
    raw_ridgeless = aggregate_raw(rl_pool)

    make_two_panel_ratio_figure(
        ratio_ridge, ratio_ridgeless,
        os.path.join(OUTPUT_DIR, "fig_mse_ratio_final"))

    make_two_panel_raw_figure(
        raw_ridge, raw_ridgeless,
        os.path.join(OUTPUT_DIR, "fig_raw_mse_final"))

    make_composite_figure(
        ratio_ridge, ratio_ridgeless, raw_ridge, raw_ridgeless,
        os.path.join(OUTPUT_DIR, "fig_combined_final"))

    print("\nFinal figures generated.")
