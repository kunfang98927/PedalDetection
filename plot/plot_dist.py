import os
import re
import glob
import argparse
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class PlotConfig:
    """Configuration for overall distribution plotting."""

    experiment_names: list[str]
    experiment_names_labels: dict[str, str]
    csv_dir: str
    out_dir: str
    fig_width: int
    fig_height: int
    dpi: int
    font_size: int
    include_plain: bool
    action_colors: dict[str, str]
    gesture_colors: dict[str, str]


DEFAULT_CONFIG = PlotConfig(
    experiment_names=["bin+a", "a", "a+m", "gt"],
    experiment_names_labels={
        "bin+a": "Binary",
        "a": "Audio",
        "a+m": "Audio+MIDI",
        "gt": "Ground Truth",
    },
    csv_dir="",
    out_dir="",
    fig_width=8,
    fig_height=4,
    dpi=1000,
    font_size=12,
    include_plain=True,
    action_colors={
        "press_frames": "#FAEAB1",
        "sustained_frames": "#D5B7C0",
        "release_frames": "#A0C3D2",
    },
    gesture_colors={
        "gesture_mountain_frames": "#CCDD5E",
        "gesture_highland_frames": "#F9EBC5",
        "gesture_hill_frames": "#F0BB78",
        "gesture_pinnacle_frames": "#798A58",
        "gesture_plain_frames": "#B38782",
    },
)

# ================= Utilities =================
def discover_experiments(csv_dir: str) -> list[str]:
    """Discover experiment names from test_*_gesture.csv files."""
    files = glob.glob(os.path.join(csv_dir, "test_*_gesture.csv"))
    names = []
    for f in files:
        match = re.match(r"test_(.+)_gesture\.csv$", os.path.basename(f))
        if match:
            names.append(match.group(1))
    return sorted(set(names))


def sum_props(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum the specified columns and normalize them to proportions."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(dtype=float)
    summed = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=0, skipna=True, min_count=1)
        .fillna(0.0)
    )
    total = float(summed.sum())
    return (summed / total) if total > 0 else summed * 0.0


def format_label(name: str) -> str:
    """Make column names more readable."""
    return name.replace("_frames", "").replace("gesture_", "").replace("_", " ").title()


def _best_text_color(hex_color: str) -> str:
    """Choose black/white text based on background brightness."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    # WCAG relative luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#222222" if luminance > 0.6 else "#FFFFFF"


def bar100_row(
    ax,
    y,
    parts: pd.Series,
    color_map: dict,
    font_size: int,
    inside_thresh: float = 0.005,  # >= 0.5% label inside
    outside_thresh: float = 0.012,  # >= 1.2% label outside
    gap: float = 0.006,  # horizontal gap for outside labels (in proportion units)
) -> None:
    """Draw one 100% stacked horizontal bar with optional outside labels."""
    left = 0.0
    for key, value in parts.items():
        width = float(value)
        if width <= 0:
            continue

        color = color_map.get(key, "#CCCCCC")
        ax.barh(
            y=y,
            width=width,
            left=left,
            height=0.6,
            color=color,
            linewidth=0.8,
        )

        label = f"{width * 100:.1f}%"
        if width >= inside_thresh:
            ax.text(
                left + width / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=font_size - 3,
                color=_best_text_color(color),
            )
        elif width >= outside_thresh:
            # Place label outside; move left if too close to right edge
            pos = left + width + gap
            align = "left"
            if pos > 0.98:
                pos = left - gap
                align = "right"
            ax.text(
                pos,
                y,
                label,
                ha=align,
                va="center",
                fontsize=font_size - 3,
                color="#333333",
                clip_on=False,
            )
            ax.annotate(
                "",
                xy=(left + width, y),
                xytext=(pos - (gap if align == "left" else -gap), y),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#777777"),
                annotation_clip=False,
            )

        left += width


def collect_overall_props(csv_path: str, include_plain: bool) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """Read a CSV and return (action_props, gesture_props)."""
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)

    action_cols = ["release_frames", "sustained_frames", "press_frames"]
    action_order = ["press_frames", "sustained_frames", "release_frames"]
    action_props = sum_props(df, action_cols).reindex(action_order).fillna(0.0)

    if include_plain:
        gesture_cols = [
            "gesture_mountain_frames",
            "gesture_highland_frames",
            "gesture_hill_frames",
            "gesture_pinnacle_frames",
            "gesture_plain_frames",
        ]
        gesture_order = gesture_cols
    else:
        gesture_cols = [
            "gesture_mountain_frames",
            "gesture_highland_frames",
            "gesture_hill_frames",
            "gesture_pinnacle_frames",
        ]
        gesture_order = gesture_cols

    gesture_props = sum_props(df, gesture_cols).reindex(gesture_order).fillna(0.0)

    return action_props, gesture_props


def build_experiment_matrices(
    experiments: list[str],
    csv_dir: str,
    include_plain: bool,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], list[str]]:
    """Collect action and gesture distributions for each experiment."""
    action_matrix: dict[str, pd.Series] = {}
    gesture_matrix: dict[str, pd.Series] = {}
    used_exps: list[str] = []

    for exp in experiments:
        csv_path = os.path.join(csv_dir, f"test_{exp}_gesture.csv")
        action_props, gesture_props = collect_overall_props(csv_path, include_plain)
        if action_props is None:
            print(f"[Skip] CSV not found: {csv_path}")
            continue
        action_matrix[exp] = action_props
        gesture_matrix[exp] = gesture_props
        used_exps.append(exp)

    return action_matrix, gesture_matrix, used_exps


def _apply_common_axis_settings(ax, used_exps: list[str], labels: list[str], font_size: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(used_exps) - 0.5)
    ax.set_yticks(range(len(used_exps)))
    ax.set_yticklabels(labels, fontsize=font_size - 2)


def plot_action_distribution(
    ax,
    action_df: pd.DataFrame,
    used_exps: list[str],
    labels: list[str],
    action_colors: dict[str, str],
    font_size: int,
) -> None:
    """Plot the overall action distribution."""
    _apply_common_axis_settings(ax, used_exps, labels, font_size)
    for i, exp in enumerate(used_exps):
        parts = action_df[exp].fillna(0.0)
        bar100_row(ax, y=i, parts=parts, color_map=action_colors, font_size=font_size)

    ax.set_xlabel("Proportion", fontsize=font_size - 2)
    ax.set_title("Overall Pedal Action Distribution", pad=20, fontsize=font_size)

    handles = [plt.Rectangle((0, 0), 1, 1, color=action_colors[k]) for k in action_df.index]
    labels_action = [format_label(k) for k in action_df.index]
    labels_action = [("Hold" if lbl == "Sustained" else lbl) for lbl in labels_action]

    ax.legend(
        handles,
        labels_action,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(labels_action),
        frameon=False,
        fontsize=font_size - 1,
    )


def plot_gesture_distribution(
    ax,
    gesture_df: pd.DataFrame,
    used_exps: list[str],
    labels: list[str],
    gesture_colors: dict[str, str],
    font_size: int,
) -> None:
    """Plot the overall gesture distribution."""
    _apply_common_axis_settings(ax, used_exps, labels, font_size)
    for i, exp in enumerate(used_exps):
        parts = gesture_df[exp].fillna(0.0)
        bar100_row(ax, y=i, parts=parts, color_map=gesture_colors, font_size=font_size)

    ax.set_xlabel("Proportion", fontsize=font_size - 2)
    ax.set_title("Overall Pedal Gesture Distribution", pad=20, fontsize=font_size)

    handles = [plt.Rectangle((0, 0), 1, 1, color=gesture_colors[k]) for k in gesture_df.index]
    labels_gesture = [f"“{format_label(k)}”" for k in gesture_df.index]

    ax.legend(
        handles,
        labels_gesture,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=max(4, len(labels_gesture)),
        frameon=False,
        fontsize=font_size - 2,
    )


def save_outputs(out_dir: str, action_df: pd.DataFrame, gesture_df: pd.DataFrame) -> None:
    """Save summary CSVs for external verification."""
    action_out = action_df.copy()
    gesture_out = gesture_df.copy()
    action_out.index.name = "action_category"
    gesture_out.index.name = "gesture_category"
    action_out.to_csv(os.path.join(out_dir, "overall_action_props_all_experiments.csv"))
    gesture_out.to_csv(os.path.join(out_dir, "overall_gesture_props_all_experiments.csv"))
    print("[Saved] overall_action_props_all_experiments.csv")
    print("[Saved] overall_gesture_props_all_experiments.csv")


def run(config: PlotConfig) -> None:
    """Run end-to-end plotting with the given config."""
    if not config.csv_dir:
        raise SystemExit("csv_dir is required. Provide --csv-dir.")
    if not config.out_dir:
        raise SystemExit("out_dir is required. Provide --out-dir.")

    os.makedirs(config.out_dir, exist_ok=True)

    experiments = config.experiment_names or discover_experiments(config.csv_dir)
    if not config.experiment_names:
        print(f"[Auto] Discover {len(experiments)} experiments: {experiments}")

    action_matrix, gesture_matrix, used_exps = build_experiment_matrices(
        experiments,
        config.csv_dir,
        config.include_plain,
    )

    if not used_exps:
        raise SystemExit("No usable CSVs found. Exiting.")

    action_df = pd.DataFrame(action_matrix)
    gesture_df = pd.DataFrame(gesture_matrix)

    plt.rcParams.update({"font.size": config.font_size, "axes.unicode_minus": False})
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(config.fig_width, config.fig_height),
        dpi=config.dpi,
        constrained_layout=True,
    )

    used_exps_labels = [config.experiment_names_labels.get(e, e) for e in used_exps]

    plot_action_distribution(
        ax=ax_top,
        action_df=action_df,
        used_exps=used_exps,
        labels=used_exps_labels,
        action_colors=config.action_colors,
        font_size=config.font_size,
    )

    plot_gesture_distribution(
        ax=ax_bot,
        gesture_df=gesture_df,
        used_exps=used_exps,
        labels=used_exps_labels,
        gesture_colors=config.gesture_colors,
        font_size=config.font_size,
    )

    out_png = os.path.join(config.out_dir, "overall_dist_compare_all_exps_gesture-by-frame.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=config.dpi)
    plt.close(fig)
    print(f"[Saved] {out_png}")

    save_outputs(config.out_dir, action_df, gesture_df)


# ================= Main =================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot overall pedal distributions.")
    parser.add_argument("--csv_dir", help="Directory with test_*_gesture.csv files")
    parser.add_argument("--out_dir", help="Output directory for plots and CSVs")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> PlotConfig:
    return PlotConfig(
        experiment_names=DEFAULT_CONFIG.experiment_names,
        experiment_names_labels=DEFAULT_CONFIG.experiment_names_labels,
        csv_dir=args.csv_dir,
        out_dir=args.out_dir,
        fig_width=DEFAULT_CONFIG.fig_width,
        fig_height=DEFAULT_CONFIG.fig_height,
        dpi=DEFAULT_CONFIG.dpi,
        font_size=DEFAULT_CONFIG.font_size,
        include_plain=DEFAULT_CONFIG.include_plain,
        action_colors=DEFAULT_CONFIG.action_colors,
        gesture_colors=DEFAULT_CONFIG.gesture_colors,
    )


if __name__ == "__main__":
    args = parse_args()
    run(build_config_from_args(args))
