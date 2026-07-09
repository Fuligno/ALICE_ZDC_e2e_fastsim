#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Parsing run_name
# ============================================================

def normalize_run_name(s: str) -> str:
    return str(s).strip()

def parse_encoder(run_name: str) -> str:
    s = str(run_name).strip()
    m = re.search(r"ENC([^_]+)_MLP", s, flags=re.IGNORECASE)
    if not m:
        return "unknown"
    return m.group(1).strip().lower()

def parse_mlp_arch(run_name: str) -> str:
    s = str(run_name).strip()
    m = re.search(r"_MLP([^_]+)_ACT", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Impossibile riconoscere MLP da run_name='{run_name}'")
    return m.group(1).strip()

# ============================================================
# Plotting
# ============================================================

def build_style_maps(df: pd.DataFrame):
    arch_order = ["32-32", "64-64", "128-128", "32-32-32", "64-64-64", "128-128-128"]
    
    arch_present = [a for a in arch_order if a in set(df["mlp_arch"])]
    arch_extra = sorted(set(df["mlp_arch"]) - set(arch_order))
    arch_present.extend(arch_extra)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        arch: color_cycle[i % len(color_cycle)]
        for i, arch in enumerate(arch_present)
    }

    return arch_present, color_map

def make_single_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    yerr_col: str | None,
    ylabel: str,
    outpath: Path,
    annotate: bool = False,
):
    # Convertiamo ms in microsecondi (x * 1000)
    df = df.copy()
    df["x_us"] = df[x_col] * 1000

    arch_present, color_map = build_style_maps(df)

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        x = row["x_us"]
        y = row[y_col]
        color = color_map[row["mlp_arch"]]
        # Usiamo un marker fisso (es. cerchio 'o') per tutti
        marker = "o"

        if yerr_col is not None and yerr_col in df.columns:
            yerr = row[yerr_col]
            ax.errorbar(
                x, y, yerr=yerr,
                fmt=marker,
                color=color,
                markersize=8,
                capsize=3,
                linestyle="None",
                alpha=0.95,
            )
        else:
            ax.scatter(
                x, y,
                c=color,
                marker=marker,
                s=80,
                alpha=0.95,
            )

        if annotate:
            ax.annotate(
                row["run_name"],
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    # Rimosso il titolo come richiesto
    ax.set_xlabel(r"Gen time [$\mu$s/sample]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Legenda solo per le architetture MLP
    color_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[arch],
            markeredgecolor=color_map[arch],
            markersize=8,
            linestyle="None",
            label=arch,
        )
        for arch in arch_present
    ]

    ax.legend(
        handles=color_handles,
        title="MLP architecture",
        loc="best",
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_time", type=str, default="runs/generation_time.csv")
    parser.add_argument("--csv_metrics", type=str, default="runs/eval_summary_ZNZP.csv")
    parser.add_argument("--outdir", type=str, default="plots_swd_vs_time_no_enc")
    parser.add_argument("--time_col", type=str, default="gen_time_ms_per_sample_mean")
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_time = pd.read_csv(args.csv_time)
    df_metrics = pd.read_csv(args.csv_metrics)

    df_time["run_name"] = df_time["run_name"].map(normalize_run_name)
    df_metrics["run_name"] = df_metrics["run_name"].map(normalize_run_name)

    df = pd.merge(df_metrics, df_time[["run_name", args.time_col]], on="run_name", how="inner")

    # Parsing e FILTRAGGIO: solo modelli senza encoder
    df["encoder"] = df["run_name"].map(parse_encoder)
    df["mlp_arch"] = df["run_name"].map(parse_mlp_arch)
    
    df = df[df["encoder"] == "none"].reset_index(drop=True)

    if len(df) == 0:
        print("Attenzione: Nessun modello con 'ENCnone' trovato.")
        return

    # Plot SWD sui 10 canali (all10)
    make_single_plot(
        df=df,
        x_col=args.time_col,
        y_col="SWD_all_all10_mean",
        yerr_col="SWD_all_all10_std",
        ylabel="SWD",
        outpath=outdir / "swd_all10_vs_time_us.png",
        annotate=args.annotate,
    )

    print(f"[OK] Plot salvato in: {outdir / 'swd_all10_vs_time_us.png'}")

if __name__ == "__main__":
    main()