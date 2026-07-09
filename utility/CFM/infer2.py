#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import (
    COND_COLS,
    TARGET_COLS,
    load_model_and_scalers,
    resolve_parquet_source,
    sample_cfm,
    split_train_test_indices,
)


DEFAULT_DATA_PATHS = [
    "../../data/dataZN/",
    "../../data/dataZP/",
]

PDG_LABELS = {2212: "protons", 2112: "neutrons"}

ZN_TARGET_COLS = TARGET_COLS[:5]
ZP_TARGET_COLS = TARGET_COLS[5:]
FIG_TARGET_GROUPS = [("ZN", ZN_TARGET_COLS), ("ZP", ZP_TARGET_COLS)]

GEANT4_COLOR = "navy"
FASTSIM_COLOR = "crimson"
COND_EPS = 1e-12


def set_global_determinism(seed: int, device: str):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _read_checked_parquet(source: str | Path, keep_check_value: int) -> tuple[pd.DataFrame, str]:
    parquet_path = resolve_parquet_source(source)
    df = pd.read_parquet(parquet_path)

    required_cols = list(COND_COLS) + list(TARGET_COLS) + ["Check"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet {parquet_path} is missing columns: {missing}")

    df = df[df["Check"] == keep_check_value].reset_index(drop=True)
    return df, parquet_path


def load_full_dataset(parquet_sources: Sequence[str | Path], keep_check_value: int = 1) -> pd.DataFrame:
    """Load the whole dataset, without splitting, for the global Geant4 distributions."""
    if len(parquet_sources) == 0:
        raise ValueError("parquet_sources cannot be empty.")

    parts = []
    for source in parquet_sources:
        df, parquet_path = _read_checked_parquet(source, keep_check_value)
        df["__source__"] = parquet_path
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


def load_merged_test_split(
    parquet_sources: Sequence[str | Path],
    test_fraction: float,
    seed: int,
    keep_check_value: int = 1,
) -> pd.DataFrame:
    """Rebuild the merged test split with the same per-source split used in training."""
    if len(parquet_sources) == 0:
        raise ValueError("parquet_sources cannot be empty.")

    test_parts = []
    for source in parquet_sources:
        df, parquet_path = _read_checked_parquet(source, keep_check_value)
        if len(df) < 2:
            raise ValueError(
                f"After filtering Check == {keep_check_value}, dataset {parquet_path} has only {len(df)} rows."
            )

        _, test_idx = split_train_test_indices(len(df), test_fraction=test_fraction, seed=seed)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        df_test["__source__"] = parquet_path
        test_parts.append(df_test)

    return pd.concat(test_parts, ignore_index=True)


@torch.no_grad()
def generate_on_conditions(
    model,
    scalers: dict,
    cond_phys: np.ndarray,
    batch_size: int,
    n_steps: int,
    device: str,
) -> np.ndarray:
    if cond_phys.ndim != 2:
        raise ValueError("cond_phys must have shape (N, cond_dim).")

    cond_mean = scalers["cond_mean"].to(device=device, dtype=torch.float32)
    cond_scale = scalers["cond_std"].to(device=device, dtype=torch.float32)

    outputs = []
    for start in range(0, cond_phys.shape[0], batch_size):
        cond_batch = torch.from_numpy(cond_phys[start : start + batch_size]).to(
            device=device, dtype=torch.float32
        )
        cond_batch_std = (cond_batch - cond_mean) / (cond_scale + COND_EPS)
        x_gen = sample_cfm(
            model=model, cond_std=cond_batch_std, scalers=scalers, n_steps=n_steps, device=device
        )
        outputs.append(x_gen.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def _safe_relative_diff_pct(gen_value: float, real_value: float) -> float:
    if not np.isfinite(real_value) or abs(real_value) < 1e-12:
        return float("nan")
    return float(100.0 * (gen_value - real_value) / real_value)


def _resolution(sigma: float, mean: float) -> float:
    if np.isfinite(sigma) and np.isfinite(mean) and abs(mean) > 1e-12:
        return float(sigma / mean)
    return float("nan")


def _resolution_metrics(real: np.ndarray, gen: np.ndarray) -> dict:
    real = np.asarray(real, dtype=float).reshape(-1)
    gen = np.asarray(gen, dtype=float).reshape(-1)
    real = real[np.isfinite(real)]
    gen = gen[np.isfinite(gen)]

    nan = float("nan")
    if real.size == 0 or gen.size == 0:
        return {
            "n_real": int(real.size), "n_gen": int(gen.size),
            "mean_real": nan, "mean_gen": nan,
            "sigma_real": nan, "sigma_gen": nan,
            "resolution_real": nan, "resolution_gen": nan,
            "bias_mean_pct": nan, "bias_sigma_pct": nan, "bias_resolution_pct": nan,
        }

    mean_real, mean_gen = float(np.mean(real)), float(np.mean(gen))
    sigma_real = float(np.std(real, ddof=1)) if real.size > 1 else nan
    sigma_gen = float(np.std(gen, ddof=1)) if gen.size > 1 else nan
    resolution_real = _resolution(sigma_real, mean_real)
    resolution_gen = _resolution(sigma_gen, mean_gen)

    return {
        "n_real": int(real.size), "n_gen": int(gen.size),
        "mean_real": mean_real, "mean_gen": mean_gen,
        "sigma_real": sigma_real, "sigma_gen": sigma_gen,
        "resolution_real": resolution_real, "resolution_gen": resolution_gen,
        "bias_mean_pct": _safe_relative_diff_pct(mean_gen, mean_real),
        "bias_sigma_pct": _safe_relative_diff_pct(sigma_gen, sigma_real),
        "bias_resolution_pct": _safe_relative_diff_pct(resolution_gen, resolution_real),
    }


def compute_bias_resolution_table(x_true: np.ndarray, x_gen: np.ndarray, pdg_value: int) -> list[dict]:
    pdg_label = PDG_LABELS.get(int(pdg_value), f"pdg_{int(pdg_value)}")

    rows = []
    for det_name, cols in FIG_TARGET_GROUPS:
        idxs = [TARGET_COLS.index(c) for c in cols]
        metrics = _resolution_metrics(
            np.sum(x_true[:, idxs], axis=1),
            np.sum(x_gen[:, idxs], axis=1),
        )
        rows.append({"pdg": int(pdg_value), "particle": pdg_label, "detector": det_name, **metrics})

    return rows


def _compute_hist_range(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0

    lo = max(0.0, float(min(np.quantile(a, 0.001), np.quantile(b, 0.001))))
    hi = float(max(np.quantile(a, 0.999), np.quantile(b, 0.999)))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = max(0.0, float(min(a.min(initial=0.0), b.min(initial=0.0))))
        hi = float(max(a.max(initial=1.0), b.max(initial=1.0)))
        if hi <= lo:
            hi = lo + 1.0

    return lo, hi


def _compute_bin_edges(lo: float, hi: float, max_bins: int) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([0.0, 1.0], dtype=float)

    n_bins = min(int(max_bins), max(1, int(np.ceil(hi - lo))))
    return np.linspace(lo, hi, n_bins + 1, dtype=float)


def _pretty_channel_name(name: str) -> str:
    return str(name).replace("_", " ")


def _draw_histograms(ax, series: Sequence[tuple[np.ndarray, str, str, float]], edges: np.ndarray, fontsize: int):
    for values, label, color, alpha in series:
        ax.hist(values, bins=edges, density=True, histtype="stepfilled", color=color, alpha=alpha, label=label)

    ax.set_xlabel("Counts", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=fontsize - 1, frameon=True)


def make_single_plot(series: Sequence[tuple[np.ndarray, str, str, float]], out_png: str, bins: int):
    """One-channel histogram: Geant4 alone, or Geant4 against FastSim."""
    fig, ax = plt.subplots(figsize=(6, 5))

    reference = series[0][0]
    comparison = series[1][0] if len(series) > 1 else reference
    edges = _compute_bin_edges(*_compute_hist_range(reference, comparison), bins)

    _draw_histograms(ax, series, edges, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def make_canvas(x_true: np.ndarray, x_gen: np.ndarray, out_png: str, bins: int):
    fig, axes = plt.subplots(4, 3, figsize=(12.0, 10.0), constrained_layout=True)
    axes_flat = axes.flatten()

    plot_specs = [name for _, cols in FIG_TARGET_GROUPS for name in cols]

    for ax, target_name in zip(axes_flat, plot_specs):
        j = TARGET_COLS.index(target_name)
        real = np.asarray(x_true[:, j]).reshape(-1)
        gen = np.asarray(x_gen[:, j]).reshape(-1)
        edges = _compute_bin_edges(*_compute_hist_range(real, gen), bins)

        _draw_histograms(
            ax,
            [(real, "Geant4", GEANT4_COLOR, 0.90), (gen, "FastSim", FASTSIM_COLOR, 0.70)],
            edges,
            fontsize=10,
        )
        ax.set_title(_pretty_channel_name(target_name), fontsize=11)
        ax.tick_params(axis="both", labelsize=9)

    for ax in axes_flat[len(plot_specs):]:
        ax.axis("off")

    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def make_bias_resolution_matrix_plot(bias_df: pd.DataFrame, out_png: str, value_col: str = "bias_resolution_pct"):
    particles = ["neutrons", "protons"]
    detectors = ["ZN", "ZP"]

    matrix = np.full((len(particles), len(detectors)), np.nan, dtype=float)
    for i, particle in enumerate(particles):
        for j, detector in enumerate(detectors):
            sel = bias_df[(bias_df["particle"] == particle) & (bias_df["detector"] == detector)]
            if len(sel) == 1:
                matrix[i, j] = float(sel.iloc[0][value_col])

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(detectors)), labels=detectors)
    ax.set_yticks(np.arange(len(particles)), labels=particles)
    fig.colorbar(im, ax=ax).set_label(f"{value_col} [%]")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "nan" if not np.isfinite(value) else f"{value:+.2f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=12)

    ax.set_title("Bias on relative resolution: generated vs test")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CFM inference on the test split fixed by training, producing one canvas per particle."
    )
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="ENCnone_MLP128-128-128_ACTsilu_lr1e-3_bs256")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=DEFAULT_DATA_PATHS,
        help="Dataset paths. Each one can be a folder or a parquet file.",
    )
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=123456)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--plots_root", type=str, default="plots")
    return parser.parse_args()


def _find_control_cols() -> list[str]:
    control_cols = [c for c in TARGET_COLS if "control" in c.lower()]
    if not control_cols:
        print("[WARN] No column with 'control' in its name. Falling back to the last ZN and ZP channels.")
        control_cols = [ZN_TARGET_COLS[-1], ZP_TARGET_COLS[-1]]
    return control_cols


def _plot_full_dataset_distributions(df_full: pd.DataFrame, control_cols: Sequence[str], out_dir: str, bins: int):
    for pdg_value, label in PDG_LABELS.items():
        df_pdg = df_full[df_full["Pdg"].astype(int) == pdg_value]
        if df_pdg.empty:
            continue

        for col in control_cols:
            real_full = df_pdg[col].to_numpy(dtype=np.float32)
            out_png = os.path.join(out_dir, f"plot_full_g4_{col}_{label}.pdf")
            make_single_plot([(real_full, "Geant4 (Full Dataset)", GEANT4_COLOR, 0.9)], out_png, bins)
            print(f"[OK] saved {out_png}")


def _save_bias_tables(bias_rows: list[dict], out_dir: str):
    bias_df = pd.DataFrame(bias_rows)

    bias_csv = os.path.join(out_dir, "bias_resolution_matrix.csv")
    bias_df.to_csv(bias_csv, index=False)
    print(f"[OK] saved {bias_csv}")

    print("\n[BIAS RESOLUTION TABLE]")
    print(
        bias_df[[
            "particle", "detector", "n_real", "mean_real", "mean_gen", "sigma_real", "sigma_gen",
            "resolution_real", "resolution_gen", "bias_mean_pct", "bias_sigma_pct", "bias_resolution_pct",
        ]].to_string(index=False)
    )

    for value_col, filename in [
        ("bias_resolution_pct", "bias_resolution_matrix.png"),
        ("bias_mean_pct", "bias_mean_matrix.png"),
        ("bias_sigma_pct", "bias_sigma_matrix.png"),
    ]:
        out_png = os.path.join(out_dir, filename)
        make_bias_resolution_matrix_plot(bias_df=bias_df, out_png=out_png, value_col=value_col)
        print(f"[OK] saved {out_png}")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(args.runs_dir, args.run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run not found: {run_dir}")

    model_path = os.path.join(run_dir, "model_best.pt")
    scalers_path = os.path.join(run_dir, "scalers.pt")
    for path in (model_path, scalers_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}")

    out_dir = os.path.join(args.plots_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    model, scalers = load_model_and_scalers(model_path=model_path, scalers_path=scalers_path, device=device)
    control_cols = _find_control_cols()

    print("\n[INFO] Loading the full dataset for the Geant4 distributions...")
    df_full = load_full_dataset(parquet_sources=args.data_paths, keep_check_value=1)
    _plot_full_dataset_distributions(df_full, control_cols, out_dir, args.bins)

    print("\n[INFO] Loading and splitting the test dataset...")
    df_test = load_merged_test_split(
        parquet_sources=args.data_paths,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
        keep_check_value=1,
    )
    print(f"[INFO] Test merged size after Check==1 and split: {len(df_test)}")
    print(f"[INFO] split_seed={args.split_seed}")
    print(f"[INFO] sample_seed={args.sample_seed}")

    set_global_determinism(args.sample_seed, device)
    bias_rows = []

    for pdg_value, label in PDG_LABELS.items():
        df_pdg = df_test[df_test["Pdg"].astype(int) == pdg_value].reset_index(drop=True)
        if df_pdg.empty:
            print(f"[WARN] No event in the merged test set for PDG={pdg_value}, skipping.")
            continue

        cond_phys = df_pdg[COND_COLS].to_numpy(dtype=np.float32)
        x_true = df_pdg[TARGET_COLS].to_numpy(dtype=np.float32)

        x_gen = generate_on_conditions(
            model=model, scalers=scalers, cond_phys=cond_phys,
            batch_size=args.batch_size, n_steps=args.n_steps, device=device,
        )

        out_png = os.path.join(out_dir, f"canvas_{label}.pdf")
        make_canvas(x_true=x_true, x_gen=x_gen, out_png=out_png, bins=args.bins)
        print(f"[OK] saved {out_png} | N={len(df_pdg)}")

        for col in control_cols:
            idx = TARGET_COLS.index(col)
            out_png = os.path.join(out_dir, f"plot_test_cmp_{col}_{label}.pdf")
            make_single_plot(
                [
                    (x_true[:, idx].reshape(-1), "Geant4", GEANT4_COLOR, 0.9),
                    (x_gen[:, idx].reshape(-1), "FastSim", FASTSIM_COLOR, 0.7),
                ],
                out_png,
                args.bins,
            )
            print(f"[OK] saved {out_png}")

        bias_rows.extend(compute_bias_resolution_table(x_true=x_true, x_gen=x_gen, pdg_value=pdg_value))

    if bias_rows:
        _save_bias_tables(bias_rows, out_dir)


if __name__ == "__main__":
    main()