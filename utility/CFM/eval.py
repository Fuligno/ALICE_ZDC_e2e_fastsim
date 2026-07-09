#!/usr/bin/env python3
import os
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

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

PDG_CODES = {"protons": 2212, "neutrons": 2112}
SUBSET_NAMES = ["all", "protons", "neutrons"]

ZN_IDXS = list(range(5))
ZP_IDXS = list(range(5, 10))

# Keys under which train.py may have stored the dataset paths, most specific first.
CONFIG_PATH_KEYS = ("resolved_parquet_paths", "data_paths_input", "data_paths")

COND_EPS = 1e-12


def set_global_determinism(seed: int, device: str):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def load_merged_test_split(
    parquet_sources: Sequence[str | Path],
    test_fraction: float,
    seed: int,
    keep_check_value: int = 1,
) -> pd.DataFrame:
    """Rebuild the merged test split with the same per-source split used in training."""
    if len(parquet_sources) == 0:
        raise ValueError("parquet_sources cannot be empty.")

    required_cols = list(COND_COLS) + list(TARGET_COLS) + ["Check", "Pdg"]
    test_parts: list[pd.DataFrame] = []

    for source in parquet_sources:
        parquet_path = resolve_parquet_source(source)
        df = pd.read_parquet(parquet_path)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet {parquet_path} is missing columns: {missing}")

        df = df[df["Check"] == keep_check_value].reset_index(drop=True)
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


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(np.sort(a[:n]) - np.sort(b[:n]))))


def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, num_projections: int = 200, seed: int = 123) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("sliced_wasserstein requires 2D arrays.")

    n = min(X.shape[0], Y.shape[0])
    if n == 0:
        return float("nan")
    X, Y = X[:n], Y[:n]

    rng = np.random.RandomState(seed)
    dirs = rng.randn(num_projections, X.shape[1])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    return float(np.mean([wasserstein_1d(X @ v, Y @ v) for v in dirs]))


def read_run_config(
    run_dir: str,
    cli_test_fraction: float,
    cli_split_seed: int,
    cli_data_paths: Optional[Sequence[str]],
) -> Tuple[List[str], float, int]:
    """CLI values win when provided; otherwise fall back to the run config.json, then to the defaults."""
    if cli_data_paths:
        return list(cli_data_paths), float(cli_test_fraction), int(cli_split_seed)

    data_paths = list(DEFAULT_DATA_PATHS)
    test_fraction = float(cli_test_fraction)
    split_seed = int(cli_split_seed)

    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return data_paths, test_fraction, split_seed

    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        print(f"[WARN] Cannot read config.json in {run_dir}: {err}")
        return data_paths, test_fraction, split_seed

    for key in CONFIG_PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, list) and value:
            data_paths = [str(x) for x in value]
            break
    else:
        if "parquet_path" in cfg:
            data_paths = [str(cfg["parquet_path"])]

    test_fraction = float(cfg.get("test_fraction", test_fraction))
    split_seed = int(cfg.get("seed", split_seed))
    return data_paths, test_fraction, split_seed


def benchmark_time_only(
    model,
    scalers: dict,
    cond_phys: np.ndarray,
    batch_size: int,
    n_steps: int,
    device: str,
) -> float:
    """Generation time in milliseconds per sample."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    generate_on_conditions(
        model=model, scalers=scalers, cond_phys=cond_phys,
        batch_size=batch_size, n_steps=n_steps, device=device,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed * 1000.0 / max(1, cond_phys.shape[0])


def compute_metrics_for_subset(x_true: np.ndarray, x_gen: np.ndarray, num_proj: int, swd_seed: int) -> dict:
    return {
        "wd_channels": np.asarray(
            [wasserstein_1d(x_true[:, j], x_gen[:, j]) for j in range(x_true.shape[1])],
            dtype=np.float64,
        ),
        "swd_zn": sliced_wasserstein(x_true[:, ZN_IDXS], x_gen[:, ZN_IDXS], num_proj, swd_seed),
        "swd_zp": sliced_wasserstein(x_true[:, ZP_IDXS], x_gen[:, ZP_IDXS], num_proj, swd_seed),
        "swd_all10": sliced_wasserstein(x_true, x_gen, num_proj, swd_seed),
    }


def summarize_seedwise_metrics(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return float(arr.mean()), sample_std(arr)
    if arr.ndim == 2:
        std = np.asarray([sample_std(arr[:, j]) for j in range(arr.shape[1])], dtype=np.float64)
        return arr.mean(axis=0), std
    raise ValueError("Unexpected array in summarize_seedwise_metrics.")


def find_run_dirs(runs_dir: str, run_name: Optional[str]) -> List[str]:
    if run_name is not None:
        candidates = [os.path.join(runs_dir, run_name)]
    else:
        candidates = [
            os.path.join(runs_dir, name)
            for name in sorted(os.listdir(runs_dir))
            if os.path.isdir(os.path.join(runs_dir, name))
        ]

    return [
        run_dir
        for run_dir in candidates
        if all(os.path.isfile(os.path.join(run_dir, f)) for f in ("model_best.pt", "scalers.pt"))
    ]


def build_csv_header() -> List[str]:
    header = ["run_name", "n_test_total", "n_test_protons", "n_test_neutrons", "gen_time_ms_per_sample"]
    for subset in SUBSET_NAMES:
        for col in TARGET_COLS:
            header += [f"WD_{subset}_{col}_mean", f"WD_{subset}_{col}_std"]
        for group in ("ZN", "ZP", "all10"):
            header += [f"SWD_{subset}_{group}_mean", f"SWD_{subset}_{group}_std"]
    return header


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CFM performance on a fixed test split: per-channel WD and "
                    "sliced WD on ZN/ZP/all10, separated into protons, neutrons and total."
    )
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="If given, evaluate only this run; otherwise every folder holding model_best.pt and scalers.pt.",
    )
    parser.add_argument(
        "--data_paths",
        nargs="*",
        default=None,
        help="One or more parquet files or folders. If omitted, the run config.json is used, then the defaults.",
    )
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed0", type=int, default=1000)
    parser.add_argument("--n_metric_seeds", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--bench_n", type=int, default=10000)
    parser.add_argument("--bench_seed", type=int, default=123456)
    parser.add_argument("--num_proj", type=int, default=200)
    parser.add_argument("--swd_seed", type=int, default=123)
    parser.add_argument("--out_csv", type=str, default="eval_summary_ZNZP.csv")
    return parser.parse_args()


def evaluate_run(run_dir: str, args, device: str, writer, csv_file) -> None:
    run_name = os.path.basename(run_dir)
    print(f"\n[INFO] Evaluating run: {run_name}")

    model, scalers = load_model_and_scalers(
        model_path=os.path.join(run_dir, "model_best.pt"),
        scalers_path=os.path.join(run_dir, "scalers.pt"),
        device=device,
    )

    data_paths, test_fraction, split_seed = read_run_config(
        run_dir=run_dir,
        cli_test_fraction=args.test_fraction,
        cli_split_seed=args.split_seed,
        cli_data_paths=args.data_paths,
    )

    df_test = load_merged_test_split(
        parquet_sources=data_paths,
        test_fraction=test_fraction,
        seed=split_seed,
        keep_check_value=1,
    )
    print(f"[INFO] test merged size after Check==1 and split: {len(df_test)}")
    print(f"[INFO] split_seed={split_seed}")

    pdg = df_test["Pdg"].astype(int).to_numpy()
    subset_masks = {
        "all": np.ones(len(df_test), dtype=bool),
        "protons": pdg == PDG_CODES["protons"],
        "neutrons": pdg == PDG_CODES["neutrons"],
    }

    cond_phys_all = df_test[COND_COLS].to_numpy(dtype=np.float32)
    x_true_all = df_test[TARGET_COLS].to_numpy(dtype=np.float32)

    bench_n = min(args.bench_n, len(df_test))
    cond_phys_bench = cond_phys_all[:bench_n]

    set_global_determinism(args.bench_seed, device)
    generate_on_conditions(
        model=model, scalers=scalers, cond_phys=cond_phys_bench[: min(1024, bench_n)],
        batch_size=args.batch_size, n_steps=args.n_steps, device=device,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    set_global_determinism(args.bench_seed, device)
    gen_ms_per_sample = benchmark_time_only(
        model=model, scalers=scalers, cond_phys=cond_phys_bench,
        batch_size=args.batch_size, n_steps=args.n_steps, device=device,
    )

    metrics_acc = {
        name: {"wd_channels": [], "swd_zn": [], "swd_zp": [], "swd_all10": []} for name in SUBSET_NAMES
    }

    for k in range(args.n_metric_seeds):
        sample_seed = args.sample_seed0 + k
        print(f"[INFO] sample_seed={sample_seed}")
        set_global_determinism(sample_seed, device)

        x_gen_all = generate_on_conditions(
            model=model, scalers=scalers, cond_phys=cond_phys_all,
            batch_size=args.batch_size, n_steps=args.n_steps, device=device,
        )

        for subset_name, mask in subset_masks.items():
            subset_metrics = compute_metrics_for_subset(
                x_true=x_true_all[mask], x_gen=x_gen_all[mask],
                num_proj=args.num_proj, swd_seed=args.swd_seed,
            )
            for key, value in subset_metrics.items():
                metrics_acc[subset_name][key].append(value)

    row = [
        run_name,
        len(df_test),
        int(subset_masks["protons"].sum()),
        int(subset_masks["neutrons"].sum()),
        gen_ms_per_sample,
    ]

    for subset_name in SUBSET_NAMES:
        wd_mean, wd_std = summarize_seedwise_metrics(metrics_acc[subset_name]["wd_channels"])
        for j in range(len(TARGET_COLS)):
            row += [float(wd_mean[j]), float(wd_std[j])]
        for key in ("swd_zn", "swd_zp", "swd_all10"):
            mean, std = summarize_seedwise_metrics(metrics_acc[subset_name][key])
            row += [float(mean), float(std)]

    writer.writerow(row)
    csv_file.flush()

    print(
        f"[OK] {run_name} | "
        f"N={len(df_test)}, Np={row[2]}, Nn={row[3]} | "
        f"SWD_all10(all)={np.mean(metrics_acc['all']['swd_all10']):.4e} | "
        f"t_gen={gen_ms_per_sample:.4f} ms/sample"
    )


def main():
    args = parse_args()

    if len(TARGET_COLS) != 10:
        raise ValueError(
            f"This script assumes 10 targets in total (5 ZN + 5 ZP). Current TARGET_COLS: "
            f"{len(TARGET_COLS)} -> {TARGET_COLS}"
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.runs_dir):
        raise FileNotFoundError(f"runs_dir not found: {args.runs_dir}")

    run_folders = find_run_dirs(args.runs_dir, args.run_name)
    if not run_folders:
        raise FileNotFoundError("No valid run found with model_best.pt and scalers.pt")

    out_csv = args.out_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(args.runs_dir, out_csv)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_csv_header())
        for run_dir in run_folders:
            evaluate_run(run_dir, args, device, writer, f)

    print(f"\nCSV written to: {out_csv}")


if __name__ == "__main__":
    main()