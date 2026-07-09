#!/usr/bin/env python3
import os
import csv
import json
import time
import argparse
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from model import load_model_and_scalers, resolve_parquet_source, sample_cfm, split_train_test_indices


DEFAULT_DATA_PATHS = [
    "../../data/dataZN/",
    "../../data/dataZP/",
]
DEFAULT_TEST_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 42

# Keys under which train.py may have stored the dataset paths, most specific first.
CONFIG_PATH_KEYS = ("resolved_parquet_paths", "data_paths_input", "data_paths")

_COND_CACHE: dict = {}


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


def _load_filtered_cond(parquet_path: str, cond_cols: Sequence[str]) -> np.ndarray:
    key = (parquet_path, tuple(cond_cols))
    if key in _COND_CACHE:
        return _COND_CACHE[key]

    df = pd.read_parquet(parquet_path, columns=list(cond_cols) + ["Check"])
    df = df[df["Check"] == 1]
    if len(df) == 0:
        raise ValueError(f"No rows with Check == 1 in parquet {parquet_path}")

    cond = df[list(cond_cols)].to_numpy(dtype=np.float32)
    _COND_CACHE[key] = cond
    return cond


def build_merged_test_conditions(
    data_paths: Iterable[str],
    cond_cols: Sequence[str],
    test_fraction: float,
    split_seed: int,
) -> np.ndarray:
    """Rebuild the merged test conditions with the same per-source split used in training."""
    test_parts = []
    for source in data_paths:
        cond_all = _load_filtered_cond(resolve_parquet_source(source), cond_cols)
        _, test_idx = split_train_test_indices(len(cond_all), test_fraction, split_seed)
        test_parts.append(cond_all[test_idx])

    if not test_parts:
        raise ValueError("No dataset available to build the test set.")
    return np.concatenate(test_parts, axis=0)


def read_run_config(
    run_dir: str,
    default_data_paths: List[str],
    default_test_fraction: float,
    default_seed: int,
) -> Tuple[List[str], float, int]:
    cfg_path = os.path.join(run_dir, "config.json")
    data_paths = list(default_data_paths)
    test_fraction = float(default_test_fraction)
    split_seed = int(default_seed)

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

    test_fraction = float(cfg.get("test_fraction", test_fraction))
    split_seed = int(cfg.get("seed", split_seed))
    return data_paths, test_fraction, split_seed


def benchmark_time_only(model, cond_std: torch.Tensor, scalers: dict, n_steps: int, device: str) -> float:
    """Generation time in milliseconds per sample."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    sample_cfm(model=model, cond_std=cond_std, scalers=scalers, n_steps=n_steps, device=device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed * 1000.0 / max(1, cond_std.shape[0])


def find_run_dirs(runs_dir: str, run_name: Optional[str]) -> List[Tuple[str, str]]:
    if run_name is not None:
        path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Run not found: {path}")
        return [(run_name, path)]

    runs = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if os.path.isdir(path) and all(
            os.path.isfile(os.path.join(path, f)) for f in ("model_best.pt", "scalers.pt")
        ):
            runs.append((name, path))
    return runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure only the CFM generation time on the test set, replicating the training split."
    )
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None, help="If given, benchmark only this run.")
    parser.add_argument("--out_csv", type=str, default="generation_time.csv")
    parser.add_argument(
        "--data_paths",
        nargs="*",
        default=None,
        help="One or more parquet files or folders. If omitted, the run config.json is used, then the defaults.",
    )
    parser.add_argument("--test_fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--bench_n", type=int, default=100000)
    parser.add_argument("--warmup_n", type=int, default=1024)
    parser.add_argument("--n_time_seeds", type=int, default=20)
    parser.add_argument("--time_seed0", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.runs_dir):
        raise FileNotFoundError(f"runs_dir not found: {args.runs_dir}")

    run_folders = find_run_dirs(args.runs_dir, args.run_name)
    if not run_folders:
        print(f"No valid run found in {args.runs_dir}")
        return

    out_csv = args.out_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(args.runs_dir, out_csv)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name",
            "n_steps",
            "bench_n",
            "n_time_seeds",
            "gen_time_ms_per_sample_mean",
            "gen_time_ms_per_sample_std",
            "total_generated_per_benchmark",
            "device",
        ])

        for run_name, run_path in run_folders:
            model, scalers = load_model_and_scalers(
                model_path=os.path.join(run_path, "model_best.pt"),
                scalers_path=os.path.join(run_path, "scalers.pt"),
                device=device,
            )

            cond_cols = list(scalers.get("cond_cols", []))
            if not cond_cols:
                raise ValueError(f"Scalers of run {run_name} do not contain 'cond_cols'.")

            if args.data_paths:
                data_paths = list(args.data_paths)
                test_fraction = float(args.test_fraction)
                split_seed = int(args.split_seed)
            else:
                data_paths, test_fraction, split_seed = read_run_config(
                    run_dir=run_path,
                    default_data_paths=DEFAULT_DATA_PATHS,
                    default_test_fraction=args.test_fraction,
                    default_seed=args.split_seed,
                )

            cond_test_phys = build_merged_test_conditions(data_paths, cond_cols, test_fraction, split_seed)

            bench_n = min(int(args.bench_n), cond_test_phys.shape[0])
            if bench_n <= 0:
                raise ValueError(f"Invalid bench_n for run {run_name}: {bench_n}")

            cond_mean = scalers["cond_mean"].to(device=device, dtype=torch.float32)
            cond_scale = scalers["cond_std"].to(device=device, dtype=torch.float32)
            cond_test_t = torch.from_numpy(cond_test_phys[:bench_n]).to(device=device, dtype=torch.float32)
            cond_test_std = (cond_test_t - cond_mean) / (cond_scale + 1e-12)

            set_global_determinism(args.time_seed0, device)
            sample_cfm(
                model=model,
                cond_std=cond_test_std[: min(args.warmup_n, bench_n)],
                scalers=scalers,
                n_steps=args.n_steps,
                device=device,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            times_ms = []
            for k in range(args.n_time_seeds):
                set_global_determinism(args.time_seed0 + k, device)
                times_ms.append(benchmark_time_only(model, cond_test_std, scalers, args.n_steps, device))

            times_ms = np.asarray(times_ms, dtype=np.float64)
            mean_ms = float(times_ms.mean())
            std_ms = sample_std(times_ms)

            writer.writerow([
                run_name,
                int(args.n_steps),
                int(bench_n),
                int(args.n_time_seeds),
                mean_ms,
                std_ms,
                int(bench_n),
                device,
            ])
            f.flush()

            print(
                f"[OK] {run_name} | time = {mean_ms:.6f} +/- {std_ms:.6f} ms/sample "
                f"(N={bench_n}, seeds={args.n_time_seeds}, device={device})"
            )

    print(f"\nCSV written to: {out_csv}")


if __name__ == "__main__":
    main()