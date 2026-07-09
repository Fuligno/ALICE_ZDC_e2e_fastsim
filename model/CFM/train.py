import os
import csv
import json
import time
import argparse
import random

import numpy as np
import torch
from torch import optim

from model import (
    load_and_preprocess_parquets,
    create_dataloaders,
    ConditionalCFMNet,
    save_model_and_scalers,
)


DEFAULT_DATA_PATHS = [
    "../../data/dataZN/",
    "../../data/dataZP/",
]

TEST_FRACTION_DEFAULT = 0.1
EPOCHS_DEFAULT = 100
SEED_DEFAULT = 42
METRICS_SEED = 12345


def parse_layers(s: str) -> list[int]:
    """Parse layer sizes from strings like '128,128,128', '128 128', '[256,256]' or '[]'."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    if len(s) == 0:
        return []

    for sep in [",", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip() != ""]
            break
    else:
        parts = [s]

    layers = [int(p) for p in parts]
    if any(h <= 0 for h in layers):
        raise ValueError("All layer sizes must be > 0, or the list must be empty.")
    return layers


def lr_to_str(lr: float) -> str:
    lr = float(lr)
    s = f"{lr:.0e}" if (lr < 1e-2 or lr >= 1e2) else f"{lr:g}"
    if "e" in s:
        base, exp = s.split("e")
        exp = exp.lstrip("+")
        if exp.startswith("-"):
            exp = "-" + (exp[1:].lstrip("0") or "0")
        else:
            exp = exp.lstrip("0") or "0"
        s = base + "e" + exp
    return s.replace(".", "p")


def layers_to_str(layers: list[int]) -> str:
    return "-".join(str(x) for x in layers) if layers else "none"


def make_run_name(
    encoder_layers: list[int],
    mlp_layers: list[int],
    activation: str,
    lr: float,
    batch_size: int,
) -> str:
    return (
        f"ENC{layers_to_str(encoder_layers)}_"
        f"MLP{layers_to_str(mlp_layers)}_"
        f"ACT{activation.lower()}_"
        f"lr{lr_to_str(lr)}_"
        f"bs{batch_size}"
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flow_matching_loss(model, x, cond, generator=None):
    """Pathwise flow matching loss:
    x_t = (1-t) x0 + t x1 with x0 ~ N(0, I), t ~ U(0, 1), target u_t = x1 - x0.
    """
    batch_size = x.shape[0]

    if generator is None:
        x0 = torch.randn_like(x)
        t = torch.rand(batch_size, 1, device=x.device, dtype=x.dtype)
    else:
        x0 = torch.randn(x.shape, device=x.device, generator=generator, dtype=x.dtype)
        t = torch.rand((batch_size, 1), device=x.device, generator=generator, dtype=x.dtype)

    x_t = (1.0 - t) * x0 + t * x
    u_t = x - x0

    v_theta = model(x_t, t, cond)
    return torch.mean((v_theta - u_t) ** 2)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x_batch, cond_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        cond_batch = cond_batch.to(device, non_blocking=True)
        batch_size = x_batch.shape[0]

        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, x_batch, cond_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / max(1, n_samples)


@torch.no_grad()
def compute_metrics(model, data_loader, device, seed: int = METRICS_SEED):
    """MSE, MAE and a 1D Wasserstein distance between v_theta and u_t over the loader."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    true_list = []
    pred_list = []

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    for x_batch, cond_batch in data_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        cond_batch = cond_batch.to(device, non_blocking=True)
        batch_size = x_batch.shape[0]

        x0 = torch.randn(x_batch.shape, device=device, generator=gen, dtype=x_batch.dtype)
        t = torch.rand((batch_size, 1), device=device, generator=gen, dtype=x_batch.dtype)

        x_t = (1.0 - t) * x0 + t * x_batch
        u_t = x_batch - x0
        v_theta = model(x_t, t, cond_batch)

        diff = v_theta - u_t
        total_mse += torch.sum(diff ** 2).item()
        total_mae += torch.sum(torch.abs(diff)).item()
        n_samples += diff.numel()

        true_list.append(u_t.cpu().reshape(-1))
        pred_list.append(v_theta.cpu().reshape(-1))

    mean_mse = total_mse / max(1, n_samples)
    mean_mae = total_mae / max(1, n_samples)

    if true_list:
        xs = np.sort(torch.cat(true_list).numpy())
        ys = np.sort(torch.cat(pred_list).numpy())
        wd = float(np.mean(np.abs(xs - ys)))
    else:
        wd = float("nan")

    return mean_mse, mean_mae, wd


def parse_args():
    parser = argparse.ArgumentParser(
        description="CFM training on the merged ZDC dataset (per-source split, then merged train/test)."
    )

    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=DEFAULT_DATA_PATHS,
        help="List of dataset paths. Each path can be a directory or a parquet file.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=TEST_FRACTION_DEFAULT,
        help=f"Test fraction for each source dataset (default {TEST_FRACTION_DEFAULT}).",
    )
    parser.add_argument(
        "--encoder_layers",
        type=str,
        default="[]",
        help="Condition encoder layers. Examples: '[]', '64', '128,64'. Default: no encoder.",
    )
    parser.add_argument(
        "--mlp_layers",
        type=str,
        default="128,128,128",
        help="Hidden layers of the generative model. Examples: '128,128,128' or '[256,256]'.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=["silu", "relu", "gelu"],
        help="Activation used in the encoder and the MLP.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default 256).")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help=f"Epochs (default {EPOCHS_DEFAULT}).")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help=f"Seed (default {SEED_DEFAULT}).")
    parser.add_argument("--device", type=str, default=None, help="Device override: 'cpu' or 'cuda'.")
    parser.add_argument("--out_root", type=str, default="runs", help="Root directory for run folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers.")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory in the DataLoaders.")

    return parser.parse_args()


def main():
    args = parse_args()

    encoder_layers = parse_layers(args.encoder_layers)
    mlp_layers = parse_layers(args.mlp_layers)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed)

    run_name = make_run_name(
        encoder_layers=encoder_layers,
        mlp_layers=mlp_layers,
        activation=args.activation,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    run_dir = os.path.join(args.out_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    model_best_path = os.path.join(run_dir, "model_best.pt")
    scalers_path = os.path.join(run_dir, "scalers.pt")
    config_json_path = os.path.join(run_dir, "config.json")

    data, scalers = load_and_preprocess_parquets(
        parquet_sources=args.data_paths,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    cond_dim = int(data["train"]["cond"].shape[1])
    x_dim = int(data["train"]["x"].shape[1])

    train_loader, test_loader = create_dataloaders(
        data,
        batch_size=args.batch_size,
        shuffle_train=True,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
    )

    model = ConditionalCFMNet(
        cond_dim=cond_dim,
        x_dim=x_dim,
        encoder_hidden_dims=encoder_layers,
        mlp_hidden_dims=mlp_layers,
        activation=args.activation,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_config = {
        "run_name": run_name,
        "data_paths_input": list(args.data_paths),
        "resolved_parquet_paths": list(scalers["resolved_paths"]),
        "split_info": list(scalers["split_info"]),
        "test_fraction": float(args.test_fraction),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "device": device,
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "cond_cols": list(scalers["cond_cols"]),
        "target_cols": list(scalers["x_cols"]),
        "n_train_total": int(data["train"]["x"].shape[0]),
        "n_test_total": int(data["test"]["x"].shape[0]),
        "architecture": model.get_config(),
    }
    with open(config_json_path, "w") as f:
        json.dump(run_config, f, indent=2)

    print("Sources:")
    for info in scalers["split_info"]:
        print(
            f"  - {info['source']} | total_after_check={info['n_total_after_check']} "
            f"| train={info['n_train']} | test={info['n_test']}"
        )
    print(f"Merged train: {data['train']['x'].shape[0]} rows")
    print(f"Merged test : {data['test']['x'].shape[0]} rows")
    print(f"cond_dim={cond_dim} | x_dim={x_dim}")

    best_test_mse = float("inf")
    best_epoch = None

    with open(metrics_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss_cfm",
            "MSE_train",
            "MSE_test",
            "MAE_train",
            "MAE_test",
            "WD_train",
            "WD_test",
            "epoch_time_sec",
            "is_best",
        ])

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()

            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            train_mse, train_mae, train_wd = compute_metrics(model, train_loader, device)
            test_mse, test_mae, test_wd = compute_metrics(model, test_loader, device)

            dt = time.perf_counter() - t0

            is_best = 0
            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_epoch = epoch
                is_best = 1

                save_model_and_scalers(
                    model=model,
                    scalers=scalers,
                    model_path=model_best_path,
                    scalers_path=scalers_path,
                )

            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "run_name": run_name,
                    "model_config": model.get_config(),
                },
                ckpt_path,
            )

            print(
                f"[{run_name}] Epoch {epoch:04d}/{args.epochs:04d} | "
                f"loss={train_loss:.6e} | "
                f"MSE tr={train_mse:.6e} te={test_mse:.6e} | "
                f"MAE tr={train_mae:.6e} te={test_mae:.6e} | "
                f"WD tr={train_wd:.6e} te={test_wd:.6e} | "
                f"time={dt:.2f}s | best={bool(is_best)}"
            )

            writer.writerow([
                epoch,
                train_loss,
                train_mse,
                test_mse,
                train_mae,
                test_mae,
                train_wd,
                test_wd,
                dt,
                is_best,
            ])
            f.flush()

    print(f"\nRun: {run_dir}")
    print(f"Best epoch: {best_epoch} | best MSE_test: {best_test_mse:.6e}")
    print(f"Best model: {model_best_path}")
    print(f"Scalers: {scalers_path}")
    print(f"Metrics: {metrics_csv}")
    print(f"Config: {config_json_path}")


if __name__ == "__main__":
    main()