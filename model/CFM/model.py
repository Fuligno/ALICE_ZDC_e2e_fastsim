import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


COND_COLS = ["Pdg", "Energy", "Vx", "Vy", "Vz", "Px", "Py", "Pz"]
TARGET_COLS = [
    "ZN_controlChannel",
    "ZN_channelQ1",
    "ZN_channelQ2",
    "ZN_channelQ3",
    "ZN_channelQ4",
    "ZP_controlChannel",
    "ZP_channelQ1",
    "ZP_channelQ2",
    "ZP_channelQ3",
    "ZP_channelQ4",
]

_PREFERRED_PARQUET_NAMES = [
    "merged_dataset.parquet",
    "proton_merged.parquet",
    "neutron_merged.parquet",
]


def resolve_parquet_source(source: Union[str, Path]) -> str:
    """Resolve a source (parquet file or directory containing one) to a file path."""
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Path not found: {source_path}")

    if source_path.is_file():
        if source_path.suffix.lower() != ".parquet":
            raise ValueError(f"Not a parquet file: {source_path}")
        return str(source_path)

    if not source_path.is_dir():
        raise ValueError(f"Invalid path: {source_path}")

    preferred_hits = [source_path / name for name in _PREFERRED_PARQUET_NAMES if (source_path / name).exists()]
    if len(preferred_hits) == 1:
        return str(preferred_hits[0])
    if len(preferred_hits) > 1:
        raise ValueError(
            f"Multiple standard parquet files found in {source_path}: {[p.name for p in preferred_hits]}. "
            "Pass the parquet file explicitly."
        )

    parquets = sorted(source_path.glob("*.parquet"))
    if len(parquets) == 1:
        return str(parquets[0])
    if len(parquets) == 0:
        raise FileNotFoundError(f"No .parquet file found in {source_path}")

    raise ValueError(
        f"Ambiguous directory {source_path}: found {len(parquets)} parquet files "
        f"({[p.name for p in parquets]}). Pass the parquet file explicitly."
    )


def _validate_required_columns(df: pd.DataFrame, required_cols: Sequence[str], parquet_path: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Parquet {parquet_path} is missing columns: {missing}")


def split_train_test_indices(n_rows: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 2:
        raise ValueError(f"At least 2 rows are required for a train/test split, found {n_rows}.")
    if not (0.0 < float(test_fraction) < 1.0):
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_rows)

    n_test = int(round(n_rows * float(test_fraction)))
    n_test = min(max(1, n_test), n_rows - 1)

    return indices[n_test:], indices[:n_test]


def _safe_standard_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def load_and_preprocess_parquets(
    parquet_sources: Sequence[str | Path],
    test_fraction: float = 0.1,
    seed: int = 42,
    cond_cols: Sequence[str] = COND_COLS,
    target_cols: Sequence[str] = TARGET_COLS,
    check_col: str = "Check",
    keep_check_value: int = 1,
):
    """Load parquet sources, filter by `check_col == keep_check_value`, split each
    source separately, merge the splits, and standardize using train statistics.

    Returns (data, scalers): float32 train/test tensors and the standardization
    parameters plus metadata needed at inference time.
    """
    if len(parquet_sources) == 0:
        raise ValueError("parquet_sources cannot be empty.")

    resolved_paths: list[str] = []
    train_cond_parts: list[np.ndarray] = []
    train_x_parts: list[np.ndarray] = []
    test_cond_parts: list[np.ndarray] = []
    test_x_parts: list[np.ndarray] = []
    split_info: list[dict] = []

    required_cols = list(cond_cols) + list(target_cols) + [check_col]

    for source in parquet_sources:
        parquet_path = resolve_parquet_source(source)
        resolved_paths.append(parquet_path)

        df = pd.read_parquet(parquet_path, columns=required_cols)
        _validate_required_columns(df, required_cols, parquet_path)

        df = df[df[check_col] == keep_check_value].reset_index(drop=True)
        if len(df) < 2:
            raise ValueError(
                f"After filtering {check_col} == {keep_check_value}, dataset {parquet_path} has only {len(df)} rows."
            )

        train_idx, test_idx = split_train_test_indices(len(df), test_fraction=test_fraction, seed=seed)
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        train_cond_parts.append(df_train[list(cond_cols)].to_numpy(dtype=np.float32))
        train_x_parts.append(df_train[list(target_cols)].to_numpy(dtype=np.float32))
        test_cond_parts.append(df_test[list(cond_cols)].to_numpy(dtype=np.float32))
        test_x_parts.append(df_test[list(target_cols)].to_numpy(dtype=np.float32))

        split_info.append(
            {
                "source": parquet_path,
                "n_total_after_check": int(len(df)),
                "n_train": int(len(df_train)),
                "n_test": int(len(df_test)),
                "test_fraction": float(test_fraction),
                "seed": int(seed),
            }
        )

    cond_train = np.concatenate(train_cond_parts, axis=0)
    x_train = np.concatenate(train_x_parts, axis=0)
    cond_test = np.concatenate(test_cond_parts, axis=0)
    x_test = np.concatenate(test_x_parts, axis=0)

    cond_mean, cond_std = _safe_standard_stats(cond_train)
    x_mean, x_std = _safe_standard_stats(x_train)

    data = {
        "train": {
            "cond": torch.from_numpy((cond_train - cond_mean) / cond_std),
            "x": torch.from_numpy((x_train - x_mean) / x_std),
        },
        "test": {
            "cond": torch.from_numpy((cond_test - cond_mean) / cond_std),
            "x": torch.from_numpy((x_test - x_mean) / x_std),
        },
    }

    scalers = {
        "cond_mean": torch.from_numpy(cond_mean),
        "cond_std": torch.from_numpy(cond_std),
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "cond_cols": list(cond_cols),
        "x_cols": list(target_cols),
        "check_col": str(check_col),
        "keep_check_value": int(keep_check_value),
        "resolved_paths": list(resolved_paths),
        "split_info": list(split_info),
    }

    return data, scalers


class CFMDataset(Dataset):
    """Pairs of standardized targets x (N, x_dim) and conditions (N, cond_dim)."""

    def __init__(self, x: torch.Tensor, cond: torch.Tensor):
        if x.shape[0] != cond.shape[0]:
            raise ValueError("x and cond must have the same number of samples.")
        if x.dim() != 2 or cond.dim() != 2:
            raise ValueError("x and cond must be 2D tensors.")

        self.x = x.float()
        self.cond = cond.float()

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.cond[idx]


def create_dataloaders(
    data: dict,
    batch_size: int = 256,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    train_dataset = CFMDataset(x=data["train"]["x"], cond=data["train"]["cond"])
    test_dataset = CFMDataset(x=data["test"]["x"], cond=data["test"]["cond"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}


def _get_activation(name: str):
    key = name.lower().strip()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}. Use one of {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]


def _validate_hidden_dims(hidden_dims: Optional[Sequence[int]], field_name: str) -> list[int]:
    if hidden_dims is None:
        return []
    dims = [int(h) for h in hidden_dims]
    if any(h <= 0 for h in dims):
        raise ValueError(f"{field_name} must contain only integers > 0.")
    return dims


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], act_cls) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    return nn.Sequential(*layers), prev


class ConditionalCFMNet(nn.Module):
    """Velocity field v_theta(x_t, t, cond) for conditional flow matching.

    Inputs: x (B, x_dim), t_flow (B, 1) in [0, 1], cond (B, cond_dim).
    Output: (B, x_dim).
    """

    def __init__(
        self,
        cond_dim: int = 8,
        x_dim: int = 10,
        encoder_hidden_dims: Optional[Sequence[int]] = None,
        mlp_hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "silu",
    ):
        super().__init__()

        self.cond_dim = int(cond_dim)
        self.x_dim = int(x_dim)
        if self.cond_dim <= 0 or self.x_dim <= 0:
            raise ValueError("cond_dim and x_dim must be > 0.")

        self.encoder_hidden_dims = _validate_hidden_dims(encoder_hidden_dims, "encoder_hidden_dims")
        self.mlp_hidden_dims = _validate_hidden_dims(
            [128, 128, 128] if mlp_hidden_dims is None else mlp_hidden_dims,
            "mlp_hidden_dims",
        )
        self.activation_name = activation.lower().strip()
        act_cls = _get_activation(self.activation_name)

        if len(self.encoder_hidden_dims) == 0:
            self.cond_encoder = None
            cond_feat_dim = self.cond_dim
        else:
            self.cond_encoder, cond_feat_dim = _build_mlp(self.cond_dim, self.encoder_hidden_dims, act_cls)

        input_dim = self.x_dim + 1 + cond_feat_dim
        body, prev = _build_mlp(input_dim, self.mlp_hidden_dims, act_cls)
        self.mlp = nn.Sequential(*body, nn.Linear(prev, self.x_dim))

    def get_config(self) -> dict:
        return {
            "cond_dim": self.cond_dim,
            "x_dim": self.x_dim,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "mlp_hidden_dims": list(self.mlp_hidden_dims),
            "activation": self.activation_name,
        }

    def forward(self, x: torch.Tensor, t_flow: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t_flow.dim() == 1:
            t_flow = t_flow.unsqueeze(-1)

        cond_feat = cond if self.cond_encoder is None else self.cond_encoder(cond)
        h = torch.cat([x, t_flow, cond_feat], dim=-1)
        return self.mlp(h)


def transform_conditions(cond: Union[pd.DataFrame, np.ndarray, torch.Tensor], scalers: dict) -> torch.Tensor:
    """Standardize conditions with the training scalers."""
    if isinstance(cond, pd.DataFrame):
        arr = cond[scalers["cond_cols"]].to_numpy(dtype=np.float32)
    elif isinstance(cond, torch.Tensor):
        arr = cond.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(cond, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    cond_mean = scalers["cond_mean"].detach().cpu().numpy().astype(np.float32)
    cond_std = scalers["cond_std"].detach().cpu().numpy().astype(np.float32)
    return torch.from_numpy((arr - cond_mean) / cond_std)


def inverse_transform_targets(x_std: np.ndarray | torch.Tensor, scalers: dict) -> torch.Tensor:
    """Map standardized targets back to physical units."""
    if isinstance(x_std, torch.Tensor):
        x_std_t = x_std.float()
    else:
        x_std_t = torch.from_numpy(np.asarray(x_std, dtype=np.float32))

    x_mean = scalers["x_mean"].detach().cpu().float()
    x_std_scale = scalers["x_std"].detach().cpu().float()
    return x_std_t * x_std_scale + x_mean


def save_model_and_scalers(
    model: nn.Module,
    scalers: dict,
    model_path: str,
    scalers_path: str,
    extra_config_path: Optional[str] = None,
    verbose: bool = False,
):
    if not hasattr(model, "get_config"):
        raise ValueError("Model does not expose get_config().")

    payload = {
        "state_dict": model.state_dict(),
        "config": model.get_config(),
    }
    torch.save(payload, model_path)
    torch.save(scalers, scalers_path)

    if extra_config_path is not None:
        with open(extra_config_path, "w") as f:
            json.dump(payload["config"], f, indent=2)

    if verbose:
        print(f"Model saved to {model_path}")
        print(f"Scalers saved to {scalers_path}")
        if extra_config_path is not None:
            print(f"Config saved to {extra_config_path}")


def load_model_and_scalers(
    model_path: str,
    scalers_path: str,
    device: str = "cpu",
):
    """Rebuild the model from the saved config and load model + scalers."""
    ckpt = torch.load(model_path, map_location=device)
    scalers = torch.load(scalers_path, map_location=device)

    config = ckpt.get("config")
    state_dict = ckpt.get("state_dict")
    if config is None or state_dict is None:
        raise ValueError("Invalid checkpoint: expected 'config' and 'state_dict'.")

    model = ConditionalCFMNet(
        cond_dim=config["cond_dim"],
        x_dim=config["x_dim"],
        encoder_hidden_dims=config.get("encoder_hidden_dims", []),
        mlp_hidden_dims=config["mlp_hidden_dims"],
        activation=config.get("activation", "silu"),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, scalers


@torch.no_grad()
def sample_cfm(
    model: nn.Module,
    cond_std: torch.Tensor,
    scalers: dict,
    n_steps: int = 100,
    device: str = "cpu",
):
    """Generate samples in physical units, conditioned on standardized conditions,
    by integrating the learned ODE with midpoint-evaluated explicit Euler steps.
    """
    model.eval()
    cond_std = cond_std.to(device=device, dtype=torch.float32)
    if cond_std.dim() == 1:
        cond_std = cond_std.unsqueeze(0)

    batch_size = cond_std.shape[0]

    x_mean = scalers["x_mean"].to(device=device, dtype=torch.float32)
    x_std = scalers["x_std"].to(device=device, dtype=torch.float32)
    x_dim = int(x_mean.shape[0])

    x = torch.randn(batch_size, x_dim, device=device, dtype=torch.float32)

    dt = 1.0 / int(n_steps)
    t = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)

    for k in range(int(n_steps)):
        t.fill_((k + 0.5) * dt)
        x = x + dt * model(x, t, cond_std)

    return x * x_std + x_mean