from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import argparse

import numpy as np
import pandas as pd


PRIMARY_COLS = ["Energy", "Vx", "Vy", "Vz", "Px", "Py", "Pz"]
RESPONSE_COLS = [
    "PhotonSum",
    "Time",
    "controlChannel",
    "channelQ1",
    "channelQ2",
    "channelQ3",
    "channelQ4",
]
OUTPUT_COLS = (
    ["Pdg"]
    + PRIMARY_COLS
    + [f"ZN_{c}" for c in RESPONSE_COLS]
    + [f"ZP_{c}" for c in RESPONSE_COLS]
    + ["Check"]
)

INFO_COLS = ["Detector", "Pdg"] + PRIMARY_COLS + ["PhotonSum", "Time"]
DUPLICATE_COLS = ["Pdg"] + PRIMARY_COLS


def _read_matrix_txt(path: str) -> np.ndarray:
    """Read a whitespace-separated numeric text file as a 2D array."""
    arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0:
        raise ValueError(f"Empty image file: {path}")
    return arr


def _infer_block_image_shape(arr: np.ndarray, n_images: int) -> Tuple[int, int]:
    """Infer (Nx, Ny) for a file holding n_images consecutive Nx x Ny blocks."""
    if n_images <= 0:
        raise ValueError("n_images must be > 0 to infer the image shape.")

    total_lines, ny = arr.shape
    if total_lines % n_images != 0:
        raise ValueError(
            f"Cannot infer Nx: total_lines={total_lines} is not divisible by n_images={n_images}. "
            "This usually means the image file and the filtered info file disagree."
        )
    return total_lines // n_images, ny


def _load_images_flexible(
    image_file: str,
    image_shape: Optional[Tuple[int, int]] = None,
    n_images: Optional[int] = None,
) -> np.ndarray:
    """Load images stored as (n*Nx) x Ny blocks, as n x (Nx*Ny) flattened rows,
    or as a single Nx x Ny image. The shape is inferred when image_shape is None.
    """
    arr = _read_matrix_txt(image_file)

    if image_shape is None:
        if n_images is None:
            raise ValueError("With image_shape=None you must pass n_images to infer the shape.")
        image_shape = _infer_block_image_shape(arr, n_images)

    nx, ny = image_shape
    h, w = arr.shape

    if w == nx * ny:
        return arr.reshape(-1, nx, ny)
    if w == ny and h % nx == 0:
        return arr.reshape(-1, nx, ny)
    if (h, w) == (nx, ny):
        return arr.reshape(1, nx, ny)

    raise ValueError(
        f"Unrecognized image layout: shape={arr.shape}. With image_shape={image_shape} "
        f"expected (n,{nx * ny}) or ({nx}*n,{ny}) or ({nx},{ny})."
    )


def _channel_masks(nx: int, ny: int, det_kind: str) -> list[np.ndarray]:
    """Pixel masks for [control, Ch1..Ch4], mirroring O2's
    SpatialPhotonResponse::getPhotonsPerChannel(). Indices are [x][y].
    """
    xx, yy = np.indices((nx, ny))
    even = (xx + yy) % 2 == 0
    odd = ~even

    kind = det_kind.upper()
    if kind == "ZN":
        nx2, ny2 = nx // 2, ny // 2
        regions = [
            (xx < nx2) & (yy < ny2),
            (xx >= nx2) & (yy < ny2),
            (xx < nx2) & (yy >= ny2),
            (xx >= nx2) & (yy >= ny2),
        ]
    elif kind == "ZP":
        stripe = (4 * xx) // nx
        regions = [stripe == i for i in range(4)]
    else:
        raise ValueError("det_kind must be 'ZN' or 'ZP'.")

    return [even] + [odd & region for region in regions]


def _compute_5_channels_from_images_o2(images: np.ndarray, det_kind: str) -> np.ndarray:
    """Return (n, 5) photon counts ordered as [Common(control), Ch1, Ch2, Ch3, Ch4]."""
    _, nx, ny = images.shape
    masks = _channel_masks(nx, ny, det_kind)
    return np.stack([images[:, mask].sum(axis=1) for mask in masks], axis=1)


def _resolve_existing_file(folder: Path, candidate_names: Iterable[str], required: bool = True) -> Optional[Path]:
    for name in candidate_names:
        path = folder / name
        if path.exists():
            return path

    if required:
        tried = ", ".join(str(folder / name) for name in candidate_names)
        raise FileNotFoundError(f"None of the expected files exist: {tried}")
    return None


def _read_csv_if_exists(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _infer_single_pdg(*dfs: Optional[pd.DataFrame]) -> int:
    pdgs: set[int] = set()
    for df in dfs:
        if df is None or df.empty or "Pdg" not in df.columns:
            continue
        values = pd.to_numeric(df["Pdg"], errors="coerce").dropna().astype(int).unique()
        pdgs.update(values.tolist())

    if not pdgs:
        raise ValueError("Cannot infer the PDG code: no rows found in the info files.")
    if len(pdgs) != 1:
        raise ValueError(f"Multiple PDG codes in the same folder: {sorted(pdgs)}")
    return int(next(iter(pdgs)))


def _choose_reference_side(
    pdg_value: int,
    df_n: Optional[pd.DataFrame] = None,
    df_p: Optional[pd.DataFrame] = None,
) -> str:
    if int(pdg_value) == 2112:
        return "N"
    if int(pdg_value) == 2212:
        return "P"

    if df_n is not None and df_p is not None and not df_n.empty and not df_p.empty:
        if "PhotonSum" in df_n.columns and "PhotonSum" in df_p.columns:
            n_sum = pd.to_numeric(df_n["PhotonSum"], errors="coerce").fillna(0).sum()
            p_sum = pd.to_numeric(df_p["PhotonSum"], errors="coerce").fillna(0).sum()
            return "N" if n_sum >= p_sum else "P"

    return "N"


def _check_duplicate_rows(df_n: pd.DataFrame, df_p: pd.DataFrame, atol: float = 1e-12) -> None:
    """The N and P rows must describe the same primary, so they may only differ in
    the detector-specific quantities.
    """
    if len(df_n) != len(df_p):
        raise AssertionError(f"Different N/P row counts: len(N)={len(df_n)} len(P)={len(df_p)}")

    for col in DUPLICATE_COLS:
        a = df_n[col].to_numpy()
        b = df_p[col].to_numpy()
        if np.issubdtype(a.dtype, np.number):
            mismatched = ~np.isclose(a, b, rtol=0.0, atol=atol)
        else:
            mismatched = a != b

        if mismatched.any():
            bad = int(np.argmax(mismatched))
            raise AssertionError(
                f"N/P rows are not duplicates in column '{col}' at row {bad}: N={a[bad]} P={b[bad]}"
            )


def _split_detector_views(df: pd.DataFrame, pdg_value: int, source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in INFO_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"File {source_name} is missing columns: {missing}")

    selected = df[df["Pdg"] == pdg_value]
    df_n = selected[selected["Detector"] == "N"].reset_index(drop=True)
    df_p = selected[selected["Detector"] == "P"].reset_index(drop=True)

    if len(df_n) != len(df_p):
        raise AssertionError(
            f"File {source_name} has a different number of N/P rows: len(N)={len(df_n)} len(P)={len(df_p)}"
        )

    if len(df_n) > 0:
        _check_duplicate_rows(df_n, df_p)
    return df_n, df_p


def _empty_output_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLS)


def _assemble_output(
    df_base: pd.DataFrame,
    zn_values: Sequence[np.ndarray],
    zp_values: Sequence[np.ndarray],
    check: int,
) -> pd.DataFrame:
    data = {"Pdg": df_base["Pdg"].to_numpy(dtype=int)}
    for col in PRIMARY_COLS:
        data[col] = df_base[col].to_numpy()
    for prefix, values in (("ZN", zn_values), ("ZP", zp_values)):
        for col, array in zip(RESPONSE_COLS, values):
            data[f"{prefix}_{col}"] = array
    data["Check"] = np.full(len(df_base), check, dtype=int)
    return pd.DataFrame(data, columns=OUTPUT_COLS)


def _warn_on_channel_sum_mismatch(label: str, reconstructed: np.ndarray, df: pd.DataFrame) -> None:
    photon_sum = pd.to_numeric(df["PhotonSum"], errors="coerce").to_numpy()
    bad = np.flatnonzero(reconstructed != photon_sum)
    if bad.size > 0:
        first = bad[0]
        print(
            f"[WARN][{label}] {bad.size}/{len(df)} rows: channel sum != PhotonSum. "
            f"Example idx={first}: sum={reconstructed[first]} PhotonSum={photon_sum[first]}"
        )


def _time_mask(df_base: pd.DataFrame, time_min: Optional[float], time_max: Optional[float]) -> Optional[np.ndarray]:
    if time_min is None and time_max is None:
        return None

    t = pd.to_numeric(df_base["Time"], errors="coerce").to_numpy()
    mask = np.ones(t.shape, dtype=bool)
    if time_min is not None:
        mask &= t >= float(time_min)
    if time_max is not None:
        mask &= t <= float(time_max)
    return mask


def _build_non_zero_output(
    df_non_zero: Optional[pd.DataFrame],
    neutron_image_file: Path,
    proton_image_file: Path,
    pdg_value: int,
    reference_side: str,
    neutron_image_shape: Optional[Tuple[int, int]] = None,
    proton_image_shape: Optional[Tuple[int, int]] = None,
    sanity_check_sum: bool = True,
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
) -> pd.DataFrame:
    if df_non_zero is None or df_non_zero.empty:
        return _empty_output_dataframe()

    df_n, df_p = _split_detector_views(df_non_zero, pdg_value, "non_zero_examples.txt")
    if len(df_n) == 0:
        return _empty_output_dataframe()

    zn_images = _load_images_flexible(str(neutron_image_file), neutron_image_shape, n_images=len(df_n))
    zp_images = _load_images_flexible(str(proton_image_file), proton_image_shape, n_images=len(df_p))

    if len(df_n) != zn_images.shape[0]:
        raise AssertionError(f"ZN mismatch: {len(df_n)} N rows vs {zn_images.shape[0]} images")
    if len(df_p) != zp_images.shape[0]:
        raise AssertionError(f"ZP mismatch: {len(df_p)} P rows vs {zp_images.shape[0]} images")

    df_base = df_n if reference_side == "N" else df_p

    mask = _time_mask(df_base, time_min, time_max)
    if mask is not None:
        df_n = df_n.loc[mask].reset_index(drop=True)
        df_p = df_p.loc[mask].reset_index(drop=True)
        df_base = df_n if reference_side == "N" else df_p
        zn_images = zn_images[mask]
        zp_images = zp_images[mask]

        if len(df_base) == 0:
            return _empty_output_dataframe()

    zn_channels = _compute_5_channels_from_images_o2(zn_images, det_kind="ZN")
    zp_channels = _compute_5_channels_from_images_o2(zp_images, det_kind="ZP")

    if sanity_check_sum:
        _warn_on_channel_sum_mismatch("ZN", zn_channels.sum(axis=1), df_n)
        _warn_on_channel_sum_mismatch("ZP", zp_channels.sum(axis=1), df_p)

    zn_values = [df_n["PhotonSum"].to_numpy(), df_n["Time"].to_numpy()] + list(zn_channels.T)
    zp_values = [df_p["PhotonSum"].to_numpy(), df_p["Time"].to_numpy()] + list(zp_channels.T)
    return _assemble_output(df_base, zn_values, zp_values, check=1)


def _build_zero_output(
    df_zero: Optional[pd.DataFrame],
    pdg_value: int,
    reference_side: str,
) -> pd.DataFrame:
    if df_zero is None or df_zero.empty:
        return _empty_output_dataframe()

    df_n, df_p = _split_detector_views(df_zero, pdg_value, "zero_examples.txt")
    if len(df_n) == 0:
        return _empty_output_dataframe()

    df_base = df_n if reference_side == "N" else df_p
    n_rows = len(df_base)
    zeros_i = np.zeros(n_rows, dtype=int)
    zeros_f = np.zeros(n_rows, dtype=float)

    values = [zeros_i, zeros_f] + [zeros_i] * 5
    return _assemble_output(df_base, values, values, check=0)


def build_and_save_merged_parquet_from_folder(
    folder_path: str,
    parquet_name: str = "merged_dataset.parquet",
    neutron_image_shape: Optional[Tuple[int, int]] = None,
    proton_image_shape: Optional[Tuple[int, int]] = None,
    engine: str = "pyarrow",
    sanity_check_sum: bool = True,
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
) -> pd.DataFrame:
    """Build one parquet from a folder holding non_zero_examples.txt, zero_examples.txt,
    neutron_image.txt (or neutro_image.txt) and proton_image.txt.

    The output has one row per primary, Pdg first and Check last (1 = non-zero example,
    0 = zero example). The time_min/time_max cut applies to the non-zero examples only;
    the zero examples are always appended with all response columns set to zero.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Invalid folder: {folder}")

    non_zero_info_file = _resolve_existing_file(folder, ["non_zero_examples.txt"], required=False)
    zero_info_file = _resolve_existing_file(folder, ["zero_examples.txt"], required=False)

    if non_zero_info_file is None and zero_info_file is None:
        raise FileNotFoundError(f"Folder {folder} contains neither non_zero_examples.txt nor zero_examples.txt")

    df_non_zero = _read_csv_if_exists(non_zero_info_file)
    df_zero = _read_csv_if_exists(zero_info_file)

    pdg_value = _infer_single_pdg(df_non_zero, df_zero)

    probe_n = probe_p = None
    if df_non_zero is not None and not df_non_zero.empty:
        probe_n, probe_p = _split_detector_views(df_non_zero, pdg_value, "non_zero_examples.txt")
    elif df_zero is not None and not df_zero.empty:
        probe_n, probe_p = _split_detector_views(df_zero, pdg_value, "zero_examples.txt")

    reference_side = _choose_reference_side(pdg_value, probe_n, probe_p)

    non_zero_out = _empty_output_dataframe()
    if df_non_zero is not None and not df_non_zero.empty:
        non_zero_out = _build_non_zero_output(
            df_non_zero=df_non_zero,
            neutron_image_file=_resolve_existing_file(folder, ["neutron_image.txt", "neutro_image.txt"]),
            proton_image_file=_resolve_existing_file(folder, ["proton_image.txt"]),
            pdg_value=pdg_value,
            reference_side=reference_side,
            neutron_image_shape=neutron_image_shape,
            proton_image_shape=proton_image_shape,
            sanity_check_sum=sanity_check_sum,
            time_min=time_min,
            time_max=time_max,
        )

    zero_out = _build_zero_output(df_zero=df_zero, pdg_value=pdg_value, reference_side=reference_side)

    parts = [df for df in (non_zero_out, zero_out) if not df.empty]
    out = pd.concat(parts, ignore_index=True) if parts else _empty_output_dataframe()

    output_path = folder / parquet_name
    out.to_parquet(output_path, engine=engine, index=False)

    n_non_zero = int((out["Check"] == 1).sum()) if not out.empty else 0
    n_zero = int((out["Check"] == 0).sum()) if not out.empty else 0
    print(
        f"Parquet saved: {output_path} ({len(out)} rows) | "
        f"PDG={pdg_value} | side_ref={reference_side} | "
        f"non_zero={n_non_zero} | zero={n_zero} | "
        f"non-zero Time in [{time_min},{time_max}]"
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build a merged parquet from one or more folders holding the standard Extract.C files."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Folders containing non_zero_examples.txt, zero_examples.txt, neutron_image.txt and proton_image.txt",
    )
    parser.add_argument(
        "--output-name",
        default="merged_dataset.parquet",
        help="Name of the parquet written inside each folder",
    )
    parser.add_argument("--time-min", type=float, default=None, help="Lower cut on the Time of the non-zero examples")
    parser.add_argument("--time-max", type=float, default=None, help="Upper cut on the Time of the non-zero examples")
    parser.add_argument("--engine", default="pyarrow", help="Engine for pandas.to_parquet (default: pyarrow)")
    args = parser.parse_args()

    for folder in args.folders:
        build_and_save_merged_parquet_from_folder(
            folder_path=folder,
            parquet_name=args.output_name,
            engine=args.engine,
            time_min=args.time_min,
            time_max=args.time_max,
        )


if __name__ == "__main__":
    main()