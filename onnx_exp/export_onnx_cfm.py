#!/usr/bin/env python3
"""End-to-end ONNX export of the CFM model (ZDC fast simulation).

The exported graph contains the whole generation pipeline:

    inputs:  cond_phys [B, cond_dim], x0 [B, x_dim]  (and u [B, x_dim] if stochastic rounding)
    output:  x_phys    [B, x_dim]                    (physical units)

Standardization, the unrolled Euler loop, de-standardization and the optional
post-processing all live inside the graph, so the consumer needs a single Run.
"""
import os
import json
import argparse

import numpy as np
import torch

from model import load_model_and_scalers


ROUND_MODES = ("none", "half_even", "half_up", "floor", "stochastic")


class EndToEndCFM(torch.nn.Module):
    """Full pipeline (cond_phys, x0[, u]) -> x_phys with an unrolled ODE loop."""

    def __init__(self, core, cond_mean, cond_std, x_mean, x_std,
                 n_steps, x_dim, cond_eps=1e-12, internal_noise=False,
                 clamp_min0=False, round_mode="none", cast_int32=False):
        super().__init__()
        if round_mode not in ROUND_MODES:
            raise ValueError(f"Unknown round_mode: {round_mode}. Use one of {ROUND_MODES}.")

        self.core = core
        self.register_buffer("cond_mean", cond_mean.view(1, -1))
        self.register_buffer("cond_std", cond_std.view(1, -1))
        self.register_buffer("x_mean", x_mean.view(1, -1))
        self.register_buffer("x_std", x_std.view(1, -1))
        self.n_steps = int(n_steps)
        self.x_dim = int(x_dim)
        self.cond_eps = float(cond_eps)
        self.internal_noise = bool(internal_noise)
        self.clamp_min0 = bool(clamp_min0)
        self.round_mode = str(round_mode)
        self.cast_int32 = bool(cast_int32)

    @property
    def needs_u(self) -> bool:
        """True if the graph takes the uniform noise tensor u ~ U(0,1) as an input."""
        return self.round_mode == "stochastic" and not self.internal_noise

    def _integrate(self, cond_std_in, x):
        dt = 1.0 / float(self.n_steps)
        for k in range(self.n_steps):
            # ones_like keeps the batch dimension dynamic in the exported graph
            t = torch.ones_like(x[:, :1]) * float((k + 0.5) * dt)
            x = x + dt * self.core(x, t, cond_std_in)
        return x

    def _postprocess(self, x_phys, u):
        if self.clamp_min0:
            x_phys = torch.clamp(x_phys, min=0.0)

        if self.round_mode == "half_even":
            x_phys = torch.round(x_phys)
        elif self.round_mode == "half_up":
            x_phys = torch.floor(x_phys + 0.5)
        elif self.round_mode == "floor":
            x_phys = torch.floor(x_phys)
        elif self.round_mode == "stochastic":
            # E[floor(x + u)] = x, so the mean photon count is preserved.
            # Applied after the clamp so that the output stays non-negative.
            uu = torch.rand_like(x_phys) if self.internal_noise else u
            x_phys = torch.floor(x_phys + uu)

        if self.cast_int32:
            x_phys = x_phys.to(torch.int32)
        return x_phys

    def forward(self, cond_phys, x0=None, u=None):
        if not self.internal_noise and x0 is None:
            raise ValueError("x0 is required unless internal_noise is enabled.")
        if self.needs_u and u is None:
            raise ValueError("u is required when round_mode='stochastic' and internal_noise is off.")

        cond_std_in = (cond_phys - self.cond_mean) / (self.cond_std + self.cond_eps)

        if self.internal_noise:
            x = torch.randn_like(cond_phys.new_zeros((cond_phys.shape[0], self.x_dim)))
        else:
            x = x0

        x = self._integrate(cond_std_in, x)
        return self._postprocess(x * self.x_std + self.x_mean, u)


def _write_mat(fh, name, matrix):
    matrix = np.asarray(matrix)
    fh.write(f"{name} {matrix.shape[0]} {matrix.shape[1]}\n")
    for row in matrix:
        fh.write(" ".join(f"{v:.10g}" for v in row) + "\n")


def _warn_on_postproc_flags(args):
    if args.cast_int32 and args.round_mode == "none":
        print("[WARN] --cast_int32 without --round_mode: the Cast truncates towards zero "
              "(1.9 -> 1). You probably also want --round_mode half_even.")
    if args.round_mode != "none" and not args.clamp_min0:
        print("[WARN] --round_mode without --clamp_min0: negative counts survive (e.g. -1 photons).")
    if args.round_mode == "stochastic" and not args.clamp_min0:
        print("[WARN] stochastic without clamp: floor(x + u) on x < 0 yields negative values. "
              "Use --clamp_min0.")


def _export_signature(internal_noise, needs_u):
    """Return the ONNX input names and dynamic-axes map for the chosen mode."""
    if internal_noise:
        names = ["cond_phys"]
    elif needs_u:
        names = ["cond_phys", "x0", "u"]
    else:
        names = ["cond_phys", "x0"]
    dynamic_axes = {name: {0: "batch"} for name in names + ["x_phys"]}
    return names, dynamic_axes


def _run_check(onnx_path, feed, x_expected, round_mode, cast_int32):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime is not installed: skipping the check.")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x_ort = sess.run(["x_phys"], feed)[0]
    diff = np.abs(x_ort.astype(np.float64) - x_expected)
    max_abs = float(np.max(diff))

    if round_mode != "none" or cast_int32:
        # With rounding the comparison is discrete: a 1e-7 float difference near a
        # boundary (x.4999999 vs x.5000001) becomes a difference of one count.
        n_tot = diff.size
        n_bad = int(np.sum(diff > 0))
        frac = n_bad / n_tot
        print("[CHECK] ORT vs PyTorch (discrete output)")
        print(f"        differing cells: {n_bad}/{n_tot} ({100 * frac:.3f}%)  max|diff|={max_abs:g}")
        if n_bad == 0:
            print("[CHECK] OK (exact match)")
        elif max_abs <= 1 and frac < 1e-3:
            print("[CHECK] OK (a few cells on a rounding boundary, |diff| <= 1)")
        else:
            print("[CHECK] MISMATCH: differences too large or too frequent")
    else:
        max_rel = float(np.max(diff / (np.abs(x_expected) + 1e-9)))
        print(f"[CHECK] ORT vs PyTorch  max|diff|={max_abs:.3e}  max rel={max_rel:.3e}")
        print("[CHECK] OK" if max_abs < 1e-3 else "[CHECK] MISMATCH")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run_dir", default="../model/CFM/runs/ENCnone_MLP128-128-128_ACTsilu_lr1e-3_bs256")
    ap.add_argument("--model_name", default="model_best.pt")
    ap.add_argument("--scalers_name", default="scalers.pt")
    ap.add_argument("--out_dir", default="onnx_export_cfm")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--n_golden", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--internal_noise", action="store_true",
                    help="Generate x0 inside the graph (cond_phys becomes the only input). Not recommended.")
    ap.add_argument("--clamp_min0", action="store_true",
                    help="Zero out negative predictions, after de-standardization.")
    ap.add_argument("--round_mode", default="none", choices=list(ROUND_MODES),
                    help="Integer rounding. 'half_even' = torch/numpy/ONNX Round; 'half_up' = C++ std::round; "
                         "'stochastic' = floor(x + u) with u ~ U(0,1) passed as an input (preserves the mean).")
    ap.add_argument("--cast_int32", action="store_true",
                    help="Emit x_phys as int32 instead of float32 (use together with --round_mode).")
    ap.add_argument("--check", action="store_true",
                    help="Compare the exported graph against PyTorch with onnxruntime.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, scalers = load_model_and_scalers(
        os.path.join(args.run_dir, args.model_name),
        os.path.join(args.run_dir, args.scalers_name),
        device="cpu",
    )
    model.eval()

    cfg = model.get_config()
    cond_dim = int(cfg["cond_dim"])
    x_dim = int(cfg["x_dim"])
    print(f"[INFO] cond_dim={cond_dim} x_dim={x_dim} n_steps={args.n_steps} "
          f"internal_noise={args.internal_noise}")

    cond_mean = scalers["cond_mean"].float()
    cond_std = scalers["cond_std"].float()
    x_mean = scalers["x_mean"].float()
    x_std = scalers["x_std"].float()
    cond_eps = 1e-12

    _warn_on_postproc_flags(args)

    e2e = EndToEndCFM(
        model, cond_mean, cond_std, x_mean, x_std,
        n_steps=args.n_steps, x_dim=x_dim, cond_eps=cond_eps,
        internal_noise=args.internal_noise, clamp_min0=args.clamp_min0,
        round_mode=args.round_mode, cast_int32=args.cast_int32,
    ).eval()
    needs_u = e2e.needs_u

    input_names, dynamic_axes = _export_signature(args.internal_noise, needs_u)

    dummy_batch = 4
    dummy = {
        "cond_phys": torch.randn(dummy_batch, cond_dim),
        "x0": torch.randn(dummy_batch, x_dim),
        "u": torch.rand(dummy_batch, x_dim),
    }
    args_tuple = tuple(dummy[name] for name in input_names)

    onnx_path = os.path.join(args.out_dir, "cfm_zdc_e2e.onnx")
    export_kwargs = dict(
        input_names=input_names,
        output_names=["x_phys"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(e2e, args_tuple, onnx_path, dynamo=False, **export_kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(e2e, args_tuple, onnx_path, **export_kwargs)
    print(f"[OK] end-to-end ONNX written to {onnx_path}")

    with open(os.path.join(args.out_dir, "config.json"), "w") as fh:
        json.dump({
            "type": "end_to_end_unrolled",
            "cond_dim": cond_dim,
            "x_dim": x_dim,
            "n_steps": args.n_steps,
            "cond_eps": cond_eps,
            "internal_noise": args.internal_noise,
            "postproc": {
                "clamp_min0": args.clamp_min0,
                "round_mode": args.round_mode,
                "cast_int32": args.cast_int32,
                "needs_u": needs_u,
                "note": "Applied inside the graph, after de-standardization; the consumer must not "
                        "reapply it. If needs_u is true, u ~ U(0,1) of shape [B, x_dim] must be "
                        "provided alongside x0.",
            },
            "output_dtype": "int32" if args.cast_int32 else "float32",
            "inputs": input_names,
            "output": "x_phys",
            "cond_cols": list(scalers["cond_cols"]),
            "x_cols": list(scalers["x_cols"]),
        }, fh, indent=2)

    # Golden reference: fixed inputs plus the expected PyTorch output.
    gen = torch.Generator().manual_seed(args.seed)
    z = torch.randn(args.n_golden, cond_dim, generator=gen)
    cond_phys = cond_mean.view(1, -1) + z * cond_std.view(1, -1)
    x0 = torch.randn(args.n_golden, x_dim, generator=gen)
    u = torch.rand(args.n_golden, x_dim, generator=gen)

    with torch.no_grad():
        if args.internal_noise:
            x_phys = e2e(cond_phys)
        elif needs_u:
            x_phys = e2e(cond_phys, x0, u)
        else:
            x_phys = e2e(cond_phys, x0)

    cond_np = cond_phys.numpy()
    x0_np = x0.numpy()
    u_np = u.numpy()
    xp_np = x_phys.numpy().astype(np.float64)

    golden_txt = os.path.join(args.out_dir, "golden.txt")
    with open(golden_txt, "w") as fh:
        fh.write(f"n_steps {args.n_steps}\n")
        fh.write(f"internal_noise {int(args.internal_noise)}\n")
        fh.write(f"stochastic {int(needs_u)}\n")
        _write_mat(fh, "cond_phys", cond_np)
        _write_mat(fh, "x0", x0_np)
        if needs_u:
            _write_mat(fh, "u", u_np)
        _write_mat(fh, "x_phys", xp_np)

    with open(os.path.join(args.out_dir, "golden.json"), "w") as fh:
        json.dump({
            "n_steps": args.n_steps,
            "internal_noise": args.internal_noise,
            "stochastic": needs_u,
            "cond_phys": cond_np.tolist(),
            "x0": x0_np.tolist(),
            "u": u_np.tolist() if needs_u else None,
            "x_phys": xp_np.tolist(),
        }, fh, indent=2)
    print(f"[OK] golden written to {golden_txt}")

    if args.check:
        if args.internal_noise:
            print("[CHECK] internal_noise=ON: ORT and PyTorch use different RNGs, "
                  "a pointwise comparison is meaningless. Skipping.")
            return
        feed = {"cond_phys": cond_np.astype(np.float32), "x0": x0_np.astype(np.float32)}
        if needs_u:
            feed["u"] = u_np.astype(np.float32)
        _run_check(onnx_path, feed, xp_np, args.round_mode, args.cast_int32)

        if args.clamp_min0 or args.round_mode != "none":
            print(f"[INFO] output: min={xp_np.min():.4g} max={xp_np.max():.4g} "
                  f"mean={xp_np.mean():.4g} | zero fraction={np.mean(xp_np == 0):.3f}")


if __name__ == "__main__":
    main()