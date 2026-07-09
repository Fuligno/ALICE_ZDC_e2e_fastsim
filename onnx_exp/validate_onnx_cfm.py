#!/usr/bin/env python3
"""Validate the end-to-end CFM ONNX model with numpy + onnxruntime (no torch).

The graph already contains standardization, the ODE loop and de-standardization,
so a single Run maps (cond_phys, x0[, u]) to x_phys.
"""
import argparse

import numpy as np
import onnxruntime as ort


def read_matrix(tokens, name):
    key = next(tokens)
    if key != name:
        raise ValueError(f"Expected {name}, found {key}")
    rows, cols = int(next(tokens)), int(next(tokens))
    values = [float(next(tokens)) for _ in range(rows * cols)]
    return np.array(values, dtype=np.float32).reshape(rows, cols)


def read_scalar(tokens, name):
    key = next(tokens)
    if key != name:
        raise ValueError(f"Expected {name}, found {key}")
    return int(next(tokens))


def compare(x_got, x_expected):
    diff = np.abs(x_got.astype(np.float64) - x_expected.astype(np.float64))
    max_abs = float(np.max(diff))
    values = x_got.astype(np.float64)

    if np.all(values == np.floor(values)):
        # Discrete output: count differing cells rather than only the max deviation.
        n_bad = int(np.sum(diff > 0))
        print(f"[GOLDEN] B={x_got.shape[0]} (discrete output) "
              f"differing cells: {n_bad}/{diff.size}  max|diff|={max_abs:g}")
        ok = (n_bad == 0) or (max_abs <= 1 and n_bad / diff.size < 1e-3)
    else:
        max_rel = float(np.max(diff / (np.abs(x_expected) + 1e-9)))
        print(f"[GOLDEN] B={x_got.shape[0]}  max|diff|={max_abs:.3e}  max rel={max_rel:.3e}")
        ok = max_abs < 1e-3

    print("[GOLDEN] OK" if ok else "[GOLDEN] MISMATCH")
    return ok


def run_golden(sess, path, internal_noise, needs_u):
    with open(path) as f:
        tokens = iter(f.read().split())

    read_scalar(tokens, "n_steps")
    read_scalar(tokens, "internal_noise")
    stochastic = bool(read_scalar(tokens, "stochastic"))

    cond = read_matrix(tokens, "cond_phys")
    x0 = read_matrix(tokens, "x0")
    u = read_matrix(tokens, "u") if stochastic else None
    x_expected = read_matrix(tokens, "x_phys")

    if internal_noise:
        print("[GOLDEN] model with internal noise: a pointwise comparison is not valid.")
        return
    if needs_u != stochastic:
        print(f"[GOLDEN] INCONSISTENT: the model {'requires' if needs_u else 'does not require'} u, "
              f"the golden {'contains it' if stochastic else 'does not'}. Regenerate the golden.")
        return

    feed = {"cond_phys": cond, "x0": x0}
    if needs_u:
        feed["u"] = u
    compare(sess.run(["x_phys"], feed)[0], x_expected)


def run_generation(sess, path, seed, internal_noise, needs_u):
    with open(path) as f:
        cond = read_matrix(iter(f.read().split()), "cond_phys")

    feed = {"cond_phys": cond}
    if not internal_noise:
        x_dim = next(i.shape[1] for i in sess.get_inputs() if i.name == "x0")
        if not isinstance(x_dim, int):
            raise ValueError("The x0 input has a dynamic feature dimension; cannot infer x_dim.")
        rng = np.random.default_rng(seed)
        feed["x0"] = rng.standard_normal((cond.shape[0], x_dim)).astype(np.float32)
        if needs_u:
            feed["u"] = rng.random((cond.shape[0], x_dim)).astype(np.float32)

    x_phys = sess.run(["x_phys"], feed)[0]
    print(f"[GEN] {cond.shape[0]} events")
    for n in range(min(5, x_phys.shape[0])):
        print(f"  ev{n}:", np.array2string(x_phys[n], precision=4, suppress_small=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="onnx_export_cfm/cfm_zdc_e2e.onnx")
    ap.add_argument("--golden", default="")
    ap.add_argument("--cond", default="")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    print("[INFO] ONNX inputs:", [(i.name, i.shape) for i in sess.get_inputs()])

    internal_noise = "x0" not in in_names
    needs_u = "u" in in_names

    if args.golden:
        run_golden(sess, args.golden, internal_noise, needs_u)
    elif args.cond:
        run_generation(sess, args.cond, args.seed, internal_noise, needs_u)
    else:
        print("Specify either --golden or --cond.")


if __name__ == "__main__":
    main()