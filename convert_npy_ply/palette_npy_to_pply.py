#!/usr/bin/env python3
"""
palette_npy_to_pply.py

Convert a finetuned palette-based .npy checkpoint into:
  1) a geometry-only binary little-endian .pply  (x, y, z, scale_*, opacity, rot_*)
  2) a sidecar binary file '.gswp' with palette + weight-SH data

The .pply is NOT a standard Inria .ply -- it deliberately omits all f_dc/f_rest
fields. Stock Brush will fail to load it. It is meant to be paired with the
.gswp sidecar and consumed by an extended Brush loader.

Sidecar binary format ('.gswp', all little-endian, packed, no compression):

  Header (32 bytes total):
    magic        u8[4]   = b"GSWP"
    version      u32     = 1
    N            u32                  # number of gaussians (matches .pply)
    splat_dim    u32                  # palette size (K_full = K+2)
    num_sh       u32                  # SH coeffs per channel (e.g. 16 for degree 3)
    low_rank     u32                  # 0 = dense weights, 1 = low-rank factors
    P            u32                  # if low_rank=1: A's inner dim, else 0
    Q            u32                  # if low_rank=1: B's inner dim, else 0
                                      # constraint: P*Q == splat_dim*(num_sh-1)

  Payload:
    palette       f32[splat_dim, 3]   # K_full x 3 RGB, row-major
    low_shs_w     f32[N, splat_dim]   # DC band weights, row-major

    if low_rank == 0:
      high_shs    f32[N, splat_dim*(num_sh-1)]  # rest-band weights, row-major
                                                # memory: per gaussian, band-major:
                                                # [b1_p0, b1_p1, ..., b1_p{K-1},
                                                #  b2_p0, ..., b{num_sh-1}_p{K-1}]
    if low_rank == 1:
      high_shs_A  f32[N, P]
      high_shs_B  f32[N, Q]
                  # reconstruction: outer product per row, then reshape to (N, P*Q)
                  # (N, P, 1) * (N, 1, Q) -> (N, P, Q) -> (N, P*Q)

Conventions:
  - Weights are stored EXACTLY as in the .npy. No shifts, no biases.
  - .pply fields exactly mirror the .npy values:
      gs['scale']  -> stored as log(scale)            (linear -> log)
      gs['alpha']  -> stored as logit(alpha)          (sigmoid -> raw)
      gs['rot']    -> stored as-is                    (Brush re-normalizes on load)
      gs['pw']     -> stored as-is

Usage:
    python palette_npy_to_pply.py truck_palette.npy
        # creates a folder ./truck_palette/ next to the .npy and writes:
        #   ./truck_palette/truck_palette.pply
        #   ./truck_palette/truck_palette.gswp

    python palette_npy_to_pply.py truck_palette.npy /tmp/scene.pply
        # explicit output path: writes /tmp/scene.pply and /tmp/scene.gswp
        # (no folder is created in this mode)

    python palette_npy_to_pply.py truck_palette.npy /tmp/scene.pply --sidecar elsewhere.gswp
        # also override the sidecar location

    python palette_npy_to_pply.py --inspect input.npy
        # print what's in the .npy and what would be written; no files
"""

import argparse
import os
import sys
import struct
import numpy as np


GSWP_MAGIC   = b"GSWP"
GSWP_VERSION = 1


def _logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _load_palette_npy(npy_path):
    """
    Load a finetuned palette .npy. The training script saves it as a Python dict
    via `np.save(fn, save_dict)`, so allow_pickle=True is required.
    Returns the dict, with all arrays cast to float32/contiguous as appropriate.
    """
    obj = np.load(npy_path, allow_pickle=True)
    # When saved as `np.save(fn, dict)`, np.load returns a 0-d array of dtype
    # object whose .item() is the dict.
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        gs = obj.item()
    else:
        raise ValueError(
            f"{npy_path} doesn't look like a palette .npy "
            f"(expected pickled dict, got {type(obj)})"
        )
    if not isinstance(gs, dict):
        raise ValueError(f"{npy_path} loaded as {type(gs)}, expected dict")
    return gs


def _validate(gs):
    """Sanity-check the loaded dict; raise on anything unexpected."""
    required = ["pw", "rot", "scale", "alpha", "palette"]
    for k in required:
        if k not in gs:
            raise KeyError(f".npy is missing required key: {k!r}")

    low_rank = bool(gs.get("low_rank", False))

    palette = np.asarray(gs["palette"], dtype=np.float32)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette has shape {palette.shape}; expected (splat_dim, 3)")
    splat_dim = palette.shape[0]

    pw = np.asarray(gs["pw"]); N = pw.shape[0]
    if pw.shape != (N, 3):
        raise ValueError(f"pw shape {pw.shape} != (N, 3)")
    if np.asarray(gs["rot"]).shape != (N, 4):
        raise ValueError(f"rot shape {np.asarray(gs['rot']).shape} != (N, 4)")
    if np.asarray(gs["scale"]).shape != (N, 3):
        raise ValueError(f"scale shape {np.asarray(gs['scale']).shape} != (N, 3)")
    alpha = np.asarray(gs["alpha"]).squeeze()
    if alpha.shape != (N,):
        raise ValueError(f"alpha shape {alpha.shape} != ({N},) after squeeze")

    if low_rank:
        for k in ("low_shs_w", "high_shs_A", "high_shs_B", "reshape_P", "reshape_Q"):
            if k not in gs:
                raise KeyError(f"low_rank=True but .npy missing key: {k!r}")
        low_shs_w = np.asarray(gs["low_shs_w"], dtype=np.float32)
        A = np.asarray(gs["high_shs_A"], dtype=np.float32)
        B = np.asarray(gs["high_shs_B"], dtype=np.float32)
        P = int(gs["reshape_P"])
        Q = int(gs["reshape_Q"])

        if low_shs_w.shape != (N, splat_dim):
            raise ValueError(
                f"low_shs_w shape {low_shs_w.shape} != (N={N}, splat_dim={splat_dim})"
            )
        if A.shape != (N, P):
            raise ValueError(f"high_shs_A shape {A.shape} != (N={N}, P={P})")
        if B.shape != (N, Q):
            raise ValueError(f"high_shs_B shape {B.shape} != (N={N}, Q={Q})")

        # Constraint: P*Q must be the rest-band size (splat_dim * (num_sh - 1)).
        # We don't know num_sh from low_rank alone, so back it out:
        if (P * Q) % splat_dim != 0:
            raise ValueError(
                f"P*Q={P*Q} not divisible by splat_dim={splat_dim}; "
                f"can't recover num_sh."
            )
        num_sh = (P * Q) // splat_dim + 1   # +1 because we excluded the DC band

    else:
        if "shs_w" not in gs:
            raise KeyError("low_rank=False but .npy missing 'shs_w'")
        shs_w = np.asarray(gs["shs_w"], dtype=np.float32)
        if shs_w.ndim != 2 or shs_w.shape[0] != N:
            raise ValueError(f"shs_w shape {shs_w.shape}; expected (N={N}, splat_dim*num_sh)")
        if shs_w.shape[1] % splat_dim != 0:
            raise ValueError(
                f"shs_w last dim {shs_w.shape[1]} not divisible by splat_dim={splat_dim}"
            )
        num_sh = shs_w.shape[1] // splat_dim

    return {
        "N": N,
        "splat_dim": splat_dim,
        "num_sh": num_sh,
        "low_rank": low_rank,
    }


def _write_pply_geometry(pply_path, gs, N):
    """Write the geometry-only .pply (no SH fields)."""
    pw    = np.ascontiguousarray(gs['pw'],    dtype=np.float32)
    rot   = np.ascontiguousarray(gs['rot'],   dtype=np.float32)
    scale = np.ascontiguousarray(gs['scale'], dtype=np.float32)   # LINEAR
    alpha = np.ascontiguousarray(np.asarray(gs['alpha']).squeeze(), dtype=np.float32)  # SIGMOID

    log_scale   = np.log(np.clip(scale, 1e-20, None)).astype(np.float32)
    raw_opacity = _logit(alpha).astype(np.float32)

    fields = (
        [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        + [('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4')]
        + [('opacity', '<f4')]
        + [('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4')]
    )
    vertex = np.empty(N, dtype=fields)
    vertex['x'], vertex['y'], vertex['z'] = pw[:, 0], pw[:, 1], pw[:, 2]
    vertex['scale_0'] = log_scale[:, 0]
    vertex['scale_1'] = log_scale[:, 1]
    vertex['scale_2'] = log_scale[:, 2]
    vertex['opacity'] = raw_opacity
    vertex['rot_0']   = rot[:, 0]
    vertex['rot_1']   = rot[:, 1]
    vertex['rot_2']   = rot[:, 2]
    vertex['rot_3']   = rot[:, 3]

    header_lines = [
        'ply',
        'format binary_little_endian 1.0',
        'comment palette-based 3DGS - geometry only; pair with .gswp sidecar',
        f'element vertex {N}',
    ]
    header_lines += [f'property float {nm}' for (nm, _) in fields]
    header_lines += ['end_header', '']
    header = '\n'.join(header_lines).encode('ascii')

    with open(pply_path, 'wb') as f:
        f.write(header)
        f.write(vertex.tobytes())


def _write_sidecar(sidecar_path, gs, info):
    """Write the .gswp sidecar binary."""
    N         = info["N"]
    splat_dim = info["splat_dim"]
    num_sh    = info["num_sh"]
    low_rank  = info["low_rank"]

    palette = np.ascontiguousarray(np.asarray(gs["palette"]), dtype=np.float32)

    if low_rank:
        low_shs_w = np.ascontiguousarray(np.asarray(gs["low_shs_w"]), dtype=np.float32)
        A         = np.ascontiguousarray(np.asarray(gs["high_shs_A"]), dtype=np.float32)
        B         = np.ascontiguousarray(np.asarray(gs["high_shs_B"]), dtype=np.float32)
        P         = int(gs["reshape_P"])
        Q         = int(gs["reshape_Q"])
    else:
        shs_w     = np.ascontiguousarray(np.asarray(gs["shs_w"]), dtype=np.float32)
        # Split into DC band (low_shs_w) and rest band (high_shs).
        # shs_w layout: (N, num_sh * splat_dim), band-major
        #   first splat_dim entries are the DC band weights for K palette colors,
        #   next splat_dim entries are band-1 weights, etc.
        shs_w_rs  = shs_w.reshape(N, num_sh, splat_dim)
        low_shs_w = np.ascontiguousarray(shs_w_rs[:, 0, :], dtype=np.float32)            # (N, splat_dim)
        high_shs  = np.ascontiguousarray(
            shs_w_rs[:, 1:, :].reshape(N, (num_sh - 1) * splat_dim), dtype=np.float32
        )   # (N, splat_dim*(num_sh-1)), band-major
        P = 0
        Q = 0

    # Header: 8 u32s (32 bytes).
    header = struct.pack(
        "<4sIIIIIII",
        GSWP_MAGIC,
        GSWP_VERSION,
        N,
        splat_dim,
        num_sh,
        1 if low_rank else 0,
        P,
        Q,
    )
    # Pad to 32 bytes (struct above is exactly 4 + 7*4 = 32 already).
    assert len(header) == 32, f"header length {len(header)} != 32"

    with open(sidecar_path, "wb") as f:
        f.write(header)
        f.write(palette.tobytes())
        f.write(low_shs_w.tobytes())
        if low_rank:
            f.write(A.tobytes())
            f.write(B.tobytes())
        else:
            f.write(high_shs.tobytes())


def convert(npy_path, pply_path=None, sidecar_path=None, verbose=True):
    # Default output: a NEW folder named after the .npy basename, containing
    # <basename>.pply and <basename>.gswp inside it.
    #
    # Example: convert("truck_palette.npy")
    #   -> creates ./truck_palette/
    #      writes  ./truck_palette/truck_palette.pply
    #      writes  ./truck_palette/truck_palette.gswp
    #
    # If `pply_path` is given explicitly, we honor it as-is (no folder created)
    # and place the sidecar next to it.
    if pply_path is None:
        npy_dir   = os.path.dirname(os.path.abspath(npy_path))
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        out_dir   = os.path.join(npy_dir, base_name)
        os.makedirs(out_dir, exist_ok=True)
        pply_path = os.path.join(out_dir, base_name + ".pply")
        if sidecar_path is None:
            sidecar_path = os.path.join(out_dir, base_name + ".gswp")
    else:
        # Explicit output path -- ensure parent dir exists, leave layout alone.
        os.makedirs(os.path.dirname(os.path.abspath(pply_path)) or ".", exist_ok=True)
        if sidecar_path is None:
            base, _ = os.path.splitext(pply_path)
            sidecar_path = base + ".gswp"

    gs = _load_palette_npy(npy_path)
    info = _validate(gs)

    if verbose:
        kind = "low-rank" if info["low_rank"] else "dense"
        extra = ""
        if info["low_rank"]:
            P = int(gs["reshape_P"]); Q = int(gs["reshape_Q"])
            extra = f"  (P={P}, Q={Q}, P*Q={P*Q} = splat_dim*(num_sh-1)={info['splat_dim']*(info['num_sh']-1)})"
        print(f"[palette_npy_to_pply] loaded {npy_path}")
        print(f"  N={info['N']:,}  splat_dim={info['splat_dim']}  "
              f"num_sh={info['num_sh']}  weights={kind}{extra}")
        print(f"  palette (first 4 rows):\n    "
              + "\n    ".join(str(np.asarray(gs['palette'])[i].round(3))
                              for i in range(min(4, info['splat_dim']))))

    _write_pply_geometry(pply_path, gs, info["N"])
    _write_sidecar(sidecar_path, gs, info)

    if verbose:
        pply_sz = os.path.getsize(pply_path)
        sc_sz   = os.path.getsize(sidecar_path)
        print(f"  wrote {pply_path}      ({pply_sz/1e6:.2f} MB)")
        print(f"  wrote {sidecar_path}  ({sc_sz/1e6:.2f} MB)")


def inspect(npy_path):
    gs = _load_palette_npy(npy_path)
    info = _validate(gs)
    print(f"[inspect] {npy_path}")
    print(f"  keys: {sorted(gs.keys())}")
    print(f"  N={info['N']:,}  splat_dim={info['splat_dim']}  num_sh={info['num_sh']}  "
          f"low_rank={info['low_rank']}")
    if info["low_rank"]:
        P = int(gs["reshape_P"]); Q = int(gs["reshape_Q"])
        print(f"  reshape_P={P}  reshape_Q={Q}  P*Q={P*Q}")
        # Compression ratio vs dense.
        dense_floats  = info['N'] * info['splat_dim'] * (info['num_sh'] - 1)
        factor_floats = info['N'] * (P + Q)
        print(f"  dense rest-band floats: {dense_floats:,}")
        print(f"  factor floats:          {factor_floats:,}  "
              f"(compression {dense_floats/factor_floats:.1f}x)")
    print(f"  palette:\n{np.asarray(gs['palette']).round(4)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .npy path (palette-based)")
    ap.add_argument("output", nargs="?", default=None,
                    help="output .pply path (default: same basename as input with .pply)")
    ap.add_argument("--sidecar", default=None,
                    help="explicit sidecar .gswp path (default: <output>.gswp)")
    ap.add_argument("--inspect", action="store_true",
                    help="print what's in the .npy; do not write files")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.input)
        return 0

    convert(args.input, args.output, sidecar_path=args.sidecar, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())