#!/usr/bin/env python3
"""
npy_to_ply.py

Convert a .npy checkpoint saved by this codebase (via save_training_params in
gsplat/gau_io.py) into a binary little-endian .ply in the Inria/Brush format.

Usage:
    python npy_to_ply.py input.npy output.ply
    python npy_to_ply.py input.npy output.ply --dc-shift 0
    python npy_to_ply.py --sanity-check input.npy

DC SHIFT — THE ONLY THING THAT MATTERS TO GET RIGHT:
    This codebase's CUDA kernel (gsplatcu/kernel.cu :: sh2Color) renders color as:
        color = SH_C0 * sh[0]           (NO +0.5 bias)
    Brush's WGSL shader renders as:
        color = SH_C0 * sh[0] + 0.5     (+0.5 bias applied at render time)
    To make Brush render the same color as this codebase, we subtract
    0.5 / SH_C0 from the DC coefficients on export. That's the default.
    Pass --dc-shift 0 to skip the shift (e.g. if your npy already uses
    Inria convention, which is not the case here — it's just an escape hatch).
"""

import argparse
import sys
import numpy as np


SH_C0 = 0.28209479177387814           # 1 / (2 * sqrt(pi))
DEFAULT_DC_SHIFT = 0.5 / SH_C0        # ~1.7724539, see module docstring


# The .npy files produced by save_training_params are numpy recarrays with
# this dtype (see gsdata_type in gau_io.py).
def _infer_sh_dim(gs):
    """gs['sh'] has shape (N, sh_dim); return sh_dim and rest_per_channel."""
    sh_dim = gs['sh'].shape[1]
    if sh_dim < 3:
        raise ValueError(f"sh_dim={sh_dim} is too small; need at least 3 DC coeffs")
    rest_total = sh_dim - 3
    if rest_total % 3 != 0:
        raise ValueError(
            f"sh_dim - 3 = {rest_total} is not divisible by 3; can't split into R/G/B rest"
        )
    return sh_dim, rest_total // 3


def _logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def npy_to_ply(npy_path, ply_path, dc_shift=DEFAULT_DC_SHIFT, verbose=True):
    """
    Load an .npy checkpoint and write a Brush-compatible binary .ply.

    Assumes the .npy was produced by save_training_params in gau_io.py, meaning:
      - gs['scale'] is LINEAR (post-exp). We re-log for PLY storage.
      - gs['alpha'] is POST-SIGMOID in [0,1]. We logit it back for PLY storage.
      - gs['rot']   is a (possibly normalized) quaternion. Stored as-is; Brush re-normalizes.
      - gs['sh']    has shape (N, sh_dim), memory order: [DC_R, DC_G, DC_B,
                                                          c1_R, c1_G, c1_B,
                                                          c2_R, c2_G, c2_B, ...]
        (this is what load_ply in gau_io.py produces via its transpose; see the
        `shs[:, 3:] = shs[:, 3:].reshape(...).transpose(...)` line there).
    """
    gs = np.load(npy_path, allow_pickle=False)
    if verbose:
        print(f"[npy_to_ply] loaded {npy_path}: N={len(gs)} gaussians")

    sh_dim, rest_per_channel = _infer_sh_dim(gs)
    if verbose:
        # Infer SH degree for info only
        coeffs_per_channel = 1 + rest_per_channel
        degree = int(np.sqrt(coeffs_per_channel)) - 1
        print(f"[npy_to_ply] sh_dim={sh_dim}, rest_per_channel={rest_per_channel}, "
              f"inferred SH degree={degree}")

    pw    = np.ascontiguousarray(gs['pw'],    dtype=np.float32)   # (N, 3)
    rot   = np.ascontiguousarray(gs['rot'],   dtype=np.float32)   # (N, 4)
    scale = np.ascontiguousarray(gs['scale'], dtype=np.float32)   # (N, 3) LINEAR
    alpha = np.ascontiguousarray(gs['alpha'], dtype=np.float32)   # (N,)   SIGMOID
    sh    = np.ascontiguousarray(gs['sh'],    dtype=np.float32)   # (N, sh_dim)

    if alpha.ndim > 1:
        alpha = alpha.squeeze()
    N = pw.shape[0]

    # --- Invert the transformations load_ply applies, so Brush reads back the right values ---

    # Scales: .npy holds linear, Inria PLY holds log.
    # Clamp to avoid log(0) from any zero-scale splats.
    log_scale = np.log(np.clip(scale, 1e-20, None)).astype(np.float32)

    # Opacity: .npy holds sigmoid, Inria PLY holds raw logit.
    raw_opacity = _logit(alpha).astype(np.float32)

    # DC: subtract dc_shift so Brush's +0.5 at render time reproduces our color.
    # our codebase: color = SH_C0 * dc
    # brush:        color = SH_C0 * dc_brush + 0.5
    # => dc_brush = dc - 0.5 / SH_C0
    dc = (sh[:, 0:3] - np.float32(dc_shift)).astype(np.float32)    # (N, 3)

    # Rest: convert from coeff-major (as stored in .npy after load_ply's transpose)
    # to channel-major (what PLY/Inria/Brush expect).
    # Memory layout in .npy for sh[:, 3:]:     [c1_R, c1_G, c1_B, c2_R, c2_G, c2_B, ...]
    # PLY expects:                              [R_c1, R_c2, ..., G_c1, G_c2, ..., B_c1, ...]
    # This is just a (rest_per_channel, 3) -> (3, rest_per_channel) transpose per splat.
    rest = sh[:, 3:]                                               # (N, 3 * rest)
    rest = rest.reshape(N, rest_per_channel, 3)                    # (N, rest, 3)  [coeff, channel]
    rest = np.transpose(rest, (0, 2, 1))                           # (N, 3, rest)  [channel, coeff]
    rest = np.ascontiguousarray(rest.reshape(N, 3 * rest_per_channel), dtype=np.float32)

    # --- Build the structured array in Inria field order ---
    fields = (
        [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        + [('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4')]
        + [('opacity', '<f4')]
        + [('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4')]
        + [('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4')]
        + [(f'f_rest_{i}', '<f4') for i in range(3 * rest_per_channel)]
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
    vertex['f_dc_0']  = dc[:, 0]
    vertex['f_dc_1']  = dc[:, 1]
    vertex['f_dc_2']  = dc[:, 2]
    for i in range(3 * rest_per_channel):
        vertex[f'f_rest_{i}'] = rest[:, i]

    # --- Write binary little-endian PLY ---
    header_lines = [
        'ply',
        'format binary_little_endian 1.0',
        f'element vertex {N}',
    ]
    header_lines += [f'property float {nm}' for (nm, _) in fields]
    header_lines += ['end_header', '']
    header = '\n'.join(header_lines).encode('ascii')

    with open(ply_path, 'wb') as f:
        f.write(header)
        f.write(vertex.tobytes())

    if verbose:
        print(f"[npy_to_ply] wrote {ply_path}")
        print(f"[npy_to_ply]   N={N}, sh_dim={sh_dim}, dc_shift={dc_shift:.7f}")


# ---------------------------------------------------------------------------
# Sanity check — quick math spot-check, no training needed
# ---------------------------------------------------------------------------

def _sanity_check(npy_path, dc_shift=DEFAULT_DC_SHIFT):
    """
    Pick a random splat and print:
      - its stored DC in the .npy (our convention)
      - the DC that will be written to the PLY (Brush convention)
      - the color each convention reproduces at viewdir-independent (DC-only) terms
    The two colors should match.
    """
    gs = np.load(npy_path, allow_pickle=False)
    i = np.random.default_rng(0).integers(0, len(gs))
    dc_ours = gs['sh'][i, 0:3].astype(np.float32)
    dc_brush = dc_ours - np.float32(dc_shift)
    color_from_ours  = SH_C0 * dc_ours                   # our codebase convention
    color_from_brush = SH_C0 * dc_brush + 0.5            # Brush's +0.5 bias
    diff = np.abs(color_from_ours - color_from_brush).max()
    print(f"[sanity] splat index {i}")
    print(f"[sanity]   stored DC (ours):   {dc_ours}")
    print(f"[sanity]   shifted DC (Brush): {dc_brush}")
    print(f"[sanity]   color (ours):       {color_from_ours}")
    print(f"[sanity]   color (Brush PLY):  {color_from_brush}")
    print(f"[sanity]   max |diff|:         {diff:.3e}")
    print(f"[sanity]   dc_shift used:      {dc_shift:.7f}")
    if diff < 1e-5:
        print("[sanity] PASS — DC shift is arithmetically correct.")
    else:
        print("[sanity] FAIL — DC shift does not recover original color.")
        print("[sanity]   this is a bug in the conversion, do not use the PLY.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="input .npy path")
    parser.add_argument("output", nargs="?", help="output .ply path (required unless --sanity-check)")
    parser.add_argument(
        "--dc-shift",
        type=float,
        default=DEFAULT_DC_SHIFT,
        help=f"value subtracted from DC coeffs on export (default: {DEFAULT_DC_SHIFT:.7f} = 0.5/SH_C0)",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="print DC-shift arithmetic check for one splat and exit; no file written",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress info messages")
    args = parser.parse_args()

    if args.sanity_check:
        _sanity_check(args.input, dc_shift=args.dc_shift)
        return 0

    if args.output is None:
        parser.error("output .ply path is required (unless --sanity-check)")

    npy_to_ply(args.input, args.output, dc_shift=args.dc_shift, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())