## example run:
##    python 

import math
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from gsplat.gau_io import *
from gsplat.simplepalettes import *
from gsplat.gausplat_dataset import GSplatDataset
from gsplat.gsmodel import (
    render_vanilla,
    FinetuneModel,
    get_finetune_params,
    get_palette_rgb,
    barycentric_coords_3d,
    bake_palette_edit,
    reconstruct_high_shs,
)

from convexhull.simplify_convexhull import *


import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#def palette_compactness_loss(sat_raw, lit_raw):
#    return torch.sigmoid(sat_raw).mean()

def palette_compactness_loss(palette_rgb):
    """
    Volume of K-anchor palette hull (K chromatic + black + white),
    decomposed as K tetrahedra fanning around the (black, white) gray-axis spine.
    """
    K = palette_rgb.shape[0]
    assert K >= 2, "Need at least 2 chromatic anchors"

    black = torch.zeros(3, device=palette_rgb.device, dtype=palette_rgb.dtype)
    white = torch.ones(3,  device=palette_rgb.device, dtype=palette_rgb.dtype)

    def tet_vol(v0, v1, v2, v3):
        M = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=0)
        return torch.abs(torch.det(M)) / 6.0

    total = palette_rgb.new_zeros(())
    for i in range(K):
        c_curr = palette_rgb[i]
        c_next = palette_rgb[(i + 1) % K]
        total = total + tet_vol(black, white, c_curr, c_next)

    return total


def hue_separation_loss(gap_raw, min_gap_fraction=0.5):
    """
    Penalize hue gaps that fall below a fraction of even spacing.
    """
    K = gap_raw.shape[0]
    gaps_positive = F.softplus(gap_raw)
    gaps = (2 * math.pi) * gaps_positive / gaps_positive.sum()
    even_gap = (2 * math.pi) / K
    min_gap = min_gap_fraction * even_gap
    violation = F.relu(min_gap - gaps)
    return (violation ** 2).mean()


def convex_hull_palette_init(frozen_renders, K, gt_alphas=None,
                             n_bins=256, alpha_thresh=0.5,
                             min_bin_count=8):
    """
    Extract K chromatic palette colors via Tan et al.'s simplified convex hull.
    """
    from scipy.spatial import ConvexHull
    from convexhull.simplify_convexhull import (
        get_faces_vertices, simplified_convex_hull
    )

    all_colors = []
    for i, view in enumerate(frozen_renders):
        rgb_np = view.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        if gt_alphas is not None and gt_alphas[i] is not None:
            mask = gt_alphas[i].reshape(-1).cpu().numpy() > alpha_thresh
        else:
            mask = rgb_np.sum(axis=1) > 0.01
        if mask.any():
            all_colors.append(rgb_np[mask])
    samples = np.concatenate(all_colors, axis=0).clip(0.0, 1.0).astype(np.float64)
    print(f"  [convex hull init] {len(samples):,} object-pixel samples")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    H, _ = np.histogramdd(samples, bins=(bin_edges, bin_edges, bin_edges))
    nonempty = H >= min_bin_count

    centers_1d = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cx, cy, cz = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing='ij')
    bin_centers = np.stack([cx, cy, cz], axis=-1)[nonempty].astype(np.float64)
    print(f"  [convex hull init] {len(bin_centers):,} non-empty bins "
          f"(n_bins={n_bins}, min_count={min_bin_count})")

    if len(bin_centers) < K + 2:
        raise RuntimeError(
            f"Not enough bins ({len(bin_centers)}) to form a hull of K+2={K+2} vertices. "
            f"Lower n_bins or min_bin_count.")

    hull = ConvexHull(bin_centers)
    print(f"  [convex hull init] initial hull: {len(hull.vertices)} vertices")

    hvertices, hfaces = get_faces_vertices(hull)
    target_size = K + 2
    simplified_mesh = simplified_convex_hull(target_size, hvertices, hfaces)
    hull_vertices = np.asarray(simplified_mesh.vs).clip(0.0, 1.0)
    print(f"  [convex hull init] simplified to {len(hull_vertices)} vertices")

    black = np.zeros(3, dtype=np.float64)
    white = np.ones(3, dtype=np.float64)

    d_black = np.linalg.norm(hull_vertices - black, axis=1)
    d_white = np.linalg.norm(hull_vertices - white, axis=1)

    idx_black = int(np.argmin(d_black))
    d_white_excl = d_white.copy()
    d_white_excl[idx_black] = np.inf
    idx_white = int(np.argmin(d_white_excl))

    drop_idx = {idx_black, idx_white}
    keep = np.array([i for i in range(len(hull_vertices)) if i not in drop_idx])
    chromatic = hull_vertices[keep]

    if len(chromatic) > K:
        gray_dist = np.linalg.norm(chromatic - chromatic.mean(axis=1, keepdims=True), axis=1)
        order = np.argsort(-gray_dist)
        chromatic = chromatic[order[:K]]
    elif len(chromatic) < K:
        gray_dist = np.linalg.norm(chromatic - chromatic.mean(axis=1, keepdims=True), axis=1)
        order = np.argsort(-gray_dist)
        pad_n = K - len(chromatic)
        chromatic = np.vstack([chromatic, chromatic[order[:1]].repeat(pad_n, axis=0)])

    print(f"  [convex hull init] dropped vertex {idx_black} (closest to black) "
          f"and {idx_white} (closest to white)")

    hues = np.array([_hue_of_rgb(c) for c in chromatic])
    sort_idx = np.argsort(hues)
    chromatic = chromatic[sort_idx]
    hues_sorted = hues[sort_idx]

    print(f"  [convex hull init] returning {len(chromatic)} chromatic colors "
          f"(CCW-sorted by hue):")
    for i, (c, h) in enumerate(zip(chromatic, hues_sorted)):
        print(f"    [{i}] {np.round(c, 3)}  hue={np.degrees(h):.1f}°")
    
    return chromatic.astype(np.float32)


def _hue_of_rgb(rgb):
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta < 1e-8:
        return 0.0
    if cmax == r:
        h_sector = ((g - b) / delta) % 6.0
    elif cmax == g:
        h_sector = (b - r) / delta + 2.0
    else:
        h_sector = (r - g) / delta + 4.0
    return (h_sector * (math.pi / 3.0)) % (2 * math.pi)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs",          required=True, help="pretrained vanilla 3DGS .npy")
    parser.add_argument("--data",        required=True, help="dataset path")
    parser.add_argument("--output",      default=None,
                        help="output path; defaults to data/<scene>_palette.npy "
                             "where <scene> is derived from --data")
    parser.add_argument("--num_palette", type=int, default=4)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--lambda_pos",     type=float, default=0.5)
    parser.add_argument("--lambda_compact", type=float, default=0.001,  # this is very sensitive
                        help="base weight for compactness loss (active in first 2/3 of training)")
    parser.add_argument("--compact_ramp_factor", type=float, default=5.0,
                        help="multiplier for lambda_compact at end of training")
    parser.add_argument("--compact_schedule_start", type=float, default=2.0/3.0,
                        help="fraction of epochs after which compactness ramp begins")
    parser.add_argument("--lambda_alpha",   type=float, default=1.0,
                        help="weight for alpha-channel supervision (Blender only)")
    parser.add_argument("--lambda_sep",     type=float, default=1.0,
                        help="weight for hue separation regularizer (palette spread)")
    parser.add_argument("--clamp_start",    type=float, default=1.0,
                        help="fraction of epochs after which barycentric weights "
                             "are clamped to non-negative and renormalized "
                             "(so weights become a proper convex combination of "
                             "palette anchors). Set <1.0 to enable; e.g. 0.7 starts "
                             "clamping at 70%% through training. Default 1.0 = off.")
    parser.add_argument("--min_gap_fraction", type=float, default=0.3,
                        help="minimum hue gap as fraction of even spacing (2π/K)")
    parser.add_argument("--resize_rate", type=float, default=1.0)
    parser.add_argument("--sh_order",    type=int, default=3)
    parser.add_argument("--low_rank",    action="store_true")
    parser.add_argument("--rank",        type=int, default=2)
    parser.add_argument("--reshape_P",   type=int, default=None)
    parser.add_argument("--reshape_Q",   type=int, default=None)
    parser.add_argument("--use_fast",    action="store_true")
    parser.add_argument("--dataset_type", type=str, default="auto",
                        choices=["auto", "colmap", "blender"])
    parser.add_argument("--convexhull_init", action="store_true",
                        help="Initialize palette from simplified convex hull of frozen renders")
    args = parser.parse_args()

    if args.output is None:
        scene_name = os.path.basename(os.path.normpath(args.data))
        args.output = os.path.join("data", f"{scene_name}_palette.npy")
        print(f"[info] --output not set; using {args.output}")

    print(f"\n{'='*60}\nPALETTE WEIGHT FINETUNING FROM VANILLA 3DGS\n{'='*60}\n")

    print(f"Loading {args.gs} ...")
    gs = np.load(args.gs, allow_pickle=True)
    if gs.ndim == 0:
        gs = gs.item()

    print(f"\nLoading dataset from {args.data} ...")
    gs_set = GSplatDataset(args.data, resize_rate=args.resize_rate,
                           dataset_type=args.dataset_type)
    n_views = len(gs_set)
    is_blender = (gs_set.dataset_type == 'blender')
    print(f"  Type: {gs_set.dataset_type} | views: {n_views}")
    print(f"  Alpha supervision: {'ON' if is_blender else 'OFF (COLMAP)'}")

    print(f"\nPre-rendering frozen colors ...")
    frozen_pws    = torch.from_numpy(gs['pw']).float().cuda()
    frozen_rots   = torch.from_numpy(gs['rot']).float().cuda()
    frozen_scales = torch.from_numpy(gs['scale']).float().cuda()
    frozen_alphas = torch.from_numpy(gs['alpha']).float().cuda()
    frozen_shs    = torch.from_numpy(gs['sh']).float().cuda()

    frozen_renders = []
    cameras        = []
    gt_alphas      = []
    for i in tqdm(range(n_views)):
        cam, _, alpha = gs_set[i]
        frozen_renders.append(render_vanilla(
            frozen_pws, frozen_shs,
            frozen_alphas, frozen_scales, frozen_rots, cam))
        cameras.append(cam)
        gt_alphas.append(alpha)
    print(f"  Done. Shape per view: {frozen_renders[0].shape}")

    init_palette_rgb = None
    if args.convexhull_init:
        print(f"\nRunning Tan et al. 16 on frozen render colors (K={args.num_palette}) ...")
        init_palette_rgb = convex_hull_palette_init(
            frozen_renders, args.num_palette, gt_alphas=gt_alphas)
        print(f"  Init Palette (RGB):\n{np.round(255.*init_palette_rgb, 3)}")

        palette_img = np.clip(palette2swatch(init_palette_rgb) * 255., 0, 255).astype(np.uint8)
        save_image_to_file(palette_img, 'int_palette.png', clobber=True)

    frozen, params, adam_params = get_finetune_params(
        gs, sh_order=args.sh_order, num_palette=args.num_palette,
        low_rank=args.low_rank, reshape_P=args.reshape_P, reshape_Q=args.reshape_Q,
        init_palette_rgb=init_palette_rgb,
    )

    K           = args.num_palette
    splat_dim   = K + 2
    num_sh_high = (args.sh_order + 1) ** 2 - 1
    N           = frozen['pws'].shape[0]
    P, Q        = params["reshape_P"], params["reshape_Q"]
    KL          = splat_dim * num_sh_high
    shs_params_per_splat = (splat_dim + P + Q) if args.low_rank else (splat_dim + KL)

    print(f"\n  Gaussians: {N:,} | K={K} (splat_dim={splat_dim}) | SH order {args.sh_order}")
    print(f"  Mode: {'low-rank' if args.low_rank else 'full rank'} | reshape {P}x{Q}")
    print(f"  SH params/splat: {shs_params_per_splat}")

    from PIL import Image as PILImage
    first_img = (frozen_renders[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255
                 ).astype(np.uint8)
    preview_path = args.output.replace('.npy', '_preview.png')
    PILImage.fromarray(first_img).save(preview_path)
    print(f"  First view saved to {preview_path}")

    model = FinetuneModel(sh_order=args.sh_order,
                          num_palette=args.num_palette, use_fast=args.use_fast)
    optimizer = optim.Adam(adam_params, lr=0.0, eps=1e-15)

    # ── Compute schedule boundary for lambda_compact ─────────────────────
    schedule_start = int(args.epochs * args.compact_schedule_start)
    clamp_epoch    = int(args.epochs * args.clamp_start)

    print(f"\n{'='*60}")
    print(f"Finetuning: {args.epochs} epochs, {n_views} views/epoch")
    print(f"  lambda_pos     = {args.lambda_pos}")
    print(f"  lambda_compact = {args.lambda_compact}  "
          f"(ramps to {args.lambda_compact * args.compact_ramp_factor} "
          f"after epoch {schedule_start})")
    print(f"  lambda_sep     = {args.lambda_sep}  (min_gap_fraction = {args.min_gap_fraction})")
    if args.clamp_start < 1.0:
        clamp_epoch = int(args.epochs * args.clamp_start)
        print(f"  clamp_start    = {args.clamp_start}  "
              f"(non-negative weight clamping starts at epoch {clamp_epoch})")
    if is_blender:
        print(f"  lambda_alpha   = {args.lambda_alpha}")
    print(f"  convexhull_init = {args.convexhull_init}")
    print(f"{'='*60}\n")

    losses_geo, losses_pos, losses_compact = [], [], []
    losses_alpha, losses_sep, losses_total = [], [], []
    t0 = time.time()

    for epoch in range(args.epochs):
        # ── Compute current lambda_compact for this epoch ──
        if epoch < schedule_start:
            lambda_compact_curr = args.lambda_compact
        else:
            ramp_progress = (epoch - schedule_start) / max(1, args.epochs - schedule_start - 1)
            ramp_progress = min(1.0, ramp_progress)
            lambda_compact_curr = args.lambda_compact * (
                1.0 + (args.compact_ramp_factor - 1.0) * ramp_progress
            )

        # ── Whether to clamp barycentric weights to non-negative this epoch ──
        clamp_active = (epoch >= clamp_epoch)

        idx_order = np.random.permutation(n_views)
        ep_geo = ep_pos = ep_compact = ep_alpha = ep_sep = ep_total = 0.0

        for idx in idx_order:
            cam = cameras[idx]
            C   = frozen_renders[idx]

            W_tilde, palette_rgb = model(
                frozen["pws"], params,
                frozen["alphas"], frozen["scales"], frozen["rots"],
                cameras[idx],
            )

            if is_blender:
                a = gt_alphas[idx]
                a_safe = a.clamp(min=1e-3)
                C_straight = (C / a_safe).clamp(0.0, 1.0)
                W_bary_straight = barycentric_coords_3d(C_straight, palette_rgb)
                if clamp_active:
                    lambda_compact_curr = 0   # if clamping starting, we stop penalizing compactness
                    W_bary_straight = W_bary_straight.clamp(min=0.0)
                    W_bary_straight = W_bary_straight / \
                        W_bary_straight.sum(dim=0, keepdim=True).clamp(min=1e-8)
                W_bary = W_bary_straight * a
            else:
                W_bary = barycentric_coords_3d(C, palette_rgb)
                if clamp_active:
                    lambda_compact_curr = 0   # if clamping starting, we stop penalizing compactness
                    W_bary = W_bary.clamp(min=0.0)
                    W_bary = W_bary / W_bary.sum(dim=0, keepdim=True).clamp(min=1e-8)

            L_geo     = F.mse_loss(W_tilde, W_bary)
            L_pos     = torch.mean(F.relu(-W_tilde))
            L_compact = palette_compactness_loss(palette_rgb)
            L_sep     = hue_separation_loss(params["gap_raw"],
                                            min_gap_fraction=args.min_gap_fraction)

            loss = (L_geo
                    + args.lambda_pos     * L_pos
                    + lambda_compact_curr * L_compact
                    + args.lambda_sep     * L_sep)

            if is_blender:
                w_alpha  = W_tilde.sum(dim=0, keepdim=True)
                L_alpha  = F.mse_loss(w_alpha, gt_alphas[idx])
                loss     = loss + args.lambda_alpha * L_alpha
                ep_alpha += L_alpha.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_geo     += L_geo.item()
            ep_pos     += L_pos.item()
            ep_compact += L_compact.item()
            ep_sep     += L_sep.item()
            ep_total   += loss.item()

        ep_geo     /= n_views
        ep_pos     /= n_views
        ep_compact /= n_views
        ep_alpha   /= n_views
        ep_sep     /= n_views
        ep_total   /= n_views
        losses_geo.append(ep_geo); losses_pos.append(ep_pos)
        losses_compact.append(ep_compact); losses_alpha.append(ep_alpha)
        losses_sep.append(ep_sep); losses_total.append(ep_total)

        msg = (f"Epoch {epoch+1:3d}/{args.epochs} | "
               f"L_geo={ep_geo:.5f}  L_pos={ep_pos:.5f}  "
               f"L_compact={ep_compact:.5f}(λ={lambda_compact_curr:.4f})  "
               f"L_sep={ep_sep:.5f}")
        if clamp_active:
            msg += "  [clamp]"
        if is_blender:
            msg += f"  L_alpha={ep_alpha:.5f}"
        msg += f"  total={ep_total:.5f} | {time.time()-t0:.1f}s"
        print(msg)

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                pal = get_palette_rgb(params["hue_base"], params["gap_raw"],
                                      params["sat_raw"], params["lit_raw"])
                P_full = torch.cat([params["black"], params["white"], pal], dim=0)
                print(f"  Palette:\n{P_full.cpu().numpy().round(3)}")

                gp = F.softplus(params["gap_raw"])
                gaps = (2 * math.pi) * gp / gp.sum()
                gaps_deg = (gaps * 180 / math.pi).cpu().numpy()
                print(f"  Hue gaps (deg): {gaps_deg.round(1)}  "
                      f"(min: {args.min_gap_fraction*360/K:.1f})")

    print(f"\nDone in {time.time()-t0:.1f}s")

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True)

    with torch.no_grad():
        pal = get_palette_rgb(params["hue_base"], params["gap_raw"],
                              params["sat_raw"], params["lit_raw"])
        P_full_np = torch.cat([params["black"], params["white"], pal],
                              dim=0).detach().cpu().numpy()

        shared = {
            "palette": P_full_np,
            "pw":      gs['pw'], "rot": gs['rot'],
            "scale":   gs['scale'], "alpha": gs['alpha'],
        }

        if args.low_rank:
            save_dict = {
                "low_shs_w":  params["low_shs_w"].detach().cpu().numpy(),
                "high_shs_A": params["high_shs_A"].detach().cpu().numpy(),
                "high_shs_B": params["high_shs_B"].detach().cpu().numpy(),
                "low_rank":   True,
                "reshape_P":  params["reshape_P"], "reshape_Q":  params["reshape_Q"],
                **shared,
            }
        else:
            high_shs = reconstruct_high_shs(params)
            shs_w_np = torch.cat([params["low_shs_w"], high_shs],
                                 dim=1).detach().cpu().numpy()
            save_dict = {"shs_w": shs_w_np, "low_rank": False, **shared}

    np.save(args.output, save_dict)
    print(f"Saved to {args.output}")
    print(f"  palette:\n{P_full_np.round(3)}")

    print(f"\nVerifying bake ...")
    shs_baked = bake_palette_edit(params, frozen)
    with torch.no_grad():
        C_baked = render_vanilla(
            frozen["pws"], shs_baked,
            frozen["alphas"], frozen["scales"], frozen["rots"], cameras[0])
        mse = F.mse_loss(C_baked, frozen_renders[0])

        if is_blender:
            mask_obj = (frozen_renders[0].sum(0) > 0.01)
            mask_bg  = ~mask_obj
            mse_obj = F.mse_loss(C_baked[:, mask_obj], frozen_renders[0][:, mask_obj]) \
                if mask_obj.any() else torch.tensor(0.0)
            mse_bg  = F.mse_loss(C_baked[:, mask_bg],  frozen_renders[0][:, mask_bg]) \
                if mask_bg.any() else torch.tensor(0.0)
            print(f"  Bake MSE — total: {mse.item():.6f}  "
                  f"object: {mse_obj.item():.6f}  background: {mse_bg.item():.6f}")
        else:
            print(f"  Bake MSE: {mse.item():.6f}")

    plot_path = args.output.replace('.npy', '_loss.png')
    plt.figure(figsize=(10, 5))
    plt.plot(losses_geo,     label='L_geo',     linewidth=1.5)
    plt.plot(losses_pos,     label='L_pos',     linewidth=1.5)
    plt.plot(losses_compact, label='L_compact', linewidth=1.5)
    plt.plot(losses_sep,     label='L_sep',     linewidth=1.5)
    if is_blender:
        plt.plot(losses_alpha, label='L_alpha', linewidth=1.5)
    plt.plot(losses_total, label='Total', linewidth=2, color='black')
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.axvline(x=schedule_start, color='gray', linestyle='--', alpha=0.5,
                label=f'compact ramp start (epoch {schedule_start})')
    if args.clamp_start < 1.0:
        plt.axvline(x=clamp_epoch, color='red', linestyle='--', alpha=0.5,
                    label=f'weight clamp start (epoch {clamp_epoch})')
    title_mode = ('low-rank ' + str(params['reshape_P']) + 'x' + str(params['reshape_Q'])
                  if args.low_rank else 'full rank')
    plt.title(f"Palette Finetuning ({title_mode}, {gs_set.dataset_type})")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"Loss plot saved to {plot_path}")