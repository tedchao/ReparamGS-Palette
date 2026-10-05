import math
import torch
import torch.nn.functional as F
import numpy as np
import gsplatcu as gsc
from gsplat.utils import *


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla 3DGS renderer — frozen, no gradients
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def render_vanilla(pws, shs, alphas, scales, rots, cam):
    """Render frozen vanilla 3DGS. Returns (3, H, W)."""
    us, pcs, depths = gsc.project(
        pws, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, False)
    cov3ds,  = gsc.computeCov3D(rots, scales, depths, False)
    cov2ds,  = gsc.computeCov2D(
        cov3ds, pcs, cam.Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, False)
    colors,  = gsc.sh2Color(shs, pws, cam.twc, False)
    cinv2ds, areas = gsc.inverseCov2D(cov2ds, depths, False)
    image, _, _, _, _ = gsc.splat_color(
        cam.height, cam.width,
        us, cinv2ds, alphas, depths, colors, areas)
    return image  # (3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# HSL palette parameterization
# ─────────────────────────────────────────────────────────────────────────────

def hsl_to_rgb(h, s, l):
    h6    = h / (2 * math.pi) * 6.0
    C     = (1.0 - torch.abs(2.0 * l - 1.0)) * s
    X     = C * (1.0 - torch.abs(h6 % 2.0 - 1.0))
    m     = l - C / 2.0
    zeros = torch.zeros_like(C)
    sectors = torch.stack([
        torch.stack([C, X, zeros], dim=-1),
        torch.stack([X, C, zeros], dim=-1),
        torch.stack([zeros, C, X], dim=-1),
        torch.stack([zeros, X, C], dim=-1),
        torch.stack([X, zeros, C], dim=-1),
        torch.stack([C, zeros, X], dim=-1),
    ], dim=1)
    idx  = h6.long().clamp(0, 5)
    rgb1 = sectors[torch.arange(len(h), device=h.device), idx]
    return (rgb1 + m.unsqueeze(-1)).clamp(0.0, 1.0)


def rgb_to_hsl(rgb):
    """
    Convert (K, 3) RGB in [0, 1] to (K,) hue [0, 2pi), sat [0, 1], lit [0, 1].
    Returns numpy arrays (works on CPU; this is only used at init).
    """
    rgb = np.asarray(rgb, dtype=np.float32).clip(0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    delta = cmax - cmin

    lit = (cmax + cmin) / 2.0

    sat = np.zeros_like(lit)
    nonzero = delta > 1e-8
    # Standard HSL saturation
    denom = 1.0 - np.abs(2.0 * lit - 1.0)
    sat[nonzero] = delta[nonzero] / np.maximum(denom[nonzero], 1e-8)
    sat = sat.clip(0.0, 1.0)

    hue = np.zeros_like(lit)
    # Compute hue per pixel based on which channel is max
    rmax = (cmax == r) & nonzero
    gmax = (cmax == g) & nonzero & ~rmax
    bmax = (cmax == b) & nonzero & ~rmax & ~gmax
    hue[rmax] = ((g[rmax] - b[rmax]) / np.maximum(delta[rmax], 1e-8)) % 6.0
    hue[gmax] = (b[gmax] - r[gmax]) / np.maximum(delta[gmax], 1e-8) + 2.0
    hue[bmax] = (r[bmax] - g[bmax]) / np.maximum(delta[bmax], 1e-8) + 4.0
    hue = hue * (math.pi / 3.0)   # convert sectors -> radians ([0, 2pi))
    hue = hue % (2 * math.pi)

    return hue, sat, lit


# ─────────────────────────────────────────────────────────────────────────────
# Low-rank high-frequency weight SH reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_high_shs(params):
    if params["low_rank"]:
        N = params["high_shs_A"].shape[0]
        A = params["high_shs_A"]
        B = params["high_shs_B"]
        return (A.unsqueeze(2) * B.unsqueeze(1)).reshape(N, -1)
    else:
        return params["high_shs_w"]


# ─────────────────────────────────────────────────────────────────────────────
# Weight splatting — standard path
# ─────────────────────────────────────────────────────────────────────────────

class WeightGSFunctionTest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pws, shs, alphas, scales, rots, cam, num_palette):
        us, pcs, depths = gsc.project(
            pws, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, False)
        cov3ds,  = gsc.computeCov3D(rots, scales, depths, False)
        cov2ds,  = gsc.computeCov2D(
            cov3ds, pcs, cam.Rcw, depths,
            cam.fx, cam.fy, cam.width, cam.height, False)
        weights, dweight_dshs, _ = \
            gsc.sh2Weight(shs, pws, cam.twc, True, num_palette)
        cinv2ds, areas = gsc.inverseCov2D(cov2ds, depths, False)
        layer_W, contrib, final_tau, patch_range_per_tile, gsid_per_patch = \
            gsc.splat_weight(cam.height, cam.width, num_palette,
                             us, cinv2ds, alphas, depths, weights, areas)
        ctx.cam         = cam
        ctx.num_palette = num_palette
        ctx.save_for_backward(
            us, cinv2ds, alphas, depths, weights,
            contrib, final_tau,
            patch_range_per_tile, gsid_per_patch,
            dweight_dshs)
        return layer_W, depths > 0.2

    @staticmethod
    def backward(ctx, dloss_dlayer_W, _):
        cam         = ctx.cam
        num_palette = ctx.num_palette
        (us, cinv2ds, alphas, depths, weights,
         contrib, final_tau, patch_range_per_tile, gsid_per_patch,
         dweight_dshs) = ctx.saved_tensors
        dloss_dweights, = gsc.splat_weight_B(
            cam.height, cam.width, num_palette,
            us, cinv2ds, alphas, depths, weights,
            contrib, final_tau,
            patch_range_per_tile, gsid_per_patch, dloss_dlayer_W)
        dloss_dshs = (dloss_dweights.permute(0, 2, 1) @
                      dweight_dshs).permute(0, 2, 1)
        N, num_sh, K = dloss_dshs.shape
        dloss_dshs = dloss_dshs.reshape(N, num_sh * K)
        return (None, dloss_dshs, None, None, None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Weight splatting — fast low-rank path
# ─────────────────────────────────────────────────────────────────────────────

class WeightGSFunctionFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pws, shs_dc, shs_A, shs_B,
                alphas, scales, rots, cam, num_palette, Q):
        us, pcs, depths = gsc.project(
            pws, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, False)
        cov3ds,  = gsc.computeCov3D(rots, scales, depths, False)
        cov2ds,  = gsc.computeCov2D(
            cov3ds, pcs, cam.Rcw, depths,
            cam.fx, cam.fy, cam.width, cam.height, False)
        weights, dweight_ddc, Ylm_stored = \
            gsc.sh2Weight_fast(
                shs_dc, shs_A, shs_B,
                pws, cam.twc, True,
                num_palette, Q)
        cinv2ds, areas = gsc.inverseCov2D(cov2ds, depths, False)
        layer_W, contrib, final_tau, patch_range_per_tile, gsid_per_patch = \
            gsc.splat_weight(
                cam.height, cam.width, num_palette,
                us, cinv2ds, alphas, depths, weights, areas)
        ctx.cam         = cam
        ctx.num_palette = num_palette
        ctx.Q           = Q
        ctx.save_for_backward(
            us, cinv2ds, alphas, depths, weights,
            contrib, final_tau,
            patch_range_per_tile, gsid_per_patch,
            dweight_ddc, Ylm_stored, shs_A, shs_B)
        return layer_W, depths > 0.2

    @staticmethod
    def backward(ctx, dloss_dlayer_W, _):
        cam         = ctx.cam
        num_palette = ctx.num_palette
        Q           = ctx.Q
        (us, cinv2ds, alphas, depths, weights,
         contrib, final_tau, patch_range_per_tile, gsid_per_patch,
         dweight_ddc, Ylm_stored, shs_A, shs_B) = ctx.saved_tensors
        dloss_dweights, = gsc.splat_weight_B(
            cam.height, cam.width, num_palette,
            us, cinv2ds, alphas, depths, weights,
            contrib, final_tau,
            patch_range_per_tile, gsid_per_patch, dloss_dlayer_W)
        dloss_dw = dloss_dweights.squeeze(1)
        dloss_ddc = dloss_dw * dweight_ddc
        dloss_dA, dloss_dB = gsc.sh2Weight_fast_B(
            shs_A, shs_B, Ylm_stored, dloss_dw,
            num_palette, Q)
        return (None, dloss_ddc, dloss_dA, dloss_dB,
                None, None, None, None, None, None)


def get_palette_rgb(hue_base, gap_raw, sat_raw, lit_raw):
    TWO_PI = 2 * math.pi
    gaps_positive = F.softplus(gap_raw)
    gaps = TWO_PI * gaps_positive / gaps_positive.sum()
    gaps_padded = F.pad(gaps[:-1], (1, 0))
    hues_raw    = (hue_base % TWO_PI) + torch.cumsum(gaps_padded, dim=0)
    hues = hues_raw % TWO_PI
    sats = torch.sigmoid(sat_raw)
    lits = torch.sigmoid(lit_raw)
    return hsl_to_rgb(hues, sats, lits)


# ─────────────────────────────────────────────────────────────────────────────
# 3D Barycentric Coordinates  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def barycentric_coords_3d(C, P, tau=1e-9, eps=1e-12):
    device, dtype = P.device, P.dtype
    K = P.shape[0]
    H, W = C.shape[1], C.shape[2]
    N = H * W

    black = torch.zeros(3, device=device, dtype=dtype)
    white = torch.ones(3,  device=device, dtype=dtype)
    P_aug = torch.cat([black.unsqueeze(0), white.unsqueeze(0), P], dim=0)
    C_flat = C.permute(1, 2, 0).reshape(N, 3)

    i_idx = torch.arange(K, device=device)
    j_idx = (i_idx + 1) % K

    v0  = (white - black).unsqueeze(0).expand(K, -1)
    v1  = P[i_idx] - black
    v2  = P[j_idx] - black
    rhs = C_flat - black

    A = torch.stack([v0, v1, v2], dim=2)

    a00=A[:,0,0]; a01=A[:,0,1]; a02=A[:,0,2]
    a10=A[:,1,0]; a11=A[:,1,1]; a12=A[:,1,2]
    a20=A[:,2,0]; a21=A[:,2,1]; a22=A[:,2,2]

    det = (a00*(a11*a22 - a12*a21)
         - a01*(a10*a22 - a12*a20)
         + a02*(a10*a21 - a11*a20))
    det_inv = 1.0 / (det + eps)

    adj = torch.stack([
         a11*a22 - a12*a21, -(a01*a22 - a02*a21),  a01*a12 - a02*a11,
        -(a10*a22 - a12*a20),  a00*a22 - a02*a20, -(a00*a12 - a02*a10),
         a10*a21 - a11*a20, -(a00*a21 - a01*a20),  a00*a11 - a01*a10,
    ], dim=1).reshape(K, 3, 3)

    coords = torch.einsum('kij,jn->kin', adj, rhs.T) * det_inv[:, None, None]
    coords = coords.permute(2, 0, 1)

    beta  = coords[:, :, 0]
    gamma = coords[:, :, 1]
    delta = coords[:, :, 2]
    alpha = 1.0 - beta - gamma - delta

    raw = torch.zeros(N, K, K + 2, device=device, dtype=dtype)
    raw[:, :, 0] = alpha
    raw[:, :, 1] = beta
    raw[:, torch.arange(K), i_idx + 2] = gamma
    raw[:, torch.arange(K), j_idx + 2] = delta

    a_p   = F.softplus(alpha)
    b_p   = F.softplus(beta)
    g_p   = F.softplus(gamma)
    d_p   = F.softplus(delta)
    denom = a_p + b_p + g_p + d_p + eps
    proj  = torch.zeros_like(raw)
    proj[:, :, 0] = a_p / denom
    proj[:, :, 1] = b_p / denom
    proj[:, torch.arange(K), i_idx + 2] = g_p / denom
    proj[:, torch.arange(K), j_idx + 2] = d_p / denom

    recon  = torch.matmul(proj, P_aug)
    errors = ((recon - C_flat.unsqueeze(1)) ** 2).sum(-1)
    probs  = F.softmax(-errors / tau, dim=1)
    W_flat = (probs.unsqueeze(-1) * raw).sum(dim=1)

    return W_flat.T.reshape(K + 2, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# FinetuneModel  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class FinetuneModel(torch.nn.Module):
    def __init__(self, sh_order=3, num_palette=4, use_fast=True):
        super().__init__()
        self.sh_order    = sh_order
        self.num_palette = num_palette
        self.splat_dim   = num_palette + 2
        self.use_fast    = use_fast
        self.mask        = None

    def forward(self, pws, params, alphas, scales, rots, cam):
        if params["low_rank"]:
            if self.use_fast:
                W_tilde, self.mask = WeightGSFunctionFast.apply(
                    pws,
                    params["low_shs_w"],
                    params["high_shs_A"],
                    params["high_shs_B"],
                    alphas, scales, rots, cam,
                    self.splat_dim,
                    params["reshape_Q"])
            else:
                high_shs = reconstruct_high_shs(params)
                shs_w    = get_shs(params["low_shs_w"], high_shs)
                W_tilde, self.mask = WeightGSFunctionTest.apply(
                    pws, shs_w, alphas, scales, rots, cam, self.splat_dim)
        else:
            high_shs = reconstruct_high_shs(params)
            shs_w    = get_shs(params["low_shs_w"], high_shs)
            W_tilde, self.mask = WeightGSFunctionTest.apply(
                pws, shs_w, alphas, scales, rots, cam, self.splat_dim)

        palette_rgb = get_palette_rgb(
            params["hue_base"],
            params["gap_raw"],
            params["sat_raw"],
            params["lit_raw"],
        )
        return W_tilde, palette_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Parameter initialization
# ─────────────────────────────────────────────────────────────────────────────

def _softplus_inv(x):
    return math.log(math.exp(x) - 1.0)


def _sigmoid_inv(x, eps=1e-4):
    """Inverse sigmoid (logit), clamped to avoid inf."""
    x = float(np.clip(x, eps, 1.0 - eps))
    return math.log(x / (1.0 - x))


def _hues_to_hsl_params(hues_np):
    """
    Convert sorted hues (radians, K values) to (hue_base, gap_raw) such that
    get_palette_rgb reconstructs them.

    The forward path computes:
      hues[k] = (hue_base + cumsum(gaps_padded)[k]) mod 2pi
    where gaps_padded = [0, gaps[0], gaps[1], ..., gaps[K-2]]
    and gaps = 2pi * softplus(gap_raw) / sum(softplus(gap_raw))

    So hues[0] = hue_base
       hues[k] - hues[k-1] = gaps[k-1]  for k >= 1
    And the K-th implied gap (hues[0]+2pi - hues[K-1]) = 2pi - sum(gaps[0..K-2])
    needs to come out positive too — that's automatic if we take sorted hues.
    """
    K = len(hues_np)
    hues_sorted = np.sort(hues_np)             # (K,) ascending in [0, 2pi)
    hue_base = float(hues_sorted[0])

    # gaps between consecutive hues (size K, last one wraps around)
    gaps = np.empty(K, dtype=np.float32)
    for k in range(K - 1):
        gaps[k] = hues_sorted[k + 1] - hues_sorted[k]
    gaps[K - 1] = (2 * math.pi) - hues_sorted[K - 1] + hues_sorted[0]

    # Forward uses softplus(gap_raw) normalized to sum 2pi, so we want
    # softplus(gap_raw[k]) ∝ gaps[k]. Pick gap_raw such that softplus(.) = gaps[k].
    gaps = np.maximum(gaps, 1e-3)              # avoid softplus_inv blowup
    gap_raw = np.array([_softplus_inv(g) for g in gaps], dtype=np.float32)

    return hue_base, gap_raw


def get_finetune_params(gs, sh_order=3, num_palette=4,
                        low_rank=False,
                        reshape_P=None, reshape_Q=None,
                        init_palette_rgb=None):
    """
    Args:
        init_palette_rgb: optional (K, 3) numpy array of RGB colors in [0, 1]
                          to initialize the chromatic palette (typically from k-means).
                          If None, falls back to even-hue init.
    """
    num_sh      = (sh_order + 1) ** 2
    num_sh_high = num_sh - 1
    splat_dim   = num_palette + 2
    K           = num_palette
    N           = gs['pw'].shape[0]
    KL          = splat_dim * num_sh_high

    # Resolve P, Q
    if reshape_P is None and reshape_Q is None:
        P = splat_dim; Q = num_sh_high
    elif reshape_P is not None and reshape_Q is not None:
        P, Q = reshape_P, reshape_Q
        assert P * Q == KL
    elif reshape_P is not None:
        P = reshape_P; assert KL % P == 0; Q = KL // P
    else:
        Q = reshape_Q; assert KL % Q == 0; P = KL // Q

    frozen = {
        "pws":       torch.from_numpy(gs['pw']).float().cuda(),
        "rots":      torch.from_numpy(gs['rot']).float().cuda(),
        "scales":    torch.from_numpy(gs['scale']).float().cuda(),
        "alphas":    torch.from_numpy(gs['alpha']).float().cuda(),
        "shs_color": torch.from_numpy(gs['sh']).float().cuda(),
    }

    low_shs_w = (1e-3 * torch.rand(N, splat_dim, device='cuda')).requires_grad_()
    if low_rank:
        high_shs_A = (1e-3 * torch.rand(N, P, device='cuda')).requires_grad_()
        high_shs_B = (1e-3 * torch.rand(N, Q, device='cuda')).requires_grad_()
        high_shs_w = None
        print(f"  High-freq SH: low-rank r=1, reshape {P}x{Q}")
    else:
        high_shs_w = (1e-3 * torch.rand(N, KL, device='cuda')).requires_grad_()
        high_shs_A = high_shs_B = None
        print(f"  High-freq SH: full rank ({KL} params/splat)")

    # ── Palette init ─────────────────────────────────────────────────────────
    if init_palette_rgb is not None:
        assert init_palette_rgb.shape == (K, 3), \
            f"init_palette_rgb must be ({K}, 3), got {init_palette_rgb.shape}"
        hues_np, sats_np, lits_np = rgb_to_hsl(init_palette_rgb)
        hue_base_val, gap_raw_np = _hues_to_hsl_params(hues_np)
        sat_raw_np = np.array([_sigmoid_inv(s) for s in sats_np], dtype=np.float32)
        lit_raw_np = np.array([_sigmoid_inv(l) for l in lits_np], dtype=np.float32)

        hue_base = torch.tensor(hue_base_val, device='cuda').requires_grad_()
        gap_raw  = torch.from_numpy(gap_raw_np).cuda().requires_grad_()
        sat_raw  = torch.from_numpy(sat_raw_np).cuda().requires_grad_()
        lit_raw  = torch.from_numpy(lit_raw_np).cuda().requires_grad_()
        print(f"  Palette init: from k-means ({K} colors)")
        print(f"    init_palette_rgb:\n{np.round(init_palette_rgb, 3)}")
    else:
        even_gap = 2.0 * math.pi / K
        hue_base = torch.tensor(0.0, device='cuda').requires_grad_()
        gap_raw  = torch.full((K,), _softplus_inv(even_gap),
                              device='cuda').requires_grad_()
        sat_raw  = torch.full((K,), 3.0, device='cuda').requires_grad_()
        lit_raw  = torch.zeros(K, device='cuda').requires_grad_()
        print(f"  Palette init: even-hue default")

    params = {
        "low_shs_w":  low_shs_w,
        "high_shs_w": high_shs_w,
        "high_shs_A": high_shs_A,
        "high_shs_B": high_shs_B,
        "low_rank":   low_rank,
        "reshape_P":  P, "reshape_Q":  Q,
        "splat_dim":  splat_dim,
        "hue_base":   hue_base, "gap_raw":  gap_raw,
        "sat_raw":    sat_raw,  "lit_raw":  lit_raw,
        "black":      torch.zeros(1, 3, device='cuda'),
        "white":      torch.ones(1,  3, device='cuda'),
    }

    adam_params = [
        {"params": [low_shs_w], "lr": 0.005,  "name": "low_shs_w"},
        {"params": [hue_base],  "lr": 0.01,   "name": "hue_base"},
        {"params": [gap_raw],   "lr": 0.01,   "name": "gap_raw"},
        {"params": [sat_raw],   "lr": 0.01,   "name": "sat_raw"},
        {"params": [lit_raw],   "lr": 0.01,   "name": "lit_raw"},
    ]
    if low_rank:
        adam_params += [
            {"params": [high_shs_A], "lr": 0.0005, "name": "high_shs_A"},
            {"params": [high_shs_B], "lr": 0.0005, "name": "high_shs_B"},
        ]
    else:
        adam_params += [
            {"params": [high_shs_w], "lr": 0.0005, "name": "high_shs_w"},
        ]

    return frozen, params, adam_params


# ─────────────────────────────────────────────────────────────────────────────
# Baking  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def bake_palette_edit(params, frozen, palette_rgb_new=None):
    with torch.no_grad():
        if palette_rgb_new is not None:
            P_chroma = palette_rgb_new
        else:
            P_chroma = get_palette_rgb(
                params["hue_base"], params["gap_raw"],
                params["sat_raw"],  params["lit_raw"],
            )
        device, dtype = P_chroma.device, P_chroma.dtype
        black  = torch.zeros(1, 3, device=device, dtype=dtype)
        white  = torch.ones(1,  3, device=device, dtype=dtype)
        P_full = torch.cat([black, white, P_chroma], dim=0)
        high_shs = reconstruct_high_shs(params)
        shs_w    = get_shs(params["low_shs_w"], high_shs)
        N         = shs_w.shape[0]
        splat_dim = P_full.shape[0]
        num_sh    = shs_w.shape[1] // splat_dim
        shs_w     = shs_w.reshape(N, num_sh, splat_dim)
        shs_baked = torch.einsum('nlk,kc->nlc', shs_w, P_full)
        shs_baked = shs_baked.reshape(N, num_sh * 3)
    return shs_baked