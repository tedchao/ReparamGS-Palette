"""
this file is modified from pytorch-ssim
https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from gsplat.utils import *
import gsplatcu as gsc

grey = torch.tensor([[128.0, 128.0]], device='cuda') / 255.



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, range_, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(
        img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = (0.01 * range_)**2
    C2 = (0.03 * range_)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, range_=1, window_size=11, size_average=True):
    channel = img1.shape[-3]
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, range_, size_average)

def palette_volume_loss(palette):
    # palette: (6, 3)
    palette_ = get_palette(palette)
    centered = palette_ - palette_.mean(dim=0, keepdim=True)  # (6, 3)
    G = centered @ centered.T  # (6, 6) Gram matrix
    det = torch.linalg.det(G + 1e-6 * torch.eye(G.shape[0], device=palette.device))  # regularized
    return -det  # maximize volume => minimize negative volume


############## loss functions ##############
def hue_separation_loss(gap_raw, min_gap_fraction=0.5):
    """
    Penalize hue gaps that fall below a fraction of even spacing.
    
    Args:
        gap_raw: (K,) raw gap params; softplus(gap_raw) defines un-normalized gaps,
                 which then sum-normalize to 2*pi (full color wheel).
        min_gap_fraction: minimum allowed gap as fraction of even spacing (2*pi/K).
                          0.5 means each gap must be at least 50% of equal spacing.
    
    Returns:
        scalar loss. Zero when all gaps >= min_gap; quadratic penalty otherwise.
    """
    import math
    K = gap_raw.shape[0]
    
    # Compute the same normalized gaps as get_palette_rgb does
    gaps_positive = F.softplus(gap_raw)
    gaps = (2 * math.pi) * gaps_positive / gaps_positive.sum()   # (K,) sums to 2pi
    
    even_gap = (2 * math.pi) / K
    min_gap = min_gap_fraction * even_gap
    
    violation = F.relu(min_gap - gaps)
    return (violation ** 2).mean()


def vertex_separation_loss(palette_rgb, min_dist=0.3):
    """
    Penalize chromatic palette colors collapsing onto each other.

    Args:
            palette_rgb: (K, 3) tensor of chromatic palette RGB values.
                                        (Don't pass black/white anchors — they're fixed and you
                                        want chromatic anchors to spread away from each other,
                                        not from black/white.)
            min_dist: minimum pairwise Euclidean distance in RGB space below
                                which the penalty kicks in. RGB diagonals span sqrt(3) ≈ 1.73,
                                so min_dist=0.15 is roughly 8% of full color span.

    Returns:
            scalar loss; 0 when all pairwise distances >= min_dist,
            increases quadratically as pairs get closer.
    """
    K = palette_rgb.shape[0]
    if K < 2:
            return palette_rgb.new_zeros(())

    # Pairwise distance matrix (K, K)
    diff = palette_rgb.unsqueeze(0) - palette_rgb.unsqueeze(1)   # (K, K, 3)
    dists = torch.linalg.norm(diff, dim=-1)                       # (K, K)

    # Mask upper triangle (i < j) to count each pair once and skip diagonal
    iu = torch.triu_indices(K, K, offset=1, device=palette_rgb.device)
    pair_dists = dists[iu[0], iu[1]]                              # (K*(K-1)/2,)

    # Hinge loss: penalty proportional to (min_dist - dist)^2 when too close
    violation = F.relu(min_dist - pair_dists)
    return (violation ** 2).mean()
    

def soft_palette_convex_regularizer(palette):
    
    def barycentric_coords(p, a, b, c, eps=1e-8):
        # Solve for (alpha, beta, gamma) in p = alpha*a + beta*b + gamma*c
        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = (v0 @ v0).clamp(min=eps)
        d01 = v0 @ v1
        d11 = (v1 @ v1).clamp(min=eps)
        d20 = v2 @ v0
        d21 = v2 @ v1

        denom = d00 * d11 - d01 * d01 + eps
        beta = (d11 * d20 - d01 * d21) / denom
        gamma = (d00 * d21 - d01 * d20) / denom
        alpha = 1.0 - beta - gamma
        return torch.stack([alpha, beta, gamma])
    
    """
    palette: (4,2) tensor (AB colors)
    returns: scalar convexity loss
    """
    num_colors = palette.shape[0]
    loss = 0.0
    idx = list(range(num_colors))

    for i in range(num_colors):
        others = idx[:i] + idx[i+1:]
        p = palette[i]
        a, b, c = (palette[others[0]], palette[others[1]], palette[others[2]])

        bary = barycentric_coords(p, a, b, c)  # (3,)
        # Only penalize if point is inside (all barycentrics >= 0)
        penalty = torch.relu(bary.min())
        loss += penalty

    return loss / num_colors

def palette_regularizer(palette):
    return vertex_separation_regularizer(palette) #+ convex * soft_palette_convex_regularizer(palette)

def grey_penalization(layer, mask):
    if mask == None:    # colmap or general scene
        return layer[0, :, :].mean()
    else:               # for blender synthetic white background dataset
        return (layer[0, :, :] * mask).sum() / mask.sum()

'''
def weights_sparsity_loss(layer):
    chroma_w = layer[1:, :, :]
    chroma_w = chroma_w / (chroma_w.sum(dim=0, keepdim=True) + 1e-8)
    NM = 1 / chroma_w.numel()
    return 1 + NM * torch.sum(- (1.0 - chroma_w) ** 2)
'''

def weights_sparsity_loss(layer, eps=1e-8):
    """
    Sparsity loss based on effective number of active chromas.
    Encourages one-hot chroma weights, scale-aware, no normalization.
    [Aksoy et al. 2017] Unmixing-Based Soft Color Segmentation for Image Manipulation.
    """
    chroma_w = layer[1:, :, :]  # (4, N, M)
    #chroma_w = layer
    s1 = chroma_w.sum(dim=0)
    s2 = (chroma_w ** 2).sum(dim=0) + eps
    loss = (s1 / s2) - 1.0
    return loss.mean()

def chroma_loss(layer, L0, gt_image, ab_palette_outer, loss_lambda=0.2):
    """
    Compute reconstruction loss with ΔE and SSIM on full Lab image
    
    Args:
        layer: (num_palette+1, H, W) predicted weights
        L0: (1, H, W) predicted lightness channel
        gt_image: (3, H, W) ground truth Lab image in [0,1]
        ab_palette_outer: (num_palette, 2) AB palette colors
        loss_lambda: weight for SSIM (default 0.2)
    
    Returns:
        scalar loss value
    """
    # 1. Reconstruct AB from weights
    grey = torch.tensor([[0.5, 0.5]], device=ab_palette_outer.device)
    ab_palette_full = torch.cat([grey, ab_palette_outer], dim=0)  # (num_palette+1, 2)
    
    # Reconstruct AB: (2, H, W) = (num_palette+1, 2).T @ (num_palette+1, H, W)
    ab_recon = torch.einsum('pc,phw->chw', ab_palette_full, layer)  # (2, H, W)
    
    # 2. Reconstruct full Lab image
    lab_recon = torch.cat([L0, ab_recon], dim=0)  # (3, H, W)
    
    # 3. ΔE loss (Euclidean distance in Lab space)
    delta_e_loss = torch.norm(lab_recon - gt_image, dim=0).mean()
    
    # 4. SSIM on Lab image (no unsqueeze needed - your SSIM handles 3D)
    ssim_loss = 1.0 - ssim(lab_recon, gt_image)
    
    # 5. Combine: 80% ΔE + 20% SSIM
    return (1.0 - loss_lambda) * delta_e_loss + loss_lambda * ssim_loss

def lightness_loss(L, gt_L):
    loss_l2_sq = ((L - gt_L) ** 2).mean()
    #loss_l1 = torch.abs(L - gt_L).mean()
    loss_ssim = 1.0 - ssim(L, gt_L)
    #return 0.8 * loss_l2_sq + 0.1 * loss_l1 + 0.1 * loss_ssim
    return 0.8 * loss_l2_sq + 0.2 * loss_ssim

'''
def lightness_loss(L, gt_L):
    """
    L2 + gradient loss for edge preservation in L channel
    This helps with structural quality without SSIM's problems
    """
    # Pixel-wise L2
    loss_l2 = ((L - gt_L) ** 2).mean()
    
    # Gradient loss (preserve edges/structure in lightness)
    diff_h_pred = L[:, :, 1:] - L[:, :, :-1]
    diff_v_pred = L[:, 1:, :] - L[:, :-1, :]
    diff_h_gt = gt_L[:, :, 1:] - gt_L[:, :, :-1]
    diff_v_gt = gt_L[:, 1:, :] - gt_L[:, :-1, :]
    
    grad_loss = (torch.abs(diff_h_pred - diff_h_gt).mean() + 
                 torch.abs(diff_v_pred - diff_v_gt).mean())
    
    return 0.7 * loss_l2 + 0.3 * grad_loss
'''
    
def weights_loss(w, gt_w):
    #return torch.abs(w - gt_w).mean()
    #return torch.norm(w - gt_w, dim=0).mean()
    return ((w - gt_w) ** 2).mean()



'''
def weights_loss(w, gt_w, L_pred, palette, image_gt_lab, loss_lambda_mse=0.5):
    """
    Weight loss with SSIM in RGB space (matches evaluation metric)
    
    Args:
        w: (num_palette+1, H, W) predicted weights in [0,1]
        gt_w: (num_palette+1, H, W) ground truth weights from barycentric (can have negatives)
        L_pred: (1, H, W) predicted lightness channel
        palette: (num_palette, 2) AB palette colors
        image_gt_lab: (3, H, W) ground truth Lab image in [0,1]
        loss_lambda_mse: weight for MSE term (default 0.5)
        loss_lambda_ssim: weight for SSIM term (default 0.5)
    
    Returns:
        scalar loss value
    """
    # 1. MSE on raw weights (preserves barycentric geometric constraint)
    mse_weights = ((w - gt_w) ** 2).mean()
    
    # 2. Reconstruct Lab image from predicted weights
    grey = torch.tensor([[0.0, 0.0]], device=palette.device)
    ab_palette_full = torch.cat([grey, palette], dim=0)  # (num_palette+1, 2)
    
    # Reconstruct AB from weights: (2, H, W) = (num_palette+1, 2).T @ (num_palette+1, H, W)
    ab_recon = torch.einsum('pc,phw->chw', ab_palette_full, w)  # (2, H, W)
    
    # Full Lab reconstruction
    lab_recon = torch.cat([L_pred, ab_recon], dim=0)  # (3, H, W)
    
    # 3. Convert Lab to RGB for SSIM (using your CUDA implementation)
    # gsc.lab2rgb expects (H, W, 3) format
    lab_recon_hwc = lab_recon.permute(1, 2, 0).contiguous()  # (H, W, 3)
    image_gt_hwc = image_gt_lab.permute(1, 2, 0).contiguous()  # (H, W, 3)
    
    rgb_recon = gsc.lab2rgb(lab_recon_hwc)  # (H, W, 3)
    rgb_gt = gsc.lab2rgb(image_gt_hwc)  # (H, W, 3)
    
    # Convert back to (C, H, W) for SSIM and add batch dimension
    rgb_recon_chw = rgb_recon.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    rgb_gt_chw = rgb_gt.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # 4. SSIM in RGB space (matches your evaluation metric!)
    ssim_rgb = 1.0 - ssim(rgb_recon_chw, rgb_gt_chw)
    
    # 5. Combine losses
    return loss_lambda_mse * mse_weights + (1-loss_lambda_mse) * ssim_rgb
'''

def sh_lightness_regularization(shs_l, L=3):
    """
    Regularization to ensure SH lightness stays in valid range [0, 1].
    
    Args:
        shs_l: SH coefficients, shape (N, 1, (L+1)^2)
        L: Maximum SH degree (default 3)
    
    Returns:
        Regularization loss (scalar)
    """
    N = shs_l.shape[0]
    
    # Extract DC component (f_00)
    f_00 = shs_l[:, 0, 0]  # Shape: (N,)
    
    # Extract higher-order components (f_ℓm for ℓ ≥ 1)
    high_order = shs_l[:, 0, 1:]  # Shape: (N, (L+1)^2 - 1)
    
    # Compute sum of squared higher-order coefficients
    # sum_{ℓ=1}^{L} sum_{m=-ℓ}^{ℓ} |f_ℓm|^2
    high_order_sum_sq = (high_order ** 2).sum(dim=1)  # Shape: (N,)
    
    # Constants
    num_coeffs = (L + 1) ** 2
    four_pi_over_n = 4 * torch.pi / num_coeffs
    
    # Constraint 1: sqrt(4π/(L+1)^2 - f_00^2) ≥ sqrt(sum |f_ℓm|^2)
    # Equivalent to: 4π/(L+1)^2 - f_00^2 ≥ sum |f_ℓm|^2
    lhs_1 = four_pi_over_n - f_00 ** 2
    rhs_1 = high_order_sum_sq
    violation_1 = torch.relu(rhs_1 - lhs_1)  # Positive if constraint violated
    
    # Constraint 2: min(4π/(L+1)^2 - f_00^2, f_00^2/((L+1)^2-1)) ≥ sum |f_ℓm|^2
    term_a = four_pi_over_n - f_00 ** 2
    term_b = f_00 ** 2 / (num_coeffs - 1)
    lhs_2 = torch.min(term_a, term_b)
    rhs_2 = high_order_sum_sq
    violation_2 = torch.relu(rhs_2 - lhs_2)  # Positive if constraint violated
    
    # Total loss: penalize violations
    loss = violation_1.mean() + violation_2.mean()
    
    return loss

if __name__ == "__main__":
    height, width = 100, 100
    image = torch.zeros([3, height, width], dtype=torch.float32).to('cuda')
    image_gt = torch.zeros([3, height, width], dtype=torch.float32).to('cuda')
    loss = gau_loss(image, image_gt)
    print(loss)
    