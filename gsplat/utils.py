import torch
import gsplatcu as gsc
import collections
import numpy as np
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    this file is copied from Plenoxels
    https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/opt/util/util.py#L78

    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def rotate_vector_by_quaternion(q, v):
    q = torch.nn.functional.normalize(q)
    u = q[:, 1:, np.newaxis]
    s = q[:, 0, np.newaxis, np.newaxis]
    v = v[:, :,  np.newaxis]
    v_prime = 2.0 * u * (u.permute(0, 2, 1) @ v) +\
        v * (s*s - (u.permute(0, 2, 1) @ u)) +\
        2.0 * torch.linalg.cross(u, v, dim=1) * s
    return v_prime.squeeze()


def compute_cov_3d_torch(scale, q):
    # Create scaling matrix
    S = torch.zeros([scale.shape[0], 3, 3], device='cuda')
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]
    # Normalize quaternion to get valid rotation
    q = torch.nn.functional.normalize(q)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Compute rotation matrix from quaternion
    R = torch.stack([
        1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x * z + y * w),
        2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)
    ], dim=1).reshape(-1, 3, 3)
    M = R @ S

    # Compute 3D world covariance matrix Sigma
    Sigma = M @ M.permute(0, 2, 1)

    return Sigma


def rainbow(scalars, scalar_min=0, scalar_max=255):
    range = scalar_max - scalar_min
    values = 1.0 - (scalars - scalar_min) / range
    # values = (scalars - scalar_min) / range  # using inverted color
    colors = torch.zeros([scalars.shape[0], 3], dtype=torch.float32, device='cuda')
    values = torch.clip(values, 0, 1)

    h = values * 5.0 + 1.0
    i = torch.floor(h).to(torch.int32)
    f = h - i
    f[torch.logical_not(i % 2)] = 1 - f[torch.logical_not(i % 2)]
    n = 1 - f

    # idx = i <= 1
    colors[i <= 1, 0] = n[i <= 1]
    colors[i <= 1, 1] = 0
    colors[i <= 1, 2] = 1

    colors[i == 2, 0] = 0
    colors[i == 2, 1] = n[i == 2]
    colors[i == 2, 2] = 1

    colors[i == 3, 0] = 0
    colors[i == 3, 1] = 1
    colors[i == 3, 2] = n[i == 3]

    colors[i == 4, 0] = n[i == 4]
    colors[i == 4, 1] = 1
    colors[i == 4, 2] = 0

    colors[i >= 5, 0] = 1
    colors[i >= 5, 1] = n[i >= 5]
    colors[i >= 5, 2] = 0
    shs = (colors - 0.5) / 0.28209479177387814
    return shs


def get_alphas_raw(x):
    """
    inverse of sigmoid
    """
    if isinstance(x, float):
        return np.log(x/(1-x))
    else:
        return torch.log(x/(1-x))


def get_alphas(x):
    return torch.sigmoid(x)


def get_scales_raw(x):
    if isinstance(x, float):
        return np.log(x)
    else:
        return torch.log(x)


def get_scales(x):
    return torch.exp(x)


def get_rots(x):
    return torch.nn.functional.normalize(x)


def get_ab_palette(x): 
    return torch.sigmoid(x)


def get_shs(low_shs, high_shs):
    return torch.cat((low_shs, high_shs), dim=1)


############
def polygon_area(pts):
    """
    Compute area of a polygon (shoelace formula).
    pts: (N,2) tensor of vertices in order (CCW or CW).
    Returns scalar area (always positive).
    """
    x = pts[:,0]
    y = pts[:,1]
    area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1)) - torch.sum(y * torch.roll(x, -1)))
    return area

def save_palette_plot(ab_palette_outer, epoch, save_dir="palettes", connect_outer=True, show_values=True):
    """
    Visualize and save the palette at a given epoch with grid + coordinate labels.

    Args:
        ab_palette_outer (torch.Tensor): (num_palette-1, 2) tensor of palette points in [0,1]
        epoch (int): current epoch number
        save_dir (str): folder to save PNGs
        connect_outer (bool): whether to connect outer vertices into a polygon
        show_values (bool): whether to display (x, y) text near each point

    Effect:
        Saves a PNG to f"{save_dir}/palette_epoch{epoch:04d}.png"
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to CPU numpy
    outer = ab_palette_outer.detach().cpu().numpy()
    cx, cy = 0.5, 0.5
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Palette at Epoch {epoch}", fontsize=11, pad=10)
    
    # ✅ Add background grid and ticks
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, linestyle="--", color="lightgray", alpha=0.7)
    ax.tick_params(axis="both", labelsize=8)
    
    # Draw square domain
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="lightgray", lw=1.2)
    
    # Draw center
    ax.scatter([cx], [cy], color="gray", s=80, zorder=3, label="Center")
    if show_values:
        ax.text(cx + 0.01, cy + 0.01, f"({cx:.2f},{cy:.2f})", fontsize=8, color="gray")
        
    # Draw outer points
    ax.scatter(outer[:, 0], outer[:, 1], color="tab:red", s=60, zorder=3, label="Palette Points")
    
    # Connect center to each outer vertex
    for (x, y) in outer:
        ax.plot([cx, x], [cy, y], color="gray", lw=1.2, alpha=0.8)
        
    # Optionally connect outer points in CCW order
    if connect_outer:
        closed_outer = np.vstack([outer, outer[0]])  # close polygon
        ax.plot(closed_outer[:, 0], closed_outer[:, 1], color="tab:blue", lw=1.2, alpha=0.8)
        
    # ✅ Annotate numeric values
    if show_values:
        for i, (x, y) in enumerate(outer):
            ax.text(x + 0.01, y + 0.01, f"{i}: ({x:.2f},{y:.2f})", fontsize=8, color="tab:red")
            
    ax.legend(fontsize=8, loc="upper right")
    
    # Save
    save_path = os.path.join(save_dir, f"palette_epoch{epoch:04d}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved palette visualization to {save_path}")
    
def init_polar_palette(num_palette, base_radius=0.25, device='cuda'):
    """
    Initialize palette with evenly spaced points around (0.5, 0.5).

    Returns:
        radii_raw: (num_palette-1,)
        deltas_raw: (num_palette-1,)
    """
    outer_size = num_palette - 1
    delta = 2 * torch.pi / outer_size
    
    # Each delta will yield equal angular spacing after softplus normalization
    deltas = torch.full((outer_size,), delta)
    deltas_raw = torch.log(torch.exp(deltas) - 1.0).to(device)  # inverse softplus
    
    # radii_raw controls distance before sigmoid -> roughly around base_radius
    # Need to invert the sigmoid effect applied later (sigmoid(outer))
    # We can approximate: sigmoid(0.5 + r_raw*cos(theta)) ≈ 0.5 + base_radius*cos(theta)
    # => radii_raw ≈ base_radius / sigmoid'(0.5) ≈ base_radius / 0.25 = 4*base_radius
    radii_raw = torch.full((outer_size,), 4 * base_radius).to(device)
    
    return radii_raw, deltas_raw

# Convert from polar to cartesian coordinates for ab palette
def polar_to_cart_ab_palette(radii_raw, deltas_raw):
    """
    radii_raw: (4,) unconstrained radii
    deltas_raw: (4,) unconstrained angle increments
    Returns:
        P: (5,2) palette, P[0]=center, P[1:]=outer CCW vertices
    """

    def safe_radius(r_raw, theta, eps=1e-6):
        """
        Constrain radius so that point stays inside [0,1]^2
        centered at z = (0.5,0.5).
        
        r_raw: (N,) unconstrained radii
        theta: (N,) angles in radians
        Returns: (N,) radii in [0, r_max(theta)]
        """
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # max allowed distances along x and y directions
        max_x = 0.5 / (cos_t.abs() + eps)
        max_y = 0.5 / (sin_t.abs() + eps)

        r_max = torch.min(max_x, max_y)

        # squash r_raw into [0, r_max] with sigmoid
        return r_max * torch.sigmoid(r_raw)
        #return r_max * (2 / torch.pi) * torch.atan(torch.exp(r_raw))
    
    # Center point
    z = torch.tensor([[0.5, 0.5]], device='cuda')

    # Make deltas positive and normalize to 2π
    deltas = F.softplus(deltas_raw)
    deltas = 2 * torch.pi * deltas / deltas.sum()

    # Cumulative CCW angles
    thetas = torch.cumsum(deltas, dim=0) - deltas[0]
    
    # Safe radii (constrained by direction)
    radii = safe_radius(radii_raw, thetas)

    # Cartesian coordinates
    x = radii * torch.cos(thetas)
    y = radii * torch.sin(thetas)
    outer = torch.stack([x, y], dim=1) + z  # (4,2)

    #outer = torch.sigmoid(outer)  # ensure within [0,1]
    return outer

# Differentiable barycentric coordinates for N points with respect to the polygon by computing each CCW triangle
def barycentric_coords(C, P, tau=1e-7, eps=1e-9):
    """
    Compute soft barycentric-like weights for an arbitrary convex polygon.

    Args:
        C: (2, H, W) tensor of ab channels
        P: (p, 2) palette vertices in CCW order (outer polygon, no center)
        tau: softmax temperature for wedge selection
        eps: numerical stability

    Returns:
        W: (p+1, H, W) weights for [center, P0, P1, ..., P_{p-1}]
    """
    device, dtype = P.device, P.dtype
    p = P.shape[0]
    assert p >= 3, "P must have at least 3 vertices"
    
    # fixed center (keep your original behavior)
    z = torch.tensor([0.5, 0.5], device=device, dtype=dtype)  # (2,)
    P_aug = torch.cat([z.unsqueeze(0), P], dim=0)             # (p+1, 2)
    
    H, W = C.shape[1:]
    N = H * W
    
    # (H, W, 2) -> (N, 2)
    C_flat = C.permute(1, 2, 0).reshape(N, 2)                 # (N, 2)
    
    # Wedges: (z, P[i], P[i+1]) for i=0..p-1  (wrap at end)
    outer = P                                                  # (p, 2)
    v0s = (outer - z).unsqueeze(0)                             # (1, p, 2)
    v1s = (torch.roll(outer, -1, 0) - z).unsqueeze(0)      # (1, p, 2)
    v2s = (C_flat - z).unsqueeze(1)                            # (N, 1, 2)
    
    # Dot products
    d00 = (v0s * v0s).sum(-1)          # (1, p)
    d01 = (v0s * v1s).sum(-1)          # (1, p)
    d11 = (v1s * v1s).sum(-1)          # (1, p)
    d20 = (v2s * v0s).sum(-1)          # (N, p)
    d21 = (v2s * v1s).sum(-1)          # (N, p)
    
    denom = d00 * d11 - d01 * d01 + eps         # (1, p)
    beta  = (d11 * d20 - d01 * d21) / denom     # (N, p)
    gamma = (d00 * d21 - d01 * d20) / denom     # (N, p)
    alpha = 1.0 - beta - gamma                  # (N, p)
    
    # Build raw per-wedge weights into unified (N, p, p+1)
    # channel 0: center (alpha)
    # channel i+1: vertex i gets beta of wedge i
    # channel (i+1)%p + 1: next vertex gets gamma of wedge i
    raw_weights = torch.zeros(N, p, p + 1, device=device, dtype=dtype)
    
    # center channel
    raw_weights[:, :, 0] = alpha  # (N, p)
    
    # indices for beta/gamma channels
    idx = torch.arange(p, device=device)
    c_beta  = idx + 1                       # (p,) -> channels 1..p
    c_gamma = ((idx + 1) % p) + 1           # (p,) -> channels 1..p (wrapped)
    
    # scatter beta and gamma per wedge i
    raw_weights[:, idx, c_beta]  = beta     # (N, p) assigned to (N, p)
    raw_weights[:, idx, c_gamma] = gamma    # (N, p) assigned to (N, p)
    
    # Projection for error estimation (softplus, renormalize per wedge)
    alpha_p = F.softplus(alpha)
    beta_p  = F.softplus(beta)
    gamma_p = F.softplus(gamma)
    denom_p = alpha_p + beta_p + gamma_p + eps
    alpha_p = alpha_p / denom_p
    beta_p  = beta_p  / denom_p
    gamma_p = gamma_p / denom_p
    
    proj_weights = torch.zeros_like(raw_weights)  # (N, p, p+1)
    proj_weights[:, :, 0] = alpha_p
    proj_weights[:, idx, c_beta]  = beta_p
    proj_weights[:, idx, c_gamma] = gamma_p
    
    # Reconstruct projected colors per wedge and compute errors
    # (N, p, p+1) @ (p+1, 2) -> (N, p, 2)
    recon = torch.matmul(proj_weights, P_aug)                 # (N, p, 2)
    errors = ((recon - C_flat.unsqueeze(1)) ** 2).sum(-1)     # (N, p)
    
    # Soft wedge selection across p wedges
    probs = F.softmax(-errors / tau, dim=1)                   # (N, p)
    
    # Blend raw barycentric weights across wedges -> (N, p+1)
    W_flat = (probs.unsqueeze(-1) * raw_weights).sum(dim=1)   # (N, p+1)
    
    # (N, p+1) -> (p+1, H, W)
    W = W_flat.T.reshape(p + 1, H, W)
    return W

'''
def barycentric_coords_(C, P, tau=1e-7, eps=1e-9):
    """
    Compute barycentric weights for image grid.

    Args:
        C: (2,H,W) tensor of ab channels
        P: (4,2) palette, P=outer CCW vertices
        tau: softmax temperature
        eps: numerical stability

    Returns:
        W: (5,H,W) barycentric weights
    """
    z = torch.tensor([0.5, 0.5], device=P.device, dtype=P.dtype)  # fixed center
    P = torch.cat([z.unsqueeze(0), P], dim=0)  # (5,2)

    H, W = C.shape[1:]
    
    # Flatten image -> (N,2)
    C_flat = C.permute(1,2,0).reshape(-1,2)  # (H*W,2)

    # Shift into coordinates relative to the center
    outer = P[1:]  # (4,2)
    v0s = (outer - z).unsqueeze(0)                # (1,4,2)
    v1s = (torch.roll(outer, -1, 0) - z).unsqueeze(0)  # (1,4,2)
    v2s = (C_flat - z).unsqueeze(1)               # (N,1,2)

    # Dot products
    d00 = (v0s * v0s).sum(-1)    # (1,4)
    d01 = (v0s * v1s).sum(-1)    # (1,4)
    d11 = (v1s * v1s).sum(-1)    # (1,4)
    d20 = (v2s * v0s).sum(-1)    # (N,4)
    d21 = (v2s * v1s).sum(-1)    # (N,4)

    denom = d00 * d11 - d01 * d01 + eps
    beta  = (d11 * d20 - d01 * d21) / denom   # (N,4)
    gamma = (d00 * d21 - d01 * d20) / denom   # (N,4)
    alpha = 1 - beta - gamma                  # (N,4)

    # Build raw weights (N,4,5)
    w0 = torch.stack([alpha[:,0], beta[:,0], gamma[:,0], torch.zeros_like(alpha[:,0]), torch.zeros_like(alpha[:,0])], dim=-1)
    w1 = torch.stack([alpha[:,1], torch.zeros_like(alpha[:,1]), beta[:,1], gamma[:,1], torch.zeros_like(alpha[:,1])], dim=-1)
    w2 = torch.stack([alpha[:,2], torch.zeros_like(alpha[:,2]), torch.zeros_like(alpha[:,2]), beta[:,2], gamma[:,2]], dim=-1)
    w3 = torch.stack([alpha[:,3], gamma[:,3], torch.zeros_like(alpha[:,3]), torch.zeros_like(alpha[:,3]), beta[:,3]], dim=-1)
    raw_weights = torch.stack([w0, w1, w2, w3], dim=1)  # (N,4,5)

    # Projection for error estimation
    alpha_proj = F.softplus(alpha)
    beta_proj  = F.softplus(beta)
    gamma_proj = F.softplus(gamma)
    denom_proj = alpha_proj + beta_proj + gamma_proj + eps
    alpha_proj /= denom_proj
    beta_proj  /= denom_proj
    gamma_proj /= denom_proj
    
    w0p = torch.stack([alpha_proj[:,0], beta_proj[:,0], gamma_proj[:,0], torch.zeros_like(alpha_proj[:,0]), torch.zeros_like(alpha_proj[:,0])], dim=-1)
    w1p = torch.stack([alpha_proj[:,1], torch.zeros_like(alpha_proj[:,1]), beta_proj[:,1], gamma_proj[:,1], torch.zeros_like(alpha_proj[:,1])], dim=-1)
    w2p = torch.stack([alpha_proj[:,2], torch.zeros_like(alpha_proj[:,2]), torch.zeros_like(alpha_proj[:,2]), beta_proj[:,2], gamma_proj[:,2]], dim=-1)
    w3p = torch.stack([alpha_proj[:,3], gamma_proj[:,3], torch.zeros_like(alpha_proj[:,3]), torch.zeros_like(alpha_proj[:,3]), beta_proj[:,3]], dim=-1)
    proj_weights = torch.stack([w0p, w1p, w2p, w3p], dim=1)  # (N,4,5)

    # Reconstruct projections + compute errors
    recon = proj_weights @ P   # (N,4,2)
    errors = ((recon - C_flat.unsqueeze(1))**2).sum(-1)  # (N,4)

    # Soft wedge selection
    probs = F.softmax(-errors/tau, dim=1)  # (N,4)

    # Blend raw barycentrics
    W_flat = (probs.unsqueeze(-1) * raw_weights).sum(1)  # (N,5)

    # Reshape back to (5,H,W)
    W = W_flat.T.reshape(5, H, W)
    
    return W
'''