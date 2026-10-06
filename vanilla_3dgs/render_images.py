import torch
import numpy as np
from PIL import Image
import os
import json
import gsplatcu as gsc
import cv2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs",          required=True)
    parser.add_argument("--cam",         required=True)
    parser.add_argument("--dir",         required=True)
    parser.add_argument("--range",       required=True, help="e.g. 0,100")
    parser.add_argument("--resize_rate", type=float, default=1.0)
    parser.add_argument("--focal_scale", type=float, default=None)
    parser.add_argument("--fps",         type=int,   default=30)
    parser.add_argument("--white_bg",    action="store_true",
                        help="composite over white using accumulated alpha (for Blender models)")
    parser.add_argument("--rgba",        action="store_true",
                        help="save straight RGB + alpha as RGBA PNG (transparent background); "
                             "gives an exact alpha for downstream FLIP background masking")
    args = parser.parse_args()

    focal_scale = args.focal_scale if args.focal_scale is not None else args.resize_rate
    os.makedirs(args.dir, exist_ok=True)

    print(f"Loading {args.gs} ...")
    gs = np.load(args.gs)
    pws    = torch.from_numpy(gs['pw']).float().cuda()
    rots   = torch.from_numpy(gs['rot']).float().cuda()
    scales = torch.from_numpy(gs['scale']).float().cuda()
    alphas = torch.from_numpy(gs['alpha']).float().cuda()
    shs    = torch.from_numpy(gs['sh']).float().cuda()
    print(f"  Gaussians: {pws.shape[0]:,}")

    with open(args.cam, "r") as f:
        camera = json.load(f)

    lr, rr = int(args.range.split(',')[0]), int(args.range.split(',')[1])
    frame_paths = []

    with torch.no_grad():
        for i in range(lr, rr + 1):
            c = camera[i]
            print(f"Rendering {c['img_name']} ...")

            width  = int(c['width']  * args.resize_rate)
            height = int(c['height'] * args.resize_rate)
            fx = c['fx'] * focal_scale
            fy = c['fy'] * focal_scale
            cx = width  / 2.0
            cy = height / 2.0

            tcw = torch.from_numpy(np.array(c['tcw'])).float().cuda()
            Rcw = torch.from_numpy(np.array(c['rotation']).T).float().cuda()
            twc = torch.linalg.inv(Rcw) @ (-tcw)

            us, pcs, depths = gsc.project(pws, Rcw, tcw, fx, fy, cx, cy, False)
            cov3ds,         = gsc.computeCov3D(rots, scales, depths, False)
            cov2ds,         = gsc.computeCov2D(cov3ds, pcs, Rcw, depths,
                                                fx, fy, height, width, False)
            colors,         = gsc.sh2Color(shs, pws, twc, False)
            cinv2ds, areas  = gsc.inverseCov2D(cov2ds, depths, False)

            splat_out = gsc.splat(height, width, us, cinv2ds, alphas, depths, colors, areas)
            image     = splat_out[0]    # (3, H, W) premultiplied
            final_tau = splat_out[2]    # (H, W) — leftover transmittance

            # alpha = coverage = 1 - leftover transmittance
            alpha = (1.0 - final_tau).clamp(0.0, 1.0)   # (H, W)

            if args.rgba:
                # un-premultiply to straight RGB, keep alpha as 4th channel
                a = alpha.unsqueeze(0)                              # (1, H, W)
                rgb_straight = (image / a.clamp(min=1e-6)).clamp(0.0, 1.0)
                rgba = torch.cat([rgb_straight, a], dim=0)          # (4, H, W)
                arr = (rgba.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                out_path = os.path.join(args.dir, c['img_name'] + '.png')
                Image.fromarray(arr, mode='RGBA').save(out_path)
            else:
                if args.white_bg:
                    # Premultiplied composite over white
                    image = image + final_tau.unsqueeze(0)
                img_np = (image.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                out_path = os.path.join(args.dir, c['img_name'] + '.png')
                Image.fromarray(img_np).save(out_path)

            frame_paths.append((out_path, height, width))

    print(f"\nCreating video from {len(frame_paths)} frames ...")
    _, H, W = frame_paths[0]
    video_path = os.path.join(args.dir, 'render.mp4')
    writer = cv2.VideoWriter(video_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             args.fps, (W, H))
    for path, _, _ in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # cv2 writer needs 3-channel BGR; if RGBA, composite over white for the video
        if frame is not None and frame.shape[-1] == 4:
            bgr   = frame[:, :, :3].astype(np.float32)
            a     = frame[:, :, 3:4].astype(np.float32) / 255.0
            frame = (bgr * a + 255.0 * (1.0 - a)).clip(0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    print(f"Video saved to {video_path}")
    print(f"Done. {len(frame_paths)} frames saved to {args.dir}")