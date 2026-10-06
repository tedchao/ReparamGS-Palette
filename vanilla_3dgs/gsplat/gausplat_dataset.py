import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image as PILImage

from gsplat.read_write_model import *


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw, path):
        self.id = id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw
        self.path = path


class GSplatDataset(Dataset):
    """
    COLMAP path: __getitem__ returns (camera, image_rgb_3ch).
    Blender path: __getitem__ returns (camera, image_rgba_4ch) with premultiplied alpha.

    For Blender, GT is RGBA where:
      channels 0-2 = R*A, G*A, B*A   (premultiplied — black where transparent)
      channel  3   = A                (1 inside object, 0 outside)

    Initialization for training split:
      - COLMAP: read points3D.bin (cached as points3D.npy on first load)
      - Blender: if points3D.npy exists at the dataset root, use it as
                 warm-start init; otherwise fall back to random init.
    """
    def __init__(self, path, resize_rate=1, device='cuda',
                 split='train', test_every=8,
                 dataset_type='auto',
                 num_random_points=100_000) -> None:
        super().__init__()
        self.device = device
        self.resize_rate = resize_rate
        self.split = split
        self.test_every = test_every
        self.num_random_points = num_random_points

        if dataset_type == 'auto':
            if (Path(path) / 'transforms_train.json').exists():
                dataset_type = 'blender'
            elif (Path(path) / 'sparse' / '0').exists():
                dataset_type = 'colmap'
            else:
                raise ValueError(f"Cannot auto-detect dataset type for {path}")
        self.dataset_type = dataset_type

        if dataset_type == 'blender':
            self._load_blender(path, split)
        elif dataset_type == 'colmap':
            self._load_colmap(path, split, test_every)
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

    # ------------------------------------------------------------------
    def _load_colmap(self, path, split, test_every):
        camera_params, image_params = read_model(Path(path, "sparse/0"), ext='.bin')

        all_cameras, all_images = [], []
        for image_param in image_params.values():
            camera_param = camera_params[image_param.camera_id]
            im_path = str(Path(path, "images", image_param.name))
            image = PILImage.open(im_path).convert('RGB')

            if self.resize_rate != 1:
                new_w = int(image.width * self.resize_rate)
                new_h = int(image.height * self.resize_rate)
                image = image.resize((new_w, new_h), PILImage.LANCZOS)

            w_scale = image.width / camera_param.width
            h_scale = image.height / camera_param.height
            fx = camera_param.params[0] * w_scale
            fy = camera_param.params[1] * h_scale
            cx = camera_param.params[2] * w_scale
            cy = camera_param.params[3] * h_scale

            Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(self.device).float()
            tcw = torch.from_numpy(image_param.tvec).to(self.device).float()
            camera = Camera(image_param.id, image.width, image.height,
                            fx, fy, cx, cy, Rcw, tcw, im_path)
            img_tensor = torchvision.transforms.functional.to_tensor(image).to(self.device).float()

            all_cameras.append(camera)
            all_images.append(img_tensor)

        total = len(all_cameras)
        if split == 'train':
            indices = [i for i in range(total) if i % test_every != 0]
        elif split == 'test':
            indices = [i for i in range(total) if i % test_every == 0]
        elif split == 'all':
            indices = list(range(total))
        else:
            raise ValueError(f"Invalid split for colmap: {split}")

        self.cameras = [all_cameras[i] for i in indices]
        self.images  = [all_images[i]  for i in indices]

        print(f"\n{'='*60}\nDataset (COLMAP): {path}")
        print(f"Total: {total} | Split: {split} ({len(indices)}) | test_every={test_every}")
        print(f"Resize rate: {self.resize_rate}\n{'='*60}\n")

        all_twcs = torch.stack([-torch.linalg.inv(c.Rcw) @ c.tcw for c in all_cameras])
        cam_dist = torch.linalg.norm(all_twcs - all_twcs.mean(dim=0), dim=1)
        self.sence_size = float(cam_dist.max()) * 1.1
        print(f"Scene size: {self.sence_size:.4f}")

        if split == 'train':
            try:
                self.gs = np.load(Path(path, "sparse/0/points3D.npy"))
            except:
                self.gs = read_points_bin_as_gau(Path(path, "sparse/0/points3D.bin"))
                np.save(Path(path, "sparse/0/points3D.npy"), self.gs)
            print(f"Loaded {len(self.gs['pw'])} COLMAP points\n")
        else:
            self.gs = None

    # ------------------------------------------------------------------
    def _load_blender(self, path, split):
        if split == 'train':
            transform_file = 'transforms_train.json'
        elif split == 'test':
            transform_file = 'transforms_test.json'
        elif split == 'val':
            transform_file = 'transforms_val.json'
        else:
            raise ValueError(f"Invalid split for blender: {split}")

        with open(Path(path) / transform_file, 'r') as f:
            meta = json.load(f)

        camera_angle_x = meta['camera_angle_x']
        self.cameras, self.images = [], []

        for idx, frame in enumerate(meta['frames']):
            file_path = frame['file_path']
            if not file_path.endswith('.png'):
                file_path += '.png'

            possible_paths = [
                Path(path) / file_path,
                Path(path) / file_path.replace('./', ''),
                Path(path) / Path(file_path).name,
            ]
            im_path = next((p for p in possible_paths if p.exists()), None)
            if im_path is None:
                raise FileNotFoundError(f"Image not found. Tried: {possible_paths}")

            image_pil = PILImage.open(im_path).convert('RGBA')
            if self.resize_rate != 1:
                new_w = int(image_pil.width * self.resize_rate)
                new_h = int(image_pil.height * self.resize_rate)
                image_pil = image_pil.resize((new_w, new_h), PILImage.LANCZOS)

            # Premultiplied RGBA: keep alpha channel, premultiply RGB by alpha
            image_np = np.asarray(image_pil).astype(np.float32) / 255.0   # (H, W, 4)
            rgb   = image_np[..., :3]
            alpha = image_np[..., 3:4]
            rgb_premult = rgb * alpha
            image_np = np.concatenate([rgb_premult, alpha], axis=-1)       # (H, W, 4)

            # (H, W, 4) -> (4, H, W)
            img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device).float()

            width, height = image_pil.width, image_pil.height
            focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
            fx = fy = focal
            cx, cy = width / 2.0, height / 2.0

            # Blender (OpenGL) c2w → OpenCV w2c
            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            Rcw = torch.from_numpy(w2c[:3, :3]).to(self.device).float()
            tcw = torch.from_numpy(w2c[:3, 3]).to(self.device).float()

            self.cameras.append(Camera(idx, width, height, fx, fy, cx, cy, Rcw, tcw, str(im_path)))
            self.images.append(img_tensor)

        print(f"\n{'='*60}\nDataset (Blender, RGBA premultiplied): {path}")
        print(f"Split: {split} ({len(self.cameras)}) | {width}x{height} | focal={focal:.2f}")
        print(f"Resize rate: {self.resize_rate}\n{'='*60}\n")

        twcs = torch.stack([c.twc for c in self.cameras])
        cam_dist = torch.linalg.norm(twcs - twcs.mean(dim=0), dim=1)
        self.sence_size = float(cam_dist.max()) * 1.1
        print(f"Scene size: {self.sence_size:.4f}")

        if split == 'train':
            bonus_path = Path(path) / 'points3D.npy'
            if bonus_path.exists():
                # Warm-start init from a previously-trained PLY's gaussian centers.
                # Format matches what _load_colmap produces / caches:
                #   {'pw': (N,3), 'sh': (N,48), 'scale': (N,3),
                #    'rot': (N,4), 'alpha': (N,) or (N,1)}
                self.gs = np.load(bonus_path, allow_pickle=True).item()
                N = len(self.gs['pw'])
                print(f"Warm-start init from points3D.npy: {N} points\n")
            else:
                self.gs = self._random_initialization(self.num_random_points)
                print(f"Random init: {self.num_random_points} points\n")
        else:
            self.gs = None

    def _random_initialization(self, num_points):
        points = (np.random.random((num_points, 3)).astype(np.float32) * 2.6) - 1.3
        
        # Compute scale from k-nearest-neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=4)   # k=4: self + 3 nearest neighbors
        avg_nn_dist = dists[:, 1:].mean(axis=1)   # exclude self (idx 0)
        
        # Scale = max(avg_nn_dist, eps) replicated for 3 axes
        scale = np.tile(avg_nn_dist[:, None], (1, 3)).astype(np.float32)
        scale = np.clip(scale, 1e-3, None)   # min scale 0.001
        
        return {
            'pw':    points,
            'sh':    np.zeros((num_points, 48), dtype=np.float32),
            'scale': scale,
            'rot':   np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (num_points, 1)),
            'alpha': np.full((num_points,), 0.1, dtype=np.float32),
        }
    
    def _initialize_sh_from_views(self, points, cameras, images, sh_dim=48):
        """
        For each point, find a view it's visible in and set DC to the projected pixel color.
        """
        SH_C0_0 = 0.28209479177387814
        
        points_t = torch.from_numpy(points).cuda().float()   # (N, 3)
        N = points_t.shape[0]
        sh = torch.zeros(N, sh_dim, device='cuda')   # band-major, (N, 48)
        
        # For each point, find its color in the first view it's visible in
        color_set = torch.zeros(N, 3, device='cuda')
        found = torch.zeros(N, dtype=torch.bool, device='cuda')
        
        for cam, image in zip(cameras, images):
            unfound = ~found
            if not unfound.any():
                break
            
            active_points = points_t[unfound]
            cam_xyz = (cam.Rcw @ active_points.T).T + cam.tcw
            z = cam_xyz[:, 2]
            valid = z > 0.01
            u = cam.fx * cam_xyz[:, 0] / z + cam.cx
            v = cam.fy * cam_xyz[:, 1] / z + cam.cy
            
            H, W = image.shape[1], image.shape[2]
            in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            
            # If image has alpha, also check it's foreground
            if image.shape[0] == 4:
                alpha = image[3]
                u_c = u.clamp(0, W - 1).long()
                v_c = v.clamp(0, H - 1).long()
                alpha_at_proj = alpha[v_c, u_c]
                in_bounds = in_bounds & (alpha_at_proj > 0.5)
                
            # For points where in_bounds, sample color from image (premultiplied for blender)
            u_c = u.clamp(0, W - 1).long()
            v_c = v.clamp(0, H - 1).long()
            rgb_at_proj = image[:3, v_c, u_c].T   # (M, 3)
            
            # Update color and found mask for newly-found points
            unfound_indices = torch.where(unfound)[0]
            new_found = unfound_indices[in_bounds]
            color_set[new_found] = rgb_at_proj[in_bounds]
            found[new_found] = True
            
        # For remaining unfound points (occluded everywhere), use mean color
        if (~found).any():
            mean_color = color_set[found].mean(dim=0) if found.any() else torch.tensor([0.5, 0.5, 0.5], device='cuda')
            color_set[~found] = mean_color
            
        # Convert RGB color to DC SH coefficient: color = 0.5 + SH_C0_0 * DC
        # → DC = (color - 0.5) / SH_C0_0
        dc = (color_set - 0.5) / SH_C0_0   # (N, 3)
        
        # SH layout: assume band-major, channels = 16 coefficients × 3 RGB
        # First 3 entries = DC for R, G, B
        sh[:, :3] = dc
        return sh.cpu().numpy().astype(np.float32)
    
    
    def __getitem__(self, index):
        return self.cameras[index], self.images[index]

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    train_set = GSplatDataset(path, split='train', test_every=8, resize_rate=1.0)
    test_set  = GSplatDataset(path, split='test',  test_every=8, resize_rate=1.0)
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")