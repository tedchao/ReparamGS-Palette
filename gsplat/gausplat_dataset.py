import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image as PILImage
from skimage.color import rgb2lab

from gsplat.read_write_model import *


def normalize_lab(lab_image):
    L = lab_image[..., 0] / 100.0
    a = (lab_image[..., 1] + 128.0) / 255.0
    b = (lab_image[..., 2] + 128.0) / 255.0
    return np.stack([L, a, b], axis=-1)


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
    __getitem__ returns (camera, image_3ch, alpha_or_None).
      - COLMAP: (cam, rgb_3ch, None)
      - Blender: (cam, premultiplied_rgb_3ch, alpha_1ch)
    """
    def __init__(self, path, color_space='rgb', resize_rate=1, device='cuda',
                 split='train', test_every=8,
                 dataset_type='auto') -> None:
        super().__init__()
        self.device = device
        self.resize_rate = resize_rate
        self.split = split
        self.test_every = test_every
        self.color_space = color_space

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
            self._load_colmap(path, split, test_every, color_space)
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

    # ------------------------------------------------------------------
    def _load_colmap(self, path, split, test_every, color_space):
        camera_params, image_params = read_model(Path(path, "sparse/0"), ext='.bin')

        all_cameras, all_images = [], []
        for image_param in image_params.values():
            i = image_param.camera_id
            camera_param = camera_params[i]
            im_path = str(Path(path, "images", image_param.name))

            image_pil = PILImage.open(im_path).convert('RGB')
            if self.resize_rate != 1:
                new_size = (int(image_pil.width * self.resize_rate),
                            int(image_pil.height * self.resize_rate))
                image_pil = image_pil.resize(new_size, PILImage.LANCZOS)

            image_np = np.asarray(image_pil).astype(np.float32) / 255.0
            if color_space == 'lab':
                image_np = normalize_lab(rgb2lab(image_np))

            w_scale = image_pil.width / camera_param.width
            h_scale = image_pil.height / camera_param.height
            fx = camera_param.params[0] * w_scale
            fy = camera_param.params[1] * h_scale
            cx = camera_param.params[2] * w_scale
            cy = camera_param.params[3] * h_scale

            Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(self.device).float()
            tcw = torch.from_numpy(image_param.tvec).to(self.device).float()
            camera = Camera(image_param.id, image_pil.width, image_pil.height,
                            fx, fy, cx, cy, Rcw, tcw, im_path)
            image = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device).float()

            all_cameras.append(camera)
            all_images.append(image)

        total = len(all_cameras)
        if split == 'train':
            indices = [i for i in range(total) if i % test_every != 0]
        elif split == 'test':
            indices = [i for i in range(total) if i % test_every == 0]
        elif split == 'all':
            indices = list(range(total))
        else:
            raise ValueError(f"Invalid split: {split}")

        self.cameras = [all_cameras[i] for i in indices]
        self.images  = [all_images[i]  for i in indices]
        self.alphas  = None  # COLMAP has no alpha

        print(f"\n{'='*60}\nDataset (COLMAP, {color_space.upper()}): {path}")
        print(f"Total: {total} | Split: {split} ({len(indices)}) | test_every={test_every}")
        print(f"Resize rate: {self.resize_rate}\n{'='*60}\n")

        all_twcs = torch.stack([cam.twc for cam in all_cameras])
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
        self.cameras, self.images, self.alphas = [], [], []

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

            image_np = np.asarray(image_pil).astype(np.float32) / 255.0   # (H, W, 4)
            rgb   = image_np[..., :3]
            alpha = image_np[..., 3]                                       # (H, W)
            rgb_premult = rgb * alpha[..., None]                           # premultiplied 3-ch

            img_tensor   = torch.from_numpy(rgb_premult).permute(2, 0, 1).to(self.device).float()
            alpha_tensor = torch.from_numpy(alpha).unsqueeze(0).to(self.device).float()  # (1, H, W)

            width, height = image_pil.width, image_pil.height
            focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
            fx = fy = focal
            cx, cy = width / 2.0, height / 2.0

            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            Rcw = torch.from_numpy(w2c[:3, :3]).to(self.device).float()
            tcw = torch.from_numpy(w2c[:3, 3]).to(self.device).float()

            self.cameras.append(Camera(idx, width, height, fx, fy, cx, cy, Rcw, tcw, str(im_path)))
            self.images.append(img_tensor)
            self.alphas.append(alpha_tensor)

        print(f"\n{'='*60}\nDataset (Blender, premultiplied RGB+alpha): {path}")
        print(f"Split: {split} ({len(self.cameras)}) | {width}x{height} | focal={focal:.2f}")
        print(f"Resize rate: {self.resize_rate}\n{'='*60}\n")

        twcs = torch.stack([c.twc for c in self.cameras])
        cam_dist = torch.linalg.norm(twcs - twcs.mean(dim=0), dim=1)
        self.sence_size = float(cam_dist.max()) * 1.1
        print(f"Scene size: {self.sence_size:.4f}\n")

        # No `gs` for Blender finetuning — we load the pretrained vanilla .npy in finetune.py
        self.gs = None

    def __getitem__(self, index):
        if self.alphas is None:
            return self.cameras[index], self.images[index], None
        return self.cameras[index], self.images[index], self.alphas[index]

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    train_set = GSplatDataset(path, color_space='rgb', split='train', test_every=8, resize_rate=0.5)
    print(f"COLMAP train: {len(train_set)}")

    blender_set = GSplatDataset('data/blender/lego', split='train')
    cam, img, alpha = blender_set[0]
    print(f"Blender: img {img.shape}, alpha {alpha.shape if alpha is not None else None}")