import torch
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import json
import os
from pathlib import Path
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsplat.gsmodel import *

torch.autograd.set_detect_anomaly(True)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="dataset path")
    parser.add_argument("--resize_rate", type=float, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dataset_type", type=str, default="auto",
                        choices=["auto", "colmap", "blender"])
    parser.add_argument("--num_random_points", type=int, default=100_000)
    args = parser.parse_args()

    set_seed(42)
    if not args.path:
        print("No dataset path given."); exit(0)

    scene_name = Path(args.path.rstrip('/')).name
    print("Try training %s..." % args.path)
    gs_set = GSplatDataset(
        args.path,
        resize_rate=args.resize_rate,
        dataset_type=args.dataset_type,
        num_random_points=args.num_random_points,
    )
    print('Loaded.')
    # Blender uses 4-channel RGBA training; COLMAP uses 3-channel RGB
    is_rgba = (gs_set.dataset_type == 'blender')

    training_params, adam_params = get_training_params(gs_set.gs)
    optimizer = optim.Adam(adam_params, lr=0.000, eps=1e-15)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8))
    txt = ax.text(50, 50, "", size=20, color='white')

    epochs = int(args.epochs)
    n = len(gs_set)
    model = GSModel(gs_set.sence_size, len(gs_set) * epochs)

    avg_losses, epoch_times = [], []
    global_start = time.time()
    print("\n" + "="*60)
    print(f"Starting training: {scene_name}")
    print(f"Dataset type: {gs_set.dataset_type} | RGBA: {is_rgba}")
    print(f"Views: {n} | epochs: {epochs}")
    print(f"Resolution: {cam0.width}x{cam0.height} | resize: {args.resize_rate}")
    print("="*60 + "\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        idxs = np.arange(n); np.random.shuffle(idxs)
        avg_loss = 0

        for i in idxs:
            cam, image_gt = gs_set[i]
            image = model(*training_params.values(), cam, return_rgba=is_rgba)
            
            import torchvision
            if epoch == 0 and i == 0:
                torchvision.utils.save_image(image[:3], 'debug_pred.png')
                torchvision.utils.save_image(image_gt[:3], 'debug_gt.png')
                if image.shape[0] == 4:
                    torchvision.utils.save_image(image[3:4], 'debug_pred_alpha.png')
                    torchvision.utils.save_image(image_gt[3:4], 'debug_gt_alpha.png')
                print('Saved debug images')
                
                
            loss = gau_loss(image, image_gt)
            loss.backward()

            model.update_density_info()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.update_pws_lr(optimizer)
            avg_loss += loss.item()
            
            if i == 0 and epoch == 0:   # only print once
                print(f"gt   shape: {image_gt.shape},   range: [{image_gt.min():.3f}, {image_gt.max():.3f}]")
                print(f"pred shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
                    
            if i == 0:
                # Display only RGB (drop alpha if present)
                disp = image[:3].detach().permute(1, 2, 0).cpu().numpy()
                img.set_data(np.clip(disp, 0, 1))
                txt.set_text("epoch %d" % epoch)
                plt.pause(0.1)

        avg_loss /= n
        avg_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print("epoch:%d avg_loss:%f time:%.2f sec" % (epoch, avg_loss, epoch_time))

        with torch.no_grad():
            #if 1 < epoch <= epochs // 2:   # use for blender type dataset
            if 1 < epoch <= 60:        # use for colmap
                if epoch % 5 == 0:
                    print("updating gaussian density...")
                    model.update_gaussian_density(training_params, optimizer)
                if epoch % 15 == 0:
                    print("resetting gaussian alpha...")
                    model.reset_alpha(training_params, optimizer)
            if epoch % 100 == 0:
                fn = f"data/{scene_name}_epoch{epoch:04d}.npy"
                save_training_params(fn, training_params)
                print("saved %s" % fn)

    global_time = time.time() - global_start
    print("\n" + "="*60)
    print(f"DONE: {scene_name} ({gs_set.dataset_type})")
    print(f"Total time: {global_time:.2f} sec ({global_time/60:.2f} min)")
    print(f"Per epoch: {np.mean(epoch_times):.2f} ± {np.std(epoch_times):.2f} sec")
    print(f"Loss: {avg_losses[0]:.6f} -> {avg_losses[-1]:.6f}")
    print("="*60 + "\n")

    os.makedirs("data/timing", exist_ok=True)
    timing_data = {
        "scene_name": scene_name, "dataset_type": gs_set.dataset_type,
        "rgba_training": is_rgba,
        "num_training_views": n,
        "image_width": cam0.width, "image_height": cam0.height,
        "resize_rate": args.resize_rate, "total_epochs": epochs,
        "total_time_sec": float(global_time),
        "avg_epoch_time_sec": float(np.mean(epoch_times)),
        "per_epoch_times": [float(t) for t in epoch_times],
        "all_losses": [float(l) for l in avg_losses],
    }
    with open(f"data/timing/timing_{scene_name}.json", "w") as f:
        json.dump(timing_data, f, indent=2)

    save_training_params(f'data/{scene_name}_final.npy', training_params)
    print(f"Final model saved to data/{scene_name}_final.npy")

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs+1), avg_losses, color="blue", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training Loss - {scene_name}"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"data/loss_curve_{scene_name}.png", dpi=150)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs+1), epoch_times, color="green", linewidth=1, alpha=0.7)
    plt.axhline(y=np.mean(epoch_times), color='red', linestyle='--',
                label=f'Mean: {np.mean(epoch_times):.2f}s')
    plt.xlabel("Epoch"); plt.ylabel("Time (s)")
    plt.title(f"Epoch Time - {scene_name}"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"data/timing_per_epoch_{scene_name}.png", dpi=150)

    print("Done!")