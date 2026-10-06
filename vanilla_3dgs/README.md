# Vanilla 3DGS

This folder contains the vanilla 3D Gaussian Splatting training pipeline, used to produce the standard 3DGS checkpoints that the palette finetuning (`finetune.py` in the repo root) starts from.

Note that vanilla 3DGS and our palette-based representation use **different rasterizers**, so this folder ships its own copy of `gsplatcu`. Both copies build a Python package with the same name — installing one replaces the other — so make sure the rasterizer from *this* folder is the one installed when training vanilla 3DGS (and reinstall the root one before palette finetuning).

## 1. Install the vanilla rasterizer

After creating and activating the conda environment (see the README in the repo root), build and install the vanilla 3DGS rasterizer from inside this folder:

```bash
cd vanilla_3dgs
pip3 install gsplatcu/.
```

## 2. Train a vanilla 3DGS

Example run:

```bash
python train.py --path ../dataset/supersplats/statue/ --epochs 150
```

The statue dataset can be downloaded [here](https://drive.google.com/drive/folders/1qA2ndQKxMGNjUO3mg9eFXP_CTukSjyeS?usp=sharing), and other example scenes can be found [here](https://drive.google.com/drive/folders/1GZxc299Z9yB_JhlW5bljQ2Yx2CXCLdFI?usp=sharing).

## 3. Render for testing

After training, you can render the trained model for testing. Example run:

```bash
python render_images.py --gs data/statue_final.npy --cam cameras/data/supersplat/statue.json --dir results/statue/ --range 0,10 --white_bg
```

The camera JSON for the statue scene can be downloaded [here](https://drive.google.com/file/d/1KQ6mMLSbBjwtfALttIYTcuZbwQqZCEJW/view?usp=sharing), and camera JSONs for other example scenes can be found [here](https://drive.google.com/drive/folders/1yTL9b62hVWhDcUa28nYp_erTH-e1tlfp?usp=sharing).

## 4. Output

After training, a checkpoint such as `statue_final.npy` is created inside the `data/` folder. This checkpoint is the input to the finetuning code in the repo root, which finetunes it into our palette-weights representation for further color grading.
