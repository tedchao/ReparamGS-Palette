# ReparamGS-Palette

Official code for the SIGGRAPH Asia 2026 conference paper:

**Reparametrizing 3D Gaussian Splatting for Real-Time Palette-based Color and Luminance Editing**

[Cheng-Kang Ted Chao](https://github.com/tedchao) and [Yotam Gingold](https://cragl.cs.gmu.edu/)

## Installation

### 1. Create the conda environment

Install [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda is fine), then create and activate the environment from the `environment.yml` in this repo:

```bash
conda env create -f environment.yml
conda activate colorfulgaussians
```

This provides Python 3.9, PyTorch, and all other Python dependencies.

### 2. Build the CUDA rasterizer

Build and install `gsplatcu`, the CUDA rasterizer used for rendering, as a PyTorch extension:

```bash
pip3 install gsplatcu/.
```

This step compiles CUDA sources, so it requires an NVIDIA GPU with the CUDA toolkit installed. You can verify the install with:

```bash
python -c "import torch, gsplatcu; print('ok')"
```

## Acknowledgements

The CUDA rasterizer in `gsplatcu/` is adapted from [EasyGaussianSplatting](https://github.com/scomup/EasyGaussianSplatting) by scomup (Liu Yang). See the license header in those source files for details.
