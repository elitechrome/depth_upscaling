# Deep Learning Depth Upscaling

This project implements a deep learning pipeline for upscaling sparse, low-resolution depth maps (224x100) to dense, high-resolution depth maps (800x600) using RGB guidance.

## Project Structure

```text
depth_upscaling/
├── data/
│   ├── simulation.py      # Camera geometry & sparse pattern simulation
│   ├── generator.py       # Synthetic scene generation (RGB-D)
│   ├── writer.py          # WebDataset shard creator
│   └── loader.py          # PyTorch-compatible WebDataset loader
├── model/
│   └── network.py         # SparseToDense-Net (Dual Encoder U-Net)
└── requirements.txt       # Dependencies
```

## Setup

1. **Prerequisites**: Ensure you have `uv` installed.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Environment**: Create and sync the virtual environment (Python 3.11).
   ```bash
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -r depth_upscaling/requirements.txt
   ```

## Data Generation

Generate synthetic WebDataset shards for testing. This script simulates the specific sensor geometry (RGB 800x600, Depth 224x100 with 1.5° sparse pattern).

```bash
# Generate 100 samples in ./dataset_shards
PYTHONPATH=. python depth_upscaling/data/writer.py --count 100 --output ./dataset_shards
```

### Option B: Real Data (ARKitScenes)
If you have the [ARKitScenes](https://github.com/apple/ARKitScenes) `depth_upsampling` dataset:
1. Download it using their scripts.
2. Convert it to our WebDataset format (simulating your sparse sensor on top of their dense GT):
   ```bash
   python depth_upscaling/data/arkit_converter.py --input /path/to/ARKitScenes/depth_upsampling --output ./arkit_shards --split Training
   ```

## Usage

### verify Data Loading
Check if the generated shards can be loaded correctly.
```bash
PYTHONPATH=. python depth_upscaling/data/loader.py
```

### Run Model Verification
Instantiate the `SparseToDenseNet` and run a forward pass with random/dummy tensors to verify architecture.
```bash
PYTHONPATH=. python depth_upscaling/model/network.py
```

## Model Architecture
- **Inputs**: 
  - RGB Image (800x600)
  - Sparse Depth (224x100) -> Upsampled internally
- **Backbone**: Dual-Stream Encoder (RGB Stream + Depth Stream)
- **Fusion**: Guide Blocks inject RGB features into the Depth stream at multiple scales.
- **Output**: Dense Depth Map (800x600)

## Depthor Integration (ICCV 2025)

We have integrated the [Depthor](https://github.com/ShadowBbBb/Depthor) model.
- **Code**: `depth_upscaling/depthor_repo`
- **Mac Check**: Use `verify_depthor.py` to test the architecture on Mac (run with mocked CUDA ops).
- **Training**: For training on NVIDIA GPUs, install the real `BpOps` extension as per Depthor's README.

```bash
# Verify Depthor
python verify_depthor.py
```
