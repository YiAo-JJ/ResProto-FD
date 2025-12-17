# ResProto-FD
Official code for the paper "ResProto-FD: Visual-Language Residual Prototype Sets for Generalized Face Forgery Detection"

## Deepfake Detection Framework

This repository implements deepfake detection using Gradient-aware Residual Prototypes (GRP) and Visual-Language Residual Learning (VLRL). The framework supports both single-GPU and multi-GPU Distributed Data Parallel (DDP) training.
Project Structure

├── dataset_fd/
│   ├── CelebDF.py        # Celeb-DF dataset loader
│   ├── DF40.py           # DF40 loader
│   ├── DFD.py            # Deepfake Detection Dataset loader
│   ├── DFDC.py           # Facebook Deepfake Detection Challenge loader
│   ├── FFpp.py           # FaceForensics++ loader
│   └── distortions.py    # Image distortion utilities
│
├── src/
│   ├── clip/             # Modified CLIP implementation
│   │   ├── __init__.py
│   │   ├── clip.py
│   │   ├── model.py
│   │   └── simple_tokenizer.py
│   │
│   ├── GRP.py            # Gradient-aware Residual Prototypes module
│   ├── RL.py             # Visual-Language Residual Learning (ResidualLearningLoss)
│   ├── model_trainer.py  # Model training logic
│   └── prompt_templates.py # Text prompt templates
│
├── environment.yaml      # Conda environment configuration
├── train.py              # Single-GPU training script
├── test.py               # Single-GPU testing script
├── train_ddp.py          # Multi-GPU DDP training entry
└── test_ddp.py           # Multi-GPU DDP testing entry

## Installation

### Create Conda environment:

```bash
conda env create -f environment.yaml
```

### Activate environment:

```bash
conda activate [ENV_NAME]
```

## Training
### Single-GPU Training

```bash
python train.py --train_dataset [train_dataset] --test_dataset [test_dataset] --manipu_type [Deepfakes,Face2Face,FaceSwap,NeuralTextures] --results_dir [results_dir] --res_lambda [res_lambda] --pts_num [pts_num] --use_Residual [use_Residual] --use_RL [use_RL] --use_GRP [use_GRP] --cluster_param [cluster_param] --gama [gama]
```
### Multi-GPU DDP Training

```bash
python -m torch.distributed.run --nproc_per_node=[NUM_GPUS] train_ddp.py [Training Params]
```

## Testing
### Single-GPU Testing

```bash
python test.py --test_dataset [test_dataset] --manipu_type [Deepfakes,Face2Face,FaceSwap,NeuralTextures] --results_dir [results_dir] --res_lambda [res_lambda] --pts_num [pts_num] --use_Residual [use_Residual] --use_RL [use_RL] --use_GRP [use_GRP] --test_level [test_level] --top_k [top_k]
```

### Multi-GPU DDP Testing

```bash
python -m torch.distributed.run --nproc_per_node=[NUM_GPUS] test_ddp.py --[Testing Params]
```

## Key Components
### Gradient-aware Residual Prototypes (GRP)

Implemented in src/GRP.py, this module learns residual prototypes to enhance feature discrimination between real and fake samples by gradient-aware optimization.
Visual-Language Residual Learning (VLRL)

Contains ResidualLearningLoss in src/RL.py, which aligns visual features with textual prompts through residual space learning.
Prompt Engineering

prompt_templates.py provides text prompts for CLIP-based joint vision-language training.
Dataset Support

Predefined loaders in dataset_fd/ for major deepfake datasets. Configure paths in respective dataset files.
Environment Configuration

