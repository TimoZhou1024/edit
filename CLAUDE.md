# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical AlphaEdit is a feasibility verification framework for train-free error correction in Medical Vision Transformers (ViTs) through null-space constrained knowledge editing. It addresses correcting errors and hallucinations without retraining.

## Commands

### Setup and Installation
```bash
cd medical_alphaedit_verification
uv sync  # Creates venv and installs dependencies
```

### Running Verification
```bash
# Synthetic verification (no external data required)
uv run python run_verification.py --mode synthetic

# With MedMNIST medical imaging data (includes fine-tuning)
uv run python run_verification.py --mode medmnist

# Force retraining (ignore checkpoint)
uv run python run_verification.py --mode medmnist --force-retrain

# Full verification (both modes)
uv run python run_verification.py --mode full --output-dir ./results

# Skip visualizations
uv run python run_verification.py --mode synthetic --no-visualizations
```

### Running Tests
```bash
uv run pytest
```

### Running Individual Modules
```bash
uv run python -m src.causal_tracing
uv run python -m src.null_space_projection
uv run python -m src.medical_alphaedit
```

## Architecture

The framework operates in two phases:

### 1. Fault Localization (src/causal_tracing.py)
- `CausalTracer`: Identifies specific image patches and transformer layers responsible for errors
- **Corrupt-Restore methodology**: Corrupt input with noise, restore clean activations at specific locations
- Distinguishes MLP vs MSA (Multi-head Self-Attention) contributions separately
- Key equations:
  - Causal Effect: `delta_y = ||y_hat - y_hat_{-p_k^l}||_2`
  - Attention Entropy: `H(A^l_k) = -sum_j A^l_{k,j} log(A^l_{k,j})`
  - Fault Score: `S_{k,l} = delta_y * exp(H(A^l_k))`

### 2. Null-Space Projected Editing (src/null_space_projection.py)
- `NullSpaceProjector`: Constrains weight updates to preserve existing knowledge
- Projects updates onto null space of correct activations covariance matrix
- Key equations:
  - Null-space projection: `P = V_0 * V_0^T`
  - Weight update: `ΔW = (v* - W*k*) * (k*^T * P) * (k* * k*^T * P + λI)^{-1}`
  - Preservation constraint: `(W + ΔW)*K_0 = W*K_0`

### Framework Integration (src/medical_alphaedit.py)
- `MedicalAlphaEdit`: Unified class combining both phases
- Workflow: fault localization → null-space projection → iterative weight updates → verification
- **Critical**: Collects activations from MLP fc1 layer (dim=3072 for ViT-B/16), not output layer
- Maintains reference database of correct activations for target retrieval

### Visualization (src/visualization.py)
- Generates heatmaps, singular value spectra, MLP/MSA fault heatmaps
- Outputs to `outputs/<timestamp>/visualizations/`

## ViT-B/16 Architecture Notes

Understanding the model structure is essential for editing:
- 12 transformer blocks, each with MSA and MLP
- MLP structure: fc1 (768→3072) → GELU → fc2 (3072→768)
- **Edit target**: `model.blocks[layer].mlp.fc2.weight` (shape: 768×3072)
- **Activation source**: `model.blocks[layer].mlp.fc1` output (dim: 3072)

## Key Data Structures

- `FaultLocalizationResult`: Contains patch/layer indices, fault scores, attention entropies, MLP/MSA distinction
- `NullSpaceProjectionResult`: Contains projection matrix P, null space basis, singular values
- `WeightUpdateResult`: Contains ΔW, original/updated weights, preservation/correction errors
- `EditingResult`: Complete workflow results including success status and metrics

## MedMNIST Verification Notes

The MedMNIST mode requires fine-tuning before AlphaEdit can work:
- Pre-trained ImageNet model has random head for PathMNIST (9 classes)
- Fine-tuning teaches task-specific features in fc1 activations
- Checkpoints saved to `checkpoints/vit_pathmnist_finetuned.pt`
- Without fine-tuning, `||P*k*|| / ||k*||` ratio is too low (~15%) for effective editing
