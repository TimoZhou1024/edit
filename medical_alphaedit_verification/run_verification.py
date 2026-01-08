#!/usr/bin/env python3
"""
Main Verification Script for Medical AlphaEdit Framework

This script performs comprehensive feasibility verification of the Medical AlphaEdit
framework, demonstrating:
1. Causal tracing for fault localization in ViTs
2. Null-space projection for knowledge preservation
3. Constrained weight updates for error correction
4. Integration with real medical imaging data (MedMNIST)

Usage:
    python run_verification.py [--mode {synthetic,medmnist,full}] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.causal_tracing import demonstrate_causal_tracing
from src.null_space_projection import demonstrate_null_space_projection
from src.medical_alphaedit import demonstrate_medical_alphaedit
from src.visualization import (
    create_all_visualizations,
    generate_summary_report,
    setup_plot_style
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_synthetic_verification(output_dir: Path) -> dict:
    """
    Run verification using synthetic data (no external dependencies).

    This mode demonstrates the mathematical correctness of all algorithms
    without requiring actual medical images or pre-trained models.

    Args:
        output_dir: Directory to save outputs

    Returns:
        Dictionary containing all verification results
    """
    print("\n" + "=" * 70)
    print("RUNNING SYNTHETIC VERIFICATION")
    print("=" * 70)

    results = {}

    # 1. Causal Tracing Verification
    print("\n[1/3] Causal Tracing Verification...")
    causal_results = demonstrate_causal_tracing()
    results['causal_tracing'] = causal_results

    # 2. Null-Space Projection Verification
    print("\n[2/3] Null-Space Projection Verification...")
    nullspace_results = demonstrate_null_space_projection()
    results['null_space'] = nullspace_results

    # 3. Framework Integration Verification
    print("\n[3/3] Framework Integration Verification...")
    framework_results = demonstrate_medical_alphaedit()
    results['framework'] = framework_results

    return results


def finetune_model(
    model,
    train_loader,
    device: str,
    num_epochs: int = 3,
    lr: float = 1e-4,
    max_batches_per_epoch: int = 200,
    verbose: bool = True
) -> dict:
    """
    Fine-tune the ViT model on the target dataset.

    This step is essential for Medical AlphaEdit to work correctly:
    - Pre-trained ImageNet model has random classification head for PathMNIST
    - Without fine-tuning, fc1 activations don't encode task-specific features
    - Result: null-space projection cannot effectively preserve knowledge

    After fine-tuning:
    - Model learns PathMNIST class boundaries
    - fc1 activations become task-relevant
    - AlphaEdit can correct individual errors without disrupting learned knowledge

    Args:
        model: ViT model to fine-tune
        train_loader: DataLoader with training data
        device: Device to use
        num_epochs: Number of training epochs
        lr: Learning rate
        max_batches_per_epoch: Max batches per epoch (for faster demo)
        verbose: Whether to print progress

    Returns:
        Dictionary with training metrics
    """
    import torch
    import torch.nn.functional as F

    model.train()

    # Use AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine annealing learning rate schedule
    total_steps = num_epochs * min(len(train_loader), max_batches_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    metrics = {
        'epochs': num_epochs,
        'lr': lr,
        'losses': [],
        'accuracies': []
    }

    if verbose:
        print(f"   Fine-tuning for {num_epochs} epochs with lr={lr}")
        print(f"   (max {max_batches_per_epoch} batches per epoch for faster demo)")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= max_batches_per_epoch:
                break

            images = images.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

        batches_processed = min(batch_idx + 1, max_batches_per_epoch)
        avg_loss = epoch_loss / batches_processed
        accuracy = correct / total
        metrics['losses'].append(avg_loss)
        metrics['accuracies'].append(accuracy)

        if verbose:
            print(f"   Epoch {epoch + 1}/{num_epochs}: "
                  f"loss={avg_loss:.4f}, accuracy={accuracy:.2%}")

    model.eval()

    if verbose:
        print(f"   Fine-tuning complete. Final accuracy: {metrics['accuracies'][-1]:.2%}")

    return metrics


def run_medmnist_verification(
    output_dir: Path,
    use_real_causal_tracing: bool = False,
    causal_tracing_max_patches: int = 20,
    checkpoint_dir: Optional[Path] = None,
    force_retrain: bool = False
) -> dict:
    """
    Run verification using MedMNIST dataset.

    This mode demonstrates the framework on actual medical imaging data,
    using a fine-tuned ViT model to show realistic error correction.

    Args:
        output_dir: Directory to save outputs
        use_real_causal_tracing: If True, use real Corrupt-Restore causal tracing
                                  (slower but more accurate). If False, use random
                                  simulation for faster demo.
        causal_tracing_max_patches: Maximum patches to analyze per layer when using
                                     real causal tracing (for speed control).
        checkpoint_dir: Directory to save/load model checkpoints. If None, uses
                        default location (medical_alphaedit_verification/checkpoints).
        force_retrain: If True, retrain even if checkpoint exists.

    Returns:
        Dictionary containing all verification results
    """
    print("\n" + "=" * 70)
    print("RUNNING MEDMNIST VERIFICATION")
    print("=" * 70)

    try:
        import torch
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        import timm
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.info("Please run: uv sync")
        return {}

    results = {}

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load MedMNIST dataset
    print("\n[1/5] Loading MedMNIST dataset...")
    try:
        import medmnist
        from medmnist import PathMNIST

        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load train and test sets
        train_dataset = PathMNIST(split='train', transform=transform, download=True, size=224)
        test_dataset = PathMNIST(split='test', transform=transform, download=True, size=224)

        # Use smaller batch size to avoid memory issues
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Number of classes: {len(train_dataset.info['label'])}")

        results['dataset'] = {
            'name': 'PathMNIST',
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'num_classes': len(train_dataset.info['label'])
        }

    except ImportError as e:
        logger.error(f"MedMNIST package not installed: {e}")
        logger.error("Please install medmnist: uv add medmnist")
        raise
    except Exception as e:
        logger.error(f"Failed to load MedMNIST: {e}")
        raise

    # Load pre-trained ViT model
    print("\n[2/5] Loading Vision Transformer model...")
    try:
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=9)
        model = model.to(device)
        model.train(False)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Model: ViT-B/16")
        print(f"   Parameters: {num_params:,}")
        print(f"   Patch size: 16x16")
        print(f"   Number of layers: 12")

        results['model'] = {
            'architecture': 'ViT-B/16',
            'parameters': num_params,
            'patch_size': 16,
            'num_layers': 12
        }

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'vit_pathmnist_finetuned.pt'

    # Check if we can load from checkpoint
    loaded_from_checkpoint = False
    if checkpoint_path.exists() and not force_retrain:
        print("\n[3/7] Loading fine-tuned model from checkpoint...")
        print(f"   Checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            finetune_metrics = checkpoint.get('metrics', {
                'epochs': 'unknown',
                'lr': 'unknown',
                'losses': [],
                'accuracies': [],
                'loaded_from_checkpoint': True
            })
            finetune_metrics['loaded_from_checkpoint'] = True
            print(f"   Loaded successfully! (trained accuracy: {checkpoint.get('accuracy', 'unknown')})")
            loaded_from_checkpoint = True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            print("   Failed to load checkpoint, will retrain...")

    if not loaded_from_checkpoint:
        # Fine-tune model on PathMNIST
        print("\n[3/7] Fine-tuning model on PathMNIST...")
        print("   (This is essential for AlphaEdit - model must learn task-specific features)")

        finetune_metrics = finetune_model(
            model=model,
            train_loader=train_loader,
            device=device,
            num_epochs=3,
            lr=1e-4,
            verbose=True
        )

        # Save checkpoint
        print(f"   Saving checkpoint to: {checkpoint_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': finetune_metrics,
            'accuracy': finetune_metrics['accuracies'][-1] if finetune_metrics['accuracies'] else None
        }, checkpoint_path)
        finetune_metrics['checkpoint_saved'] = str(checkpoint_path)

    results['finetuning'] = finetune_metrics

    # Test accuracy after fine-tuning
    print("\n[4/7] Testing model performance after fine-tuning...")
    correct = 0
    total = 0
    misclassified_samples = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit for demo
                break

            images = images.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for i in range(len(labels)):
                if predictions[i] != labels[i]:
                    misclassified_samples.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'true_label': labels[i].item(),
                        'predicted': predictions[i].item()
                    })

            correct += (predictions == labels).sum().item()
            total += len(labels)

    initial_accuracy = correct / total
    print(f"   Post-finetuning accuracy: {initial_accuracy:.2%}")
    print(f"   Misclassified samples found: {len(misclassified_samples)}")

    results['post_finetuning_performance'] = {
        'accuracy': initial_accuracy,
        'correct': correct,
        'total': total,
        'misclassified_count': len(misclassified_samples)
    }

    # Demonstrate causal tracing on a sample
    print("\n[5/7] Demonstrating causal tracing on misclassified sample...")

    if len(misclassified_samples) > 0:
        from src.causal_tracing import CausalTracer

        # Get a misclassified sample
        sample_info = misclassified_samples[0]

        # Reload the sample
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx == sample_info['batch_idx']:
                sample_image = images[sample_info['sample_idx']].unsqueeze(0).to(device)
                sample_label = labels[sample_info['sample_idx']].item()
                break

        print(f"   Analyzing sample: true={sample_label}, predicted={sample_info['predicted']}")

        # Initialize causal tracer
        tracer = CausalTracer(model, device)

        num_patches = 196
        num_layers = 12

        if use_real_causal_tracing:
            # ═══════════════════════════════════════════════════════════════
            # Real Corrupt-Restore causal tracing
            # ═══════════════════════════════════════════════════════════════
            print("   Running REAL fault localization (layer-level detailed tracing)...")
            print("   This may take a few minutes...")

            import time
            start_time = time.time()

            # Run detailed fault localization (layer-level) and save CSV
            fault_result, detailed_records = tracer.localize_faults_detailed(
                image=sample_image,
                verbose=True,
                include_msa=False
            )

            csv_path = output_dir / 'causal_tracing_detailed.csv'
            tracer.save_detailed_records_csv(
                records=detailed_records,
                output_path=str(csv_path),
                corrupted_dist=fault_result.corrupted_dist
            )

            elapsed_time = time.time() - start_time
            print(f"   Fault localization completed in {elapsed_time:.1f} seconds")
            print(f"   Detailed tracing CSV saved to: {csv_path}")

            # Extract results
            critical_components = fault_result.critical_components

            print("   Top critical layers (REAL, detailed):")
            for rank, (patch_idx, layer_idx, component_type) in enumerate(critical_components[:5], 1):
                if component_type == 'mlp' and fault_result.mlp_fault_scores is not None:
                    score = fault_result.mlp_fault_scores[0, layer_idx]
                elif component_type == 'msa' and fault_result.msa_fault_scores is not None:
                    score = fault_result.msa_fault_scores[0, layer_idx]
                else:
                    score = fault_result.fault_scores[0, layer_idx]
                print(f"      {rank}. Layer {layer_idx}, {component_type.upper()}: score = {score:.4f}")

            # Extract the most critical layer for editing (prefer MLP components)
            mlp_components = [(p, layer, t) for p, layer, t in critical_components if t == 'mlp']
            if mlp_components:
                critical_layer = mlp_components[0][1]
            else:
                critical_layer = critical_components[0][1] if critical_components else 11

            results['causal_tracing'] = {
                'method': 'real_corrupt_restore_detailed',
                'sample_true_label': sample_label,
                'sample_predicted': sample_info['predicted'],
                'num_patches_analyzed': 1,
                'num_layers_analyzed': num_layers,
                'elapsed_time_seconds': elapsed_time,
                'corrupted_distance': fault_result.corrupted_dist,
                'critical_components': [
                    {'patch': patch, 'layer': layer, 'type': comp_type}
                    for patch, layer, comp_type in critical_components[:5]
                ],
                'top_components': [(patch, layer) for patch, layer, _ in critical_components[:5]],
                'critical_layer_for_editing': critical_layer,
                'detailed_csv': str(csv_path)
            }

        else:
            # ═══════════════════════════════════════════════════════════════
            # Simulated fault scores (fast demo mode)
            # ═══════════════════════════════════════════════════════════════
            print("   Running SIMULATED fault localization (fast demo mode)...")
            print("   (Use --real-causal-tracing flag for actual Corrupt-Restore analysis)")

            # Simulate fault scores
            fault_scores = np.random.exponential(0.1, (num_patches, num_layers))

            # Find top components
            flat_scores = fault_scores.flatten()
            top_5_indices = np.argsort(flat_scores)[-5:][::-1]

            print("   Top 5 critical components (SIMULATED):")
            for rank, idx in enumerate(top_5_indices, 1):
                patch_idx = idx // num_layers
                layer_idx = idx % num_layers
                score = fault_scores[patch_idx, layer_idx]
                print(f"      {rank}. Patch {patch_idx}, Layer {layer_idx}: score = {score:.4f}")

            # Extract critical layer from simulated results
            critical_layer = top_5_indices[0] % num_layers

            results['causal_tracing'] = {
                'method': 'simulated',
                'sample_true_label': sample_label,
                'sample_predicted': sample_info['predicted'],
                'num_patches_analyzed': num_patches,
                'num_layers_analyzed': num_layers,
                'top_components': [(idx // num_layers, idx % num_layers) for idx in top_5_indices],
                'critical_layer_for_editing': critical_layer
            }

        tracer.cleanup()

    # ═══════════════════════════════════════════════════════════════════════
    # Use Causal Tracing result to determine target layer for editing
    # ═══════════════════════════════════════════════════════════════════════
    if 'causal_tracing' in results and 'critical_layer_for_editing' in results['causal_tracing']:
        target_layer = results['causal_tracing']['critical_layer_for_editing']
        print(f"\n   *** Using Causal Tracing result: editing Layer {target_layer} ***")
    else:
        target_layer = 11  # Default to last layer
        print(f"\n   *** No Causal Tracing result, using default Layer {target_layer} ***")

    # Demonstrate null-space projection with fc1 activations
    print("\n[6/7] Demonstrating null-space projection with fc1 activations...")
    from src.null_space_projection import NullSpaceProjector
    from src.medical_alphaedit import MedicalAlphaEdit

    projector = NullSpaceProjector()
    # target_layer is now determined by Causal Tracing above

    # Collect fc1 activations using hook (CRITICAL FIX)
    print("   Collecting fc1 activations from correctly classified samples...")
    fc1_activations_by_class: Dict[int, List[np.ndarray]] = {}
    hook_storage = {}

    # Register hook on target layer's fc1
    fc1_module = model.blocks[target_layer].mlp.fc1

    def capture_fc1(module, inp, out):
        hook_storage['fc1'] = out.detach()

    hook = fc1_module.register_forward_hook(capture_fc1)

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx >= 10:  # Collect from more batches
                    break

                images = images.to(device)
                labels = labels.squeeze()

                outputs = model(images)
                predictions = outputs.argmax(dim=1).cpu()

                # fc1_out shape: [batch, num_patches, fc1_dim]
                fc1_out = hook_storage['fc1'].cpu().numpy()
                # CLS token activation: [batch, fc1_dim]
                cls_activations = fc1_out[:, 0, :]

                for i in range(len(labels)):
                    label = int(labels[i].item())
                    if predictions[i] == label:  # Correctly classified
                        if label not in fc1_activations_by_class:
                            fc1_activations_by_class[label] = []
                        if len(fc1_activations_by_class[label]) < 50:
                            fc1_activations_by_class[label].append(cls_activations[i])
    finally:
        hook.remove()

    # Report collection results
    total_collected = sum(len(v) for v in fc1_activations_by_class.values())
    print(f"   Collected {total_collected} fc1 activations across {len(fc1_activations_by_class)} classes")

    # Compute null-space for each class with enough samples
    null_space_results = {}
    for class_id, activations in fc1_activations_by_class.items():
        if len(activations) >= 10:
            K_0 = np.array(activations).T  # [fc1_dim x num_samples]
            proj_result = projector.compute_null_space_projection(K_0, verbose=False)
            null_space_results[class_id] = {
                'num_samples': len(activations),
                'fc1_dim': K_0.shape[0],
                'null_space_dim': proj_result.null_space_dim,
                'rank': proj_result.rank
            }
            print(f"   Class {class_id}: {len(activations)} samples, "
                  f"fc1_dim={K_0.shape[0]}, null_dim={proj_result.null_space_dim}")

    results['null_space'] = {
        'target_layer': target_layer,
        'classes_analyzed': len(null_space_results),
        'total_samples': total_collected,
        'class_results': null_space_results
    }

    # Demonstrate actual error correction
    print("\n[7/7] Demonstrating error correction...")
    print(f"   Target layer for MLP editing: {target_layer} (from Causal Tracing)")

    if len(misclassified_samples) > 0 and len(fc1_activations_by_class) > 0:
        # Initialize MedicalAlphaEdit framework with layer from Causal Tracing
        editor = MedicalAlphaEdit(model=model, device=device, target_layer=target_layer)

        # Manually set reference activations (already collected)
        for class_id, activations in fc1_activations_by_class.items():
            if len(activations) >= 5:
                editor.reference_activations[class_id] = np.array(activations).T

        # Find a misclassified sample whose target class has reference activations
        correction_attempted = False
        for sample_info in misclassified_samples[:20]:  # Try up to 20 samples
            target_class = sample_info['true_label']

            if target_class in editor.reference_activations:
                # Reload the sample
                sample_image = None
                for batch_idx, (images, labels) in enumerate(test_loader):
                    if batch_idx == sample_info['batch_idx']:
                        sample_image = images[sample_info['sample_idx']].unsqueeze(0).to(device)
                        break

                if sample_image is None:
                    continue

                print(f"\n   Attempting to correct: pred={sample_info['predicted']} -> target={target_class}")

                # Get null-space projection info
                proj_result = editor.get_projection_matrix(target_class, verbose=False)
                if proj_result:
                    print(f"   Null space dimension for class {target_class}: {proj_result.null_space_dim}")

                    # ═══════════════════════════════════════════════════════════
                    # Method 1: MLP Editing (using Causal Tracing target layer)
                    # ═══════════════════════════════════════════════════════════
                    print(f"\n   --- Method 1: MLP Editing (Layer {target_layer}) ---")

                    # Save original weights for restoration
                    original_fc2_weight = model.blocks[target_layer].mlp.fc2.weight.data.clone()

                    try:
                        mlp_result = editor.correct_error(
                            image=sample_image,
                            target_label=target_class,
                            verbose=True,
                            use_causal_tracing=False  # Already did causal tracing
                        )

                        mlp_success = mlp_result.success
                        mlp_weight_change = mlp_result.total_weight_change
                        mlp_preservation = mlp_result.preservation_metrics.get('avg_preservation_error', 'N/A')

                        print(f"   MLP Editing: {'SUCCESS' if mlp_success else 'FAILED'}")
                        print(f"   Weight change: {mlp_weight_change:.6f}")

                    except Exception as e:
                        print(f"   MLP Editing failed: {e}")
                        mlp_success = False
                        mlp_weight_change = 0
                        mlp_preservation = 'error'

                    # Restore original weights for fair comparison
                    model.blocks[target_layer].mlp.fc2.weight.data = original_fc2_weight

                    # ═══════════════════════════════════════════════════════════
                    # Method 2: Head Editing (direct classification head)
                    # ═══════════════════════════════════════════════════════════
                    print("   --- Method 2: Head Editing (Classification Head) ---")

                    # Save original head weights for potential restoration
                    original_head_weight = model.head.weight.data.clone()

                    try:
                        head_result = editor.correct_error_head(
                            image=sample_image,
                            target_label=target_class,
                            verbose=True
                        )

                        head_success = head_result.success
                        head_weight_change = head_result.total_weight_change

                        print(f"   Head Editing: {'SUCCESS' if head_success else 'FAILED'}")
                        print(f"   Weight change: {head_weight_change:.6f}")

                    except Exception as e:
                        print(f"   Head Editing failed: {e}")
                        head_success = False
                        head_weight_change = 0

                    # Restore head weights if needed for subsequent tests
                    if not head_success:
                        model.head.weight.data = original_head_weight

                    # ═══════════════════════════════════════════════════════════
                    # Method 1b: Null-space Direct (end-to-end with preservation)
                    # ═══════════════════════════════════════════════════════════
                    print(f"   --- Method 1b: Null-space Direct (Layer {target_layer}) ---")

                    # Restore head weights first
                    model.head.weight.data = original_head_weight
                    # Restore fc2 weights
                    model.blocks[target_layer].mlp.fc2.weight.data = original_fc2_weight

                    try:
                        nullspace_direct_result = editor.correct_error_nullspace_direct(
                            image=sample_image,
                            target_label=target_class,
                            num_steps=100,
                            lr=0.05,
                            l2_weight=0.0001,
                            nullspace_weight=0.1,  # Soft constraint
                            verbose=True
                        )

                        nullspace_direct_success = nullspace_direct_result.success
                        nullspace_direct_weight_change = nullspace_direct_result.total_weight_change
                        nullspace_direct_pres_error = nullspace_direct_result.preservation_metrics.get('preservation_error', 0)

                        print(f"   Null-space Direct: {'SUCCESS' if nullspace_direct_success else 'FAILED'}")
                        print(f"   Weight change: {nullspace_direct_weight_change:.6f}")
                        print(f"   Preservation error: {nullspace_direct_pres_error:.6f}")

                    except Exception as e:
                        print(f"   Null-space Direct failed: {e}")
                        import traceback
                        traceback.print_exc()
                        nullspace_direct_success = False
                        nullspace_direct_weight_change = 0
                        nullspace_direct_pres_error = 0

                    # ═══════════════════════════════════════════════════════════
                    # Method 3: Direct MLP Optimization (end-to-end)
                    # ═══════════════════════════════════════════════════════════
                    print(f"   --- Method 3: Direct MLP Optimization (Layer {target_layer}) ---")

                    # Restore head weights first
                    model.head.weight.data = original_head_weight
                    # Restore fc2 weights
                    model.blocks[target_layer].mlp.fc2.weight.data = original_fc2_weight

                    try:
                        direct_result = editor.correct_error_direct(
                            image=sample_image,
                            target_label=target_class,
                            num_steps=50,
                            lr=0.01,
                            verbose=True
                        )

                        direct_success = direct_result.success
                        direct_weight_change = direct_result.total_weight_change

                        print(f"   Direct MLP: {'SUCCESS' if direct_success else 'FAILED'}")
                        print(f"   Weight change: {direct_weight_change:.6f}")

                    except Exception as e:
                        print(f"   Direct MLP Optimization failed: {e}")
                        import traceback
                        traceback.print_exc()
                        direct_success = False
                        direct_weight_change = 0

                    # ═══════════════════════════════════════════════════════════
                    # Comparison Summary
                    # ═══════════════════════════════════════════════════════════
                    print("\n   === Correction Method Comparison ===")
                    print(f"   MLP Editing (v* + null-space):     {'[SUCCESS]' if mlp_success else '[FAILED]'}")
                    print(f"   Null-space Direct (end-to-end):    {'[SUCCESS]' if nullspace_direct_success else '[FAILED]'}")
                    print(f"   Direct MLP Optimization:           {'[SUCCESS]' if direct_success else '[FAILED]'}")
                    print(f"   Head Editing:                      {'[SUCCESS]' if head_success else '[FAILED]'}")

                    results['correction'] = {
                        'target_layer_from_causal_tracing': target_layer,
                        'mlp_editing': {
                            'success': mlp_success,
                            'weight_change': float(mlp_weight_change),
                            'preservation_error': mlp_preservation if isinstance(mlp_preservation, str) else float(mlp_preservation)
                        },
                        'nullspace_direct': {
                            'success': nullspace_direct_success,
                            'weight_change': float(nullspace_direct_weight_change),
                            'preservation_error': float(nullspace_direct_pres_error)
                        },
                        'direct_mlp_optimization': {
                            'success': direct_success,
                            'weight_change': float(direct_weight_change)
                        },
                        'head_editing': {
                            'success': head_success,
                            'weight_change': float(head_weight_change)
                        },
                        'original_prediction': sample_info['predicted'],
                        'target_label': target_class,
                        'null_space_dim': proj_result.null_space_dim,
                        'causal_tracing_guided': True
                    }

                    correction_attempted = True
                    break

        if not correction_attempted:
            print(f"   Could not find suitable sample for correction demo")
            print(f"   Available classes in reference: {list(editor.reference_activations.keys())}")

    return results


def run_full_verification(
    output_dir: Path,
    use_real_causal_tracing: bool = False,
    causal_tracing_max_patches: int = 20,
    checkpoint_dir: Optional[Path] = None,
    force_retrain: bool = False
) -> dict:
    """
    Run both synthetic and MedMNIST verification.

    Args:
        output_dir: Directory to save outputs
        use_real_causal_tracing: If True, use real Corrupt-Restore causal tracing
        causal_tracing_max_patches: Maximum patches to analyze per layer
        checkpoint_dir: Directory to save/load model checkpoints
        force_retrain: If True, retrain even if checkpoint exists

    Returns:
        Combined verification results
    """
    print("\n" + "=" * 70)
    print("RUNNING FULL VERIFICATION (SYNTHETIC + MEDMNIST)")
    print("=" * 70)

    results = {
        'synthetic': run_synthetic_verification(output_dir / 'synthetic'),
        'medmnist': run_medmnist_verification(
            output_dir / 'medmnist',
            use_real_causal_tracing=use_real_causal_tracing,
            causal_tracing_max_patches=causal_tracing_max_patches,
            checkpoint_dir=checkpoint_dir,
            force_retrain=force_retrain
        )
    }

    return results


def main():
    """Main entry point for verification script."""
    parser = argparse.ArgumentParser(
        description="Medical AlphaEdit Feasibility Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run synthetic verification only (no external dependencies)
    python run_verification.py --mode synthetic

    # Run verification with MedMNIST dataset (simulated causal tracing, fast)
    python run_verification.py --mode medmnist

    # Run verification with REAL causal tracing (slower but accurate)
    python run_verification.py --mode medmnist --real-causal-tracing

    # Force retraining even if checkpoint exists
    python run_verification.py --mode medmnist --force-retrain

    # Use custom checkpoint directory
    python run_verification.py --mode medmnist --checkpoint-dir ./my_checkpoints

    # Run full verification
    python run_verification.py --mode full

    # Specify custom output directory
    python run_verification.py --mode full --output-dir ./my_outputs
        """
    )

    parser.add_argument(
        '--mode',
        choices=['synthetic', 'medmnist', 'full'],
        default='synthetic',
        help='Verification mode: synthetic (no deps), medmnist (requires data), full (both)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Directory to save output files'
    )

    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualization plots'
    )

    parser.add_argument(
        '--real-causal-tracing',
        action='store_true',
        help='Use real Corrupt-Restore causal tracing instead of simulation (slower but accurate)'
    )

    parser.add_argument(
        '--causal-tracing-max-patches',
        type=int,
        default=20,
        help='Maximum patches to analyze per layer when using real causal tracing (default: 20)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=None,
        help='Directory to save/load model checkpoints (default: ./checkpoints)'
    )

    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining even if checkpoint exists'
    )

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("MEDICAL ALPHAEDIT FEASIBILITY VERIFICATION")
    print("=" * 70)
    print(f"\nMode: {args.mode}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    if args.mode in ['medmnist', 'full']:
        print(f"Real causal tracing: {args.real_causal_tracing}")
        if args.real_causal_tracing:
            print(f"Causal tracing max patches: {args.causal_tracing_max_patches}")
        print(f"Force retrain: {args.force_retrain}")
        if args.checkpoint_dir:
            print(f"Checkpoint directory: {args.checkpoint_dir}")

    # Run verification based on mode
    if args.mode == 'synthetic':
        results = run_synthetic_verification(output_dir)
    elif args.mode == 'medmnist':
        results = run_medmnist_verification(
            output_dir,
            use_real_causal_tracing=args.real_causal_tracing,
            causal_tracing_max_patches=args.causal_tracing_max_patches,
            checkpoint_dir=args.checkpoint_dir,
            force_retrain=args.force_retrain
        )
    else:
        results = run_full_verification(
            output_dir,
            use_real_causal_tracing=args.real_causal_tracing,
            causal_tracing_max_patches=args.causal_tracing_max_patches,
            checkpoint_dir=args.checkpoint_dir,
            force_retrain=args.force_retrain
        )

    # Save results to JSON
    results_file = output_dir / 'verification_results.json'

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    serializable_results = convert_to_serializable(results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate visualizations
    if not args.no_visualizations:
        print("\nGenerating visualizations...")
        try:
            viz_dir = output_dir / 'visualizations'
            created_files = create_all_visualizations(results, viz_dir)
            print(f"Created {len(created_files)} visualization files")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")

    # Generate summary report
    print("\nGenerating summary report...")
    report_file = output_dir / 'verification_report.txt'
    report_text = generate_summary_report(results, report_file)
    print(report_text)

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles generated:")
    for f in output_dir.rglob('*'):
        if f.is_file():
            print(f"  - {f.relative_to(output_dir)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
