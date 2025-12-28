"""
Medical AlphaEdit Framework Integration

This module integrates causal tracing and null-space projection into
a unified framework for train-free error correction in Medical ViTs.

The workflow follows Section 4.3 of the paper:
1. Extract patch embeddings and intermediate activations
2. Compute fault scores to identify critical components
3. Retrieve reference activations from correct samples
4. Apply null-space projected weight updates
5. Verify correction while maintaining general performance

Key Fix: Activations are now collected from MLP fc1 layer (dim=3072)
instead of output layer (dim=num_classes), enabling proper null-space editing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from .causal_tracing import CausalTracer, FaultLocalizationResult
from .null_space_projection import NullSpaceProjector, WeightUpdateResult, NullSpaceProjectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EditingResult:
    """Container for the complete editing workflow results."""
    success: bool
    original_prediction: np.ndarray
    corrected_prediction: np.ndarray
    target_label: int
    fault_localization: Optional[FaultLocalizationResult]
    weight_updates: List[WeightUpdateResult]
    num_edits: int
    total_weight_change: float
    preservation_metrics: Dict[str, float]


class MedicalAlphaEdit:
    """
    Medical AlphaEdit Framework for Train-Free Error Correction.

    This framework combines:
    1. Causal tracing for fault localization
    2. Null-space projection for knowledge-preserving edits
    3. Iterative refinement for complete error correction

    Key features:
    - No retraining required
    - Preserves performance on correct samples
    - Provides theoretical guarantees via null-space constraints

    IMPORTANT: This implementation collects activations from MLP fc1 layer
    (dimension 3072 for ViT-B/16) to ensure sufficient null-space for editing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        regularization: float = 1e-6,
        max_edits: int = 10,
        convergence_threshold: float = 0.01,
        target_layer: int = 10
    ):
        """
        Initialize the Medical AlphaEdit framework.

        Args:
            model: Vision Transformer model to edit (timm ViT)
            device: Device for computation
            regularization: Lambda for numerical stability
            max_edits: Maximum number of iterative edits
            convergence_threshold: Threshold for considering correction complete
            target_layer: Which transformer block to edit (0-11 for ViT-B)
        """
        self.model = model
        self.device = device
        self.max_edits = max_edits
        self.convergence_threshold = convergence_threshold
        self.target_layer = target_layer

        # Initialize components
        self.causal_tracer = CausalTracer(model, device)
        self.null_space_projector = NullSpaceProjector(regularization)

        # Storage for correct activations (reference database)
        # Now stores fc1 activations [3072 x num_samples] per class
        self.reference_activations: Dict[int, np.ndarray] = {}

        # Cache for projection matrices per class
        self.projection_cache: Dict[int, NullSpaceProjectionResult] = {}

        # Track edit history
        self.edit_history: List[EditingResult] = []

        # Verify model structure
        self._verify_model_structure()

    def _verify_model_structure(self):
        """Verify the model has expected ViT structure."""
        if not hasattr(self.model, 'blocks'):
            logger.warning("Model does not have 'blocks' attribute. "
                          "Expected timm ViT model structure.")
            return

        num_blocks = len(self.model.blocks)
        if self.target_layer >= num_blocks:
            logger.warning(f"target_layer={self.target_layer} >= num_blocks={num_blocks}. "
                          f"Setting target_layer to {num_blocks - 1}")
            self.target_layer = num_blocks - 1

        # Check MLP structure
        block = self.model.blocks[self.target_layer]
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc1'):
            fc1 = block.mlp.fc1
            logger.info(f"Target layer {self.target_layer} MLP fc1: "
                       f"in={fc1.in_features}, out={fc1.out_features}")
        else:
            logger.warning("Could not find MLP.fc1 in target block")

    def build_reference_database(
        self,
        dataloader: torch.utils.data.DataLoader,
        target_layer: Optional[int] = None,
        max_samples_per_class: int = 100
    ) -> Dict[int, int]:
        """
        Build reference database from correctly classified samples.

        CRITICAL: This method collects fc1 output activations (dim=3072)
        instead of model output logits (dim=num_classes).

        Args:
            dataloader: DataLoader with (image, label) pairs
            target_layer: Which layer to collect from (uses self.target_layer if None)
            max_samples_per_class: Maximum samples to store per class

        Returns:
            Dictionary mapping class labels to sample counts
        """
        if target_layer is not None:
            self.target_layer = target_layer

        logger.info(f"Building reference database from layer {self.target_layer} fc1...")

        self.model.eval()
        class_activations: Dict[int, List[np.ndarray]] = {}
        hook_storage = {}

        # Register hook on target layer's fc1
        fc1_module = self.model.blocks[self.target_layer].mlp.fc1

        def capture_fc1(module, inp, out):
            hook_storage['fc1'] = out.detach()

        hook = fc1_module.register_forward_hook(capture_fc1)

        try:
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(self.device)
                    # Handle different label formats
                    if isinstance(labels, torch.Tensor):
                        if labels.dim() > 1:
                            labels = labels.squeeze()
                        labels_np = labels.cpu().numpy()
                    else:
                        labels_np = np.array(labels)

                    # Forward pass captures fc1 output via hook
                    outputs = self.model(images)
                    predictions = outputs.argmax(dim=1).cpu().numpy()

                    # fc1_out shape: [batch, num_patches, fc1_dim]
                    # For ViT-B/16: [batch, 197, 3072]
                    fc1_out = hook_storage['fc1'].cpu().numpy()

                    # Use CLS token activation (index 0)
                    # Shape: [batch, 3072]
                    cls_activations = fc1_out[:, 0, :]

                    for i, (pred, label) in enumerate(zip(predictions, labels_np)):
                        label_int = int(label)
                        if pred == label_int:  # Correctly classified
                            if label_int not in class_activations:
                                class_activations[label_int] = []

                            if len(class_activations[label_int]) < max_samples_per_class:
                                class_activations[label_int].append(cls_activations[i])

                    # Check if we have enough samples
                    total_samples = sum(len(v) for v in class_activations.values())
                    if total_samples >= max_samples_per_class * len(class_activations):
                        if len(class_activations) >= 5:  # At least 5 classes
                            break

        finally:
            hook.remove()

        # Convert to matrices [fc1_dim x num_samples]
        self.reference_activations = {}
        class_counts = {}

        for label, acts in class_activations.items():
            if len(acts) > 0:
                # Stack as [num_samples, fc1_dim] then transpose to [fc1_dim, num_samples]
                self.reference_activations[label] = np.array(acts).T
                class_counts[label] = len(acts)
                logger.info(f"  Class {label}: {len(acts)} samples, "
                           f"shape {self.reference_activations[label].shape}")

        total = sum(class_counts.values())
        logger.info(f"Reference database built: {total} samples across {len(class_counts)} classes")

        # Clear projection cache since activations changed
        self.projection_cache = {}

        return class_counts

    def get_projection_matrix(self, target_class: int, verbose: bool = False) -> Optional[NullSpaceProjectionResult]:
        """
        Get or compute null-space projection matrix for a class.

        Args:
            target_class: Class label
            verbose: Whether to print details

        Returns:
            NullSpaceProjectionResult or None if class not in database
        """
        if target_class in self.projection_cache:
            return self.projection_cache[target_class]

        if target_class not in self.reference_activations:
            logger.warning(f"No reference activations for class {target_class}")
            return None

        K_0 = self.reference_activations[target_class]
        result = self.null_space_projector.compute_null_space_projection(K_0, verbose=verbose)

        self.projection_cache[target_class] = result
        return result

    def retrieve_target_activation(
        self,
        k_star: np.ndarray,
        target_class: int,
        method: str = 'nearest'
    ) -> Optional[np.ndarray]:
        """
        Retrieve target activation v* for error correction.

        v* is the desired fc2 output, computed from the nearest correct activation.

        Args:
            k_star: Faulty fc1 activation [fc1_dim]
            target_class: The correct class label
            method: 'nearest' for nearest-neighbor, 'gradient' for optimization

        Returns:
            Target activation v* [fc2_out_dim] or None
        """
        if target_class not in self.reference_activations:
            logger.warning(f"No reference activations for class {target_class}")
            return None

        K_0 = self.reference_activations[target_class]  # [fc1_dim x num_samples]

        # Get fc2 weight matrix
        fc2 = self.model.blocks[self.target_layer].mlp.fc2
        W = fc2.weight.data.cpu().numpy()  # [fc2_out_dim x fc1_dim]

        if method == 'nearest':
            # Find nearest neighbor in activation space
            # K_0.T has shape [num_samples, fc1_dim]
            distances = np.linalg.norm(K_0.T - k_star, axis=1)
            nearest_idx = np.argmin(distances)
            k_nearest = K_0[:, nearest_idx]

            # v* = W @ k_nearest (what fc2 would output for the correct activation)
            v_star = W @ k_nearest
            return v_star

        elif method == 'gradient':
            return self._optimize_target_activation(k_star, W, target_class)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _optimize_target_activation(
        self,
        k_star: np.ndarray,
        W: np.ndarray,
        target_class: int,
        num_steps: int = 100,
        lr: float = 0.1
    ) -> np.ndarray:
        """
        Optimize v* to maximize probability of target class using gradient descent.

        Args:
            k_star: Faulty activation [fc1_dim]
            W: Weight matrix [fc2_out_dim x fc1_dim]
            target_class: Target class label
            num_steps: Optimization steps
            lr: Learning rate

        Returns:
            Optimized v* [fc2_out_dim]
        """
        # Initial v* from current W @ k*
        v_init = W @ k_star
        v = torch.tensor(v_init, requires_grad=True, dtype=torch.float32, device='cpu')
        optimizer = torch.optim.Adam([v], lr=lr)

        # Get classifier head weights
        head_W = self.model.head.weight.data.cpu()  # [num_classes, hidden_dim]
        head_b = self.model.head.bias.data.cpu() if self.model.head.bias is not None else None

        target = torch.tensor([target_class], dtype=torch.long)

        for step in range(num_steps):
            optimizer.zero_grad()

            # Simulate forward through head
            if head_b is not None:
                logits = F.linear(v.unsqueeze(0), head_W, head_b)
            else:
                logits = F.linear(v.unsqueeze(0), head_W)

            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

        return v.detach().numpy()

    def get_fc1_activation(self, image: torch.Tensor) -> np.ndarray:
        """
        Get fc1 output activation for an image.

        Args:
            image: Input image tensor [1, C, H, W]

        Returns:
            fc1 activation [fc1_dim] (CLS token)
        """
        hook_storage = {}
        fc1_module = self.model.blocks[self.target_layer].mlp.fc1

        def capture_fc1(module, inp, out):
            hook_storage['fc1'] = out.detach()

        hook = fc1_module.register_forward_hook(capture_fc1)

        try:
            with torch.no_grad():
                self.model(image)

            # [1, num_patches, fc1_dim] -> [fc1_dim]
            fc1_out = hook_storage['fc1'][0, 0, :].cpu().numpy()
            return fc1_out
        finally:
            hook.remove()

    def get_fc2_weights(self, layer_idx: Optional[int] = None) -> Tuple[str, np.ndarray]:
        """
        Get fc2 weight matrix from the specified layer.

        Args:
            layer_idx: Layer index (uses self.target_layer if None)

        Returns:
            Tuple of (layer_name, weight_matrix)
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        fc2 = self.model.blocks[layer_idx].mlp.fc2
        layer_name = f"blocks.{layer_idx}.mlp.fc2.weight"
        W = fc2.weight.data.cpu().numpy()  # [hidden_dim x fc1_dim] = [768 x 3072]
        return layer_name, W

    def apply_weight_update(
        self,
        delta_W: np.ndarray,
        layer_idx: Optional[int] = None
    ):
        """
        Apply weight update to fc2 of the target layer.

        Args:
            delta_W: Weight update matrix [hidden_dim x fc1_dim]
            layer_idx: Layer index (uses self.target_layer if None)
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        fc2 = self.model.blocks[layer_idx].mlp.fc2

        # Apply update
        with torch.no_grad():
            fc2.weight.add_(torch.tensor(delta_W, device=fc2.weight.device, dtype=fc2.weight.dtype))

    def correct_error(
        self,
        image: torch.Tensor,
        target_label: int,
        verbose: bool = True,
        use_causal_tracing: bool = False
    ) -> EditingResult:
        """
        Main error correction workflow using fc1/fc2 layer editing.

        Implements the complete Medical AlphaEdit pipeline:
        1. Get fc1 activation k* for the error sample
        2. Retrieve or compute target activation v*
        3. Get null-space projection P from reference activations
        4. Compute and apply constrained weight update to fc2
        5. Verify correction

        All operations work in proper dimensions:
        - k*: [fc1_dim] = [3072] for ViT-B/16
        - v*: [hidden_dim] = [768] for ViT-B/16
        - W (fc2.weight): [hidden_dim x fc1_dim] = [768 x 3072]
        - P: [fc1_dim x fc1_dim] = [3072 x 3072]

        Args:
            image: Input image that produces error
            target_label: Correct label for this image
            verbose: Whether to print progress
            use_causal_tracing: Whether to run full causal tracing (slow)

        Returns:
            EditingResult with complete correction information
        """
        self.model.train(False)
        image = image.to(self.device)

        # Get original prediction
        with torch.no_grad():
            original_output = self.model(image)
            original_pred = original_output.cpu().numpy()

        if verbose:
            logger.info(f"Original prediction: {original_pred.argmax()}, Target: {target_label}")

        # Step 1: Fault Localization (optional, for analysis)
        fault_result = None
        if use_causal_tracing:
            if verbose:
                logger.info("Step 1: Localizing faults...")
            fault_result = self.causal_tracer.localize_faults(image, top_m=5, verbose=verbose)
            if verbose:
                logger.info(f"Found {len(fault_result.critical_components)} critical components")

        # Step 2: Get null-space projection for target class
        if verbose:
            logger.info("Step 2: Computing null-space projection...")

        projection_result = self.get_projection_matrix(target_label, verbose=verbose)
        if projection_result is None:
            logger.error(f"Cannot correct: no reference activations for class {target_label}")
            return EditingResult(
                success=False,
                original_prediction=original_pred,
                corrected_prediction=original_pred,
                target_label=target_label,
                fault_localization=fault_result,
                weight_updates=[],
                num_edits=0,
                total_weight_change=0,
                preservation_metrics={'error': 'no_reference_activations'}
            )

        P = projection_result.projection_matrix  # [fc1_dim x fc1_dim]

        if verbose:
            logger.info(f"  Null space dimension: {projection_result.null_space_dim}")
            logger.info(f"  Projection matrix P shape: {P.shape}")

        # Step 3: Iterative Weight Updates
        if verbose:
            logger.info("Step 3: Applying iterative weight updates...")

        weight_updates = []
        total_weight_change = 0.0

        for edit_num in range(self.max_edits):
            # Check if already corrected
            with torch.no_grad():
                current_output = self.model(image)
                current_pred = current_output.argmax().item()

            if current_pred == target_label:
                if verbose:
                    logger.info(f"Correction achieved after {edit_num} edits")
                break

            # Get faulty fc1 activation k* (CLS token)
            k_star = self.get_fc1_activation(image)  # [fc1_dim] = [3072]

            if verbose and edit_num == 0:
                logger.info(f"  k* (fc1 activation) shape: {k_star.shape}")

            # Retrieve target activation v*
            v_star = self.retrieve_target_activation(k_star, target_label, method='nearest')
            if v_star is None:
                logger.warning(f"Could not retrieve target activation for class {target_label}")
                break

            if verbose and edit_num == 0:
                logger.info(f"  v* (target activation) shape: {v_star.shape}")

            # Get fc2 weight matrix W
            layer_name, W = self.get_fc2_weights()  # [hidden_dim x fc1_dim] = [768 x 3072]

            if verbose and edit_num == 0:
                logger.info(f"  W (fc2.weight) shape: {W.shape}")

            # Verify dimensions are correct (no resizing needed!)
            assert k_star.shape[0] == W.shape[1] == P.shape[0], \
                f"Dimension mismatch: k*={k_star.shape}, W={W.shape}, P={P.shape}"
            assert v_star.shape[0] == W.shape[0], \
                f"Output dimension mismatch: v*={v_star.shape}, W={W.shape}"

            # Compute weight update using null-space projection
            update_result = self.null_space_projector.compute_weight_update(
                W, k_star, v_star, P, verbose=(verbose and edit_num == 0)
            )

            weight_updates.append(update_result)
            total_weight_change += np.linalg.norm(update_result.delta_W)

            # Apply update to fc2.weight
            self.apply_weight_update(update_result.delta_W)

            if verbose:
                logger.info(f"Edit {edit_num + 1}: ||ΔW||={np.linalg.norm(update_result.delta_W):.4e}, "
                           f"correction_err={update_result.correction_error:.4e}")

        # Get final prediction
        with torch.no_grad():
            final_output = self.model(image)
            final_pred = final_output.cpu().numpy()

        success = final_pred.argmax() == target_label

        # Compute preservation metrics
        preservation_metrics = {
            'total_edits': len(weight_updates),
            'total_weight_change': total_weight_change,
            'null_space_dim': projection_result.null_space_dim,
            'covariance_rank': projection_result.rank,
            'avg_correction_error': float(np.mean([u.correction_error for u in weight_updates])) if weight_updates else 0,
            'avg_preservation_error': float(np.mean([u.preservation_error for u in weight_updates])) if weight_updates else 0
        }

        result = EditingResult(
            success=success,
            original_prediction=original_pred,
            corrected_prediction=final_pred,
            target_label=target_label,
            fault_localization=fault_result,
            weight_updates=weight_updates,
            num_edits=len(weight_updates),
            total_weight_change=total_weight_change,
            preservation_metrics=preservation_metrics
        )

        self.edit_history.append(result)

        if verbose:
            logger.info(f"Correction {'successful' if success else 'failed'}: "
                       f"Original={original_pred.argmax()}, Final={final_pred.argmax()}, Target={target_label}")

        return result

    def measure_preservation(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Measure knowledge preservation on held-out samples.

        Measures specificity: accuracy on samples that were NOT edited.

        Args:
            dataloader: DataLoader for testing
            num_samples: Number of samples to test

        Returns:
            Dictionary with preservation metrics
        """
        self.model.train(False)
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                if total >= num_samples:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += len(labels)

        accuracy = correct / total if total > 0 else 0

        return {
            'preservation_accuracy': accuracy,
            'samples_tested': total,
            'correct': correct
        }

    def cleanup(self):
        """Clean up resources."""
        self.causal_tracer.cleanup()


def demonstrate_medical_alphaedit():
    """
    Demonstrate the Medical AlphaEdit framework with synthetic components.

    This provides a complete walkthrough of the algorithm without
    requiring actual medical imaging data.

    Key dimensions for ViT-B/16:
    - fc1_dim: 3072 (MLP expansion)
    - hidden_dim: 768 (ViT hidden size)
    - num_classes: varies (e.g., 9 for PathMNIST)
    """
    print("=" * 70)
    print("MEDICAL ALPHAEDIT FRAMEWORK DEMONSTRATION")
    print("=" * 70)

    np.random.seed(42)

    # Simulate the framework workflow with correct dimensions
    print("\n1. FRAMEWORK INITIALIZATION")
    print("-" * 50)
    print("Initializing components:")
    print("  - Causal Tracer: For fault localization")
    print("  - Null-Space Projector: For knowledge preservation")
    print("  - Reference Database: For target activation retrieval")

    # Correct dimensions for ViT-B/16
    fc1_dim = 3072      # fc1 output dimension (MLP expansion)
    hidden_dim = 768    # ViT hidden dimension (fc2 output)
    num_classes = 10
    num_correct_samples = 50

    print(f"\nConfiguration (ViT-B/16 dimensions):")
    print(f"  fc1 dimension: {fc1_dim}")
    print(f"  hidden dimension: {hidden_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Reference samples per class: {num_correct_samples}")

    print("\n2. BUILDING REFERENCE DATABASE (fc1 activations)")
    print("-" * 50)

    # Simulate reference fc1 activations for each class
    # These are collected from correctly classified samples
    reference_db = {}
    for cls in range(num_classes):
        # Create class-specific activation patterns in fc1 space
        class_center = np.random.randn(fc1_dim) * 0.5
        class_activations = []
        for _ in range(num_correct_samples):
            # Each sample's fc1 output (CLS token)
            activation = class_center + np.random.randn(fc1_dim) * 0.1
            class_activations.append(activation)
        reference_db[cls] = class_activations

    print(f"Reference database: {len(reference_db)} classes")
    print(f"  Each sample: fc1 activation of shape [{fc1_dim}]")
    print(f"  Matrix K_0 per class: [{fc1_dim} x {num_correct_samples}]")

    print("\n3. SIMULATING ERROR CORRECTION WORKFLOW")
    print("-" * 50)

    # Simulate an error case
    true_label = 3
    predicted_label = 7

    print(f"Error case:")
    print(f"  True label: {true_label}")
    print(f"  Predicted label: {predicted_label}")

    # Step 1: Fault Localization
    print("\nStep 1: Fault Localization")
    num_patches = 196
    num_layers = 12
    fault_scores = np.random.exponential(0.1, (num_patches, num_layers))
    fault_scores[50:55, 8:10] = np.random.exponential(0.5, (5, 2)) + 0.5

    flat_scores = fault_scores.flatten()
    top_5_indices = np.argsort(flat_scores)[-5:][::-1]

    print(f"  Top 5 critical components:")
    for rank, idx in enumerate(top_5_indices, 1):
        patch_idx = idx // num_layers
        layer_idx = idx % num_layers
        score = fault_scores[patch_idx, layer_idx]
        print(f"    {rank}. Patch {patch_idx}, Layer {layer_idx}: score = {score:.4f}")

    # Step 2: Null-Space Projection (using fc1 activations)
    print("\nStep 2: Null-Space Projection")

    # Use correct class fc1 activations
    K_0 = np.array(reference_db[true_label]).T  # [fc1_dim x num_samples] = [3072 x 50]
    print(f"  Correct activations matrix K_0: {K_0.shape}")

    # Compute covariance and SVD
    covariance = K_0 @ K_0.T  # [3072 x 3072]
    U, S, Vt = np.linalg.svd(covariance, full_matrices=True)
    rank = np.sum(S > 1e-10)
    null_dim = fc1_dim - rank

    print(f"  Covariance matrix: {covariance.shape}")
    print(f"  Covariance rank: {rank}")
    print(f"  Null space dimension: {null_dim}")
    print(f"  Expected: ~{fc1_dim - num_correct_samples} (fc1_dim - num_samples)")

    # Construct projection matrix P
    if null_dim > 0:
        V_0 = Vt[rank:, :].T  # Null space basis
        P = V_0 @ V_0.T  # [fc1_dim x fc1_dim]
    else:
        P = np.zeros((fc1_dim, fc1_dim))

    print(f"  Projection matrix P: {P.shape}")

    # Step 3: Weight Update (on fc2)
    print("\nStep 3: Weight Update Computation (fc2)")

    # Simulate fc2 weight matrix: [hidden_dim x fc1_dim] = [768 x 3072]
    W = np.random.randn(hidden_dim, fc1_dim) * 0.02

    # Faulty fc1 activation k* (from the misclassified sample)
    k_star = reference_db[predicted_label][0].copy()  # [3072]
    k_star += np.random.randn(fc1_dim) * 0.2  # Add some noise

    # Current wrong fc2 output
    wrong_output = W @ k_star  # [768]
    print(f"  k* (fc1 activation): shape {k_star.shape}")
    print(f"  W (fc2.weight): shape {W.shape}")
    print(f"  W @ k* (fc2 output): shape {wrong_output.shape}")

    # Target activation v* (from nearest correct sample)
    k_nearest = reference_db[true_label][0]  # Nearest correct fc1 activation
    v_star = W @ k_nearest  # What fc2 would output for correct activation

    print(f"  v* (target fc2 output): shape {v_star.shape}")

    # Compute update using null-space constraint
    residual = v_star - W @ k_star  # [768]
    Pk_star = P @ k_star  # [3072]
    regularization = 1e-6

    print(f"\n  Residual ||v* - W @ k*||: {np.linalg.norm(residual):.4f}")
    print(f"  ||P @ k*|| / ||k*||: {np.linalg.norm(Pk_star) / np.linalg.norm(k_star):.4f}")

    # Compute weight update ΔW
    if np.linalg.norm(Pk_star) > 1e-10:
        k_star_T_P = k_star @ P  # [3072]
        denominator_matrix = np.outer(k_star, k_star_T_P) + regularization * np.eye(fc1_dim)
        denominator_inv = np.linalg.inv(denominator_matrix)
        delta_W = np.outer(residual, k_star_T_P @ denominator_inv)  # [768 x 3072]
    else:
        delta_W = np.zeros_like(W)

    print(f"  ΔW shape: {delta_W.shape}")
    print(f"  ||ΔW||: {np.linalg.norm(delta_W):.6f}")

    # Apply update
    W_new = W + delta_W

    # Verify correction
    corrected_output = W_new @ k_star
    correction_error = np.linalg.norm(corrected_output - v_star)
    print(f"\n  Correction error ||(W + ΔW) @ k* - v*||: {correction_error:.2e}")

    # Verify preservation on correct samples
    print("\n  Preservation check on correct samples:")
    for cls_idx in range(3):
        sample = reference_db[cls_idx][0]
        original_out = W @ sample
        new_out = W_new @ sample
        change = np.linalg.norm(new_out - original_out)
        relative_change = change / (np.linalg.norm(original_out) + 1e-10)
        print(f"    Class {cls_idx}: relative output change = {relative_change:.2e}")

    print("\n4. FRAMEWORK METRICS SUMMARY")
    print("-" * 50)

    metrics = {
        "Fault Localization": {
            "Patches analyzed": num_patches,
            "Layers analyzed": num_layers,
            "Critical components found": 5
        },
        "Null-Space Projection": {
            "Feature dimension": fc1_dim,
            "Covariance rank": rank,
            "Null space dimension": null_dim
        },
        "Weight Update": {
            "Update norm": np.linalg.norm(delta_W),
            "Correction achieved": correction_error < 1e-4,
            "Correction error": correction_error,
            "fc2 weight shape": f"{W.shape}"
        }
    }

    for category, values in metrics.items():
        print(f"\n{category}:")
        for key, value in values.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("MEDICAL ALPHAEDIT DEMONSTRATION COMPLETE")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    results = demonstrate_medical_alphaedit()
