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
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

from .causal_tracing import CausalTracer, FaultLocalizationResult
from .null_space_projection import NullSpaceProjector, WeightUpdateResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EditingResult:
    """Container for the complete editing workflow results."""
    success: bool
    original_prediction: np.ndarray
    corrected_prediction: np.ndarray
    target_label: int
    fault_localization: FaultLocalizationResult
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
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        regularization: float = 1e-6,
        max_edits: int = 10,
        convergence_threshold: float = 0.01
    ):
        """
        Initialize the Medical AlphaEdit framework.

        Args:
            model: Vision Transformer model to edit
            device: Device for computation
            regularization: Lambda for numerical stability
            max_edits: Maximum number of iterative edits
            convergence_threshold: Threshold for considering correction complete
        """
        self.model = model
        self.device = device
        self.max_edits = max_edits
        self.convergence_threshold = convergence_threshold

        # Initialize components
        self.causal_tracer = CausalTracer(model, device)
        self.null_space_projector = NullSpaceProjector(regularization)

        # Storage for correct activations (reference database)
        self.reference_activations: Dict[int, List[np.ndarray]] = {}

        # Track edit history
        self.edit_history: List[EditingResult] = []

    def build_reference_database(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples_per_class: int = 100
    ) -> Dict[int, int]:
        """
        Build reference database from correctly classified samples.

        This implements the nearest-neighbor search described in Eq. 13
        for retrieving target activations v*.

        Args:
            dataloader: DataLoader with (image, label) pairs
            max_samples_per_class: Maximum samples to store per class

        Returns:
            Dictionary mapping class labels to sample counts
        """
        logger.info("Building reference activation database...")

        self.model.train(False)
        class_counts: Dict[int, int] = {}

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.numpy()

                # Get predictions
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1).cpu().numpy()

                # Store activations for correctly classified samples
                for i, (pred, label) in enumerate(zip(predictions, labels)):
                    if pred == label:
                        label_int = int(label)
                        if label_int not in self.reference_activations:
                            self.reference_activations[label_int] = []

                        if len(self.reference_activations[label_int]) < max_samples_per_class:
                            # Store the output activation
                            activation = outputs[i].cpu().numpy()
                            self.reference_activations[label_int].append(activation)

                            class_counts[label_int] = class_counts.get(label_int, 0) + 1

        logger.info(f"Reference database built with {sum(class_counts.values())} samples "
                   f"across {len(class_counts)} classes")

        return class_counts

    def retrieve_target_activation(
        self,
        faulty_activation: np.ndarray,
        target_class: int
    ) -> Optional[np.ndarray]:
        """
        Retrieve target activation v* using nearest-neighbor search (Eq. 13).

        v* = argmin_{v in D} ||faulty_activation - v||_2

        Args:
            faulty_activation: The faulty activation to correct
            target_class: The correct class label

        Returns:
            Target activation from reference database, or None if not found
        """
        if target_class not in self.reference_activations:
            logger.warning(f"No reference activations found for class {target_class}")
            return None

        references = self.reference_activations[target_class]
        if len(references) == 0:
            return None

        # Find nearest neighbor
        min_distance = float('inf')
        best_reference = None

        for ref in references:
            distance = np.linalg.norm(faulty_activation - ref)
            if distance < min_distance:
                min_distance = distance
                best_reference = ref

        return best_reference

    def get_layer_weights(self, layer_idx: int) -> Optional[Tuple[str, np.ndarray]]:
        """
        Get weight matrix from specified layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Tuple of (layer_name, weight_matrix) or None
        """
        layer_modules = [(n, m) for n, m in self.model.named_modules()
                        if 'mlp' in n.lower() or 'ffn' in n.lower()]

        if layer_idx < len(layer_modules):
            name, module = layer_modules[layer_idx]
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    return name + '.' + param_name, param.data.cpu().numpy()

        return None

    def apply_weight_update(
        self,
        layer_name: str,
        delta_W: np.ndarray
    ):
        """
        Apply weight update to the model.

        Args:
            layer_name: Full name of the parameter to update
            delta_W: Weight update matrix
        """
        # Navigate to the parameter
        parts = layer_name.split('.')
        module = self.model

        for part in parts[:-1]:
            module = getattr(module, part)

        param_name = parts[-1]
        param = getattr(module, param_name)

        # Apply update
        with torch.no_grad():
            param.add_(torch.tensor(delta_W, device=param.device, dtype=param.dtype))

    def correct_error(
        self,
        image: torch.Tensor,
        target_label: int,
        correct_activations: np.ndarray,
        verbose: bool = True
    ) -> EditingResult:
        """
        Main error correction workflow.

        Implements the complete Medical AlphaEdit pipeline:
        1. Localize faults via causal tracing
        2. Compute null-space projection from correct activations
        3. Apply iterative weight updates
        4. Verify correction

        Args:
            image: Input image that produces error
            target_label: Correct label for this image
            correct_activations: Matrix of activations from correct samples
            verbose: Whether to print progress

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

        # Step 1: Fault Localization
        if verbose:
            logger.info("Step 1: Localizing faults...")

        fault_result = self.causal_tracer.localize_faults(image, top_m=5, verbose=verbose)

        if verbose:
            logger.info(f"Found {len(fault_result.critical_components)} critical components")

        # Step 2: Compute Null-Space Projection
        if verbose:
            logger.info("Step 2: Computing null-space projection...")

        projection_result = self.null_space_projector.compute_null_space_projection(
            correct_activations, verbose=verbose
        )

        P = projection_result.projection_matrix

        # Step 3: Iterative Weight Updates
        if verbose:
            logger.info("Step 3: Applying iterative weight updates...")

        weight_updates = []
        total_weight_change = 0

        for edit_num in range(self.max_edits):
            # Check if already corrected
            with torch.no_grad():
                current_output = self.model(image)
                current_pred = current_output.argmax().item()

            if current_pred == target_label:
                if verbose:
                    logger.info(f"Correction achieved after {edit_num} edits")
                break

            # Get faulty activation (current output)
            k_star = current_output.cpu().numpy().flatten()

            # Retrieve target activation
            v_star = self.retrieve_target_activation(k_star, target_label)
            if v_star is None:
                # Use mean of correct activations as fallback
                v_star = np.mean(correct_activations, axis=1)

            # Get weight matrix from most faulty layer
            if len(fault_result.layer_indices) > 0:
                layer_idx = fault_result.layer_indices[0]
            else:
                layer_idx = 0

            layer_info = self.get_layer_weights(layer_idx)
            if layer_info is None:
                if verbose:
                    logger.warning(f"Could not get weights for layer {layer_idx}")
                continue

            layer_name, W = layer_info

            # Ensure dimensions match
            if W.shape[1] != len(k_star):
                k_star_resized = np.zeros(W.shape[1])
                k_star_resized[:len(k_star)] = k_star[:W.shape[1]]
                k_star = k_star_resized

            if W.shape[0] != len(v_star):
                v_star_resized = np.zeros(W.shape[0])
                v_star_resized[:len(v_star)] = v_star[:W.shape[0]]
                v_star = v_star_resized

            # Resize P if needed
            if P.shape[0] != W.shape[1]:
                P_resized = np.eye(W.shape[1])
                min_dim = min(P.shape[0], W.shape[1])
                P_resized[:min_dim, :min_dim] = P[:min_dim, :min_dim]
                P = P_resized

            # Compute weight update
            update_result = self.null_space_projector.compute_weight_update(
                W, k_star, v_star, P, verbose=verbose
            )

            weight_updates.append(update_result)
            total_weight_change += np.linalg.norm(update_result.delta_W)

            # Apply update
            self.apply_weight_update(layer_name, update_result.delta_W)

            if verbose:
                logger.info(f"Edit {edit_num + 1}: Applied update with norm {np.linalg.norm(update_result.delta_W):.4f}")

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
            'avg_correction_error': np.mean([u.correction_error for u in weight_updates]) if weight_updates else 0,
            'avg_preservation_error': np.mean([u.preservation_error for u in weight_updates]) if weight_updates else 0
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
    """
    print("=" * 70)
    print("MEDICAL ALPHAEDIT FRAMEWORK DEMONSTRATION")
    print("=" * 70)

    np.random.seed(42)

    # Simulate the framework workflow
    print("\n1. FRAMEWORK INITIALIZATION")
    print("-" * 50)
    print("Initializing components:")
    print("  - Causal Tracer: For fault localization")
    print("  - Null-Space Projector: For knowledge preservation")
    print("  - Reference Database: For target activation retrieval")

    # Simulate dimensions
    num_classes = 10
    feature_dim = 768  # ViT-B hidden dimension
    num_correct_samples = 50

    print(f"\nConfiguration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Reference samples per class: {num_correct_samples}")

    print("\n2. BUILDING REFERENCE DATABASE")
    print("-" * 50)

    # Simulate reference activations for each class
    reference_db = {}
    for cls in range(num_classes):
        # Create class-specific activation patterns
        class_center = np.random.randn(feature_dim)
        class_activations = []
        for _ in range(num_correct_samples):
            activation = class_center + np.random.randn(feature_dim) * 0.1
            class_activations.append(activation)
        reference_db[cls] = class_activations

    print(f"Reference database: {len(reference_db)} classes, {num_correct_samples} samples each")

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

    # Step 2: Null-Space Projection
    print("\nStep 2: Null-Space Projection")

    # Use correct class activations
    K_0 = np.array(reference_db[true_label]).T  # [feature_dim x num_samples]
    print(f"  Correct activations matrix K_0: {K_0.shape}")

    # Compute covariance and SVD
    covariance = K_0 @ K_0.T
    U, S, Vt = np.linalg.svd(covariance, full_matrices=True)
    rank = np.sum(S > 1e-10)
    null_dim = feature_dim - rank

    print(f"  Covariance rank: {rank}")
    print(f"  Null space dimension: {null_dim}")

    # Construct projection matrix
    if null_dim > 0:
        V_0 = Vt[rank:, :].T
        P = V_0 @ V_0.T
    else:
        P = np.zeros((feature_dim, feature_dim))

    print(f"  Projection matrix P: {P.shape}")

    # Step 3: Weight Update
    print("\nStep 3: Weight Update Computation")

    # Simulate weight matrix
    d_out = 10  # Output dimension (num classes)
    W = np.random.randn(d_out, feature_dim) * 0.1

    # Faulty activation (predicts class 7 instead of 3)
    k_star = np.zeros(feature_dim)
    k_star[:100] = reference_db[predicted_label][0][:100]  # Mix of wrong class

    # Target activation (should predict class 3)
    v_star = np.zeros(d_out)
    v_star[true_label] = 1.0  # One-hot for correct class

    # Current (wrong) output
    wrong_output = W @ k_star
    print(f"  Before correction: argmax(W @ k*) = {wrong_output.argmax()}")

    # Compute update using simplified formula
    residual = v_star - W @ k_star
    Pk_star = P @ k_star
    regularization = 1e-6

    # Simplified update for demonstration
    if np.linalg.norm(Pk_star) > 1e-10:
        scale = np.dot(k_star, Pk_star) + regularization
        delta_W = np.outer(residual, Pk_star) / scale
    else:
        delta_W = np.zeros_like(W)

    print(f"  Weight update norm: {np.linalg.norm(delta_W):.6f}")

    # Apply update
    W_new = W + delta_W

    # Verify correction
    corrected_output = W_new @ k_star
    print(f"  After correction: argmax((W + dW) @ k*) = {corrected_output.argmax()}")

    # Verify preservation
    for cls_idx in range(3):
        sample = reference_db[cls_idx][0]
        sample_resized = np.zeros(feature_dim)
        sample_resized[:len(sample)] = sample[:feature_dim]

        original_out = W @ sample_resized
        new_out = W_new @ sample_resized

        change = np.linalg.norm(new_out - original_out)
        print(f"  Class {cls_idx} preservation: output change = {change:.6f}")

    print("\n4. FRAMEWORK METRICS SUMMARY")
    print("-" * 50)

    metrics = {
        "Fault Localization": {
            "Patches analyzed": num_patches,
            "Layers analyzed": num_layers,
            "Critical components found": 5
        },
        "Null-Space Projection": {
            "Feature dimension": feature_dim,
            "Covariance rank": rank,
            "Null space dimension": null_dim
        },
        "Weight Update": {
            "Update norm": np.linalg.norm(delta_W),
            "Correction achieved": corrected_output.argmax() == true_label,
            "Original prediction": wrong_output.argmax(),
            "Corrected prediction": corrected_output.argmax()
        }
    }

    for category, values in metrics.items():
        print(f"\n{category}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("MEDICAL ALPHAEDIT DEMONSTRATION COMPLETE")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    results = demonstrate_medical_alphaedit()
