"""
Null-Space Projection Module for Knowledge-Preserving Weight Editing

This module implements the null-space constrained weight editing mechanism
described in Section 4.2 of the paper.

Key Equations:
- SVD of covariance: Sigma = U * Lambda * U^T (Eq. 6)
- Null-space projection matrix: P = V_0 * V_0^T (Eq. 11)
- Weight update: Delta_W = (v* - W*k*) * (k*^T * P) * (k* * k*^T * P + lambda*I)^{-1} (Eq. 12)

The core insight is that by projecting weight updates onto the null space of
correct activations, we can correct errors without disrupting learned knowledge.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NullSpaceProjectionResult:
    """Container for null-space projection results."""
    projection_matrix: np.ndarray
    null_space_basis: np.ndarray
    null_space_dim: int
    singular_values: np.ndarray
    rank: int
    condition_number: float


@dataclass
class WeightUpdateResult:
    """Container for weight update results."""
    delta_W: np.ndarray
    original_W: np.ndarray
    updated_W: np.ndarray
    target_activation: np.ndarray
    faulty_activation: np.ndarray
    preservation_error: float
    correction_error: float


class NullSpaceProjector:
    """
    Implements null-space projection for knowledge-preserving weight editing.

    The projector computes a projection matrix P that maps weight updates
    onto directions orthogonal to the span of correct activations.
    """

    def __init__(self, regularization: float = 1e-6, rank_threshold: float = 1e-10,
                 use_relative_threshold: bool = True, relative_threshold: float = 1e-6):
        """
        Initialize the null-space projector.

        Args:
            regularization: Lambda parameter for numerical stability (Eq. 12)
            rank_threshold: Absolute threshold for determining numerical rank
            use_relative_threshold: If True, use relative threshold based on max singular value
            relative_threshold: Threshold relative to max singular value (e.g., 1e-6 means
                              singular values < max_sv * 1e-6 are considered zero)
        """
        self.regularization = regularization
        self.rank_threshold = rank_threshold
        self.use_relative_threshold = use_relative_threshold
        self.relative_threshold = relative_threshold

    def compute_covariance_matrix(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix Sigma = K_0 * K_0^T

        Args:
            activations: Correct activations K_0 [d x m] where d is feature dim, m is num samples

        Returns:
            Covariance matrix [d x d]
        """
        if activations.ndim == 1:
            activations = activations.reshape(-1, 1)

        # Keep activations as [d x m] - do not transpose
        # The covariance is K_0 @ K_0.T which gives [d x d]
        covariance = activations @ activations.T
        return covariance

    def compute_null_space_projection(
        self,
        correct_activations: np.ndarray,
        verbose: bool = True
    ) -> NullSpaceProjectionResult:
        """
        Compute null-space projection matrix P = V_0 * V_0^T (Eq. 11)

        The projection matrix P projects vectors onto the null space of the
        covariance matrix of correct activations. This ensures that weight
        updates do not affect the model's behavior on correct samples.

        Args:
            correct_activations: Matrix K_0 of correct activations [d x m]
            verbose: Whether to print detailed information

        Returns:
            NullSpaceProjectionResult containing projection matrix and metadata
        """
        if verbose:
            logger.info(f"Computing null-space projection for activations shape: {correct_activations.shape}")

        # Compute covariance matrix
        covariance = self.compute_covariance_matrix(correct_activations)

        if verbose:
            logger.info(f"Covariance matrix shape: {covariance.shape}")

        # Perform SVD: Sigma = U * diag(singular_values) * V^T
        U, singular_values, Vt = np.linalg.svd(covariance, full_matrices=True)

        # Determine numerical rank using appropriate threshold
        if self.use_relative_threshold and len(singular_values) > 0 and singular_values[0] > 0:
            # Use relative threshold: singular values < max_sv * relative_threshold are zero
            threshold = singular_values[0] * self.relative_threshold
            rank = np.sum(singular_values > threshold)
        else:
            # Use absolute threshold
            rank = np.sum(singular_values > self.rank_threshold)

        if verbose:
            logger.info(f"Numerical rank: {rank} out of {len(singular_values)}")
            logger.info(f"Top 5 singular values: {singular_values[:5]}")

        # Extract null space basis (columns of V corresponding to zero singular values)
        null_space_dim = len(singular_values) - rank
        if null_space_dim > 0:
            # V_0 contains the last (d - rank) columns of V
            V_0 = Vt[rank:, :].T  # Transpose to get columns
        else:
            # If full rank, null space is empty
            V_0 = np.zeros((covariance.shape[0], 1))
            null_space_dim = 0

        if verbose:
            logger.info(f"Null space dimension: {null_space_dim}")

        # Compute projection matrix P = V_0 * V_0^T (Eq. 11)
        projection_matrix = V_0 @ V_0.T

        # Compute condition number for stability analysis
        if singular_values[-1] > 0:
            condition_number = singular_values[0] / singular_values[-1]
        else:
            condition_number = float('inf')

        return NullSpaceProjectionResult(
            projection_matrix=projection_matrix,
            null_space_basis=V_0,
            null_space_dim=null_space_dim,
            singular_values=singular_values,
            rank=rank,
            condition_number=condition_number
        )

    def compute_weight_update(
        self,
        W: np.ndarray,
        k_star: np.ndarray,
        v_star: np.ndarray,
        P: np.ndarray,
        verbose: bool = True
    ) -> WeightUpdateResult:
        """
        Compute constrained weight update Delta_W (Eq. 12)

        Delta_W = (v* - W*k*) * (k*^T * P) * (k* * k*^T * P + lambda*I)^{-1}

        This formulation guarantees:
        1. Error Correction: (W + Delta_W) * k* = v*
        2. Knowledge Preservation: (W + Delta_W) * K_0 = W * K_0

        Args:
            W: Original weight matrix [d_out x d_in]
            k_star: Faulty activation vector [d_in]
            v_star: Target corrected activation [d_out]
            P: Null-space projection matrix [d_in x d_in]

        Returns:
            WeightUpdateResult containing the update and verification metrics
        """
        if verbose:
            logger.info(f"Computing weight update:")
            logger.info(f"  W shape: {W.shape}")
            logger.info(f"  k* shape: {k_star.shape}")
            logger.info(f"  v* shape: {v_star.shape}")
            logger.info(f"  P shape: {P.shape}")

        # Ensure correct dimensions
        k_star = k_star.flatten()
        v_star = v_star.flatten()

        # Compute the residual (v* - W*k*)
        current_output = W @ k_star
        residual = v_star - current_output

        if verbose:
            logger.info(f"  Residual norm: {np.linalg.norm(residual):.6f}")

        # Project k* onto null space: P * k*
        Pk_star = P @ k_star

        if verbose:
            logger.info(f"  ||P*k*|| / ||k*||: {np.linalg.norm(Pk_star) / (np.linalg.norm(k_star) + 1e-10):.6f}")

        # Compute (k* * k*^T * P + lambda*I)
        # This is a scalar when k* is a vector
        k_star_T_P = k_star @ P  # [d_in]
        denominator_matrix = np.outer(k_star, k_star_T_P) + self.regularization * np.eye(len(k_star))

        # Compute pseudo-inverse for stability
        try:
            denominator_inv = np.linalg.inv(denominator_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed, using pseudo-inverse")
            denominator_inv = np.linalg.pinv(denominator_matrix)

        # Compute Delta_W = residual * (k*^T * P) * inverse_term
        # This is an outer product: [d_out] x [d_in]
        delta_W = np.outer(residual, k_star_T_P @ denominator_inv)

        # Compute updated weights
        updated_W = W + delta_W

        # Verify correction: (W + Delta_W) * k* should equal v*
        corrected_output = updated_W @ k_star
        correction_error = np.linalg.norm(corrected_output - v_star)

        if verbose:
            logger.info(f"  Correction error: {correction_error:.6e}")

        # For preservation verification, we need the original correct activations
        # Here we just compute the projection of delta_W to verify it's in null space
        preservation_error = np.linalg.norm((np.eye(P.shape[0]) - P) @ delta_W.T)

        if verbose:
            logger.info(f"  Preservation error (should be ~0): {preservation_error:.6e}")

        return WeightUpdateResult(
            delta_W=delta_W,
            original_W=W,
            updated_W=updated_W,
            target_activation=v_star,
            faulty_activation=k_star,
            preservation_error=preservation_error,
            correction_error=correction_error
        )


def verify_null_space_properties(
    P: np.ndarray,
    K_0: np.ndarray,
    delta_W: np.ndarray,
    tolerance: float = 1e-6
) -> Dict[str, bool]:
    """
    Verify that the null-space projection satisfies required properties.

    Properties to verify:
    1. P is symmetric: P = P^T
    2. P is idempotent: P^2 = P
    3. Delta_W * K_0 approximately equals 0 (preservation)

    Args:
        P: Projection matrix
        K_0: Correct activations matrix
        delta_W: Weight update matrix
        tolerance: Numerical tolerance for verification

    Returns:
        Dictionary of verification results
    """
    results = {}

    # Check symmetry
    symmetry_error = np.linalg.norm(P - P.T)
    results['symmetric'] = symmetry_error < tolerance
    results['symmetry_error'] = symmetry_error

    # Check idempotence
    P_squared = P @ P
    idempotence_error = np.linalg.norm(P_squared - P)
    results['idempotent'] = idempotence_error < tolerance
    results['idempotence_error'] = idempotence_error

    # Check preservation property
    if K_0.ndim == 1:
        K_0 = K_0.reshape(-1, 1)
    preservation = delta_W @ K_0
    preservation_error = np.linalg.norm(preservation)
    results['preserves_correct'] = preservation_error < tolerance * np.linalg.norm(K_0)
    results['preservation_error'] = preservation_error

    return results


def demonstrate_null_space_projection():
    """
    Demonstrate the null-space projection algorithm with synthetic data.

    This function verifies the mathematical correctness of the null-space
    projection and weight update equations from the paper.
    """
    print("=" * 70)
    print("NULL-SPACE PROJECTION DEMONSTRATION")
    print("=" * 70)

    np.random.seed(42)

    # Setup dimensions
    d_in = 64   # Input dimension (activation size)
    d_out = 64  # Output dimension (weight matrix output)
    m = 20      # Number of correct samples

    print(f"\n1. SETUP")
    print("-" * 50)
    print(f"Feature dimension: {d_in}")
    print(f"Output dimension: {d_out}")
    print(f"Number of correct samples: {m}")

    # Generate synthetic correct activations K_0
    # These represent activations from correctly classified samples
    K_0 = np.random.randn(d_in, m)

    print(f"Correct activations K_0 shape: {K_0.shape}")

    print("\n2. NULL-SPACE PROJECTION COMPUTATION")
    print("-" * 50)

    projector = NullSpaceProjector(regularization=1e-6)

    # Compute null-space projection
    result = projector.compute_null_space_projection(K_0, verbose=True)

    print(f"\nProjection matrix P shape: {result.projection_matrix.shape}")
    print(f"Null space dimension: {result.null_space_dim}")
    print(f"Covariance rank: {result.rank}")
    print(f"Condition number: {result.condition_number:.2e}")

    print("\n3. PROJECTION MATRIX PROPERTIES")
    print("-" * 50)

    P = result.projection_matrix

    # Check symmetry
    symmetry_error = np.linalg.norm(P - P.T)
    print(f"Symmetry check ||P - P^T||: {symmetry_error:.2e}")

    # Check idempotence (projection property)
    P_squared = P @ P
    idempotence_error = np.linalg.norm(P_squared - P)
    print(f"Idempotence check ||P^2 - P||: {idempotence_error:.2e}")

    # Verify that P projects onto null space of K_0
    # P * K_0 should be approximately zero
    projection_of_K0 = P @ K_0
    nullspace_check = np.linalg.norm(projection_of_K0) / np.linalg.norm(K_0)
    print(f"Null-space check ||P*K_0|| / ||K_0||: {nullspace_check:.2e}")

    print("\n4. WEIGHT UPDATE COMPUTATION")
    print("-" * 50)

    # Create synthetic weight matrix
    W = np.random.randn(d_out, d_in) * 0.1

    # Create faulty activation (the error case)
    k_star = np.random.randn(d_in)

    # Current (wrong) output
    wrong_output = W @ k_star

    # Target (correct) output
    v_star = wrong_output + np.random.randn(d_out) * 0.5  # Shift by correction

    print(f"Original weight matrix W shape: {W.shape}")
    print(f"Faulty activation k* shape: {k_star.shape}")
    print(f"Target activation v* shape: {v_star.shape}")

    # Compute weight update
    update_result = projector.compute_weight_update(W, k_star, v_star, P, verbose=True)

    print(f"\nWeight update Delta_W shape: {update_result.delta_W.shape}")
    print(f"Update magnitude ||Delta_W||: {np.linalg.norm(update_result.delta_W):.6f}")

    print("\n5. VERIFICATION OF CONSTRAINTS")
    print("-" * 50)

    # Constraint 1: Error Correction
    # (W + Delta_W) * k* should equal v*
    W_new = update_result.updated_W
    corrected_output = W_new @ k_star
    correction_error = np.linalg.norm(corrected_output - v_star)
    print(f"Error Correction: ||(W + Delta_W)*k* - v*|| = {correction_error:.2e}")

    # Constraint 2: Knowledge Preservation
    # (W + Delta_W) * K_0 should equal W * K_0
    original_on_K0 = W @ K_0
    updated_on_K0 = W_new @ K_0
    preservation_error = np.linalg.norm(updated_on_K0 - original_on_K0)
    relative_preservation_error = preservation_error / np.linalg.norm(original_on_K0)
    print(f"Knowledge Preservation: ||(W + Delta_W)*K_0 - W*K_0|| = {preservation_error:.2e}")
    print(f"Relative preservation error: {relative_preservation_error:.2e}")

    print("\n6. ITERATIVE EDITING SIMULATION")
    print("-" * 50)

    # Simulate multiple edits
    num_edits = 5
    W_current = W.copy()
    cumulative_preservation_error = 0

    print(f"Simulating {num_edits} sequential edits...")

    for i in range(num_edits):
        # Generate new faulty case
        k_i = np.random.randn(d_in)
        v_i = np.random.randn(d_out)

        # Recompute projection (in practice, would update incrementally)
        # For simplicity, keep same P here
        update_i = projector.compute_weight_update(W_current, k_i, v_i, P, verbose=False)

        W_current = update_i.updated_W

        # Check preservation after this edit
        current_on_K0 = W_current @ K_0
        preservation_i = np.linalg.norm(current_on_K0 - original_on_K0)
        cumulative_preservation_error = preservation_i

        print(f"  Edit {i+1}: Correction error = {update_i.correction_error:.2e}, "
              f"Cumulative preservation = {preservation_i:.2e}")

    print("\n7. SUMMARY STATISTICS")
    print("-" * 50)

    final_on_K0 = W_current @ K_0
    final_preservation = np.linalg.norm(final_on_K0 - original_on_K0) / np.linalg.norm(original_on_K0)

    print(f"Initial weight norm: {np.linalg.norm(W):.4f}")
    print(f"Final weight norm: {np.linalg.norm(W_current):.4f}")
    print(f"Total weight change: {np.linalg.norm(W_current - W):.4f}")
    print(f"Final relative preservation error: {final_preservation:.2e}")

    print("\n" + "=" * 70)
    print("NULL-SPACE PROJECTION VERIFICATION COMPLETE")
    print("=" * 70)

    return {
        "null_space_dim": result.null_space_dim,
        "rank": result.rank,
        "symmetry_error": symmetry_error,
        "idempotence_error": idempotence_error,
        "correction_error": correction_error,
        "preservation_error": preservation_error,
        "final_preservation": final_preservation
    }


if __name__ == "__main__":
    results = demonstrate_null_space_projection()
    print(f"\nResults summary: {results}")
