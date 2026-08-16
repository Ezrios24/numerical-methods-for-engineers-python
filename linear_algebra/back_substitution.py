import numpy as np


def backsub(A, B):
    """
    Solve the upper-triangular system A X = B using backward substitution.

    The matrix A must be square, upper-triangular, and have non-zero diagonal
    entries. B can be passed as a 1D array (n,) or as a column vector (n, 1).

    Parameters
    ----------
    A : np.ndarray
        An n x n upper-triangular coefficient matrix.
    B : np.ndarray
        Right-hand side vector of length n or shape (n, 1).

    Returns
    -------
    np.ndarray
        Solution vector X of length n.

    Raises
    ------
    ValueError
        If A is not square, dimensions mismatch, or A has a zero on its diagonal.

    Notes
    -----
    Adapted from:
    NUMERICAL METHODS: Matlab Programs
    (c) 2004 by John H. Mathews and Kurtis D. Fink
    Complementary Software to accompany the textbook:
    NUMERICAL METHODS: Using Matlab, Fourth Edition
    ISBN: 0-13-065248-2
    Prentice-Hall Pub. Inc.
    One Lake Street
    Upper Saddle River, NJ 07458
    """
    # --- Input validation ---
    # Ensure A is a 2D square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (2D with equal rows and columns).")

    n = A.shape[0]

    # Ensure B matches the matrix dimension
    B_flat = np.asarray(B).flatten()
    if len(B_flat) != n:
        raise ValueError(
            f"Length of B ({len(B_flat)}) must match matrix size ({n})."
        )

    # Check for zero diagonal entries to avoid division by zero
    if np.any(np.diag(A) == 0):
        raise ValueError(
            "Matrix A has a zero diagonal entry. "
            "Back substitution requires a non-singular upper-triangular matrix."
        )

    # --- Back substitution ---
    # Initialise solution vector with zeros
    X = np.zeros(n)

    # 1. Solve for the last variable (no unknown terms to its right)
    X[n - 1] = B_flat[n - 1] / A[n - 1, n - 1]

    # 2. Iterate backwards from row n-2 down to row 0
    for k in range(n - 2, -1, -1):
        # Sum the products of already-solved variables:
        #   A[k, k+1:n] * X[k+1:n]
        # This represents the contribution of known variables to equation k.
        sum_known = np.dot(A[k, k + 1:n], X[k + 1:n])

        # Rearranged equation:
        #   A[k,k] * X[k] = B[k] - (sum of known terms)
        X[k] = (B_flat[k] - sum_known) / A[k, k]

    return X

# Define the upper-triangular system:
# 4x₁ - x₂ + 2x₃ + 3x₄ = 20
#      -2x₂ + 7x₃ - 4x₄ = -7
#             6x₃ + 5x₄ = 4
#                    3x₄ = 6

A = np.array([
    [4.0, -1.0,  2.0,  3.0],
    [0.0, -2.0,  7.0, -4.0],
    [0.0,  0.0,  6.0,  5.0],
    [0.0,  0.0,  0.0,  3.0]
], dtype=float)

B = np.array([20.0, -7.0, 4.0, 6.0])

X = backsub(A, B)

print("Solution found by back substitution:")
for i, xi in enumerate(X, start=1):
    print(f"  x{i} = {xi:.4f}")

print("\nVerification (A @ X):")
print(f"  {A @ X}")
