import numpy as np


def backsub(A, B):
    """
    Solves upper triangular system A X = B using back substitution.
    Input: A (n x n upper triangular matrix), B (n x 1 or 1D array)
    Output: X (solution vector, n x 1)
    """
    # Ensure B is a 1D array for simplicity
    B = np.asarray(B).flatten()
    n = len(B)
    X = np.zeros(n)
    
    X[n-1] = B[n-1] / A[n-1, n-1]
    
    for k in range(n-2, -1, -1):
        # sum of A(k, k+1:n) * X(k+1:n)
        X[k] = (B[k] - np.dot(A[k, k+1:n], X[k+1:n])) / A[k, k]
        
    return X


A = np.array([
    [4.0, -1.0,  2.0,  3.0],
    [0.0, -2.0,  7.0, -4.0],
    [0.0,  0.0,  6.0,  5.0],
    [0.0,  0.0,  0.0,  3.0]
], dtype=float)

# Right-hand side vector B
B = np.array([20.0, -7.0, 4.0, 6.0])

X = backsub(A, B)

print("Solution X:", X)