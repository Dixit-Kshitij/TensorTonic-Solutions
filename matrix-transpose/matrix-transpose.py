import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    m,n = A.shape
    B = np.zeros((n,m))
    for i in range (1,m+1):
        for j in range (1,n+1):
            B[j-1][i-1] = A[i-1][j-1]

    return B
