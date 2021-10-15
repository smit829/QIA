
import numpy as np
from numpy import linalg as la
import time
import os


def ls_probs(m, n, A):

    r"""Function generating the length-squared (LS) probability distributions for sampling matrix A.
    Args:
        m (int): number of rows of matrix A
        n (int): row n of columns of matrix A
        A (array[complex]): most general case is a rectangular complex matrix
    Returns:
        tuple: Tuple containing the row-norms, LS probability distributions for rows and columns,
        and Frobenius norm.
    """

    # populates array with the row-norms squared of matrix A
    row_norms = np.zeros(m)
    for i in range(m):
        row_norms[i] = np.abs(la.norm(A[i, :]))**2

    # Frobenius norm of A
    A_Frobenius = np.sqrt(np.sum(row_norms))

    LS_prob_rows = np.zeros(m)

    # normalized length-square row probability distribution
    for i in range(m):
        LS_prob_rows[i] = row_norms[i] / A_Frobenius**2

    LS_prob_columns = np.zeros((m, n))

    # populates array with length-square column probability distributions
    # LS_prob_columns[i]: LS probability distribution for selecting columns from row A[i]
    for i in range(m):
        LS_prob_columns[i, :] = [np.abs(k)**2 / row_norms[i] for k in A[i, :]]

    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius


def sample_C(A, m, n, r, c, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius):

    r"""Function used to generate matrix C by performing LS sampling of rows and columns of matrix A.
    Args:
        A (array[complex]): rectangular, in general, complex matrix
        m (int): number of rows of matrix A
        n (int): number of columns of matrix A
        r (int): number of sampled rows
        c (int): number of sampled columns
        row_norms (array[float]): norm of the rows of matrix A
        LS_prob_rows (array[float]): row LS probability distribution of matrix A
        LS_prob_columns (array[float]): column LS probability distribution of matrix A
        A_Frobenius (float): Frobenius norm of matrix A
    Returns:
        tuple: Tuple containing the singular values (sigma), left- (w) and right-singular vectors (vh) of matrix C,
        the sampled rows (rows), the column LS prob. distribution (LS_prob_columns_R) of matrix R and split running
        times for the FKV algorithm.
    """

    tic = time.time()
    # sample row indices from row length_square distribution
    rows = np.random.choice(m, r, replace=True, p=LS_prob_rows)

    columns = np.zeros(c, dtype=int)
    # sample column indices
    for j in range(c):
        # sample row index uniformly at random
        i = np.random.choice(rows, replace=True)
        # sample column from length-square distribution of row A[i]
        columns[j] = np.random.choice(n, 1, p=LS_prob_columns[i])

    toc = time.time()
    rt_sampling_C = toc - tic

    # building the lenght-squared distribution to sample columns from matrix R
    R_row = np.zeros(n)
    LS_prob_columns_R = np.zeros((r, n))

    for s in range(r):
        R_row[:] = A[rows[s], :] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))
        R_row_norm = np.abs(la.norm(R_row[:]))**2
        LS_prob_columns_R[s, :] = [np.abs(k)**2 / R_row_norm for k in R_row[:]]

    tic = time.time()
    # creates empty array for R and C matrices. We treat R as r x c here, since we only need columns later
    R_C = np.zeros((r, c))
    C = np.zeros((r, c))

    # populates array for matrix R with the submatrix of A defined by sampled rows/columns
    for s in range(r):
        for t in range(c):
            R_C[s, t] = A[rows[s], columns[t]]

        # renormalize each row of R
        R_C[s,:] = R_C[s,:] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))

    # creates empty array of column norms
    column_norms = np.zeros(c)

    # computes column Euclidean norms
    for t in range(c):
        for s in range(r):
            column_norms[t] += np.abs(R_C[s, t])**2

    # renormalize columns of C
    for t in range(c):
        C[:, t] = R_C[:, t] * (A_Frobenius / np.sqrt(column_norms[t])) / np.sqrt(c)

    toc = time.time()
    rt_building_C = toc - tic

    tic = time.time()
    # Computing the SVD of sampled C matrix
    w, sigma, vh = la.svd(C, full_matrices=False)

    toc = time.time()
    rt_svd_C = toc - tic

    return w, rows, sigma, vh, LS_prob_columns_R, rt_sampling_C, rt_building_C, rt_svd_C


