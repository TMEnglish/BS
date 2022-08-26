def inverse_power(W, e_vector, n_iterations=5):
    """
    Attempts to improve solution `e_vector` for an eigenvector of `W`.
        
    Returns 
    * the eigenvalue corresponding to ...
    * an eigenvector `v` with elements summing to 1, and
    * the maximum absolute error of `e_value * v` relative to `W @ v`.
        
    The given eigenvector is improved (ordinarily) by iterating the
    inverse power method `n_iterations` times.
    """
    e_vector /= fsum(e_vector)
    best_e_vector = e_vector
    best_e_value, best_error = rayleigh_quotient(W, best_e_vector)
    A = np.array(W)
    diag_indices = np.diag_indices(A.shape[0])
    A[diag_indices] -= best_e_value
    #
    # The inverse power method fails (and `solve` raises an exception)
    # when subtraction of the approximate eigenvalue from the main
    # diagonal of `W` produces a singular matrix. Suppress warnings
    # from `solve`: they advise of possible numerical inaccuracy.
    for _ in range(n_iterations):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                e_vector = linalg.solve(A, e_vector)
            except:
                break
        e_vector /= fsum(e_vector)
        e_value, error = rayleigh_quotient(W, e_vector)
        if error < best_error:
            best_error = error
            best_e_value = e_value
            best_e_vector = e_vector
            A[diag_indices] = W[diag_indices] - best_e_value
    return best_e_value, best_e_vector, best_error