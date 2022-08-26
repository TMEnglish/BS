def inverse_power(W, e_vector, n_iter=5):
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
    for _ in range(n_iter):
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


def rayleigh_quotient(W, v):
    """
    Calculates the Rayleigh quotient of square matrix `W` and vector `v`.
    
    Returns
    * the Rayleigh quotient `e_value` (the approximate eigenvalue
      corresponding to approximation `v` of an eigenvector of `W`)
    * the maximum absolute error of `e_value * v` relative to `W @ v`,
      the matrix product of `W` and `v`
    
    The latter value is undefined if any element of `W @ v` is zero.
    """
    # Calculate the vector dot product using the numerically stable
    # `fsum` to sum the elements of the pointwise product of vectors.
    def dot(u, v):
        return math.fsum(u * v)
    #
    # The Rayleigh quotient is (v.T @ W @ v) / (v.T @ v). 
    Wv_product = np.array([dot(row, v) for row in W])
    e_value = dot(v, Wv_product) / dot(v, v)
    #
    # Calculate the maximum absolute relative error, assuming that all
    # elements of `Wv_product` are nonzero. NumPy will issue a warning
    # if any of the elements are zero.
    relative_errors = (e_value * v - Wv_product) / Wv_product
    mare = np.max(np.abs(relative_errors))
    return e_value, mare