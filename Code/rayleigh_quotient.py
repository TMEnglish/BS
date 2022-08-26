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