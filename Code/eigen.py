from scipy.sparse.linalg import eigs


def dominant_eig(a, rtol=1e-14, max_iterations=10**5):
    """
    Returns a dominant eigenvector of `a` along with its error.
    
    Assume: Vectors associated with largest real eigenvalue have only
    non-negative real elements.
    """
    # Try to get a good approximation of the eigenvector rapidly. The
    # maximum number of iterations is set to twice the default value.
    v0 = rand.rand(a.shape[0])
    try:
        val, vec = eigs(a, 1, which='LR', v0=v0, maxiter=20*len(a))
        v0 = np.abs(vec.flatten())
    except:
        warnings.warn('`eigs` failed in `dominant_eig`')
    # Now improve the approximate eigenvector with power iteration.
    v, error = power_iteration(a, v0, rtol, max_iterations)
    return v, error


def power_iteration(a, v0, rtol=1e-14, max_iterations=10**5):
    """
    Returns a dominant eigenvector of `a` along with its error.
    
    The iteration starts with `v0` as an approximate eigenvector of the
    matrix `a`. It stops when a block of `n_block` iterations does not
    reduce the error in the approximate eigenvector, as measured by
    the function `eigen_error()`.
    """
    tmp = np.array(v0)
    max_exponent = 896
    safe_exponent = 510 - math.ceil(math.log2(len(v)))
    best_error = eigen_error(a, bias_exponents(tmp, safe_exponent))
    best_v = np.array(tmp)
    v = np.array(tmp)
    n_iterations = 0

    while best_error > rtol and n_iterations < max_iterations:
        for _ in range(1000):
            bias_exponents(v, max_exponent)
            v += np.matmul(a, v, out=tmp)
            n_iterations += 1
        tmp[:] = v
        error = eigen_error(a, bias_exponents(tmp, safe_exponent))
        if error < best_error:
            best_v[:] = v
            best_error = error
    return best_v, best_error


def rayleigh_quotient(a, v, product=None):
    """
    Returns Rayleigh quotient for square matrix `a` and vector `v`.
    
    The Rayleigh quotient is the approximate eigenvalue corresponding
    to apprroximate eigenvector `v` of matrix `a`.
    
    The `product` parameter is either `None` or an array of the same
    shape as `v`. In the later case, `product` is set to the matrix
    product of `a` and `v`.
    """
    if product is None:
        product = np.matmul(a, v)
    else:
        np.matmul(a, v, out=product)
    return np.matmul(v, product) / np.dot(v, v)


def eigen_error(a, v):
    """
    Returns the error of the approximate eigenvector `v` of `a`.
    
    The error is the maximum absolute error of `e_value * v` relative
    to the matrix product `a @ v`.
    """
    product = np.empty_like(v)
    e_value = rayleigh_quotient(a, v, product=product)
    error = maximum_absolute_relative_error(e_value * v, product)
    return error