def largest_real_eig(W):
    """
    Returns largest real eigenvalue of `W` and associated eigenvector.
    
    The eigenpair is obtained using the `eig` function in the SciPy
    linear algebra package.
    """
    # Use the `eig` function of SciPy's linear algebra package to obtain
    # all eigenvalues and eigenvectors of `W`. Ignore eigenvalues with
    # nonzero imaginary parts, and also their associated eigenvectors.
    # Select the largest real eigenvalue and its associated eigenvector.
    e_values, e_vectors = linalg.eig(W)
    real = e_values.imag == 0
    e_values = e_values[real]
    e_vectors = e_vectors[:,real]
    largest = np.argmax(e_values.real)
    e_value = e_values[largest].real
    e_vector = e_vectors[:,largest].real
    return e_value, e_vector