from psutil import cpu_count
import ray
ray.shutdown()
#ray.init()


def eigen_test(a, eigenvector):
    """
    Checks accuracy of approximate `eigenvector` of array `a`.
    
    Vector `v` is divided by its Euclidean norm to obtain `e_vector`. A
    triple is returned:
        (1) the eigenvalue `e_value` associated with `e_vector`,
        (2) eigenvector `e_vector`, and
        (3) the maximum absolute error of `e_value * e_vector` relative
            to the matrix product `np.dot(a, e_vector)`.
    """
    e_vector = eigenvector
    product = np.dot(a, e_vector)
    e_value = np.dot(e_vector, product) / np.dot(e_vector, e_vector)
    error = maximum_absolute_relative_error(e_value*e_vector, product)
    return e_value, e_vector, error


def dominant_eig(a, v0=None):
    """
    Calculates the dominant eigenpair for matrix `a`.
    
    ASSUME: Vectors associated with largest real eigenvalue have only
    non-negative real elements.
    
    A triple `(eigenvalue, eigenvector, error)` is returned, with the
    approximation `error` determined by `eigen_test(a, eigenvector)`.
    """
    if v0 is None:
        v = rand.random(len(a))
    else:
        v = v0 / fsum(v0)
    try:
        val, vec = eigs(a, 1, which='LR', v0=v, maxiter=20*len(a))
        v = np.abs(vec.flatten())
    except:
        warnings.warn('eigs failed in eigenpair')
    return power_iteration(a, v)

        
@ray.remote
class Dot(object):
    def __init__(self, a):
        self.a = np.array(a)
        
    def dot(self, v):
        return np.dot(self.a, v)
    
    def change(self, a):
        self.a = np.array(a)
    
    def stop(self):
        ray.actor.exit_actor()


class ParallelDot(object):
    def __init__(self, operator, n_actors=None):
        if n_actors is None:
            n_actors = cpu_count()
        n, _ = operator.shape
        self.actor = np.empty(n_actors, dtype=object)
        start = 0
        for i in range(n_actors):
            n_rows = (n - start) // (n_actors - i)
            self.actor[i] = Dot.remote(operator[start: start+n_rows])
            start += n_rows
        
    def __call__(self, v):
        results = ray.get([a.dot.remote(v) for a in self.actor])
        return np.concatenate(results)
    
    def __del__(self):
        for a in self.actor:
            a.stop.remote()

######################################################################
        

def power_iteration(a, v, n_block=2000):
    """
    Returns results of power iteration beginning with `v`.
    
    The iteration starts with `v` as an approximate eigenvector of the
    matrix `a`. It stops when a block of `n_block` iterations does not
    reduce the error in the approximate eigenvector, as measured by the
    function `eigen_test()`. When the iteration ends, a triple
    
        `(eigenvalue, eigenvector, error)`
        
    is returned, where `error` is determined by `eigen_test()`.
    """
    last_error = np.inf
    e_value, e_vector, error = eigen_test(a, v)
    tmp = np.empty_like(v)
    safe_exponent = 510 - math.ceil(math.log2(len(v)))

    while error < last_error:
        last_error = error
        for _ in range(n_block):
            _, max_exponent = frexp(v.max())
            if max_exponent > 768 or max_exponent < safe_exponent:
                bias_exponents(v, 630, max_exponent)
            v += np.dot(a, v, out=tmp)
        bias_exponents(v, safe_exponent)
        e_value, e_vector, error = eigen_test(a, v)
    return e_value, e_vector, error

######################################################################

def mp_power_iteration(a, v0, n_iterations):
    """
    Returns results of power iteration starting with `v0`.
    
    The calculations are done with multiprecision floating-point
    numbers. The multiplication of matrix `a` by the curren eigenvector
    approximation is parallelized.
    """
    if type(v0[0]) is mp.mpf:
        v = np.array(v0)
    else:
        v = mp_float(v0)
    if not type(a[0,0]) is mp.mpf:
        a = mp_float(a)
    op = ParallelDot(a)
    for _ in range(n_iterations):
        v += op(v)
    del op
    return eigen_test(a, v)
            

def rayleigh_quotient_iteration(A, mu, v, maxiter=2):
    """    
    Works if `A` contains multiprecision floats, but is very slow.
    
    For order 401, one iteration with 40 digits of working precision
    took 18 minutes. In the case that I tested, the maximum absolute
    relative error was reduced from 1.60e-15 to 3.19e-31.
    """
    n = len(A)
    A = mp.matrix(A.tolist())
    v = mp.matrix(v.tolist())
    v /= mp.norm(v)
    I = mp.eye(n)
    mu = mp.mpf(mu)
    for i in range(maxiter):
        v = (A - mu * I)**-1 * v
        v /= mp.norm(v)
        product = A * v
        mu = (v.T * product)[0] / (v.T * v)[0]
        mare = maximum_absolute_relative_error(mu*v, product)
        if mare < 1e-18:
            break
    return mu, np.array(v.tolist()), mare