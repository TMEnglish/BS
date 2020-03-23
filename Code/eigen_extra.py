from psutil import cpu_count
import ray
        
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

def mp_power_iteration(a, v0, n_iterations):
    """
    Returns results of power iteration starting with `v0`.
    
    The calculations are done with multiprecision floating-point
    numbers. The multiplication of matrix `a` by the curren eigenvector
    approximation is parallelized.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ray.init()
    if not type(a[0,0]) is mp.mpf:
        a = mp_float(a)
    v = mp_float(v0)
    op = ParallelDot(a)
    for _ in range(n_iterations):
        v += op(v)
    del op
    ray.shutdown()
    return v, eigen_error(a, v)
            

def rayleigh_quotient_iteration(A, mu, v, maxiter=2):
    """
    Square matrix `A`. Start with approximate eigenpair `(mu, v)`.
    
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
        # The following is the calculation done by `eigen_test`.
        product = A * v
        mu = (v.T * product)[0] / (v.T * v)[0]
        mare = maximum_absolute_relative_error(mu*v, product)
        if mare < 1e-18:
            break
    return mu, np.array(v.tolist()), mare


##################################
# I originally intended to parallelize the following implementation
# of matrix inversion by Gauss-Jordan elimination, and use it to
# invert matrices containing multiprecision floats. The approach to
# parallelization is shown above in `ParallelDot`.
#

def print_matrix(a, title):
    print(title)
    for row in a:
        print(' '.join(['{:6.3f}'.format(x) for x in row]))

# Here, apart from A, the parameters are references to previously
# allocated storage.
def invert_matrix(A, aug, pivot_col, tmp):
    """
    Returns the inverse of matrix `A`.
    """
    n, n_cols = aug.shape
    row_indices = np.arange(n)
    aug[:, :n] = A
    aug[:, n:] = np.eye(n)

    for row in range(n):
        # Exclude completed columns.
        a = aug[:, row:]
        #
        # Divide the pivot row by the pivot element. 
        pivot_row = a[row]
        pivot_row /= pivot_row[0]
        pivot_col[:] = a[:, 0]

        # Operate on on all non-pivot rows.
        out = tmp[:len(pivot_row)]
        for i in range(n):
            if i != row:
                a[i,:] -= np.dot(pivot_col[i], pivot_row, out)

    return aug[:,n:]


class MatrixInvert(object):
    """
    Example:
    
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]).astype(float)
    invert = MatrixInvert(A) 
    Ainv = invert(A)
    print_matrix(Ainv, 'Result of invert(A)')
    
    Output:
     Result of invert(A)
     0.750  0.500  0.250
     0.500  1.000  0.500
     0.250  0.500  0.750   
    """
    def __init__(self, A):
        basetype = type(A[0][0])
        n = len(A)
        self.row_indices = np.arange(n)
        n_cols = 2 * n
        self.aug = np.empty((n, n_cols), dtype=basetype)
        self.pivot_col = np.empty((3, n), dtype=basetype)[1]
        self.tmp = np.empty((3, n_cols), dtype=basetype)[1]

    def __call__(self, A, trace=False):
        """
        Returns the inverse of matrix `A`.
        """
        result = invert_matrix(A, self.aug, self.pivot_col, self.tmp)
        if trace:
            print_matrix(result, 'returned from invert_matrix')
        return result
        ################################
        
        n, n_cols = self.aug.shape
        self.aug[:, :n] = A
        self.aug[:, n:] = np.eye(n)
        pivot_col = self.pivot_col

        #if trace: print_matrix(self.aug, 'aug *')

        for row in range(n):
            # Exclude completed columns.
            a = self.aug[:, row:]
            #
            # Divide the pivot row by the pivot element. 
            pivot_row = a[row]
            pivot_row /= pivot_row[0]
            pivot_col[:] = a[:, 0]

            #if trace: print_matrix(a, 'a {}a'.format(row))

            # Operate on on all non-pivot rows.
            
            #which = self.row_indices != row
            #out = self.tmp[:(n-1)*n_cols].reshape(n-1, n_cols)
            #a[which,:] -= np.outer(pivot_col[which], pivot_row, out)
            out = self.tmp[:n_cols]
            #print('out.shape', out.shape)
            for i in prange(n):
                a[i,:] -= (i != row) * np.dot(pivot_col[i], pivot_row, out)
               
            #if trace: print_matrix(a, 'a {}b'.format(row))
                
            n_cols -= 1

        return self.aug[:,n:]