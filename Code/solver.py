###############################
# Need to include `ivp_solution` here
###############################
def runge_kutta(derivative, initial_freqs, n_years, max_step_size=1/128):
    times = np.linspace(0, n_years, n_years + 1)
    return ivp_solution(derivative, initial_freqs, times, max_step_size)


######################################################################
##############################
# The following still might be used to test the above.
###############################

        
def solver(derivative, initial_freqs, n_years, steps_per_year=2**10,
           zeroing=False, hold_zero=False, normed=True):
    """
    Uses the Euler forward method to solve the initial value problem
    when `zeroing` is false.
    
    To replicate the results of B&S, set `steps_per_year` to 1 and
    `zeroing` to 'incorrect'. WRONG!!!!
    """
    # The step size is the reciprocal of the number of steps per year.
    step_size = 1 / steps_per_year
    #
    # Allocate storage for solutions in years 0, 1, ..., `n_years`.
    solution = np.empty((n_years + 1, len(initial_freqs)))
    #
    # The 1-D array `s` always contains the latest solutions.
    s = convert(initial_freqs, float)
    if zeroing:
        # Zero solutions that are small in relation to sum of solutions.
        zeros = s < 1e-9 * math.fsum(s)
        s[zeros] = 0
    solution[0] = s
    #
    for year in range(1, n_years + 1):
        for _ in range(steps_per_year):
            # Update the solutions in `s`.
            s += step_size * derivative(None, s)
            #
            if zeroing:
                if hold_zero:
                    zeros = np.logical_or(zeros, s < 1e-9 * math.fsum(s))
                else:
                    zeros = s < 1e-9 * math.fsum(s)
                s[zeros] = 0
        #
        # Scale solutions to keep the maximum of their exponents at zero.
        # Scaling by an integer power of two maintains full precision,
        # because the mantissas of the solutions are unchanged.
        s *= 2 ** -round(math.log(np.max(s), 2))
        #
        # Save the year-end solutions.
        solution[year] = s
    if normed:
        # Normalize the solutions in each year.
        solution /= solution.sum(axis=1)[:,None]
    return solution


def alt_solver(derivative, initial_freqs, n_years, steps_per_year=2**10,
               threshold=None, hold_zero=False, normed=True):
    """
    Uses the Euler forward method to solve the initial value problem
    when `zeroing` is false.
    
    To replicate the results of B&S, set `steps_per_year` to 1 and
    `zeroing` to 'incorrect'.
    """
    # The step size is the reciprocal of the number of steps per year.
    step_size = 1 / steps_per_year
    #
    # Allocate storage for solutions in years 0, 1, ..., `n_years`.
    solution = np.empty((n_years + 1, len(initial_freqs)))
    #
    # The 1-D array `s` always contains the latest solutions represented
    # as multiprecision floating point numbers.
    s = mp_float(initial_freqs)
    if not threshold is None:
        # Zero solutions that are small in relation to sum of solutions.
        zeros = s < threshold * mp.fsum(s)
        s[zeros] = 0
    solution[0] = s
    #
    for year in range(1, n_years + 1):
        for _ in range(steps_per_year):
            # Update the solutions in `s`. The derivative calculation
            # is done with 64-bit floating point numbers.
            s += step_size * derivative(None, convert(s, float))
            #
            if zeroing:
                if hold_zero:
                    zeros = zeros or s < threshold * mp.fsum(s)
                else:
                    zeros = s < threshold * mp.fsum(s)
                s[zeros] = 0
        solution[year] = s
    if normed:
        # Normalize the solutions in each year.
        solution /= solution.sum(axis=1)[:,None]
    return solution
