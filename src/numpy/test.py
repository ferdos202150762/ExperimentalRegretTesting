import numpy as np

def project_onto_simplex(v):
    """
    Projects a vector v onto the probability simplex.
    Ensures the result is non-negative and sums to 1.
    """
    v = np.asarray(v)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u + (1 - cssv) / (np.arange(n) + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def Uniform_distribution_L_ball(vector, epsilon=0.1, n_samples=1):
    """
    Sample approximately uniformly from the intersection of:
    - An L∞ ball of radius epsilon around `vector`, and
    - The probability simplex (non-negative, sums to 1).
    
    Parameters:
    -----------
    vector : np.array
        Center of the L∞ ball. Must lie in the probability simplex.
    epsilon : float
        Radius of the L∞ ball.
    n_samples : int
        Number of samples to generate.
    
    Returns:
    --------
    np.array
        Shape (n_samples, len(vector)) if n_samples > 1,
        Shape (len(vector),) if n_samples == 1
    """
    vector = np.asarray(vector)
    dim = len(vector)
    samples = []


    # Sample uniform noise in L∞ ball
    noise = np.random.uniform(-epsilon, epsilon, size=dim)
    perturbed = vector + noise
    # Project back to the simplex
    projected = project_onto_simplex(perturbed)
    


    return  np.array(projected)

vector = np.array([0.3, 0.4, 0.3])
samples = Uniform_distribution_L_ball(vector, epsilon=0.1, n_samples=1)
print(samples)
#print(samples.sum(axis=1))  # should be all close to 1