import numpy as np

def gaussian_wave_packet(x, x0=0.0, sigma=1.0):
    """
    Generates a 1D normalized Gaussian wave packet.
    
    Parameters:
    x (numpy.ndarray): The spatial grid.
    x0 (float): The center of the wave packet.
    sigma (float): The spread (standard deviation) of the wave packet.
    
    Returns:
    numpy.ndarray: The normalized wave function array.
    """
    normalization = 1.0 / (np.pi * sigma**2)**0.25
    return normalization * np.exp(-(x - x0)**2 / (2 * sigma**2))

def get_probability_density(psi):
    """
    Calculates the probability density from a wave function using Born's rule.
    """
    return np.abs(psi)**2

def measure_position(x, psi):
    """
    Simulates a position measurement causing wave function collapse.
    
    Returns:
    tuple: (psi_collapsed, measured_position)
    """
    dx = x[1] - x[0]
    prob_density = get_probability_density(psi)
    prob_normalized = prob_density * dx
    
    # Ensure probabilities sum exactly to 1 (handles minor floating point errors)
    prob_normalized = prob_normalized / np.sum(prob_normalized)
    
    # The quantum measurement (random choice weighted by probability)
    measured_index = np.random.choice(len(x), p=prob_normalized)
    measured_position = x[measured_index]
    
    # The wave function collapse (Dirac delta approximation)
    psi_collapsed = np.zeros_like(psi)
    psi_collapsed[measured_index] = 1.0 / np.sqrt(dx)
    
    return psi_collapsed, measured_position