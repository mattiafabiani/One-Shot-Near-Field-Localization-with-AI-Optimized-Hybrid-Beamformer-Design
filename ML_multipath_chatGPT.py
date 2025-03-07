import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Simulation parameters
f0 = 25e9                   # carrier frequency
k = 2 * np.pi / (3e8 / f0)   # wave number
c = 3e8                      # speed of light
wavelength = c / f0          # wavelength
d = wavelength / 2           # antenna spacing
N = 128                      # number of antennas
range_limits = [1, 10]       # range limits in meters
angle_limits_deg = [-80, 80]  # angle limits in degrees
SNR_dB = [0, 5, 10, 15, 20]  # SNR values to iterate over
N_trials = 50               # number of trials per SNR value
N_ang = 128                   # grid resolution for angle
N_r = 30                     # grid resolution for range

# Function to generate complex Gaussian noise
def CN_realization(mean, std_dev, size=1):
    return np.random.normal(mean, std_dev, size) + 1j * np.random.normal(mean, std_dev, size)

# Steering vector functions
delta = lambda n: (2 * n - N + 1) / 2
a = lambda theta: np.exp(-1j * k * delta(np.arange(N)) * d * theta)  # angle only
b = lambda theta, r: np.array([np.exp(-1j * k * (np.sqrt(r**2 + delta(n)**2 * d**2 - 2 * r * theta * delta(n) * d) - r)) for n in range(N)]).T

# Function for likelihood estimation with nested loops for (theta, r, theta_scat, r_scat)
def estimate_likelihood(y, N_ang, N_r, angle_grid, range_grid):
    ML_bins = np.zeros((N_ang, N_r, N_ang, N_r), dtype=float)  # 4D array for likelihood

    for i, theta in enumerate(angle_grid):
        for j, r in enumerate(range_grid):
            for k, theta_scat in enumerate(angle_grid):
                for l, r_scat in enumerate(range_grid):
                    # Generate the estimated signals for both direct and scattered paths
                    y_test = b(np.sin(theta), r) * s + b(np.sin(theta_scat), r_scat) * s
                    # Likelihood computation (magnitude squared)
                    ML_bins[i, j, k, l] = np.abs(np.dot(y, y_test.conj().T))**2
    
    # Find the indices of maximum likelihood estimate
    idx = np.unravel_index(np.argmax(ML_bins), ML_bins.shape)
    estimated_theta = angle_grid[idx[0]]
    estimated_r = range_grid[idx[1]]
    estimated_theta_scat = angle_grid[idx[2]]
    estimated_r_scat = range_grid[idx[3]]
    
    return estimated_theta, estimated_r, estimated_theta_scat, estimated_r_scat, ML_bins

# Main iteration loop over SNR values
for snr_db in SNR_dB:
    snr = 10 ** (snr_db / 10)
    sigma_n = 1 / np.sqrt(snr)
    
    for trial in tqdm(range(N_trials)):
        # Generate random positions and scatterer
        p = np.random.uniform(low=0, high=20, size=(2,)) - [10, 0]
        r_true = np.linalg.norm(p)
        while r_true < range_limits[0] or r_true > range_limits[1]:
            p = np.random.uniform(low=0, high=20, size=(2,)) - [10, 0]
            r_true = np.linalg.norm(p)

        r_scat = np.random.uniform(0, 1.5)
        theta_scat = np.random.uniform(0, 2 * np.pi)
        x1 = p[0] + r_scat * np.cos(theta_scat)
        y1 = p[1] + r_scat * np.sin(theta_scat)
        r_scat = np.linalg.norm([x1, y1])
        while r_scat < range_limits[0] or r_scat > range_limits[1]:
            r_scat = np.random.uniform(0, 1.5)
            theta_scat = np.random.uniform(0, 2 * np.pi)
            x1 = p[0] + r_scat * np.cos(theta_scat)
            y1 = p[1] + r_scat * np.sin(theta_scat)
            r_scat = np.linalg.norm([x1, y1])
        
        # Calculate true angles
        theta_true = np.sin(np.arctan2(p[1], p[0]))
        theta_scat_true = np.sin(np.arctan2(y1, x1))
        
        # Generate noise
        s = 1  # signal
        n = CN_realization(mean=0, std_dev=sigma_n, size=N)

        # Generate the received signal (multipath: direct + scattered)
        y_direct = b(theta_true, r_true) * s
        y_scatter = b(theta_scat_true, r_scat) * s
        y = y_direct + y_scatter + n  # Total received signal

        # Generate grids for angle and range
        angle_grid = np.deg2rad(np.linspace(angle_limits_deg[0], angle_limits_deg[1], N_ang))  # Angle grid in radians
        range_grid = np.linspace(range_limits[0], range_limits[1], N_r)  # Range grid in meters
        
        est_theta, est_r, est_theta_scat, est_r_scat, ML_bins = estimate_likelihood(
            y, N_ang, N_r, angle_grid, range_grid)

        # (Optional) Plotting likelihood estimation
        if trial == 0:  # Plot only for the first trial in each SNR
            plt.figure(figsize=(12, 6))
            
            # 4D likelihood visualization (for simplicity, we can slice the 4D array)
            plt.subplot(1, 2, 1)
            plt.imshow(np.max(ML_bins, axis=(2, 3)).T, extent=[np.degrees(angle_limits_deg[0]), np.degrees(angle_limits_deg[1]),
                        range_limits[0], range_limits[1]], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Likelihood (max over scattered paths)')
            plt.scatter(np.degrees(est_theta), est_r, color='red', label='ML estimate')
            plt.scatter(np.degrees(np.arcsin(theta_true)), r_true, color='green', label='True')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Range (meters)')
            plt.title(f'Max Likelihood Estimation, SNR={snr_db} dB')
            plt.legend()
            
            # Additional plotting for scattered paths could be added here
            
            plt.tight_layout()
            plt.show()

# Print final estimates (example for the last iteration)
print(f'Estimated Direct Path: Angle = {np.degrees(est_theta):.2f} degrees, Range = {est_r:.2f} meters')
print(f'Estimated Scattered Path: Angle = {np.degrees(est_theta_scat):.2f} degrees, Range = {est_r_scat:.2f} meters')
