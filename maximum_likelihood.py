import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,                   default='', help='Unique experiment identifier')
parser.add_argument('--N', type=int,                    default=128, help='Number of antennas.')
parser.add_argument('--generate_dataset', type=int,     default=0, help='Generate Dataset.')
parser.add_argument('--logdir', type=str,               default='saved_models', help='Directory to log data to')
args = parser.parse_args()

#---------- SIMULATION PARAMETERS -----------
f0 = 300e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
c = 3e8
wavelength = c / f0         # Wavelength
d = wavelength * 4          # Antenna spacing
N = args.N                  # Number of antennas
SNR_dB = list(range(0, 25, 5))
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

array_length = d*(N-1)
near_field_boundary = 2*array_length**2/wavelength
print(f'Near-field up to {near_field_boundary:.1f} m')
angle_true = float(input("Enter angle (in degrees): "))
r_true = float(input("Enter range (in meters): "))

theta_true = np.radians(angle_true)  # True angle
theta = np.sin(theta_true)

range_limits = [.1,1]
N_ang, N_r = 128, 60

rng_seed = 42
np.random.seed(rng_seed)

def CN_realization(mean, std_dev, size=1):
    return np.random.normal(mean, std_dev, size) + 1j * np.random.normal(mean, std_dev, size)

# Steering vector dependent only on the angle
delta = lambda n: (2 * n - N + 1) / 2
a = lambda theta: np.exp(-1j * k * delta(np.arange(N)) * d * theta)
b = lambda theta, r: np.array([np.exp(-1j*k*(np.sqrt(r**2 + delta(n)**2*d**2 - 2*r*theta*delta(n)*d) - r)) for n in range(N)]).T

snr = SNR[2]  # 20 dB
sigma_n = 1 / np.sqrt(snr)
# s = CN_realization(mean=0, std_dev=1)
s = 1
n = CN_realization(mean=0, std_dev=sigma_n, size=N)

y = a(theta) * s + n  # Far-field signal
y1 = b(theta, r_true) * s + n  # Near-field signal

# Angle and Range grid for estimation
ang_grid = np.linspace(-np.pi/2, np.pi/2, N_ang)  # Angle grid in radians
range_grid = np.linspace(range_limits[0], range_limits[1], N_r)  # Range grid in meters

# Maximum Likelihood Estimation over angle and range for near-field
ML_bins_near = np.zeros((len(ang_grid), len(range_grid)), dtype=float)
for i, ang in enumerate(ang_grid):
    for j, r in enumerate(range_grid):
        y1_test = b(np.sin(ang), r) * s
        ML_bins_near[i, j] = np.abs(np.dot(y1, y1_test.conj().T))**2


# Find the indices of the maximum likelihood estimate for near-field
idx_ang_near, idx_range_near = np.unravel_index(np.argmax(ML_bins_near), ML_bins_near.shape)

# Display the results
plt.figure(figsize=(8, 6))

# Near-field plot
estimated_angle_near = np.degrees(ang_grid[idx_ang_near])
estimated_range_near = range_grid[idx_range_near]
rmse_angle = np.abs(estimated_angle_near - theta_true/np.pi*180)
rmse_range = np.abs(estimated_range_near - r_true)

# plt.subplot(1, 2, 2)
plt.contourf(np.degrees(ang_grid), range_grid, ML_bins_near.T, levels=50, cmap='viridis')
# plt.imshow(ML_bins_near.T, extent=[np.degrees(ang_grid).min(), np.degrees(ang_grid).max(), range_grid.min(), range_grid.max()], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Likelihood')
plt.scatter(np.degrees(ang_grid[idx_ang_near]), range_grid[idx_range_near], color='red', label='ML estimate')
plt.scatter(np.degrees(theta_true), r_true, color='green', label='True')
plt.xlabel('Angle (degrees)')
plt.ylabel('Range (meters)')
plt.title(f'Near-field Maximum Likelihood Estimation, SNR={snr}')
plt.text(np.degrees(ang_grid[idx_ang_near]) + 1, range_grid[idx_range_near] + 0.5,
         f"({estimated_angle_near:.2f}°, {estimated_range_near:.2f}m)", color='white', fontsize=10, ha='left')
plt.text(np.degrees(ang_grid[idx_ang_near]) - 20, range_grid[idx_range_near] + 3,
         f"Error (angle): {rmse_angle:.2f}°\nError (range): {rmse_range:.2f} m", color='white', fontsize=10, ha='left')
# plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
# plt.savefig(f'imgs/ML_estimate_r{r_true}_t{theta_true/np.pi*180}.png', dpi=300, bbox_inches='tight')

# Print the estimated angles and range
# estimated_angle_far = np.degrees(ang_grid[idx_ang_far])
estimated_angle_near = np.degrees(ang_grid[idx_ang_near])
estimated_range_near = range_grid[idx_range_near]
# print(f'Estimated Far-field Angle of Arrival: {estimated_angle_far:.2f} degrees')
print(f'\nEstimated Near-field Angle of Arrival: {estimated_angle_near:.2f} degrees')
print(f'Estimated Near-field Range: {estimated_range_near:.2f} meters')
