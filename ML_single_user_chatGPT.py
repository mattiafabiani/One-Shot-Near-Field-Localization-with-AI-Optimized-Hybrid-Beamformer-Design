import matplotlib.pyplot as plt
# import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os
from utils import CN_realization

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,                   default='', help='Unique experiment identifier')
parser.add_argument('--N', type=int,                    default=128, help='Number of antennas.')
parser.add_argument('--spacing', type=float,                    default=0.5, help='antenna spacing (in wavelength).')
parser.add_argument('--n_trials', type=int,             default=10, help='Number of trials for each SNR.')
parser.add_argument('--plot_likelihood', type=int,      default=0, help='Flag to display likelihood plot')
args = parser.parse_args()

# Simulation parameters
f0 = 300e9                   # carrier frequency
k = 2 * np.pi / (3e8 / f0)  # wave number
c = 3e8
wavelength = c / f0         # Wavelength
d = wavelength / 2          # Antenna spacing
# d = args.spacing*wavelength          # Antenna spacing
N = args.N                  # Number of antennas
SNR_dB = list(range(0, 25, 5))
SNR = [10 ** (snr_db / 10) for snr_db in SNR_dB]
save_path = f'saved_models/maximum_likelihood/{int(f0/1e9)}GHz_{N}N_{args.n_trials}trials/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.random.seed(42)

array_length = d * (N - 1)
near_field_boundary = 2 * array_length ** 2 / wavelength
print(f'Near-field up to {near_field_boundary:.1f} m'
      f'\nArray length: {array_length*100} cm,\nwavelength = {wavelength*1000} mm')
exit()

# Steering vector dependent only on the angle
delta = lambda n: (2 * n - N + 1) / 2
a = lambda theta: np.exp(-1j * k * delta(np.arange(N)) * d * theta)
b = lambda theta, r: np.array([np.exp(-1j*k*(np.sqrt(r**2 + delta(n)**2*d**2 - 2*r*theta*delta(n)*d) - r)) for n in range(N)]).T

# Angle and range limits
angle_limits_deg = [-90, 90]  # Angle limits in degrees
range_limits = [1, 10]        # Range limits in meters

# Function to simulate multiple cases for each SNR
def run_simulation(SNR, N_trials=100, plot_likelihood=False):
    angle_tolerance = 5       # Angular tolerance in degrees
    range_tolerance = 3       # Range tolerance in meters
    N_ang, N_r = 128, 20       # Grid sizes for angle and range
    
    results = []
    results1 = []
    
    for snr in SNR:
        print(f'SNR = {10*np.log10(snr)} dB')
        sigma_n = 1 / np.sqrt(snr)
        rmse_angles = []
        rmse_ranges = []
        true_ranges, pred_ranges = [], []
        true_angles, pred_angles = [], []
        
        for trial in tqdm(range(N_trials)):
            # True angle and range (randomly chosen)
            # angle_true_deg = np.random.uniform(angle_limits_deg[0], angle_limits_deg[1])
            # r_true = np.random.uniform(range_limits[0], range_limits[1])
            # theta_true = np.radians(angle_true_deg)
            # theta = np.sin(theta_true)

            p = np.random.uniform(low=0,high=2*range_limits[1],size=(2,)) - [range_limits[1],0]
            r_true =  np.linalg.norm(p)
            while r_true < range_limits[0] or r_true > range_limits[1]:
                p = np.random.uniform(low=0,high=2*range_limits[1],size=(2,)) - [range_limits[1],0]
                r_true =  np.linalg.norm(p)
            theta_true = np.pi/2 - np.arctan2(p[1],p[0])
            angle_true_deg = np.rad2deg(theta_true)
            theta = np.sin(theta_true)
            
            # Simulate received signal with noise
            s = 1
            n = CN_realization(mean=0, std_dev=sigma_n, size=N)
            # y = a(theta) * s + n
            y1 = b(theta, r_true) * s + n
            
            # Maximum Likelihood Estimation for near-field
            ang_grid = np.deg2rad(np.linspace(angle_limits_deg[0], angle_limits_deg[1], N_ang))  # Angle grid in radians
            range_grid = np.linspace(range_limits[0], range_limits[1], N_r)  # Range grid in meters
            ML_bins_near = np.zeros((len(ang_grid), len(range_grid)), dtype=float)
            for i, ang in enumerate(ang_grid):
                for j, r in enumerate(range_grid):
                    y1_test = b(np.sin(ang), r) * s
                    ML_bins_near[i, j] = np.abs(np.dot(y1, y1_test.conj().T))**2
            
            # Find the indices of the maximum likelihood estimate
            idx_ang_near, idx_range_near = np.unravel_index(np.argmax(ML_bins_near), ML_bins_near.shape)
            estimated_angle_near = np.degrees(ang_grid[idx_ang_near])
            estimated_range_near = range_grid[idx_range_near]
            
            # Compute RMSE for angle and range
            rmse_angle = (estimated_angle_near - angle_true_deg)**2
            rmse_range = (estimated_range_near - r_true)**2

            if np.abs(estimated_angle_near - angle_true_deg) > 50:
                tqdm.write(f'angle: {np.sqrt(np.mean(rmse_angles)):.2f}, r: {np.sqrt(np.mean(rmse_ranges)):.2f}, theta: {angle_true_deg:.1f}, range: {r_true:.1f} --------------- X')
                # plt.figure(figsize=(8, 6))
                # # plt.contourf(np.degrees(ang_grid), range_grid, ML_bins_near.T, levels=50, cmap='viridis')
                # plt.imshow(ML_bins_near.T, extent=[np.degrees(ang_grid).min(), np.degrees(ang_grid).max(), range_grid.min(), range_grid.max()], 
                #     aspect='auto', origin='lower', cmap='viridis')
                # plt.colorbar(label='Likelihood')
                # plt.scatter(np.degrees(ang_grid[idx_ang_near]), range_grid[idx_range_near], color='red', label='ML estimate')
                # plt.scatter(angle_true_deg, r_true, color='green', label='True')
                # plt.xlabel('Angle (degrees)')
                # plt.ylabel('Range (meters)')
                # plt.title(f'Near-field Maximum Likelihood Estimation, SNR={10*np.log10(snr):.1f} dB')
                # plt.text(np.degrees(ang_grid[idx_ang_near]) + 1, range_grid[idx_range_near] + 0.5,
                #          f"({estimated_angle_near:.2f}°, {estimated_range_near:.2f}m)", color='white', fontsize=10, ha='left')
                # plt.text(np.degrees(ang_grid[idx_ang_near]) - 20, range_grid[idx_range_near] + 3,
                #          f"Error (angle): {rmse_angle:.2f}°\nError (range): {rmse_range:.2f} m", color='white', fontsize=10, ha='left')
                # plt.legend()
                # plt.tight_layout()
                # plt.show()
                continue
            rmse_angles.append(rmse_angle)
            rmse_ranges.append(rmse_range)
            true_ranges.append(r_true)
            pred_ranges.append(estimated_range_near)
            true_angles.append(angle_true_deg)
            pred_angles.append(estimated_angle_near)
            cur_rmse = (np.sqrt(np.mean(rmse_angles)), np.sqrt(np.mean(rmse_ranges)))
            tqdm.write(f'rmse (deg,m): ({cur_rmse[0]:.1f},{cur_rmse[1]:.1f}), theta (pred,true): ({estimated_angle_near:.1f},{angle_true_deg:.1f}), range (pred,true): ({estimated_range_near:.1f},{r_true:.1f})')
            results1.append({
            'SNR': int(10 * np.log10(snr)),
            'r_pred': estimated_range_near,
            'r_true': r_true,
            'theta_pred': estimated_angle_near,
            'theta_true': angle_true_deg,
            'Test (r)': rmse_range,
            'Test (theta)': rmse_angle,
            })
            # SNR,r_pred,r_true,theta_pred,theta_true,Test (r),Test (theta),Test (pos)
            
            # Plot likelihood if the flag is set
            if plot_likelihood and trial == 0: #and np.mod(trial,30) == 0:  # Only plot for the first trial
                plt.figure(figsize=(8, 6))
                # plt.contourf(np.degrees(ang_grid), range_grid, ML_bins_near.T, levels=50, cmap='viridis')
                plt.imshow(ML_bins_near.T, extent=[np.degrees(ang_grid).min(), np.degrees(ang_grid).max(), range_grid.min(), range_grid.max()], 
                    aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(label='Likelihood')
                plt.scatter(np.degrees(ang_grid[idx_ang_near]), range_grid[idx_range_near], color='red', label='ML estimate')
                plt.scatter(angle_true_deg, r_true, color='green', label='True')
                plt.xlabel('Angle (degrees)')
                plt.ylabel('Range (meters)')
                plt.title(f'Near-field Maximum Likelihood Estimation, SNR={10*np.log10(snr):.1f} dB')
                plt.text(np.degrees(ang_grid[idx_ang_near]) + 1, range_grid[idx_range_near] + 0.5,
                         f"({estimated_angle_near:.2f}°, {estimated_range_near:.2f}m)", color='white', fontsize=10, ha='left')
                plt.text(np.degrees(ang_grid[idx_ang_near]) - 20, range_grid[idx_range_near] + 3,
                         f"Error (angle): {rmse_angle:.2f}°\nError (range): {rmse_range:.2f} m", color='white', fontsize=10, ha='left')
                plt.legend()
                plt.tight_layout()
                plt.show()
        print(f'RMSE (deg) = {np.sqrt(np.mean(rmse_angles)):.2f}')
        print(f'RMSE (m) = {np.sqrt(np.mean(rmse_ranges)):.2f}')
        
        # Store results for this SNR
        results.append({
            'SNR': 10 * np.log10(snr),
            'RMSE_Angle': np.sqrt(np.mean(rmse_angles)),
            'RMSE_Range': np.sqrt(np.mean(rmse_ranges))
        })
    
    return pd.DataFrame(results), pd.DataFrame(results1)

# Run the simulation
results, results1 = run_simulation(SNR, N_trials=args.n_trials, plot_likelihood=args.plot_likelihood)
# results.to_csv(save_path+'results.csv',index=False)
# results1.to_csv(save_path+'test_score.csv',index=False)

# Display the results for each SNR
for index, res in results.iterrows():
    print(f"SNR: {int(res['SNR']):d} dB, RMSE Angle: {res['RMSE_Angle']:.2f}°, RMSE Range: {res['RMSE_Range']:.2f} m")
