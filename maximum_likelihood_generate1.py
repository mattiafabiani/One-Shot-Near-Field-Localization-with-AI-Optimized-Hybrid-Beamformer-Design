from utils import pol2dist, CN_realization, pol2cart
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# fix the seed for reproducibility
rng_seed = 1
np.random.seed(rng_seed)

#---------- SIMULATION PARAMETERS -----------
f0 = 300e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
c = 3e8                     # speed of light
wavelength = c / f0         # Wavelength
d = wavelength / 2          # Antenna spacing
# N = int(input('Number of antennas: '))                     # Number of antennas
N = 256
range_limits = [.1,2]       # range limits
angle_limits_deg = [-90,90]
# N_ang, N_r = int(input('Grid size (angle): ')), int(input('Grid size (range): '))        # ML grid
N_ang, N_r = 128, 10
ch_realizations = 1000       # channel realizations (for CRLB and ML)
ch_realizations = 30       # channel realizations (for CRLB and ML)
SNR_dB = list(range(0, 25, 5)) # SNR_dB
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

## Near-field boundary [m]
array_length = d*(N-1)
near_field_boundary = 2*array_length**2/wavelength
print(f'\nNear-field up to {near_field_boundary:.1f} m (strong near-field region: {near_field_boundary/10:.1f} m)')

# Steering vector dependent only on the angle
delta = lambda n: (2 * n - N + 1) / 2
a = lambda theta, r: np.array(np.exp(-1j*k*(np.sqrt(r**2 + delta(np.arange(N))**2*d**2 - 2*r*theta*delta(np.arange(N))*d) - r))).T
# b = lambda theta, r: np.array([np.exp(-1j*k*(np.sqrt(r**2 + delta(n)**2*d**2 - 2*r*theta*delta(n)*d) - r)) for n in range(N)]).T
# a = lambda theta, r: np.array([np.exp(-1j*k*(np.sqrt(r**2 + delta(n)**2*d**2 - 2*r*theta*delta(n)*d) - r)) for n in range(N)]).T

## maximum likelihood init
ang_grid = np.linspace(-np.pi/2, np.pi/2, N_ang)  # Angle grid in radians
range_grid = np.linspace(range_limits[0], range_limits[1], N_r)  # Range grid in meters
rmse_angle_deg = np.zeros((len(SNR),ch_realizations),dtype=float)
rmse_angle_sin = np.zeros((len(SNR),ch_realizations),dtype=float)
rmse_r = np.zeros((len(SNR),ch_realizations),dtype=float)
rmse_pos = np.zeros((len(SNR),ch_realizations),dtype=float)
pts = np.zeros((len(SNR),ch_realizations,2),dtype=float)
p_pred = np.zeros((len(SNR)*ch_realizations,2),dtype=float)
p_true = np.zeros((len(SNR)*ch_realizations,2),dtype=float)

rmse_angle_deg = []
rmse_r = []
rmse_pos = []

kk = 0
for idx_snr, snr in enumerate(SNR):
    # snr = 100
    sigma_n = 1 / np.sqrt(snr)
    print(f'{idx_snr+1}/{len(SNR)}, SNR = {SNR_dB[idx_snr]} dB')

    
    cur_rmse_angle_deg = []
    cur_rmse_angle_sin = []
    cur_rmse_r = []
    cur_rmse_pos = []
    true_ranges, pred_ranges = [], []
    true_angles, pred_angles = [], []

    for i in tqdm(range(ch_realizations)):
        # s = CN_realization(mean=0, std_dev=1)
        s = 1
        n = CN_realization(mean=0, std_dev=sigma_n, size=N)
        
        p = np.random.uniform(low=0,high=2*range_limits[1],size=(2,)) - [range_limits[1],0]
        r =  np.linalg.norm(p)
        while r < range_limits[0] or r > range_limits[1]:
            p = np.random.uniform(low=0,high=2*range_limits[1],size=(2,)) - [range_limits[1],0]
            r =  np.linalg.norm(p)
        theta_deg = 90 - np.rad2deg(np.arctan2(p[1],p[0]))
        theta = np.sin(np.pi/2 - np.arctan2(p[1],p[0]))
        pts[idx_snr,i,:] = p
        
        # pts[idx_snr,i,:] = np.stack([np.arcsin(theta),r])
        
        # uplink received signal
        y_ = a(theta,r) * s + n

        # Maximum Likelihood Estimation - near-field
        ML_bins_near = np.zeros((N_ang,N_r), dtype=float)
        for ii, angg in enumerate(ang_grid):
            for j, rr in enumerate(range_grid):
                y1_test = a(np.sin(angg), rr) * s
                ML_bins_near[ii, j] = np.abs(np.dot(y_, y1_test.conj().T))**2

        # Find the indices of the maximum likelihood estimate for near-field
        idx_ang_near, idx_range_near = np.unravel_index(np.argmax(ML_bins_near), ML_bins_near.shape)
        estimated_angle_near = np.degrees(ang_grid[idx_ang_near])
        estimated_range_near = range_grid[idx_range_near]
        
        if False:
            print(f'\n\npos: {p}, r: {r:.1f}, theta: {theta_deg:.1f}, r_pred: {estimated_range_near:.1f}, theta_pred: {estimated_angle_near:.1f}')
            plt.contourf(np.degrees(ang_grid), range_grid, ML_bins_near.T, cmap='viridis')
            # plt.imshow(ML_bins_near.T, extent=[np.degrees(ang_grid).min(), np.degrees(ang_grid).max(), range_grid.min(), range_grid.max()], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Likelihood')
            plt.scatter(np.degrees(ang_grid[idx_ang_near]), range_grid[idx_range_near], color='red', label='ML estimate')
            plt.scatter(theta_deg, r, color='green', label='True')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Range (meters)')
            plt.title(f'Near-field Maximum Likelihood Estimation, SNR={snr}')
            plt.text(np.degrees(ang_grid[idx_ang_near]) + 1, range_grid[idx_range_near] + 0.5,
                    f"({estimated_angle_near:.2f}°, {estimated_range_near:.2f}m)", color='white', fontsize=10, ha='left')
            plt.text(np.degrees(ang_grid[idx_ang_near]) - 20, range_grid[idx_range_near] + 3,
                    f"Error (angle): {np.abs(estimated_angle_near - theta_deg):.2f}°\nError (range): {np.abs(estimated_range_near - r):.2f} m", color='white', fontsize=10, ha='left')
            # plt.grid(True)
            plt.legend()

            # plt.tight_layout()
            plt.show()
            if kk == 4:
                exit()

        # if 
        cur_rmse_angle_deg.append((estimated_angle_near - theta_deg)**2)
        cur_rmse_r.append((estimated_range_near - r)**2)

        # calculate euclidean distance from polar coordinates
        theta_pred = np.sin(estimated_angle_near/180*np.pi)
        r_pred = estimated_range_near
        # rmse_pos[idx_snr,i] = pol2dist(r,theta,r_pred,theta_pred)
        cur_rmse_pos.append(pol2dist(r,theta,r_pred,theta_pred)**2)
        # p_pred[kk,:], p_true[kk,:] = pol2cart(np.reshape(r_pred,(1,)),np.reshape(theta_pred,(1,))), pol2cart(np.reshape(r,(1,)),np.reshape(theta,(1,)))
        kk = kk + 1

        
        # Compute RMSE for angle and range
        true_ranges.append(r)
        pred_ranges.append(estimated_range_near)
        true_angles.append(theta_deg)
        pred_angles.append(estimated_angle_near)
    if True:
        # Plot true vs predicted range and angle in real-time
        plt.figure(figsize=(10, 5))

        # Subplot 1: True vs Predicted Range
        plt.subplot(1, 2, 1)
        plt.scatter(pred_ranges, true_ranges, color='blue', label='Predicted vs True Range')
        plt.plot([range_limits[0], range_limits[1]], [range_limits[0], range_limits[1]], 'r--', label='Ideal')
        plt.xlabel('True Range (m)')
        plt.ylabel('Predicted Range (m)')
        plt.xlim([range_limits[0], range_limits[1]])
        plt.ylim([range_limits[0], range_limits[1]])
        plt.legend()

        # Subplot 2: True vs Predicted Angle
        plt.subplot(1, 2, 2)
        plt.scatter(pred_angles, true_angles, color='green', label='Predicted vs True Angle')
        plt.plot(angle_limits_deg, angle_limits_deg, 'r--', label='Ideal')
        plt.xlabel('True Angle (degrees)')
        plt.ylabel('Predicted Angle (degrees)')
        plt.xlim(angle_limits_deg)
        plt.ylim(angle_limits_deg)
        plt.legend()

        plt.suptitle(f'SNR = {10*np.log10(snr):.1f} dB')
        plt.tight_layout()
        # plt.pause(0.1)  # Display the plot for a brief moment
        # plt.close()  # Close the plot after the pause
        plt.show()

    
    rmse_angle_deg.append(np.sqrt(np.mean(cur_rmse_angle_deg)))
    rmse_r.append(np.sqrt(np.mean(cur_rmse_r)))
    rmse_pos.append(np.sqrt(np.mean(cur_rmse_pos)))
    print(f'theta: {rmse_angle_deg}\nr: {rmse_r}\npos: {rmse_pos}')

# import matplotlib.pyplot as plt
# plt.scatter(p_pred[:, 0], p_pred[:, 1], c='k', label='Predicted')  # p_pred scatter (black points)
# plt.scatter(p_true[:, 0], p_true[:, 1], c='r', label='True')  # p_true scatter (red points)
# for i in range(len(p_pred)):
#     plt.text(p_pred[i, 0], p_pred[i, 1], str(i+1), fontsize=9, color='k') # add number for each point
#     plt.text(p_true[i, 0], p_true[i, 1], str(i+1), fontsize=9, color='r')
#     plt.plot([p_pred[i, 0], p_true[i, 0]], [p_pred[i, 1], p_true[i, 1]], 'k--',linewidth=.5) # connect p_pred and p_true with a dashed line
# plt.xlim([-10,10])
# plt.ylim([-10,10])
# plt.grid()
# plt.axis('equal')
# plt.legend()
# plt.show()

# print('rmse r',rmse_r)
# print('rmse theta', rmse_angle_deg)
# print('rmse pos', rmse_pos)
# plt.figure()
# plt.plot(rmse_r)
# plt.figure()
# plt.plot(rmse_angle_deg)
# plt.figure()
# plt.plot(rmse_pos)
# plt.show()
# exit()

# rmse_angle_deg_mean = np.mean(rmse_angle_deg,axis=1)
# rmse_angle_rad_mean = np.mean(rmse_angle_sin,axis=1)
# rmse_r_mean = np.mean(rmse_r,axis=1)
# rmse_pos_mean = np.mean(rmse_pos,axis=1)

# print(f'rmse_angle: {rmse_angle_deg_mean}')
# print(f'rmse_r: {rmse_r_mean}')
# print(f'rmse_pos: {rmse_pos_mean}')
exit()

# save results
foldername = 'saved_models/maximum_likelihood/'
data = {
            'Test (r)': rmse_r,
            'Test (theta)': rmse_angle_deg,
            'Test (pos)': rmse_pos
        }
df = pd.DataFrame(data,index=SNR_dB)
df.index.name = 'SNR [dB]'
print(df)
exit()
df.to_csv(os.path.join(foldername,f'rmse_ML_N{N}_{N_ang}_{N_r}_{ch_realizations}pts.csv'),index=False)
# np.save(foldername+f'rmse_r_N{N}_{N_ang}_{N_r}.npy',rmse_r_mean,allow_pickle=True)
# np.save(foldername+f'rmse_theta_deg_N{N}_{N_ang}_{N_r}.npy',rmse_angle_deg_mean,allow_pickle=True)
# np.save(foldername+f'rmse_theta_rad_N{N}_{N_ang}_{N_r}.npy',rmse_angle_rad_mean,allow_pickle=True)
# np.save(foldername+f'rmse_pos_N{N}_{N_ang}_{N_r}.npy',rmse_pos_mean,allow_pickle=True)