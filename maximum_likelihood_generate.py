from utils import pol2dist, CN_realization, pol2cart
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

# fix the seed for reproducibility
rng_seed = 42
np.random.seed(rng_seed)

#---------- SIMULATION PARAMETERS -----------
f0 = 25e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
c = 3e8                     # speed of light
wavelength = c / f0         # Wavelength
d = wavelength / 2          # Antenna spacing
N = 128#int(input('Number of antennas: '))                     # Number of antennas
range_limits = [1,10]       # range limits
# N_ang, N_r = int(input('Grid size (angle): ')), int(input('Grid size (range): '))        # ML grid
N_ang, N_r = 128, 50
ch_realizations = 2000       # channel realizations (for CRLB and ML)
ch_realizations = 50       # channel realizations (for CRLB and ML)
SNR_dB = list(range(0, 25, 5)) # SNR_dB
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

## Near-field boundary [m]
array_length = d*(N-1)
near_field_boundary = 2*array_length**2/wavelength
print(f'\nNear-field up to {near_field_boundary:.1f} m (strong near-field region: {near_field_boundary/10:.1f} m)')

# Steering vector dependent only on the angle
delta = lambda n: (2 * n - N + 1) / 2
a = lambda theta, r: np.array(np.exp(-1j*k*(np.sqrt(r**2 + delta(np.arange(N))**2*d**2 - 2*r*theta*delta(np.arange(N))*d) - r))).T

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

k = 0
for idx_snr, snr in enumerate(SNR):
    sigma_n = 1 / np.sqrt(snr)
    print(f'{idx_snr+1}/{len(SNR)}, SNR = {SNR_dB[idx_snr]} dB')
    for i in tqdm(range(ch_realizations)):
        s = CN_realization(mean=0, std_dev=1)
        n = CN_realization(mean=0, std_dev=sigma_n, size=N)
        
        r = np.random.uniform(range_limits[0], range_limits[1])
        r_norm = 2 * (r - range_limits[0]) / (range_limits[1] - range_limits[0]) - 1
        theta = np.random.uniform(-90,90)
        theta = np.sin(np.deg2rad(theta))
        pts[idx_snr,i,:] = np.stack([np.arcsin(theta),r])
        
        # uplink received signal
        y_ = a(theta,r) * s + n
        y = y_.reshape((N,1))

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
        rmse_angle_deg[idx_snr,i] = np.abs(estimated_angle_near - np.arcsin(theta)/np.pi*180)
        rmse_angle_sin[idx_snr,i] = np.abs(np.sin(estimated_angle_near/180*np.pi) - theta)
        rmse_r[idx_snr,i] = np.abs(estimated_range_near - r)

        # calculate euclidean distance from polar coordinates
        theta_pred = np.sin(estimated_angle_near/180*np.pi)
        r_pred = estimated_range_near
        rmse_pos[idx_snr,i] = pol2dist(r,theta,r_pred,theta_pred)
        p_pred[k,:], p_true[k,:] = pol2cart(np.reshape(r_pred,(1,)),np.reshape(theta_pred,(1,))), pol2cart(np.reshape(r,(1,)),np.reshape(theta,(1,)))
        k = k + 1

import matplotlib.pyplot as plt
plt.scatter(p_pred[:, 0], p_pred[:, 1], c='k', label='Predicted')  # p_pred scatter (black points)
plt.scatter(p_true[:, 0], p_true[:, 1], c='r', label='True')  # p_true scatter (red points)
for i in range(len(p_pred)):
    plt.text(p_pred[i, 0], p_pred[i, 1], str(i+1), fontsize=9, color='k') # add number for each point
    plt.text(p_true[i, 0], p_true[i, 1], str(i+1), fontsize=9, color='r')
    plt.plot([p_pred[i, 0], p_true[i, 0]], [p_pred[i, 1], p_true[i, 1]], 'k--',linewidth=.5) # connect p_pred and p_true with a dashed line
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.grid()
plt.axis('equal')
plt.legend()
plt.show()

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

rmse_angle_deg_mean = np.mean(rmse_angle_deg,axis=1)
rmse_angle_rad_mean = np.mean(rmse_angle_sin,axis=1)
rmse_r_mean = np.mean(rmse_r,axis=1)
rmse_pos_mean = np.mean(rmse_pos,axis=1)

# print(f'rmse_angle: {rmse_angle_deg_mean}')
# print(f'rmse_r: {rmse_r_mean}')
# print(f'rmse_pos: {rmse_pos_mean}')
# exit()

# save results
foldername = 'saved_models/maximum_likelihood/'
data = {
            'Test (r)': rmse_r_mean,
            'Test (theta)': rmse_angle_deg_mean,
            'Test (pos)': rmse_pos_mean
        }
df = pd.DataFrame(data,index=SNR_dB)
df.index.name = 'SNR [dB]'
print(df)
# exit()
df.to_csv(os.path.join(foldername,f'rmse_ML_N{N}_{N_ang}_{N_r}.csv'),index=False)
# np.save(foldername+f'rmse_r_N{N}_{N_ang}_{N_r}.npy',rmse_r_mean,allow_pickle=True)
# np.save(foldername+f'rmse_theta_deg_N{N}_{N_ang}_{N_r}.npy',rmse_angle_deg_mean,allow_pickle=True)
# np.save(foldername+f'rmse_theta_rad_N{N}_{N_ang}_{N_r}.npy',rmse_angle_rad_mean,allow_pickle=True)
# np.save(foldername+f'rmse_pos_N{N}_{N_ang}_{N_r}.npy',rmse_pos_mean,allow_pickle=True)