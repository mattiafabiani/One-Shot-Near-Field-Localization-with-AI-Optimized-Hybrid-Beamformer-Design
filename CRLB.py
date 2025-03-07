import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

# fix the seed for reproducibility
rng_seed = 42
np.random.seed(rng_seed)

#---------- SIMULATION PARAMETERS -----------
f0 = 25e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
c = 3e8                     # speed of light
wavelength = c / f0         # Wavelength
d = wavelength / 2          # Antenna spacing
N = 128                     # Number of antennas
range_limits = [1,20]       # range limits
N_ang, N_r = 128, 10        # ML grid
ch_realizations = 100       # channel realizations (for CRLB and ML)
SNR_dB = list(range(0, 25, 5)) # SNR_dB
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

## Near-field boundary [m]
array_length = d*(N-1)
near_field_boundary = 2*array_length**2/wavelength
print(f'Near-field up to {near_field_boundary:.1f} m')


def CN_realization(mean, std_dev, size=1):
    return np.random.normal(mean, std_dev, size) + 1j * np.random.normal(mean, std_dev, size)

# Steering vector dependent only on the angle
delta = lambda n: (2 * n - N + 1) / 2
a = lambda theta, r: np.array(np.exp(-1j*k*(np.sqrt(r**2 + delta(np.arange(N))**2*d**2 - 2*r*theta*delta(np.arange(N))*d) - r))).T
def derivative_theta(theta, r, h=1e-5):
    return (a(theta + h, r) - a(theta - h, r)) / (2 * h)
def derivative_r(theta, r, h=1e-5):
    return (a(theta, r + h) - a(theta, r - h)) / (2 * h)



## CRLB init
# P = sigma_s
J = np.ones((2,2),dtype=float)
CRB = np.zeros((len(SNR),ch_realizations,2,2),dtype=float)

## maximum likelihood init
ang_grid = np.linspace(-np.pi/2, np.pi/2, N_ang)  # Angle grid in radians
range_grid = np.linspace(range_limits[0], range_limits[1], N_r)  # Range grid in meters
rmse_angle_deg = np.zeros((len(SNR),ch_realizations),dtype=float)
rmse_angle_sin = np.zeros((len(SNR),ch_realizations),dtype=float)
rmse_r = np.zeros((len(SNR),ch_realizations),dtype=float)
pts = np.zeros((len(SNR),ch_realizations,2),dtype=float)
    
dataset_crlb = dict()
ii = 0
for idx_snr, snr in enumerate(SNR):
    sigma_n = 1 / np.sqrt(snr)
    print('\n')
    print(snr)
    for i in range(ch_realizations):
        s = CN_realization(mean=0, std_dev=1)
        n = CN_realization(mean=0, std_dev=sigma_n, size=N)
        
        r = np.random.uniform(range_limits[0], range_limits[1])
        r_norm = 2 * (r - range_limits[0]) / (range_limits[1] - range_limits[0]) - 1
        theta = np.random.uniform(-1,1)
        pts[idx_snr,i,:] = np.stack([np.arcsin(theta),r])
        
        # uplink received signal
        y_ = a(theta,r) * s + n
        y = y_.reshape((N,1))

        da_dtheta_ = derivative_theta(theta,r)
        da_dr_ = derivative_r(theta,r)

        A = a(theta,r).reshape((N,1))
        H = np.eye(N) - A@np.linalg.pinv(A)
        D = np.c_[da_dtheta_,da_dr_] # shape=(N,2)
        R = np.dot(y, y.T.conj()) / y.shape[0]

        D_H_D = D.conj().T@H@D
        # PARAP = A.conj().T@np.linalg.inv(R)@A * P
        FIM = np.real(D_H_D )#* (np.kron(J,PARAP.T))
        CRB[idx_snr,i,:,:] = 1/2*sigma_n**2 * np.linalg.inv(FIM)



        # Maximum Likelihood Estimation over angle and range for near-field
        ML_bins_near = np.zeros((len(ang_grid), len(range_grid)), dtype=float)
        for k, ang in enumerate(ang_grid):
            for j, rr in enumerate(range_grid):
                y1_test = a(np.sin(ang), rr) * s
                ML_bins_near[k, j] = np.abs(np.dot(y_, y1_test.conj().T))**2


        # Find the indices of the maximum likelihood estimate for near-field
        idx_ang_near, idx_range_near = np.unravel_index(np.argmax(ML_bins_near), ML_bins_near.shape)
        estimated_angle_near = np.degrees(ang_grid[idx_ang_near])
        estimated_range_near = range_grid[idx_range_near]
        rmse_angle_deg[idx_snr,i] = np.abs(estimated_angle_near - theta/np.pi*180)
        rmse_angle_sin[idx_snr,i] = np.abs(np.sin(estimated_angle_near/180*np.pi) - theta)
        rmse_r[idx_snr,i] = np.abs(estimated_range_near - r)

        ii += 1
        if np.mod(ii,10) == 0:
            print(ii)

rmse_angle_deg_mean = np.mean(rmse_angle_deg,axis=1)
rmse_angle_rad_mean = np.mean(rmse_angle_sin,axis=1)
rmse_r_mean = np.mean(rmse_r,axis=1)

CRB_avg = np.mean(CRB,axis=1)
CRB_theta = CRB_avg[:,0,0]
CRB_r = CRB_avg[:,1,1]

CRB_r_8RF = [8.4, 4.7, 2.5, 1.8, 0.7] # figure 3b
CRB_theta_8RF = [0.6, .33, .18, .11, .06] # figure 3a

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(SNR_dB, rmse_r_mean,'-sk',label=f'fully-digital (ML {N_ang}-{N_r} grid)')
plt.plot(SNR_dB,CRB_r,'-or',label='CRLB fully-digital')
plt.plot(SNR_dB,CRB_r_8RF,'-og',label='CRLB (hybrid 8 RF chains)')
plt.xticks(SNR_dB)
# plt.ylim([0,12])
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE (r)')
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(SNR_dB, rmse_angle_rad_mean,'-sk',label=f'fully-digital (ML {N_ang}-{N_r} grid)')
plt.plot(SNR_dB,np.sin(CRB_theta),'-or',label='CRLB fully-digital')
plt.plot(SNR_dB,CRB_theta_8RF,'-og',label='CRLB (hybrid 8 RF chains)')
plt.xticks(SNR_dB)
plt.yscale('log')
# plt.ylim([0.04,1])
plt.grid(True,which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE (theta)')
plt.legend()
plt.show