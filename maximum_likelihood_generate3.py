import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from utils import pol2dist, CN_realization, pol2cart

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,                   default='', help='Unique experiment identifier')
parser.add_argument('--N', type=int,                    default=128, help='Number of antennas.')
parser.add_argument('--generate_dataset', type=int,     default=0, help='Generate Dataset.')
parser.add_argument('--logdir', type=str,               default='saved_models', help='Directory to log data to')
args = parser.parse_args()

#---------- SIMULATION PARAMETERS -----------
f0 = 25e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
c = 3e8
wavelength = c / f0         # Wavelength
d = wavelength / 2          # Antenna spacing
N = args.N                  # Number of antennas
SNR_dB = list(range(0, 25, 5))
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

array_length = d*(N-1)
near_field_boundary = 2*array_length**2/wavelength
print(f'Near-field up to {near_field_boundary:.1f} m')

range_limits = [1,10]
ch_realizations = 10
N_ang, N_r = 128, 10

rng_seed = 42
np.random.seed(rng_seed)

def CN_realization(mean, std_dev, size=1):
    return np.random.normal(mean, std_dev, size) + 1j * np.random.normal(mean, std_dev, size)
def steering_vector_b(theta, r,N):
    """
    Calcola il vettore di steering b(theta, r) per un dato angolo e distanza.

    Args:
    theta (float): L'angolo in radianti (valore di np.sin()).
    r (float): La distanza radiale dalla sorgente.

    Returns:
    np.ndarray: Un array complesso che rappresenta il vettore di steering.
    """
    steering_vector = []
    delta = lambda n: (2 * n - N + 1) / 2
    
    for n in range(N):
        # Calcolo della distanza per ogni antenna
        delta_n = delta(n)  # Calcolo del delta
        distance = np.sqrt(r**2 + delta_n**2 * d**2 - 2 * r * theta * delta_n * d)
        
        # Calcolo dell'elemento di steering
        steering_element = np.exp(-1j * k * (distance - r))
        
        # Aggiunta dell'elemento alla lista
        steering_vector.append(steering_element)
        
        # Debug info
        print(f"n: {n}, delta_n: {delta_n}, distance: {distance}, steering_element: {steering_element}")
    
    # Convertiamo la lista in un array numpy
    steering_vector = np.array(steering_vector)
    
    return steering_vector.T  # Trasposta come nel lambda originale
# Steering vector dependent only on the angle
delta = lambda n: (2 * n - N + 1) / 2
a = lambda theta: np.exp(-1j * k * delta(np.arange(N)) * d * theta)
b = lambda theta, r: np.array([np.exp(-1j*k*(np.sqrt(r**2 + delta(n)**2*d**2 - 2*r*theta*delta(n)*d) - r)) for n in range(N)]).T

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

s = 1 # deterministic transmitted signal by the near-field source
kk = 0

# Angle and Range grid for estimation
ang_grid = np.linspace(-np.pi/2, np.pi/2, N_ang)  # Angle grid in radians
range_grid = np.linspace(range_limits[0], range_limits[1], N_r)  # Range grid in meters

for idx_snr, snr in enumerate(SNR):
    sigma_n = 1 / np.sqrt(snr)
    print(f'{idx_snr+1}/{len(SNR)}, SNR = {SNR_dB[idx_snr]} dB')

    cur_rmse_angle_deg = []
    cur_rmse_r = []
    cur_rmse_pos = []

    for idx_ch in tqdm(range(ch_realizations)):
        # noise realization
        n = CN_realization(mean=0, std_dev=sigma_n, size=N)

        # generate (theta,r)
        p = np.random.uniform(low=0,high=20,size=(2,)) - [10,0]
        r =  np.linalg.norm(p)
        while r < range_limits[0] or r > range_limits[1]:
            p = np.random.uniform(low=0,high=20,size=(2,)) - [10,0]
            r =  np.linalg.norm(p)
        theta_deg = 90 - np.rad2deg(np.arctan2(p[1],p[0]))
        theta = np.sin(np.pi/2 - np.arctan2(p[1],p[0]))

        # Near-field signal
        y1 = b(theta, r) * s + n

        # Maximum Likelihood Estimation over angle and range for near-field
        ML_bins_near = np.zeros((len(ang_grid), len(range_grid)), dtype=float)
        for i, ang in enumerate(ang_grid):
            for j, rr in enumerate(range_grid):
                print(steering_vector_b(np.sin(ang), rr,N)[:3])
                # y1_test = b(np.sin(ang), rr) * s
                # print(y1_test[:2])
                exit()
                ML_bins_near[i, j] = np.abs(np.dot(y1, y1_test.conj().T))**2


        # Find the indices of the maximum likelihood estimate for near-field
        idx_ang_near, idx_range_near = np.unravel_index(np.argmax(ML_bins_near), ML_bins_near.shape)
        estimated_angle_near = np.degrees(ang_grid[idx_ang_near])
        estimated_range_near = range_grid[idx_range_near]
        # cur_rmse_angle_deg.append((estimated_angle_near - theta_deg)**2)
        # cur_rmse_r.append((estimated_range_near - r)**2)

        # # calculate euclidean distance from polar coordinates
        # theta_pred = np.sin(estimated_angle_near/180*np.pi)
        # r_pred = estimated_range_near
        # # rmse_pos[idx_snr,i] = pol2dist(r,theta,r_pred,theta_pred)
        # cur_rmse_pos.append(pol2dist(r,theta,r_pred,theta_pred))
        # p_pred[k,:], p_true[k,:] = pol2cart(np.reshape(r_pred,(1,)),np.reshape(theta_pred,(1,))), pol2cart(np.reshape(r,(1,)),np.reshape(theta,(1,)))
        # k = k + 1
    

        if True:
            # Display the results
            plt.figure(figsize=(8, 6))

            # Near-field plot
            print(f'\nEstimated Near-field Angle of Arrival: {estimated_angle_near:.2f} degrees')
            print(f'Estimated Near-field Range: {estimated_range_near:.2f} meters')

            # plt.subplot(1, 2, 2)
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

            plt.tight_layout()
            plt.show()
            # plt.savefig(f'imgs/ML_estimate_r{r}_t{theta_true/np.pi*180}.png', dpi=300, bbox_inches='tight')

            if idx_ch == 1:
                exit()

    rmse_angle_deg.append(np.sqrt(np.mean(cur_rmse_angle_deg)))
    rmse_r.append(np.sqrt(np.mean(cur_rmse_r)))
    rmse_pos.append(np.sqrt(np.mean(cur_rmse_pos)))

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
