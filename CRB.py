import numpy as np
import matplotlib.pyplot as plt

# Constants from the paper
c = 3e8  # Speed of light (m/s)
f0 = 25e9  # Carrier frequency (Hz)
N = 10  # Number of antennas (adjust as per the paper)
d = c / (2 * f0)  # Element spacing (meters)
wavelength = c / f0  # Wavelength (meters)
SNR_dB = np.linspace(-10, 30, 5)  # SNR range in dB
SNR = 10 ** (SNR_dB / 10)  # Linear SNR

# Parameters from the paper
M = 1  # Number of FDA subarrays (typically M=1)
theta_true = np.radians(30)  # True angle in radians (adjust if necessary)
r_true = 1000  # True range in meters (adjust if necessary)

# Fisher Information Matrix components
def fisher_information_r(SNR, N, d, theta_true, r_true, wavelength):
    # Placeholder for the actual FIM expression for range
    # Replace with actual expression from the paper
    return (N * SNR * (d * np.cos(theta_true)) ** 2) / (wavelength ** 2 * r_true ** 2)

def fisher_information_theta(SNR, N, d, theta_true, r_true, wavelength):
    # Placeholder for the actual FIM expression for angle
    # Replace with actual expression from the paper
    return (N * SNR * (d ** 2) * np.sin(2 * theta_true)) / (wavelength ** 2 * r_true)

# CRLB Calculation
CRLB_r = 1 / fisher_information_r(SNR, N, d, theta_true, r_true, wavelength)  # CRLB for range
CRLB_theta = 1 / fisher_information_theta(SNR, N, d, theta_true, r_true, wavelength)  # CRLB for angle

# Plotting the CRLB for range and angle
plt.figure(figsize=(10, 6))
plt.plot(SNR_dB, CRLB_r, label='CRLB for Range (m)')
plt.xlabel('SNR (dB)')
plt.ylabel('CRLB (dB)')
plt.title('CRLB for Range and Angle Estimation')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(SNR_dB, CRLB_theta, label='CRLB for Angle (deg)')
plt.xlabel('SNR (dB)')
plt.ylabel('CRLB (dB)')
plt.title('CRLB for Range and Angle Estimation')
plt.legend()
plt.grid(True)

plt.show()
