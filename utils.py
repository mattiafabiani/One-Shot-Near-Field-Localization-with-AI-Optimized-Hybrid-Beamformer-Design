import numpy as np

def pol2dist(r1, theta1, r2, theta2):
    '''
    This function calculates the euclidean distance between two points, defined in polar coordinates.
    r1, r2: distance of two points from origin
    theta1, theta2: defined as sin(angle)
    '''
    x1, y1 = r1 *np.sqrt(1-theta1**2), r1 * theta1
    x2, y2 = r2 *np.sqrt(1-theta2**2), r2 * theta2
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

    
def CN_realization(mean, std_dev, size=1):
    return np.random.normal(mean, std_dev, size) + 1j * np.random.normal(mean, std_dev, size)

def path_loss_model(dist,fc=28e9,n=2):
    return 10*n*np.log10(4*np.pi/3e8*fc) + 10*n*np.log10(dist)

def pol2cart(r,theta):
    '''
    input: array of
        - r: range
        - theta: angle in degrees
    output: array of
        - pos: cartesian position
    '''
    # x = r * np.cos(np.deg2rad(theta))
    # y = r * np.sin(np.deg2rad(theta))
    y = r * np.sqrt(1-theta**2)
    x = r * theta
    pos = np.stack((x, y), axis=1) # shape (batch,2)
    return pos