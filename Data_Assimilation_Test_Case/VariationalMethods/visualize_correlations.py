"""
  Module: Visualize the effect of diffusion operators
  Author: Olivier Goux
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2023 CERFACS
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from operators_to_be_completed import Bmatrix


# Compute discrete Fourier trasnform
def DFT(x):
    n = np.size(x)
    q =  np.arange(n)
    p =   np.array([k/2 if k%2==0 else n-(k+1)/2 for k in np.arange(n)])
    omega = np.exp(-2*1j*np.pi/n)
    F =omega**(p[:, np.newaxis] * q[np.newaxis, :]) / np.sqrt(n)
    return F @ x
#---------------------------------------------
# Initialize B
#---------------------------------------------

n      = 1000         # space dimension 
type   = 'diffusion' # diffusion or diagonal
D      = 10          # Daley length scale
M      = 4        # Smoothness parameter

# Initialize the operator associated to B
B = Bmatrix(n, 1, type = type, D = D, M = M)


#---------------------------------------------
# Correlation function
#---------------------------------------------

# Visualize the correlation function by appyling B to a 'dirac' 
dirac = np.zeros(n)
dirac[0] = 1
corr_function = B.dot(dirac)

# Look at Fourier transform
f_dirac = DFT(dirac)
f_corr_function = DFT(corr_function)

#---------------------------------------------
# Correlated noise
#---------------------------------------------

# Generate white noise
noise = randn(n)

# Generate noise of covariance B 
corr_noise = B.sqrtdot(noise)

# Look at Fourier transforms
f_noise = DFT(noise)
f_corr_noise = DFT(corr_noise)


#---------------------------------------------
# Display
#---------------------------------------------

# Abscissas
dist = np.arange(n)
freq = (np.arange(n)+1)//2

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12,6))

ax1.plot(dist, dirac, label = "dirac")
ax1.plot(dist, corr_function, label = "correlation function")
ax1.set_xlabel("Distance")
ax1.set_xlim([None,n//2])
ax1.legend()

ax2.loglog(freq, f_dirac, label = "dirac (spectrum)")
ax2.loglog(freq, f_corr_function, label = "correlation function (spectrum)")
ax2.legend()
ax2.grid(True, which = 'both', ls = '--')
ax2.set_ylim([1E-10, 10])
ax2.set_xlabel("Spatial scale")

ax3.plot(dist, noise, label = "white noise", linewidth = 0.5)
ax3.plot(dist, corr_noise, label = "correlated noise", linewidth = 2)
ax3.set_xlabel("Distance")
ax3.set_xlim([None,n//2])
ax3.legend()

ax4.semilogx(freq, f_noise, label = "white noise (spectrum)")
ax4.semilogx(freq, f_corr_noise, label = "correlated noise (spectrum)")
ax4.legend()
ax4.grid(True, which = 'both', ls = '--')
ax4.set_xlabel("Spatial scale")


fig.set_tight_layout(True)
