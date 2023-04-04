"""
  Module: 4D-Variational Data Assimilation
  Authors: Selime Gurol
         : Olivier Goux - Plots
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""

import sys
import math
from numpy.lib.function_base import append
sys.path.append('../Model')

from numpy.core.numeric import zeros_like
from numpy import\
(round, shape,
copy, zeros, array,
dot,# Matrix-matrix or matrix-vector product
eye,# To generate an identity matrix
ones, # To generate an array full of ones
random,
concatenate,
linspace,  # To get space and time position indices for observations 
where,
min,
max,
arange)
from numpy.linalg import \
(inv,# To invert a matrix
norm) # To compute the Euclidean norm
from numpy.random import randn # To generate samples from a normalized Gaussian
import matplotlib.pyplot as plt # To plot a graph
import matplotlib.animation as anim # To plot a graph
from operators import obs_operator, Rmatrix, Bmatrix
from operators import Hessian4dVar, Precond
from models import lorenz95 
from solvers import pcg, Bcg

def nonlinear_funcval(x):
    eo = y - obs.gop(x)
    eb = x-xb
    J = eb.dot(B.invdot(eb)) + eo.dot(R.invdot(eo))
    return J

def quadratic_funcval(x, dx):
    eo = obs.gop(x) - y + obs.tlm_gop(x, dx)
    eb = x-xb+dx
    J = eb.dot(B.invdot(eb)) + eo.dot(R.invdot(eo))
    return J    

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (1) Initialization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
random.seed(1)
n  = 100           # state space dimension 
nt = 10            # number of time steps

# Model class initialization
dt = 0.025        # time step for 4th order Runge-Kutta
F = 8            # Forcing term
model = lorenz95(F,dt)

# Observation class initialization
sigmaR = 1e-2# observation error std
total_space_obs = 10# total number of observations at fixed time 
total_time_obs = 5 # total number of observations at fixed location 
space_inds_obs = round(linspace(0, n, total_space_obs, endpoint = False)).astype(int) # observation locations in space
time_inds_obs = round(linspace(0, nt, total_time_obs, endpoint = False)).astype(int) # observation locations along the time 
m = total_space_obs*total_time_obs
obs = obs_operator(sigmaR, space_inds_obs, n, time_inds_obs, nt, model)
R = Rmatrix(sigmaR)

# Background class initialization
sigmaB = 0.8 # background error std
#B = Bmatrix(n, sigmaB,'diagonal')
B = Bmatrix(n, sigmaB,'diffusion', D=10, M=4) 

# Minimization initialization
max_outer = 10 # number of maximum outer loops
max_inner = 500 # number of maximum inner loops
tol = 1e-6  # tolerance for the inner loop
tol_grad = 1e-6 # tolerance for the outer loop
In = eye(n)
F = Precond(B) # Define the preconditioner

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (2) Generate the truth
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xt = 3.0*ones(n)+randn(n) # the true state
xt = model.traj(xt,5000)  # spin-up

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (3) Generate the background
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xb = xt + B.sqrtdot(randn(n))
#The square root of B can be used to create correlated errors
#xb = xt + B.sqrtdot(randn(n)) 

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (4) Generate the observations
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = obs.generate_obs(xt)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Variational data assimilation
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter_outer = 0 
dxs = []
xa = copy(xb)   # Choose the initial vector
quadcost = math.log10(quadratic_funcval(xa, zeros_like(xa)))
print('')
print('iter', '  CGiter', '        f(x)', '          ||grad(x)||')
iter = 0
while iter_outer < max_outer: # Gauss-Newton loop
    A = Hessian4dVar(obs,R,B, xa)    # Define the Hessian matrix (Binv + HtRinvH)
    d = obs.misfit(y, xa) # misfit calculation (y - G(xa))
    b = B.invdot(xb-xa) + obs.adj_gop(xa, R.invdot(d))
    print('{:<9d}{:<9d}{:<20.2f}{:<9.2f}'.format(iter_outer, iter, nonlinear_funcval(xa), norm(b)))
    if norm(b) < tol_grad:
        break
    # Calculate the increment dx such that 
    # (Binv + HtRinvH) dx = Binv(xb - x) + Ht Rinv d
    dxs, error, iter, flag = pcg(A, zeros_like(xa), b, F, max_inner, tol)
    dx = dxs[-n:]
    for i in range(iter):
        qval = math.log10(quadratic_funcval(xa, dxs[i*n:(i+1)*n]))
        quadcost = append(quadcost, qval)
    xa += dx
    iter_outer += 1

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Diagnostics
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('')
print('||xt - xb||_2 / ||xt||_2 = ', norm(xt - xb)/norm(xt))
print('||xt - xa||_2 / ||xt||_2 = ', norm(xt - xa)/norm(xt))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (6) Plots
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Generate the model trajectories starting from the background and the true state
xb_traj = [xb]
xt_traj = [xt]
xa_traj = [xa]
for i in range(nt-1):
    xb_traj.append(model.traj(xb_traj[-1],1))
    xt_traj.append(model.traj(xt_traj[-1],1))
    xa_traj.append(model.traj(xa_traj[-1],1))
xb_traj = array(xb_traj)
xt_traj = array(xt_traj)
xa_traj = array(xa_traj)

xb_err = xb_traj - xt_traj
xa_err = xa_traj - xt_traj
y_err = y - obs.gop(xt)


### TRAJECTORIES PLOT
#------------------------------------------------------------------------------


# Find a location with an observation and a location without an observation
obs_ind = total_space_obs//2
x_obs = space_inds_obs[obs_ind]

if obs_ind+1 <= total_space_obs:
    x_notobs = int((space_inds_obs[obs_ind] + space_inds_obs[obs_ind+1])//2)
else:
    x_notobs = int((space_inds_obs[obs_ind] + n)//2)


fig, ((ax_obs, ax_obs_err), (ax_notobs, ax_notobs_err)) = plt.subplots(nrows = 2,
                                                    ncols = 2, figsize = (20,10))

trange = range(0,nt)
ax_obs.plot(trange, xt_traj[:,x_obs],'-k', label='Truth')
ax_obs.plot(trange, xb_traj[:,x_obs],'-.b', label='Background')
ax_obs.plot(time_inds_obs, y[obs_ind::total_space_obs],'og', label='Observations')
ax_obs.plot(trange, xa_traj[:,x_obs],'-r', label='Analysis')
ax_obs.legend()
ax_obs.set_xlabel('time')
ax_obs.set_ylabel('Temperature')
ax_obs.set_title("Trajectory at x=" + str(x_obs) + " (location with observations)")
##
ax_obs_err.axhline(0, color='k', linewidth=0.5)
ax_obs_err.plot(trange, xb_traj[:,x_obs] - xt_traj[:,x_obs],'-.b', label='Background error')
ax_obs_err.plot(time_inds_obs, y[obs_ind::total_space_obs] - xt_traj[time_inds_obs,x_obs],
            'og', label='Observation error')
ax_obs_err.plot(trange, xa_traj[:,x_obs]- xt_traj[:,x_obs],'-r', label='Analysis error')
ax_obs_err.legend()
ax_obs_err.set_xlabel('time')
ax_obs_err.set_ylabel('Temperature')
ax_obs_err.set_title("Trajectory at x=" + str(x_obs)+ " (location with observations)")
##
##
ax_notobs.plot(trange, xt_traj[:,x_notobs],'-k', label='Truth')
ax_notobs.plot(trange, xb_traj[:,x_notobs],'-.b', label='Background')
ax_notobs.plot(trange, xa_traj[:,x_notobs],'-r', label='Analysis')
ax_notobs.legend()
ax_notobs.set_xlabel('time')
ax_notobs.set_ylabel('Temperature')
ax_notobs.set_title("Trajectory at x=" + str(x_notobs)+ " (location without observations)")
##
ax_notobs_err.axhline(0, color='k', linewidth=0.5)
ax_notobs_err.plot(trange, xb_traj[:,x_notobs] - xt_traj[:,x_notobs],'-.b', label='Background error')
ax_notobs_err.plot(trange, xa_traj[:,x_notobs]- xt_traj[:,x_notobs],'-r', label='Analysis error')
ax_notobs_err.legend()
ax_notobs_err.set_xlabel('time')
ax_notobs_err.set_ylabel('Temperature')
ax_notobs_err.set_title("Trajectory at x=" + str(x_notobs)+ " (location without observations)")

plt.tight_layout()
plt.show()
#------------------------------------------------------------------------------


# QUADRATIC COST FUNCTION
#------------------------------------------------------------------------------
plt.figure()
plt.plot(quadcost,'r*')
plt.ylabel("Quadratic cost function")
plt.xlabel("Number of CG iterations")
plt.show()
#------------------------------------------------------------------------------


### ANIMATED PLOT
#------------------------------------------------------------------------------

fig, (ax, ax_err)  = plt.subplots(ncols =2, figsize = (15,7))

l_xt, = ax.plot(xt_traj[0,:],'-k', label = "Truth")
l_xb, = ax.plot(xb_traj[0,:], '-.b', label = "Background")
l_y, = ax.plot(space_inds_obs, y[:total_space_obs], 'og', label='Observations')
l_xa, = ax.plot(xa_traj[0,:],'-r', label = "Analysis")
ax.set_xlabel("x-coordinate")
ax.legend()


ax_err.axhline(0, color='k',linewidth = 0.5)
l_xb_err, = ax_err.plot(xb_err[0,:], '-.b', label = "Background error")
l_y_err, = ax_err.plot(space_inds_obs,y_err[:total_space_obs], 'og', label='Observation error')
l_xa_err, = ax_err.plot(xa_err[0,:],'-r', label = "Analysis error")
ax_err.legend()
ax_err.set_ylim([min(xb_err)*1.1, max(xb_err)*1.1])
ax_err.set_xlabel("x-coordinate")

fig.suptitle('t=00'  )


def animate(i):
    l_xt.set_ydata(xt_traj[i,:]) 
    l_xb.set_ydata(xb_traj[i,:])  
    l_xb_err.set_ydata(xb_err[i,:])  
    if i in time_inds_obs:
        i_obs = where(time_inds_obs==i)[0][0]
        l_y.set_ydata(y[total_space_obs*i_obs:total_space_obs*(i_obs+1)])  
        l_y_err.set_ydata(y_err[total_space_obs*i_obs:total_space_obs*(i_obs+1)])  
    else:
        l_y.set_ydata(ones(total_space_obs)*1000)  
        l_y_err.set_ydata(ones(total_space_obs)*1000)  
    l_xa.set_ydata(xa_traj[i,:])  
    l_xa_err.set_ydata(xa_err[i,:])  
    fig.suptitle('t=' + str(i).zfill(2) )
    return l_xt, l_xb, l_y, l_xa, l_xb_err, l_y_err, l_xa_err, 

t = concatenate((arange(1,nt), [0]))
ani_plot = anim.FuncAnimation(
    fig, animate, frames=t, interval = 1000 )
#------------------------------------------------------------------------------



