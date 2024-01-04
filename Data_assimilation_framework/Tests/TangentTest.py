"""
  Module: Tangent Tests
  Author: Selime Gurol
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""

from numpy import random, round, linspace, shape
from numpy.linalg import norm
import sys
sys.path.append('../Model')
sys.path.append('../VariationalMethods')

from models import lorenz95
from operators import obs_operator

def test_lorenz_model():
    #TLM Test for Lorenz95
    x = random.randn(40)
    p = random.randn(40)
    alpha = [1e-8*10**i for i in range(9)]
    dt = 0.025  # time step for 4th order Runge-Kutta
    F = 8 # Forcing term
    model = lorenz95(F,dt)
    y = [norm(model.l95(x + alpha[i]*p) - model.l95(x)) / norm(model.tlm_l95(x, alpha[i]*p)) for i in range(len(alpha))]
    print("TLM Test for Lorenz95:")
    print('alpha', '         ratio')
    [print('{:<9.1e}{:<5.12f}'.format(alpha[i], y[i])) for i in range(len(alpha))]

    
def test_Runge_Kutta():
    #TLM Test  for 4th order Runge-Kutta
    x = random.randn(40)
    p = random.randn(40)
    alpha = [1e-8*10**i for i in range(9)]
    dt = 0.025  # time step for 4th order Runge-Kutta
    F = 8 # Forcing term
    model = lorenz95(F,dt)
    y = [norm(model.RKstep(x + alpha[i]*p) - model.RKstep(x)) / norm(model.dRKstep(x, alpha[i]*p)) for i in range(len(alpha))]
    print("TLM Test for Lorenz95:")
    print('alpha', '         ratio')
    [print('{:<9.1e}{:<5.12f}'.format(alpha[i], y[i])) for i in range(len(alpha))]

def test_traj():
    #TLM Test  for model traj
    x = random.randn(40)
    p = random.randn(40)
    nt = 4
    alpha = [1e-8*10**i for i in range(9)]
    dt = 0.025  # time step for 4th order Runge-Kutta
    F = 8 # Forcing term
    model = lorenz95(F,dt)
    y = [norm(model.traj(x + alpha[i]*p,nt) - model.traj(x,nt)) / norm(model.tlm_traj(x, alpha[i]*p,nt)) for i in range(len(alpha))]
    print("TLM Test for model traj:")
    print('alpha', '         ratio')
    [print('{:<9.1e}{:<5.12f}'.format(alpha[i], y[i])) for i in range(len(alpha))]    

def test_hop():
    n = 40
    x = random.randn(n)
    p = random.randn(n)
    alpha = [1e-8*10**i for i in range(9)]
    sigmaR = 0.4 # observation error std
    total_space_obs = 30 # total number of observations at fixed time
    space_inds_obs = round(linspace(1, n, total_space_obs, endpoint = False)).astype(int) # observation locations in the space 
    obs = obs_operator(sigmaR, space_inds_obs, n)

    y = [norm(obs.hop(x + alpha[i]*p) - obs.hop(x)) / norm(obs.tlm_hop(alpha[i]*p)) for i in range(len(alpha))]
    print("TLM Test for observation operator:")
    print('alpha', '         ratio')
    [print('{:<9.1e}{:<5.12f}'.format(alpha[i], y[i])) for i in range(len(alpha))]

def test_gop():
    n = 40
    nt = 10
    x = random.randn(n)
    p = random.randn(n)
    alpha = [1e-8*10**i for i in range(9)]
    sigmaR = 0.4 # observation error std
    total_space_obs = 5 # total number of observations at fixed time 
    total_time_obs = 2 # total number of observations at fixed location 
    space_inds_obs = round(linspace(0, n, total_space_obs, endpoint = False)).astype(int) # observation locations in the space
    time_inds_obs = round(linspace(0, nt, total_time_obs, endpoint = False)).astype(int) # observation locations along the time

    dt = 0.025        # time step for 4th order Runge-Kutta
    F = 8             # Forcing term
    model = lorenz95(F,dt)

    obs = obs_operator(sigmaR, space_inds_obs, n, time_inds_obs, nt, model)

    y = [norm(obs.gop(x + alpha[i]*p) - obs.gop(x)) / norm(obs.tlm_gop(x, alpha[i]*p)) for i in range(len(alpha))]
    print("TLM Test for generalized observation operator:")
    print('alpha', '         ratio')
    [print('{:<9.1e}{:<5.12f}'.format(alpha[i], y[i])) for i in range(len(alpha))]

if __name__ == '__main__':
    test_lorenz_model()
    test_Runge_Kutta()
    test_traj()
    test_hop()
    test_gop()

