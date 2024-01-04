"""
  Module: Adjoint Tests
  Author: Selime Gurol
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""

# Choose random vectors u and w
# Define: v = A*u
#         z = A'*w
# Test if 
# v'*w = (Au)'*w = u'*(A'*w) = u'*z

from numpy import random, round, linspace, shape
import sys
sys.path.append('../Model')
sys.path.append('../VariationalMethods')

from models import lorenz95
from operators import obs_operator

def test_lorenz_model():
    #Adjoint Test for Lorenz Model
    xx = random.randn(40)
    u = random.randn(40)
    w = random.randn(40)
    dt = 0.025  # time step for 4th order Runge-Kutta
    F = 8 # Forcing term
    model = lorenz95(F,dt)
    v = model.tlm_l95(xx, u)
    z = model.ad_l95(xx, w)
    diff = v.dot(w) - u.dot(z)
    print("Adjoint Test for Lorenz Model:", diff)

def test_Runge_Kutta():
    #Adjoint Test for 4th order Runge-Kutta
    xx = random.randn(40)
    u = random.randn(40)
    w = random.randn(40)
    dt = 0.025  # time step for 4th order Runge-Kutta
    F = 8 # Forcing term
    model = lorenz95(F,dt)
    v = model.dRKstep(xx, u)
    z = model.aRKstep(xx, w)
    diff = v.dot(w) - u.dot(z)
    print("Adjoint Test for Runge-Kutta:", diff)

def test_model_traj():
    #Adjoint Test for 4th order Runge-Kutta
    xx = random.randn(40)
    u = random.randn(40)
    w = random.randn(40)
    dt = 0.025  # time step for 4th order Runge-Kutta
    F = 8 # Forcing term
    model = lorenz95(F,dt)
    nt = 4
    v = model.tlm_traj(xx, u, nt)
    z = model.ad_traj(xx, w, nt)
    diff = v.dot(w) - u.dot(z)
    print("Adjoint Test for model traj:", diff)    

def test_hop():
    n = 40
    total_space_obs = 5 # total number of observations at fixed time 
    u = random.randn(n)
    w = random.randn(total_space_obs)
    sigmaR = 0.4 # observation error std
    space_inds_obs = round(linspace(1, n, total_space_obs, endpoint = False)).astype(int) # observation locations in the space
    obs = obs_operator(sigmaR, space_inds_obs, n)
    v = obs.tlm_hop(u)
    z = obs.adj_hop(w)
    diff = v.dot(w) - u.dot(z)
    print("Adjoint Test for Hop:", diff)

def test_gop():
    n = 10
    nt = 20
    
    sigmaR = 0.4 # observation error std
    total_space_obs = 10 # total number of observations at fixed time 
    total_time_obs = 10 # total number of observations at fixed location 
    space_inds_obs = round(linspace(0, n, total_space_obs, endpoint = False)).astype(int) # observation locations in the space
    time_inds_obs = round(linspace(0, nt, total_time_obs, endpoint = False)).astype(int) # observation locations along the time

    x = random.randn(n)
    u = random.randn(n)
    w = random.randn(total_time_obs*total_space_obs)

    dt = 0.025        # time step for 4th order Runge-Kutta
    F = 8             # Forcing term
    model = lorenz95(F,dt)

    obs = obs_operator(sigmaR, space_inds_obs, n, time_inds_obs, nt, model)

    v = obs.tlm_gop(x,u)
    z = obs.adj_gop(x,w)
    diff = v.dot(w) - u.dot(z)
    print("Adjoint Test for Gop:", diff)



if __name__ == '__main__':
    test_lorenz_model()
    test_Runge_Kutta()
    test_model_traj()
    test_hop()
    test_gop()

