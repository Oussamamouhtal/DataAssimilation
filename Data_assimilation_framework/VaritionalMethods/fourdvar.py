"""
  Module: 4D-Variational Data Assimilation
  Authors: Selime Gurol
         : Oussama Mouhtal - Plot for scaled Spectral LMP within CG,
          and deflated CG
             
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""

import sys
import math
from time import time
from numpy.lib.function_base import append
sys.path.append('../Model')

import numpy as np
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
plt.ioff()
import matplotlib.animation as anim # To plot a graph
from operators import obs_operator, Rmatrix, Bmatrix, init_p, EmptyLambdaError, get_ritzpair, plotRitzValues 
from operators import Hessian4dVar, Precond, PrecondHessian4dVar, spectrale_LMP, Split_Prec_Hessian, MatrixOperations
from models import lorenz95 
from solvers import pcg, Bcg, Lanczos, CG, deflated_CG



def fourDvar(n, m_t, Nt, max_outer, list_max_inner , method, selectedTHETA = None , IP = False):       
    """
    fourDvar implements the minimization process in data assimilation.
    Parameters:
        n               : state space dimension (int)
        m_t             : total number of observations at a fixed time (int)
        Nt              : total number of observations at a fixed location (int)
        max_outer       : number of maximum outer loops (Gauss-Newton loop) (int)
        list_max_inner       : number of maximum inner loops (Conjugate Gradient loop) (list of integer)
        method          : define the method to solve the inner loops (str):
                            Unprecon_CG : solving linear system without scaled LMP
                            Spectral    : solving linear system with scaled spectral LMP
                            Deflated_CG : solving linear system with Deflated CG
        selectedTHETA   : choose the scaling parameter (str):
                            lambda_k     : choose the smallest converged ritz value.
                            ThetaOpt     : choose theta that minimizes the error in energy norm 
                                         at the first iteration
                            mediane      : choose the mediane between the smallest converged rirz 
                            value and 1. 
        IP              :  Define a specific choice of the initial guess when solving inner loops (bool)
        
    """
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
    n  = n  # state space dimension 
    nt = 10          # number of time steps

    # Model class initialization
    dt = 0.025        # time step for 4th order Runge-Kutta
    F = 8            # Forcing term
    model = lorenz95(F,dt)

    # Observation class initialization
    sigmaR = 1e-2# observation error std
    total_space_obs = m_t # total number of observations at fixed time 
    total_time_obs = Nt # total number of observations at fixed location 
    space_inds_obs = round(linspace(0, n, total_space_obs, endpoint = False)).astype(int) # observation locations in space
    time_inds_obs = round(linspace(0, nt, total_time_obs, endpoint = False)).astype(int) # observation locations along the time
    m = total_space_obs*total_time_obs
    obs = obs_operator(sigmaR, space_inds_obs, n, time_inds_obs, nt, model)
    R = Rmatrix(sigmaR)

    # Background class initialization
    sigmaB = 0.8 # background error std
    #B = Bmatrix(n, sigmaB,'diagonal')
    B = Bmatrix(n, sigmaB,'diffusion', D=5, M=4) 

    # Minimization initialization
    tol = 1e-19  # tolerance for the inner loop
    tol_grad = 1e-6 # tolerance for the outer loop
    In = eye(n)

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                          (2) Generate the truth
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    xt = 3.0*ones(n)+randn(n) # the true state
    xt = model.traj(xt,5000)  # spin-up

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                          (3) Generate the background
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #The square root of B can be used to create correlated errors
    xb = xt + B.sqrtdot(randn(n))

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                          (4) Generate the observations
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y = obs.generate_obs(xt)

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                          (5) Variational data assimilation
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iter_outer = 0 
    dxs = []
    xa = copy(xb)   # Choose the initial vector for the outer loop
    du = zeros_like(xa) # Choose the initial vector for the inner loops
    quadcost = []
    iter = 0
    dx = zeros_like(xa)
    Precond_list = [eye(n)]
    while iter_outer < max_outer: # Gauss-Newton loop
        
        d = obs.misfit(y, xa) # misfit calculation (y - G(xa))
        b = B.invdot(xb-xa) + obs.adj_gop(xa, R.invdot(d))
        btilde = B.sqrtdot(b)
            
        
        if selectedTHETA == 'ThetaOpt' and iter_outer >= 1:
            bwidetilde = copy(btilde)
            for Y in Precond_list:
                bwidetilde = Y.dot(bwidetilde)
            bTAb = Atilde.dot(bwidetilde)
            bTAb = np.dot(bwidetilde, bTAb)
        else: 
            bwidetilde = None
            bTAb = None

        Atilde = PrecondHessian4dVar(obs,R,B, xa,n)  # Construct the B-preconditioned Hessian
        if norm(b) < tol_grad:
            break
        ###################### Calculate the increment dx such that with differents method  ################
        # (Binv + HtRinvH) dx = Binv(xb - x) + Ht Rinv d

        if iter_outer == 0:
            T_Lanczos , V_Lanczos, dxs, iter, flag, beta_Lanczos = CG(Atilde, du, btilde,  list_max_inner[iter_outer],tol, get_T = True)
            lambda_ , eignvect = get_ritzpair(T_Lanczos, V_Lanczos, iter, beta_Lanczos)    # Converged ritz pair

        
        else :
    
            if method == 'Unprecon_CG':
                if IP:
                    """
                    Solving system with du = S_k Lambda_k^{-1} S_k^{T} b 
                    such that the spectral information are comming from the previous system
                    """
                    initialGuessClass = init_p(Atilde, btilde, lambda_, eignvect)
                    du = initialGuessClass.initialGuess()     # x_0 = tildeS_k tildeLambda_k^{-1} tildeS_k^{T} b 

                dxs, iter = CG(Atilde, du, btilde, list_max_inner[iter_outer], tol, get_T = False)
                




            if method == 'Deflated_CG':
                """
                Solving inner loops with deflated cg. The deflated CG will not be recycled from inner loops
                """
                dxs, iter = deflated_CG(Atilde, btilde, list_max_inner[iter_outer], tol , eignvect)

            
            if method == 'Spectral_LMP':
                """
                Solving inner loops with a spectral LMP constructed 
                using old spectral information from the previous system
                """
    
                F = spectrale_LMP(lambda_, eignvect, bwidetilde, bTAb, selectedTHETA)   
                for Y in Precond_list:
                    btilde = Y.dot(btilde)    # btilde = F{i-1}...F1 @ F0 b
                if IP :
                    initialGuessClass = init_p(btilde, lambda_, eignvect)
                    du = initialGuessClass.initialGuess()     
                    du = F.invdot(du)      # x_0 = U.dot(du)      

                btilde = F.dot(btilde)        # btilde = FiF{i-1}...F1 @ F0 b
                Precond_list.append(F)


                 

                ### New right hand side
                
                
                
                Atilde = Split_Prec_Hessian(Atilde, Precond_list)  #  Atilde = Fi...F1 @ F0 @ A @ F0 @ F1...Fi
                AtildeCalcul = MatrixOperations(Atilde, n, n)
                _, lambda_A  = AtildeCalcul.calculate_lambda()
                T_Lanczos , V_Lanczos, dxs, iter, flag, beta_Lanczos, listRitzValues = CG(Atilde, 
                                        du, btilde, list_max_inner[iter_outer], tol, get_T = True, PlotRitz=True)
                
                # Plot Ritz value for the preconditioned System over CG iteration
                if False:
                    listLambda_A = [lambda_A for i in range(iter+1)]
                    CGiteration = [i+1 for i in range(iter+1)]
                    plotRitzValues(listRitzValues, listLambda_A, CGiteration)
                lambda_ , eignvect = get_ritzpair(T_Lanczos, V_Lanczos, iter, beta_Lanczos)    # Converged ritz pair

        

        if len(lambda_) == 0 and iter_outer < max_outer - 1: # Check if lambda_ is not empty 
            raise EmptyLambdaError(f"Error : Ritz values didn't converge at the outer iteration {iter_outer}")

        ###################################################################################################################
        dx = dxs[-n:]
        for Y in Precond_list[::-1]:
            dx = Y.dot(dx)
        dx = B.sqrtdot(dx)

        for i in range(iter+2):          
            iterate = dxs[i*n:(i+1)*n]
            for Y in Precond_list[::-1]:
                iterate = Y.dot(iterate)
            qval = math.log10(quadratic_funcval(xa, B.sqrtdot(iterate)))
            quadcost = append(quadcost, qval)


        xa += dx
        iter_outer += 1

    return quadcost
    #------------------------------------------------------------------------------

