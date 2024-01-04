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
from operators import obs_operator, Rmatrix, Bmatrix, OptimizerTheta, init_p, exactSpectral_Precond, get_ritzpair 
from operators import Hessian4dVar, Precond, PrecondHessian4dVar, spectrale_LMP, rSpectral_Precond, Split_Prec_Hessian
from models import lorenz95 
from solvers import pcg, Bcg, Lanczos, CG, deflated_CG



def fourDvar(n, m_t, Nt, max_outer, max_inner, estimateEigen , method, rmethod= 'basic', selectedTHETA = 'eigenvalue', 
             k=10, p=10, Index_scalling = -1, IP = False, theta_iter = 1, iterForFopti = 2, projec = False):       
    """
    fourDvar implements the minimization process in data assimilation.
    Parameters:
        n               : state space dimension (int)
        m_t             : total number of observations at a fixed time (int)
        Nt              : total number of observations at a fixed location (int)
        max_outer       : number of maximum outer loops (Gauss-Newton loop) (int)
        max_inner       : number of maximum inner loops (Conjugate Gradient loop) (int)
        estimateEigen   : choose the way to estimate the eigen-pair (str)
        method          : define the method to solve the inner loops (str)
        rmethod         : define the algorithm to use when estimating eigenvalues with randomized SVD (str)
        selectedTHETA   : choose the scaling parameter (str)
        k               : number of selected eigenvalues and their corresponding eigenvectors (int)
        p               : oversampling parameter used when sampling a matrix for RSVD (int)
        IP              : define a specific choice of the initial guess when solving inner loops (bool)
        theta_iter      : Define any scaling parameter for the spectral LMP
        iterForFopti    : Provide the iteration at which to minimize the error in energy norm 
                        with respect to theta (int)
        projec          : Reorthogonalize the residual when choosing a specific initial guess (bool)
        
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
    tol = 1e-26  # tolerance for the inner loop
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
    while iter_outer < max_outer: # Gauss-Newton loop
        
        
        Atilde = PrecondHessian4dVar(obs,R,B, xa,n)  # Construct the B-preconditioned Hessian
        d = obs.misfit(y, xa) # misfit calculation (y - G(xa))
        b = B.invdot(xb-xa) + obs.adj_gop(xa, R.invdot(d))
        
        
        btilde = B.sqrtdot(b)
        if norm(b) < tol_grad:
            break
        ###################### Calculate the increment dx such that with differents method  ################
        # (Binv + HtRinvH) dx = Binv(xb - x) + Ht Rinv d
        if estimateEigen == 'Exact':

            F = exactSpectral_Precond(Atilde, n, k, selectedTHETA, Index_scalling, btilde, 
                                      theta_iter) # spectral LMP operator with true eigen-pair
            initialGuessClass = init_p(Atilde, btilde, F.tetas, F.U)   # class for intialisation inner loops
            
            if method == 'Unprecon_CG':
                """
                Solving inner loops without second level preconditioner
                """
    
                Precond_list = [eye(n)] 
                if IP: 
                    du = initialGuessClass.initialGuess()  # x_0 = S_k Lambda_k^{-1} S_k^{T} b
                dxs, iter = CG(Atilde,du,btilde,max_inner,tol,Eigenvec = F.U,get_T = False, projec = projec)
                

            if method == 'Spectral':
                """
                Solving inner loops with second level preconditioner
                """
                Precond_list = [F]
                if IP:
                    du = initialGuessClass.initialGuess()     # x_0 = S_k Lambda_k^{-1} S_k^{T} b 
                    du = F.invdot(du)    # x_0 = U.dot(du)

                ### The preconditioned matrix
                Atilde = Split_Prec_Hessian(Atilde, Precond_list)
                ### New right hand side
                btilde = F.dot(btilde)
                
                dxs, iter = CG(Atilde,du, btilde,max_inner,tol, k, get_T = False)
            

            if method == 'Spectral_with_optimized_theta':
                """
                Solving inner loops with a second-level preconditioner. 
                Theta scaling is chosen such that it minimizes the square
                error in the energy norm at the given iterate <iterForFopti>
                """
                thetaOpt = OptimizerTheta(Atilde, k, btilde, iterForFopti, n, np.zeros_like(btilde))
                theta_iter = thetaOpt.Optimize()
                F = exactSpectral_Precond(Atilde, n, k, 'anyTheta', Index_scalling, btilde, theta_iter)
                Precond_list = [F]
                ### The preconditioned matrix
                Atilde = Split_Prec_Hessian(Atilde, Precond_list)
                ### New right hand side
                btilde = F.dot(btilde)
                
                dxs, iter = CG(Atilde,np.zeros_like(btilde), btilde,max_inner,tol)



        if estimateEigen == 'PreviousSystem':
            if iter_outer == 0:
                T_Lanczos , V_Lanczos, dxs, iter, flag, beta_Lanczos = CG(Atilde, du, btilde,  max_inner,tol, get_T = True)
                lambda_ , eignvect = get_ritzpair(T_Lanczos, V_Lanczos, iter, beta_Lanczos)    # Converged ritz pair
                Precond_list = [eye(n)]
            else :
                
                if method == 'Unprecon_CG':
                    """
                    Solving system with du = S_k Lambda_k^{-1} S_k^{T} b 
                    such that the spectral information are comming from the previous system
                    """
                    if IP:
                        initialGuessClass = init_p(Atilde, btilde, lambda_, eignvect)
                        du = initialGuessClass.initialGuess()     # x_0 = tildeS_k tildeLambda_k^{-1} tildeS_k^{T} b 

                        
                    T_Lanczos , V_Lanczos, dxs, iter, flag, beta_Lanczos = CG(Atilde, du, btilde, max_inner,tol,
                                                                              Eigenvec = eignvect, get_T = True,projec = projec)
                    lambda_ , eignvect = get_ritzpair(T_Lanczos, V_Lanczos, iter, beta_Lanczos)    # Converged ritz pair




                if method == 'Deflated_CG':
                    """
                    Solving inner loops with deflated cg
                    """
                    initialGuessClass = init_p(Atilde, btilde, lambda_, eignvect)
                    du = initialGuessClass.deflated_cg_initialGuess(eignvect)
                    dxs, iter = deflated_CG(Atilde, du, btilde, max_inner, tol , eignvect) 
            


                if method == 'Spectral':
                    """
                    Solving inner loops with a spectral LMP constructed 
                    using old spectral information from the previous system
                    """
                    if selectedTHETA == 'eigenvalue':
                        #Spectral LMP operator constructed with an approximate eigen-pair. Choose the lowest Ritz value as scaling
                        F = spectrale_LMP(lambda_, eignvect,teta = lambda_[0])  
                        Precond_list.append(F)
                    else:
                        F = spectrale_LMP(lambda_, eignvect)   # Choose 1.0 as scalling
                        Precond_list.append(F)                    
                    if IP :
                        initialGuessClass = init_p(Atilde, btilde, lambda_, eignvect)
                        du = initialGuessClass.initialGuess()     
                        du = F.invdot(du)      # x_0 = U.dot(du)       
                    ### New right hand side
                    for Y in Precond_list:
                        btilde = Y.dot(btilde)    # btilde = Fi...F1 @ F0 b
                    Atilde = Split_Prec_Hessian(Atilde, Precond_list)  #  Atilde = Fi...F1 @ F0 @ A @ F0 @ F1...Fi
                    T_Lanczos , V_Lanczos, dxs, iter, flag, beta_Lanczos = CG(Atilde, du, btilde, max_inner, tol, get_T = True)
                    lambda_ , eignvect = get_ritzpair(T_Lanczos, V_Lanczos, iter, beta_Lanczos)



        if estimateEigen == 'Randomestimat':                

            if method == 'Spectral':
                
                Omega = np.random.normal(size=(n,k+p))
                # spectral LMP operator with RSVD
                F = rSpectral_Precond(Atilde,k,p,Omega,selectedTHETA, rmethod , Index_scalling, theta_iter) 
                initialGuessClass = init_p(Atilde, btilde, F.tetas, F.U)
                Precond_list = [F]

                if IP:
                    du = initialGuessClass.initialGuess()            # x_0 = S_k Lambda_k^{-1} S_k^{T} b 
                    du = F.invdot(du)    # x_0 = U.dot(du)

                Atilde = Split_Prec_Hessian(Atilde, Precond_list)    # The preconditioned matrix

                ### New right hand side
                btilde = F.dot(btilde)
                dxs, iter= CG(Atilde,du, btilde,max_inner,tol, k)
            
       
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

