"""
  Module: Operators for data assimilation
  Authors: Selime Gurol 
         : Olivier Goux - Diffusion operator for B
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""

from numpy.core.numeric import zeros, zeros_like
from numpy import eye, copy, shape, array, ndarray, ones, sqrt, pi
from numpy.random import randn, seed #, default_rng
from numpy.linalg import norm
from numpy import fft
from scipy import linalg, fft as sp_fft
from scipy import sparse as scp
from scipy.special import gamma
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt # To plot a graph

from solvers import pcg

from scipy import signal

class obs_operator:
    def __init__(self,sigmaR,space_inds,n, time_inds=[], nt=[], model=[]):
        self.sigmaR = sigmaR
        self.space_inds = space_inds
        self.time_inds = time_inds
        self.n = n
        self.nt = nt
        self.model = model

    def generate_obs(self, xt):
        # Generates observations from the truth trajectory
        x = copy(xt)
        y = []
        xrange = range(len(x))
        counter = 0
        verbose = 0
        if self.nt:
            # 4DVAR CASE
            for ii in range(self.nt):
                if ii in self.time_inds:
                    new_obs = self.hop(x) + self.sigmaR*randn(len(self.space_inds))
                    y.append(new_obs) 
                    counter += 1
                    if verbose:
                        plt.figure()
                        plt.plot(xrange, x,'-b')
                        plt.plot(self.space_inds, new_obs, '+g')
                        plt.show()
                if counter == len(self.time_inds):
                    break
                x = self.model.traj(x,1)
        else:
            # 3DVAR CASE
            y = self.hop(x) + self.sigmaR*randn(len(self.space_inds))

        return array(y).flatten()     

    def misfit(self, y, xt):
        # d = y - H(x)
        # For 4DVAR y includes all observations 
        # within the assimilation window 
        x = copy(xt)
        l = len(self.time_inds)
        nx = len(self.space_inds)
        if not l:
            # 3DVAR CASE
            return y - self.hop(xt)
        else:
            # 4DVAR CASE
            yy = copy(y.reshape((l,nx)))
            d = []
            counter = 0
            for ii in range(self.nt):
                if ii in self.time_inds:
                    d.append(yy[counter] - self.hop(x))
                    counter += 1
                if counter == len(self.time_inds):
                    break
                x = self.model.traj(x,1)   
            return array(d).flatten()   

    def hop(self, x):
        # Observation operator as a selection operator
        hx = x[self.space_inds]
        return hx

    def tlm_hop(self,dx):
        # Tangent linear model: Jacobian of hop
        dy = dx[self.space_inds]
        return dy

    def adj_hop(self,ay):
        # Adjoint model for hop
        ax = zeros(self.n) if len(ay.shape) == 1 else zeros((self.n, ay.shape[1]))
        ax[self.space_inds] = ay
        return ax 

    def gop(self, xt):
        # Generalized observation operator (H(M(x)))
        x = copy(xt)
        gx = []
        counter = 0
        for ii in range(self.nt):
            if ii in self.time_inds:
                gx.append(self.hop(x))
                counter += 1
            if counter == len(self.time_inds):
                break
            x = self.model.traj(x,1)
        return array(gx).flatten()

    def tlm_gop(self,xt, dxt):
        # Tangent linear model for the gop
        x = copy(xt)
        dx = copy(dxt)
        dgx = []
        counter = 0
        for ii in range(self.nt):
            if ii in self.time_inds:
                dgx.append(self.tlm_hop(dx))
                counter += 1
            if counter == len(self.time_inds):
                break
            dx = self.model.tlm_traj(x,dx,1)   
            x = self.model.traj(x,1) 
        return array(dgx).flatten()

    def adj_gop(self,xt, axt):
        # Adjoint model for the gop
        x = copy(xt)
        l = len(self.time_inds)
        nx = len(self.space_inds)
        agx = copy(axt.reshape((l,nx)))
        ax = zeros((self.time_inds[-1]+1,len(xt)))
        counter = 0
        traj_xx = []

        # Forward run
        for ii in range(self.nt):
            traj_xx.append(x)
            x = self.model.traj(x,1)
        for ii in range(self.time_inds[-1],-1,-1):
            if ii in self.time_inds[::-1]:
                ax[ii] += self.adj_hop(agx[l-counter-1])
                counter += 1
            if counter == len(self.time_inds):
                break
            ax[ii-1] = self.model.ad_traj(traj_xx[ii-1],ax[ii],1)
        
        return array(ax[0]).flatten()
            
class Rmatrix:
    def __init__(self, sigmaR):
        self.sigmaR = sigmaR
    def invdot(self,d):
        y = d/(self.sigmaR*self.sigmaR) 
        return y

class Bmatrix:
    def __init__(self, n, sigmaB, type, D = 10, M=4):
        self.D = D
        self.M = M
        self.type = type
        if type == 'diffusion' and D == 0:
            self.type = 'diagonal'
        else:
            assert (D>0) and (M>=2) and (M%2==0)
            
            self.h = h = 1 # grid resolution
            l = self.D/sqrt(2*M-3)
            
            #Initialize finite differences matrix T = I -2*(l/h)**2 * Laplacian
            self.T = scp.diags(
                [(1 + 2*(l/h)**2)*ones(n), -(l/h)**2*ones(n), -(l/h)**2*ones(n)],
                           [0,-1,1], format = 'csr')
            self.T += scp.csr_matrix(([-(l/h)**2, -(l/h)**2], ([n-1, 0], [0, n-1]) ))
            
            # Compute normalization factor
            self.normalization = 1
            self.sigmaB = 1
            dirac = zeros(n)
            dirac[n//2] = 1
            self.normalization = 1/sqrt(max(self.dot(dirac)))
            
        self.sigmaB = sigmaB
            
    def invdot(self,x):
        if norm(x) == 0.:
            return zeros_like(x)
    
        # Apply inverse std
        y = x / self.sigmaB
        
        if self.type == 'diffusion':
            # Apply inverse normalization
            y /= self.normalization
            
            # M/2 step of inverse diffusion
            for k in range(self.M//2):
                y = self.T.dot(y)
                
            # Apply grid step
            y *= self.h
                      
            # M/2 step of inverse diffusion
            for k in range(self.M//2):
                y = self.T.dot(y)
            
            # Apply inverse normalization
            y /= self.normalization
            
        # Apply inverse std
        y /= self.sigmaB
        return y
    

    def dot(self,x):
        if norm(x) == 0.:
            return zeros_like(x)
    
        # Apply  std
        y = x * self.sigmaB
        
        if self.type == 'diffusion':
            # Apply normalization
            y *= self.normalization
            
            # M/2 step of  diffusion
            for k in range(self.M//2):
                y = scp.linalg.spsolve(self.T,y)
                
            # Apply inverse of grid step
            y /= self.h
                      
            # M/2 step of  diffusion
            for k in range(self.M//2):
                y = scp.linalg.spsolve(self.T,y)
            
            # Apply inverse normalization
            y *= self.normalization
            
        # Apply inverse std
        y *= self.sigmaB
        return y
        
        
    def sqrtdot(self,x):
        if norm(x) == 0.:
            return zeros_like(x)
        
        if self.type == 'diffusion':
            # Apply inverse of grid step
            y = x/ sqrt(self.h)
                      
            # M/2 step of  diffusion
            for k in range(self.M//2):
                y = scp.linalg.spsolve(self.T,y)
            
            # Apply inverse normalization
            y *= self.normalization
            
        # Apply inverse std
        y *= self.sigmaB
        
        return y
        
class Hessian3dVar:
    def __init__(self,obs,R,B):
        self.obs = obs
        self.R = R
        self.B = B
    def dot(self,dx):
        w = self.R.invdot(self.obs.tlm_hop(dx))
        htrinvh_dx = self.obs.adj_hop(w)
        binv_dx = self.B.invdot(dx)
        dy =  binv_dx +  htrinvh_dx
        return dy

class Hessian4dVar:
    def __init__(self,obs,R,B,xt):
        self.obs = obs
        self.R = R
        self.B = B
        self.xt = copy(xt)
    def dot(self,dx):
        w = self.R.invdot(self.obs.tlm_gop(self.xt,dx))
        gtrinvg_dx = self.obs.adj_gop(self.xt,w)
        binv_dx = self.B.invdot(dx)
        dy =  binv_dx +  gtrinvg_dx
        return dy        

class Precond:
    def __init__(self, F):
        self.F = F

    def dot(self,x):
        y = (self.F).dot(x)
        return y 





    

    

