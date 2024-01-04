"""
  Module: Physical dynamics, Lorenz-96 model
  Authors: Selime Gurol 
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""
import numpy as np

class lorenz95:
#Lorenz 96 system with the tangent linear and adjoint model
#Time propogation with 4th order Runge-Kutta, its tangent linear and adjoint model
    def __init__(self, forcing, dt):
        # dt: time step for Runge-Kutta
        self.F = forcing
        self.dt = dt

    def l95(self,xx):
        F = self.F
        n=xx.shape[0]
        dxdt = np.zeros_like(xx)
        for i in range(n):
            im2 = i-2
            im1 = i-1
            ip1 = i+1
            if im2 < 0: im2 += n
            if im1 < 0: im1 += n
            if ip1 >= n: ip1 -=n
            dxdt[i] = (xx[ip1] - xx[im2]) * xx[im1] - xx[i] + F
        return dxdt

    # Tangent linear model of Lorenz dynamics
    def tlm_l95(self, xx, dx):
        F = self.F
        n=xx.shape[0]
        ddxdt = np.zeros_like(xx)
        for i in range(n):
            im2 = i-2
            im1 = i-1
            ip1 = i+1
            if im2 < 0: im2 += n
            if im1 < 0: im1 += n
            if ip1 >= n: ip1 -=n
            ddxdt[i] = (dx[ip1] - dx[im2]) * xx[im1] + (xx[ip1] - xx[im2])* dx[im1] - dx[i]
        return ddxdt

    # Adjoint model of Lorenz dynamics
    def ad_l95(self,xx, ax):
        F = self.F
        n=xx.shape[0]
        adxdt = np.zeros_like(xx)
        for i in range(n):
            im2 = i-2
            im1 = i-1
            ip1 = i+1
            if im2 < 0: im2 += n
            if im1 < 0: im1 += n
            if ip1 >= n: ip1 -=n
            adxdt[im2] -= xx[im1]*ax[i]
            adxdt[im1] += xx[ip1]*ax[i]
            adxdt[im1] -= xx[im2]*ax[i]
            adxdt[ip1] +=xx[im1]*ax[i]
            adxdt[i] -=ax[i]
        return adxdt    

    # 4th order Runge-Kutta 
    def RKstep(self,xx):
        dt = self.dt
        k1 = self.l95(xx)
        k2 = self.l95(xx+(dt/2.0)*k1)
        k3 = self.l95(xx+(dt/2.0)*k2)
        k4 = self.l95(xx+dt*k3)
        return xx+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4) 
    
    # Jacobian of 4th order Runge-Kutta 
    def dRKstep(self,xx, dx):
        dt = self.dt
        k1 = self.l95(xx)
        dk1 = self.tlm_l95(xx,dx)
        k2 = self.l95(xx+(dt/2.0)*k1)
        dk2 = self.tlm_l95(xx+(dt/2.0)*k1,dx+(dt/2.0)*dk1)
        k3 =  self.l95(xx+(dt/2.0)*k2)
        dk3 = self.tlm_l95(xx+(dt/2.0)*k2,dx+(dt/2.0)*dk2)
        dk4 = self.tlm_l95(xx+dt*k3,dx+dt*dk3)
        return dx+(dt/6.0)*(dk1+2.0*dk2+2.0*dk3+dk4)

    # Adjoint of 4th order Runge-Kutta     
    def aRKstep(self, xx, axp):
        dt = self.dt
        x0 = np.copy(xx)
        k1 = self.l95(x0)
        x1 = x0 + (dt/2.0)*k1
        k2 = self.l95(x1)
        x2 = x0+(dt/2.0)*k2
        k3 = self.l95(x2)
        x3 = x0+dt*k3

        ak4,ak3,ak2,ak1,ax,jak4,jak3,jak2 = np.zeros(8)
        ak4 += axp*(dt/6.0)
        ak3 += axp*(dt/3.0)
        ak2 += axp*(dt/3.0)
        ak1 += axp*(dt/6.0)
        ax += axp

        jak4 += self.ad_l95(x3, ak4)
        ak3 += dt*jak4
        ax += jak4

        jak3 += self.ad_l95(x2, ak3)
        ak2 += (dt/2.0)*jak3
        ax += jak3

        jak2 += self.ad_l95(x1, ak2)
        ak1 += (dt/2.0)*jak2
        ax += jak2

        ax += self.ad_l95(x0, ak1)

        return ax 

    def traj(self,x,nt):
        # nt: number of time steps
        xx = np.copy(x)
        for i in range(np.abs(nt)):
            xx = self.RKstep(xx)
        return xx

    def tlm_traj(self,x,dx,nt):
        xx = np.copy(x)
        dxx = np.copy(dx)
        for i in range(np.abs(nt)):
            dxx = self.dRKstep(xx,dxx)
            xx = self.RKstep(xx)
        return dxx

    def ad_traj(self,x,ax,nt):
        axx = np.copy(ax)
        xx = np.copy(x)
        traj_xx = []
        # Save model trajectory
        for i in range(np.abs(nt)):
            traj_xx.append(xx)
            xx = self.RKstep(xx)   
        # Reverse mode    
        for i in range(np.abs(nt)):
            axx = self.aRKstep(traj_xx[nt-i-1],axx)
        return axx

        

 
