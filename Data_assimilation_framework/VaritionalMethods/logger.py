from fourdvar import fourDvar
import matplotlib.pyplot as plt


# ===================== Define All Parameters for fourDvar =====================
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

# ===================== The Problem Parameters =====================
n = 120
m_t = 15
Nt = 3

# ===================== The Solvers Parameters =====================
max_inner = 25  # max iteration for cg loop
max_outer = 3   # max iteration for Gauss Newton loop

# ===================== Methods for Estimating Eigen-Pair =====================
'''
The parameter <estimateEigen> is for choosing
the way to estimate the eigen-pair:

    Exact          : for true eigen-pair
    PreviousSystem : use the eigen-pair corresponding 
                     to the previous system
    Randomestimat  : for eigen-pair estimated with RSVD
'''
estimateEigen = ['Exact', 'PreviousSystem', 'Randomestimat']

# ===================== Methods for Solving Linear System =====================
'''
The parameter <method> chooses the way to solve the linear
system:

    Unprecon_CG                 : solving linear system without preconditioning
    Spectral                    : solving linear system with spectral LMP
    Spectral_with_optimized_theta:  Solving the linear system involves employing 
                                    spectral LMP and incorporating a scaling parameter 
                                    obtained through an optimization problem. 
                                    In this optimization, the objective function is 
                                    the squared error in energy norm at 
                                    a specified iteration  <iterForFopti> 
    Deflated_CG                 : solving linear system with Deflated CG
'''
method = ['Unprecon_CG', 'Spectral', 'Spectral_with_optimized_theta', 'Deflated_CG']

# ===================== Methods for RSVD =====================
'''
The parameter <rmethod> for choosing the method to estimate the eigen-pair using randomness:

    basic       :
    single_pass :
    nystrom     :
    nystrom_sp  :
'''
rmethod = ['basic', 'single_pass', 'nystrom', 'nystrom_sp']

# ===================== Choose the Scaling Parameter =====================
'''
The parameter <selectedTHETA> for choosing the scaling parameter:

    eigenvalue     : choose the scaling parameter as an eigenvalue
                     the parameter <Index_scalling> is for 
                     selecting an eigenvalue
    optimizedTheta : choose theta that minimizes the error in energy norm 
                     at the first iteration
    anyTheta       : choose any scaling parameter, and <theta_iter> 
                     represents the given choice of this parameter. 
                     By default <theta_iter>  = 1.0
'''
selectedTHETA = ['eigenvalue', 'optimizedTheta', 'anyTheta']
Index_scalling = -1  # Index_scalling = 0, ..., k-1 
theta_iter = 1
k = 10
p = 10














# ==================== Example of Different Computations ====================

# ======================================  Exact eigen-pair ==================

#out_Unprec_with_IP= fourDvar(n,m_t, Nt, max_outer, max_inner, 
#                           'Exact', 'Unprecon_CG' , IP = False, projec=False)     
#out_True_Spectrale_with_scalling_k  = fourDvar(n,m_t, Nt, max_outer, max_inner, 
#                            'Exact', 'Spectral' ,selectedTHETA = 'eigenvalue', 
#                            IP = False, Index_scalling = -1) 
#out_True_Spectrale_with_scalling_1  = fourDvar(n,m_t, Nt, max_outer, max_inner,
#                            'Exact', 'Spectral' ,selectedTHETA = 'eigenvalue', 
#                            IP = False, Index_scalling = 0) 
#out_True_Spectrale_with_scalling_opt  = fourDvar(n,m_t, Nt, max_outer, max_inner, 
#                            'Exact', 'Spectral' ,selectedTHETA = 'optimizedTheta', 
#                            IP = False) 
#out_True_Spectrale  = fourDvar(n,m_t, Nt, max_outer, max_inner, 'Exact', 
#                            'Spectral_with_optimized_theta', iterForFopti = 1) 



# ============================  Ritz pair from previous system ==================

out_Unprec= fourDvar(n,m_t, Nt, max_outer, max_inner, 
                            'PreviousSystem', 'Unprecon_CG' , IP = False, projec=False)    

out_Spectrale_with_scalling  = fourDvar(n,m_t, Nt, max_outer, max_inner, 
                            'PreviousSystem', 'Spectral' ,selectedTHETA = 'eigenvalue') 
                            # scaling parameter represent the lowest ritz value converged 

out_Spectrale = fourDvar(n,m_t, Nt, max_outer, max_inner,
                            'PreviousSystem', 'Spectral' ,selectedTHETA = 'anyTheta',
                              IP = False) 

out_deflated = fourDvar(n,m_t, Nt, max_outer, max_inner, 'PreviousSystem', 
                            'Deflated_CG') 

# ==================== Displays ====================

plt.xlabel('total CG itérations')
plt.ylabel('Quadratic cost function')
plt.yscale('log')
plt.axvline(x = 0, color = 'black')
for i in range(max_outer-1):
    plt.axvline(x = (i+1)*(max_inner+1), color = 'black')

plt.plot(out_Unprec,':d',label='No second level preconditionner')
plt.plot(out_Spectrale_with_scalling,':*',label= 'scaled spectrale LMP')
plt.plot(out_Spectrale,':*',label= 'spectrale LMP'+ r'   $\theta = 1$')
plt.plot(out_deflated,':*',label= 'Deflated CG')

plt.legend()
plt.show()

            