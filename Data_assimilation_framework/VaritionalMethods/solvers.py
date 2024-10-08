"""
  Module: Iterative Solvers 
  Author: Selime Gurol
        : Oussama Mouhtal - Conjugate gradient and deflated conjugate gradient
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
  Updates: 2022 - Lanczos code - Jean-Guillaume De Damas
"""

import numpy as np
from scipy import linalg
from time import time
def solve_lower_triangular(matrix, vector):
    n = len(vector)
    solution = [0] * n

    for i in range(n):
        solution[i] = (vector[i] - sum(matrix[i][j] * solution[j] for j in range(i))) / matrix[i][i]

    return solution

def solve_upper_triangular(matrix, vector):
    n = len(vector)
    solution = [0] * n

    for i in range(n - 1, -1, -1):
        solution[i] = (vector[i] - sum(matrix[i][j] * solution[j] for j in range(i + 1, n))) / matrix[i][i]

    return solution



def pcg(A,x0,b,F,maxit,tol, ortho=True,verbose = False):
    """ 
    Preconditioned Conjugate Gradient Algorithm
        pcg solves the symmetric positive definite linear system 
            A x  = b 
        with a preconditioner F
        A     : n-by-n symmetric and positive definite matrix
        b     : n dimensional right hand side vector
        F     : n-by-n preconditioner matrix (an approximation to inv(A))
        maxit : maximum number of iterations
        tol   : error tolerance on the residual 
    """
    flag = 0
    x = np.copy(x0)
    X = x
    r = b - A.dot(x)
    nrmb = np.linalg.norm(b)
    error = np.linalg.norm(r)/nrmb

    if error < tol:
        return x, error, 0, flag

    for i in range(maxit-1):
        z = F.dot(r)
        rho = np.dot(r.T,z)

        if ortho:
            if i == 0:
                R = np.array([r/(rho**0.5)]).T
                Z = np.array([z/(rho**0.5)]).T
            else:
                R = np.append(R, np.array([r/(rho**0.5)]).T, axis=1)
                Z = np.append(Z, np.array([z/(rho**0.5)]).T, axis=1)

        if verbose:
            print("Iter ", i, " rho :", rho)
        if i > 0:
            beta = rho / rho_prev
            p = z + beta*p
        else:
            p = np.copy(z)
        q = A.dot(p)
        curvature = np.dot(p.T, q)
        if verbose:
            print("   curvature :", curvature)
        alpha = rho / curvature
        x = x + alpha*p
        X = np.concatenate((X,x))
        r = r - alpha*q

        # Re orthogonalisation
        if ortho:
            r = r - np.dot(R,np.dot(Z.T,r))
            #print(np.dot(Z.T,r))    

        error =  np.linalg.norm(r)/nrmb
        if error < tol:
            return X, error, i, flag   
        rho_prev = rho 

    if error > tol: 
        #print("No convergence")
        flag = 1
    return X, error, i, flag



def deflated_CG(A,b, maxit, tol , W, reortho = True):
    """ 
    Deflated Conjugate Gradient Algorithm solves the symmetric positive definite linear system 
            A x  = b 
        A     : n-by-n symmetric and positive definite matrix
        b     : n dimensional right hand side vector
        maxit : maximum number of iterations
        tol   : error tolerance on the residual 
        W     : deflation space
        reortho: Reorthogonalization step
    """
    n, k = W.shape
    #### Maintaining matrices in memory
    A_W = np.zeros((n,k)) # form the matrix A @ W 
    for j in range(k):
        A_W[:,j] = A.dot(W[:,j])
    W_A_W =  A_W.T @ W    # form the matrix W^T A W
    W_A_W = np.linalg.cholesky(W_A_W) # form the square root of W^T A W
    #### Construct the initial guess ######
    forw_backw_sub = solve_lower_triangular(W_A_W,  W.T.dot(b))                  # forward substitution
    forw_backw_sub = solve_upper_triangular(W_A_W.T, forw_backw_sub)   # backward substitution
    x = W.dot(forw_backw_sub)
    ######################################
    flag = 0                      # initialize a flag variable to check if the algorithm converged
    r = b - A.dot(x)  
    nrmb = np.linalg.norm(b)

        
    mu = solve_lower_triangular(W_A_W, A_W.T.dot(r))         # forward substitution
    mu = solve_upper_triangular(W_A_W.T, mu)              # backward substitution
    p = np.copy(r) - W.dot(mu)                        # set the initial search direction to r - W.dot(mu)
    rs_old = np.dot(r, r)         # compute the initial squared residual
    nrmb = np.linalg.norm(b)
    nrmr = np.linalg.norm(r)
    
    X = x 
    for i in range(maxit):  


        #### Re-orthogonalisation
        if reortho:
            forw_backw_sub = W.T.dot(r)
            r = r - W.dot(forw_backw_sub)
        
        q = A.dot(p) 
        curvature = np.dot(p, q)          
        alpha = rs_old / curvature      # compute the step size
        x = x + alpha * p               # update the current approximation x
        r = r - alpha * q               # update the residual



        X = np.concatenate((X, x))
        rs_new = np.dot(r, r)           # compute the new squared residual
        res = np.sqrt(rs_new) 
        beta = (rs_new / rs_old)
        
        if  res / nrmb < tol :  
            print(res / nrmb, i)
            break
        mu = solve_lower_triangular(W_A_W, A_W.T.dot(r))         # forward substitution
        mu = solve_upper_triangular(W_A_W.T, mu)       # backward substitution
        p = r + beta * p  - W.dot(mu)                             # update the search direction
        rs_old = rs_new 
 
    
    if res > tol:
        flag = 1
    
    return  X, i



def CG(A, x, b, maxit, tol, ortho = True, get_T = False, PlotRitz = False):
    """ 
    Conjugate Gradient Algorithm
        CG solves the symmetric positive definite linear system 
            A x  = b 
        A     : n-by-n symmetric and positive definite matrix
        b     : n dimensional right hand side vector
        maxit : maximum number of iterations
        tol   : error tolerance on the residual 
    """
    
    n = b.shape[0]
    flag = 0                      # initialize a flag variable to check if the algorithm converged
    r = b - A.dot(x)
    p = np.copy(r)                         # set the initial search direction to r
    rs_old = np.dot(r, r)         # compute the initial squared residual   
    nrmb = np.linalg.norm(b)

    ### Construct the Lancsoz matrices T and V
    if get_T:
        beta = 0
        alpha_old = 1
        T = np.zeros((maxit+1,maxit+1))
        V = np.zeros((n,maxit+1))
        V[:,0] = r / np.sqrt(rs_old)

    X = x 
    RitzValues = []  # For plotting ritz values
    for i in range(maxit):  
        if ortho:
            if i == 0:
                R = np.array([r/(rs_old**0.5)]).T
            else:
                R = np.append(R, np.array([r/(rs_old**0.5)]).T, axis=1)
            


        
        q = A.dot(p) 
        curvature = np.dot(p, q)      
        alpha = rs_old / curvature      # compute the step size
        x = x + alpha * p               # update the current approximation x
        r = r - alpha * q               # update the residual

        X = np.concatenate((X, x))
        # Tridiagonal matrix 
        if get_T:
            T[i,i] = (1 / alpha) + (beta / alpha_old)
            if PlotRitz:
                ritz , _ = np.linalg.eigh(T[:i + 1,:i + 1])
                RitzValues.append(ritz)
        # Re orthogonalization
        if ortho:
            for j in range(i+1):
                proj = np.dot(R[:,j].T,r)
                r = r - proj*R[:,j]
        rs_new = np.dot(r, r)           # compute the new squared residual
        beta = (rs_new / rs_old)

        
        res = np.sqrt(rs_new)
        if res/nrmb < tol:    
            #print(res/nrmb, i)
            break
        
        p = r + beta * p  # update the search direction
        if get_T:
            V[:,i + 1] = ((-1)**(i + 1)) * (r / np.sqrt(rs_new))
            beta_ritz = np.sqrt(beta) / alpha
            T[i,i+1] = beta_ritz
            T[i+1,i] = beta_ritz
            alpha_old = alpha 

        rs_old = rs_new 
 
    
    if res > tol:
        flag = 1
    if get_T and PlotRitz:
        return T[:i + 1,:i + 1] ,  V[:,:i+1] , X, i, flag, beta_ritz, RitzValues
    elif get_T:
        return T[:i + 1,:i + 1] ,  V[:,:i+1] , X, i, flag, beta_ritz
    else:
        return  X, i

def Bcg(B,HtRinvH,x0,b,maxit,tol,ortho=True,verbose = False):
    """ 
    B-Right Preconditioned Conjugate Gradient Algorithm
        Bcg_right solves the linear system 
             (I + HtRinvH B) v = b
        with an inner product < , >_B
        HtRinvH : n-by-n symmetric and positive definite matrix
        B       : n-by-n symmetric and (positive definite matrix)!
        b       : n dimensional right hand side vector
        maxit   : maximum number of iterations
        tol     : error tolerance on the residual 
    """
    flag = 0
    x = np.copy(x0)
    xh = np.copy(x0)
    X = x
    #r = b - np.matmul(A,x)
    #r  = b - x - HtRinvH.dot(B.dot(x))
    r = b
    z  = B.dot(r)
    h  = r

    nrmb = (np.dot(b.T, B.dot(b)))**0.5
    error = ((np.dot(r.T, z))**0.5)/nrmb
    if error < tol:
        return x, error, 0, flag

    for i in range(maxit-1):
        z = B.dot(r)
        rho = np.dot(r.T,z) 
        if ortho:
            if i == 0:
                R = np.array([r/(rho**0.5)]).T
                Z = np.array([z/(rho**0.5)]).T
            else:
                R = np.append(R, np.array([r/(rho**0.5)]).T, axis=1)
                Z = np.append(Z, np.array([z/(rho**0.5)]).T, axis=1)

        if verbose:
            print("Iter ", i, " rho :", rho)
        if i > 0:
            beta = rho / rho_prev
            p = z + beta*p
            h = r + beta*h
        else:
            p = z
            h = r
        #q = h + HtRinvH.dot(p)
        q = h + HtRinvH.dot(p)
        curvature = np.dot(p.T, q)
        if verbose:
            print("   curvature :", curvature)
        if curvature < 0:
            print('negative curvature')    
        alpha = rho / curvature
        x = x + alpha*p
        X = np.concatenate((X,x))
        r = r - alpha*q

        # Re orthogonalisation
        if ortho:
            r = r - np.dot(R,np.dot(Z.T,r))
            #print(np.dot(Z.T,r)) 

        error =  (np.dot(r.T, B.dot(r)))**0.5/nrmb
        if error < tol:
            return X, error, i, flag   
        rho_prev = rho 

    if error > tol: 
        #print("No convergence")
        flag = 1
    return X, error, i, flag
    
    
def Lanczos(A,x0,b,maxit,tol,ortho=True, get_T = False):
    """ 
    Preconditioned Lanczos Algorithm
        Lanczos solves the symmetric positive definite linear system 
            A x  = b 
        A     : n-by-n symmetric and positive definite matrix
        b     : n dimensional right hand side vector
        maxit : maximum number of iterations
        tol   : error tolerance on the residual 
    """
    n = b.shape[0]
    
    flag = 0
    nrmb = np.linalg.norm(b)
    r = b - A.dot(x0)
    beta0 = np.linalg.norm(r)
    beta = 0
    v = r/beta0
    V = np.zeros((n,maxit))
    V[:,0] = v
    T = np.zeros((maxit,maxit))
    X = x0
    Ritz_value = []
    abscisses = []
    #print('iter','   res', '     res_beta','       beta')

    for i in range(maxit-1):
        if i > 0:
            w = A.dot(v) - beta*V[:,i-1]
        elif i ==0:
            w = A.dot(v)
        alpha = np.dot(w.T,v)
        T[i,i] = alpha
        w = w - alpha*v
        beta = np.linalg.norm(w)
        
        #Tinv  = np.linalg.inv(T[:i+1,:i+1])
        #y = Tinv[:,0]*beta0
        s = np.zeros(i+1)
        s[0] = beta0
        y =  np.linalg.solve(T[:i+1,:i+1], s)
        x = x0 + np.dot(V[:,:i+1],y)
        X = np.concatenate((X,x))
        error = np.linalg.norm(b-A.dot(x))/nrmb
        lambda_ , eignvect2 = np.linalg.eigh(T[:i + 1,:i + 1])
        Ritz_value.append(lambda_)
        abscisses.append(i)
        #if  beta < 1e-12 or error < tol:
        #    break
                            
        T[i,i+1] = beta
        T[i+1,i] = beta
        v = w/beta
        
        # Re-orthogonalisation
        if ortho:
            for j in range(i+1):
                proj = np.dot(V[:,j].T,v)
                v = v - proj*V[:,j]
        
        V[:,i+1] = v
        
        # Check that V is orthogonal
        #Q = np.dot(V[:,:i+1].T,V[:,:i+1])
        
    if error > tol:
        flag = 1
        
  
    return  X, error, i, flag


if __name__ == '__main__':
    n = 1000
    A = np.random.rand(n,n)
    A = np.matmul(A, A.T)
    P = np.eye(n)
    b = np.random.rand(n)
    x0 = np.zeros_like(b)
    xstar = np.linalg.solve(A, b)
    maxit = 1000
    tol   = 1e-8
    print(" > Solution with Conjugate Gradient Algorithm")
    [cg_sol, error, iter, flag] = pcg(A,x0,b,P,maxit,tol)
    print("|| xstar - cg_sol || : ", np.linalg.norm(xstar - cg_sol[-n:]),'\n')  

    print(" > Solution with Lanczos Algorithm")
    [lanczos_sol, error, iter, flag] = Lanczos(A,x0,b,maxit,tol)
    print("|| xstar - lanczos_sol || : ", np.linalg.norm(xstar-lanczos_sol[-n:]),'\n') 
    #print("|| cg_sol - lanczos_sol || : ", np.linalg.norm(cg_sol[-n:]-lanczos_sol[-n:]),'\n') 

    # Data Assimilation Set-Up
    #H matrix
    m = 8
    Hmat = np.eye(n)
    inds=np.random.permutation(m)
    Hmat = Hmat[:,inds]
    Hmat = Hmat.T 
    #R matrix
    Imat = np.eye(n)
    sigma_r = 0.3
    Rinv = sigma_r*np.eye(m)
    RinvH  = np.matmul(Rinv, Hmat)
    #Hessian 
    HtRinvH = np.matmul(Hmat.T, RinvH)
    
    #B matrix
    B12 = np.random.rand(n,n)
    B = np.dot(B12, B12.T)
    A = np.linalg.inv(B) + HtRinvH

    xstar = np.linalg.solve(A, b)
    print("Data Assimilation Test-Case")
    print(" > Solution with Conjugate Gradient Algorithm")
    [cg_sol, error, iter, flag] = pcg(A,x0,b,B,maxit,tol)
    print("|| xstar - cg_sol || : ", np.linalg.norm(xstar-cg_sol[-n:]),'\n') 
    print(" > Solution with B - Conjugate Gradient Algorithm")
    [cgright_sol, error, iter, flag] = Bcg(B,HtRinvH,x0,b,maxit,tol)
    print("|| xstar - cgright_sol || : ", np.linalg.norm(xstar-cgright_sol[-n:]))
    

    
    
