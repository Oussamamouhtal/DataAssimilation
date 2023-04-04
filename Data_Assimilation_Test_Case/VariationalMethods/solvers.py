"""
  Module: Iterative Solvers 
  Author: Selime Gurol
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
  Updates: 2022 - Lanczos code - Jean-Guillaume De Damas
"""
import numpy as np

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
    
    
def Lanczos(A,x0,b,maxit,tol,ortho=True):
    """ 
    Preconditioned Lanczos Algorithm
        Lanczos solves the symmetric positive definite linear system 
            A x  = b 
        A     : n-by-n symmetric and positive definite matrix
        b     : n dimensional right hand side vector
        maxit : maximum number of iterations
        tol   : error tolerance on the residual 
    """
    flag = 0
    nrmb = np.linalg.norm(b)
    n = b.shape[0]
    r = b - A.dot(x0)
    beta0 = np.linalg.norm(r)
    beta = 0
    v = r/beta0
    V = np.zeros((n,maxit))
    V[:,0] = v
    T = np.zeros((maxit,maxit))
    X = x0
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
        if beta < 1e-12 or error < tol:
            return X, error, i, flag
        
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
    
    return X,error,i, flag


if __name__ == '__main__':
    n = 30
    A = np.random.rand(n,n)
    A = np.matmul(A, A.T)
    P = np.eye(n)
    b = np.random.rand(n)
    x0 = np.zeros_like(b)
    xstar = np.linalg.solve(A, b)
    maxit = 100
    tol   = 1e-8
    print(" > Solution with Conjugate Gradient Algorithm")
    [cg_sol, error, iter, flag] = pcg(A,x0,b,P,maxit,tol)
    print("|| xstar - cg_sol || : ", np.linalg.norm(xstar-cg_sol[-n:]),'\n')  

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
    

    
    
