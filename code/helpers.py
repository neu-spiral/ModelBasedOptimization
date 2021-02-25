import torch
import matplotlib.pyplot as plt
import pickle
import time
import argparse 
import numpy as np
import sympy as sym
import math
from numpy import linalg as LA
import logging
from scipy.interpolate import interp1d

torch.manual_seed(1993)

def solve_ga_bisection(a, p, epsilon=1.e-8):
    """Return the solution of (x/a)^(p-1)+x=1, via bi-section method."""
    if a>0.:
        U = 1
        L = 0.
        error = epsilon + 1
        f = lambda x:(x/a)**(p-1)+x-1
        while error>epsilon:
            C = (L+U)/2. 
            if f(C)*f(U) <= 0:
                L = C
            else:
                U = C
            
            error = (U-L)
           
            
        return C
    else:
        return 0.

def pNormProxOp(V, rho, p=2, eps=1.e-6, g_est=None):
    """
        Return the p-norm prox operator for the vector V
               argmin_X ||X||_p + rho/2 \|X - V\|_2^2
    """
    def g_func(a, p, bisect=True):
        "Return the solution of (x/a)^(p-1) + x = 1"
        if a>0:
            x = sym.Symbol("x", positive=True)
            sols = sym.solvers.solve((x/a)**(p-1)+x-1, x)
            if len(sols) == 0:
                sol = 0.
            else:
                sol = max(sols)
            return sol
        else:
            return  0.

    t_start = time.time()
    b_size, data_size = V.size()

    if b_size != 1:
        raise Exception('V must has a batch dimenstion of one.')
    V = V.squeeze(0)

    if p == 2:
        return EuclidianProxOp(V, 1./rho)

    elif p == 1:
        return ell1normProxOp(V, rho)
    elif p == -2:
        return norm2squaredProxOp(V, rho)
    signs = torch.sign(V)
    V_normalized = rho * torch.abs(V)
    vec_size = V_normalized.size()[0]
    q =  p/(p-1.)
    if torch.norm(V_normalized, p=q) < 1:
        U = torch.zeros( vec_size )
        U = U.unsqueeze(0)
        return U 
    upper_bound = torch.norm(V_normalized, p=p)
    lower_bound = 0.0
    U =  torch.zeros(vec_size, dtype=torch.float)

    #estimator for g function
    if g_est is None:
        g_est = estimate_gFunction(p)
    for k in  range( math.ceil(math.log2(1./eps)) ):

        mid_bound = 0.5 * (upper_bound + lower_bound )
        for j in range(vec_size):
           try:
              U_j = V_normalized[j] * g_est(mid_bound * V_normalized[j] ** ((2.0-p) / (p-1.0)) )
           except ValueError:
              #if argument of g_est is above the given values (during estimating ga) the output is 1
              U_j = V_normalized[j] 

           U[j] = U_j

        #compute norm
        U_norm = torch.norm(U, p=p)

        #update bounds
        if U_norm < mid_bound:
            upper_bound = mid_bound
        else:
            lower_bound = mid_bound


    #logging.debug('Computed the proximal operator in {0:0.2f}(s)'.format(time.time() - t_start) )
    U = U.unsqueeze(0)
    return U * signs / rho

def norm2squaredProxOp(V, rho):
    """
        Return the 2-norm prox operator for the vector V
             argmin_X ||X||_2^2 + rho/2 \|X - V\|_2^2
    """
    return rho / (rho + 2) * V

def EuclidianProxOp(V, rho):
     """
        Return the 2-norm prox operator for the vector V
             argmin_X rho ||X||_2 + 1/2 \|X - V\|_2^2
     """
     V_norm = torch.norm(V, 2)
     if V_norm < rho:
         return torch.zeros( V.size() ).unsqueeze(0)
     return (1 - rho / V_norm ) * V.unsqueeze(0)

def ell1normProxOp(V, rho):
    """
        Return the 2-norm prox operator for the vector V
             argmin_X ||X||_1 + rho/2 \|X - V\|_2^2
    """
    V_proj =  torch.max(V - 1./rho, V * 0.0) -  torch.max(-1. * V - 1./rho, V * 0.0)
    return V_proj.unsqueeze(0)

def estimate_gFunction(p, eps=1.e-8, kind='linear'):
    """
        Return an estimator for  the g function that is the solution of (x/a)^(p-1)+x=1.
    """

    g_inv = lambda x: (1. - x ) ** (1./ (1. - p)) * x
    #Generate a set of x and a pairs
    x = np.arange(0, 1, eps)
    a = g_inv( x )
    return interp1d(a, x, kind=kind)



    
    
     
      
def _testOpt(U, V, rho, p):
    V  = V.squeeze(0)
    U = U.squeeze(0)
    vec_size = V.size()[0]
    norm_U =  torch.norm(U, p=p)
    return [(U[i] / norm_U) ** (p-1.0) + rho * (U[i] - V[i]) for i in range(vec_size)]
        
def clearFile(file):
    "Delete all contents of a file"
    with open(file,'w') as f:
        f.write("")

def dumpFile(fname, obj):
    """
       Dump picklable object obj to the file fname."
    """
    with open(fname,'wb') as f:
        pickle.dump(obj,  f)

def loadFile(fname):
    """
       Load the object dumped in fname.
    """
    with open(fname, 'rb') as current_file:
        obj  = pickle.load(current_file)
    return obj
     
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="vector size", default=10)
    parser.add_argument("--p", type=float, help="p norm", default=2.)
    parser.add_argument("--rho", type=float, help="rho", default=1.0)
    args = parser.parse_args()
 
#    for p in [1.5, 2.5,4,5]:
#        estimate_g = estimate_gFunction(p)
#        dumpFile('interpolations/p' + str(p),  estimate_g)
#        print('Estimation and saving done for {}'.format(p))

    if args.p not in [1, 2]:
        g_est = estimate_gFunction( args.p )

    t_s = time.time()
#    logging.getLogger().setLevel(logging.INFO) 
    V = torch.randn(1, args.n)
    V =  torch.abs(V)

    
    U_p = pNormProxOp(V, rho=args.rho, p=args.p, g_est = g_est)
    #U = EuclidianProxOp(V, args.rho)
#    print (U_p.size())
    t_e = time.time()
    print("Time taken is ", t_e - t_s)
