import torch
import time
import argparse 
import numpy as np
import sympy as sym
import math

torch.manual_seed(1993)
def pNormProxOp(V, rho, p=2, eps=1.e-9):
    """
        Return the p-norm prox operator for the vector V
               argmin_X ||X||_p + rho/2 \|X - V\|_2^2
    """
    def g_func(a, p):
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


    V = V.squeeze(0)

    signs = torch.sign(V)
    V_normalized = rho * torch.abs(V)
    vec_size = V_normalized.size()[0]
    q =  p/(p-1.)
    if torch.norm(V_normalized, p=q) < 1:
        return torch.zeros( vec_size )
    upper_bound = torch.norm(V_normalized, p=p)
    lower_bound = 0.0
    U =  torch.zeros( vec_size )
    for k in  range( math.ceil(math.log2(1./eps)) ):
        mid_bound = 0.5 * (upper_bound + lower_bound )
        U =  [V_normalized[j] * g_func(mid_bound * V_normalized[j] ** ((2.0-p) / (p-1.0)), p ) for j in range(vec_size)]  
        U = torch.tensor(U, dtype=torch.float)
        if torch.norm(U, p=p) < mid_bound:
            upper_bound = mid_bound
        else:
            lower_bound = mid_bound
    U = U.unsqueeze(0)
    return U * signs / rho


def EuclidianProxOp(V, rho):
     """
        Return the 2-norm prox operator for the vector V
             argmin_X ||X||_2 + rho/2 \|X - V\|_2^2
     """
     vec_size = V.size()[0]
     V_norm = torch.norm(V, 2)
     if V_norm < rho:
         return torch.zeros( vec_size )
     return (1 - rho / V_norm ) * V
     
      
def _testOpt(U, V, rho, p):
    vec_size = V.size()[0]
    norm_U =  torch.norm(U, p=p)
    print( [U[i] for i in range(vec_size)])
    return [(U[i] / norm_U) ** (p-1.0) + rho * (U[i] - V[i]) for i in range(vec_size)]
        
     
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="vector size")
    parser.add_argument("--p", type=float, help="p norm", default=2.)
    parser.add_argument("--rho", type=float, help="rho", default=1.0)
    args = parser.parse_args()

    t_s = time.time()
    V = torch.randn(1,args.n)
    V =  torch.abs(V)
    U_p = pNormProxOp(V, rho=args.rho, p=args.p)
    U = EuclidianProxOp(V, args.rho)
    print (U.size(), U_p.size())
    t_e = time.time()
    print (   _testOpt(U_p, V, rho=args.rho, p=args.p) )
    print ("Time {} seconds".format(t_e - t_s) )
