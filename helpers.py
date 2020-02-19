import torch
from sympy.solvers import solve
import math

def pNormProxOp(V, rho, p=2, eps=1.e-6):
    """Return the p-norm prox operator for the vector V
               argmin_X ||X||_p + rho/2 \|X - V\|_2^2
    """
    def g_func(a, p):
        "Return the solution of (x/a)^(p-1) + x = 1"
        if a>0:
            x = Symbol("x", positive=True)
            sols = solve((x/a)**(p-1)+x-1, x)
            if len(sols) == 0:
                sol = 0.
            else:
                sol = max(sols)
            return sol
        else:
            return  0.
    
    signs = torch.sign(V)
    V_normalized = rho * V._abs()
    vec_size = V_normalized.size()
    q =  p/(p-1.)
    if torch.norm(V_normalized, p=q) < 1:
        return torch.zeros( vec_size )
    upper_bound = torch.norm(V_normalized, p=p)
    lower_bound = 0.0
    for k in math.ceil( range( math.log2(1./eps) )):
        mid_bound = 0.5 * (upper_bound + lower_bound )
        U = torch.tensor( [V_normalized[j] * g_func(mid_bound * V_normalized[j] ** ((2.0-p) / (p-1.0)) for j in range(vec_size)] )
        if torch.norm(U, p=p) < mid_bound:
            upper_bound = mid_bound
        else:
            lower_bound = mid_bound
    
    return U * signs / rho
        
     
    
