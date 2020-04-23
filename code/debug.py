import argparse
from Net import AEC
import torch
from torch.utils.data import Dataset, DataLoader
from ADMM import LocalSolver


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=2)
    args = parser.parse_args() 

    model = AEC(args.m, args.m_prime)

    theta_k = model.getParameters(True)

    z = theta_k ** 2
    
    l = []

    l.append( torch.sum(z[0] * 3) )
    l.append( torch.sum(z[1] * 2) )

   # print(l)
    model.zero_grad()
    print(theta_k[0].is_leaf)
    l[0].backward()
    print(theta_k[0].grad)
    #print(theta_k[0] )

    #l_tens = torch.tensor(l, requires_grad=True)
    #print(l_tens)

    
    


