import argparse
from Net import AEC
import torch
from torch.utils.data import Dataset, DataLoader
from ADMM import ADMM


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=2)
    args = parser.parse_args() 



    model = AEC(args.m, args.m_prime)

    theta_k = model.getParameters()
    
    theta =  theta_k 
    for para in theta:
        para.requires_grad = True
    y = theta[0] * 2

    z = torch.mean(y.view(-1))
    z.backward()
    print(theta[0].grad, theta_k[0].grad.requires_grad)


