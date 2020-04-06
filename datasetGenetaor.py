import argparse
from helpers import dumpFile
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.linalg import matrix_rank
import math
torch.manual_seed(1993)


class unlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx, :]


class labeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        dataset_x, dataset_y = dataset
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
    def __len__(self):
        return len(self.dataset_x)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset_x[idx, :],  self.dataset_y[idx, :]
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help='Dataset size')
    parser.add_argument("--m", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')
    parser.add_argument("--noiseLevel", type=float, help='Covariance for the noise', default=1.e-3)
    parser.add_argument("--outliers", type=float, help='A real nuber between 0 and 1, the portion of data points that are outliers.', default=0.0)
    parser.add_argument("--outfile", type=str, help="Outfile")
    parser.add_argument("--problem_type", choices=['labeled', 'unlabeled'], default='unlabeled')
    args = parser.parse_args()

    
    data_m_primeDim = torch.randn(args.n, args.m_prime)
    #Generate a weight matrix
    W = torch.randn(args.m_prime, args.m)
    #project data_m_primeDim to a higher dimensional space via W
    data = torch.matmul(data_m_primeDim, W)
    #add noise
    noise_distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros( args.n * args.m), args.noiseLevel * torch.eye( args.n * args.m) )
    data = data + noise_distr.sample().view(args.n, args.m)

    

    #add outliers
  
    #distribution of outliers values
    outliers_distr = torch.distributions.multivariate_normal.MultivariateNormal(10.0 * torch.ones( args.n * args.m), args.noiseLevel * torch.eye( args.n * args.m) )
    #distribution of outliers indices
    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones(args.n) * args.outliers)
    
    outliers =  outliers_indices_distr.sample()
    outliers_ind = torch.nonzero( outliers>0)
    outliers = outliers.unsqueeze(1)
    #add outliers 
    data = data + outliers * outliers_distr.sample().view(args.n, args.m)
    
    print (torch.norm(data -  torch.matmul(data_m_primeDim, W), p=2) **2) 

    if args.problem_type == 'labeled':
        data = data_m_primeDim, data
    if args.outfile == None:
        outfile = 'IN' + str(args.n) + 'by' + str(args.m) + '_' + 'lower' + str(args.m_prime) + 'noise' + str(args.noiseLevel) + 'outliers' + str(args.outliers)
        if args.problem_type == 'labeled':
            outfile += args.problem_type
    else:
        outfile = args.outfile
    #create a Dataset object 
    if args.problem_type == 'unlabeled':
        dataset = unlabeledDataset( data )
    else:
        dataset = labeledDataset( data )
   # torch.save(data, outfile)
    dumpFile(outfile, dataset) 
    torch.save(outliers_ind, outfile + 'outliers')
    
