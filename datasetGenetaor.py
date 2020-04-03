import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.linalg import matrix_rank
import math
torch.manual_seed(1993)




class RandomDataset_1D(Dataset):
    def __init__(self, dataset_size, dimension, sigma=1.e-2):
        self.dataset_size = dataset_size
        self.input_tensor = torch.randn(dataset_size, dimension)
        self.dim = dimension
    def __len__(self):
        return self.dataset_size
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.input_tensor[idx, :]
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help='Dataset size')
    parser.add_argument("--m", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')
    parser.add_argument("--noiseLevel", type=float, help='Covariance for the noise', default=1.e-2)
    parser.add_argument("--outliers", type=float, help='A real nuber between 0 and 1, the portion of data points that are outliers.', default=0.0)
    parser.add_argument("--outfile", type=str, help="Outfile")
    args = parser.parse_args()

    
    data_lowDim = torch.randn(args.n, args.m_prime)
    #Generate a weight matrix
    W = torch.randn(args.m_prime, args.m)
    print(matrix_rank(  np.matrix(W) ))
    #project data_lowDim to a higher dimensional space via W
    data = torch.matmul(data_lowDim, W)
    #add noise
    noise_distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros( args.n * args.m), args.noiseLevel * torch.eye( args.n * args.m) )
    data = data + noise_distr.sample().view(args.n, args.m)

    #add outliers
    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones(args.n) * args.outliers)
    outliers =  outliers_indices_distr.sample()
    outliers_ind = torch.nonzero( outliers>0)
    outliers = outliers.unsqueeze(1)
    print (outliers_ind)
    data = data + outliers * (torch.randn(args.n, args.m)  + 1.0)
    

    if args.outfile == None:
        outfile = 'IN' + str(args.n) + 'by' + str(args.m) + '_' + 'lower' + str(args.m_prime) + 'noise' + str(args.noiseLevel) + 'outliers' + str(args.outliers)
    else:
        outfile = args.outfile
    torch.save(data, outfile)
    torch.save(outliers_ind, outfile + 'outliers')
    
