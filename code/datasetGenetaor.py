import argparse
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from helpers import dumpFile
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.linalg import matrix_rank
import math
torch.manual_seed(1993)


class MySampler(torch.utils.data.Sampler):
    def __init__(self, dataset_len):
        super(MySampler).__init__()
        self.dataset_len = dataset_len
    def __len__(self):
        return self.dataset_len
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
        # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.dataset_len 

        else:  # in a worker process
            per_worker = int(math.ceil((self.dataset_len - 1) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.dataset_len) 
        return iter(range(iter_start, iter_end)) 
      
    


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
    parser.add_argument("--outfile_prefix", type=str, help="Outfile", default="data/synthetic/")
    parser.add_argument("--problem_type", choices=['labeled', 'unlabeled'], default='unlabeled')
    args = parser.parse_args()

    
    data_m_primeDim = torch.randn(args.n, args.m_prime)
    #Generate a weight matrix
    W = torch.randn(args.m_prime, args.m)
    #project data_m_primeDim to a higher dimensional space via W
    data = torch.matmul(data_m_primeDim, W)

    #add noise
    #noise_distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros( args.n * args.m), args.noiseLevel * torch.eye( args.n * args.m) )
    data = data + torch.randn(args.n, args.m) * args.noiseLevel

     

    #add outliers
  
    #distribution of outliers values
   # outliers_distr = torch.distributions.multivariate_normal.MultivariateNormal(10.0 * torch.ones( args.n * args.m), args.noiseLevel * torch.eye( args.n * args.m) )
    #distribution of outliers indices
    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones(args.n) * args.outliers)
    
    outliers =  outliers_indices_distr.sample()
    outliers_ind = torch.nonzero( outliers>0)
    outliers = outliers.unsqueeze(1)
    #add outliers 
    data = data + outliers * (torch.randn(args.n, args.m) * args.noiseLevel + 10.)

    print (torch.norm(data -  torch.matmul(data_m_primeDim, W), p=2) **2) 

    if args.problem_type == 'labeled':
        data = data_m_primeDim, data

    outfile = args.outfile_prefix + 'IN' + str(args.n) + 'by' + str(args.m) + '_' + 'lower' + str(args.m_prime) + 'noise' + str(args.noiseLevel) + 'outliers' + str(args.outliers)
    if args.problem_type == 'labeled':
        outfile += args.problem_type
    #create a Dataset object 
    if args.problem_type == 'unlabeled':
        dataset = unlabeledDataset( data )
    else:
        dataset = labeledDataset( data )
   # torch.save(data, outfile)

    dumpFile(outfile, dataset) 
    torch.save(outliers_ind, outfile + 'outliers')
  
    #Visualize
    #outliers_ind_list = []
    #for i in range(outliers_ind.size()[0]):
    #    outliers_ind_list.append( outliers_ind[i].item())
    #non_outliers_list = [i for i in range(args.n) if i not in outliers_ind_list]
    
    #plt.plot(data[non_outliers_list,0] ,data[non_outliers_list,1], 'o', label='points', linewidth=3, markersize=10)
    #plt.plot(data[outliers_ind_list,0] ,data[outliers_ind_list,1], 'or', label='outliers', linewidth=0.5, markersize=10)
    #plt.legend()
    #plt.savefig(outfile + '_fig.pdf', format='pdf')



    
