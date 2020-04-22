import argparse
import matplotlib
matplotlib.use('Agg')

import torchvision 
import torchvision.transforms as transforms
from torchvision import datasets

import matplotlib.pyplot as plt
from helpers import dumpFile, loadFile
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
    parser.add_argument("--dataset_name", type=str, default='MNIST', help="The name of the dataset to download")
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--outliers", type=float, help='A real nuber between 0 and 1, the portion of data points that are outliers.', default=0.0)
    parser.add_argument("--data_dir", type=str, default='data/', help="Directory to download data")
    parser.add_argument("--outfile", type=str, help="Outfile")
    parser.add_argument("--problem_type", choices=['labeled', 'unlabeled'], default='unlabeled')
    args = parser.parse_args()


    my_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,) )] )

    #Download data
    my_dataset_class = eval('datasets.' + args.dataset_name)
    my_dataset = my_dataset_class(args.data_dir, train=False, download=True, transform=my_transform)

    if args.local_rank != None:
        torch.distributed.init_process_group(backend='gloo',
                                         init_method='env://')
   
    dumpFile(args.outfile, my_dataset)

    #Sampler
    #data_sampler  = torch.utils.data.distributed.DistributedSampler(my_dataset, rank=args.local_rank)
  #  my_dataset = loadFile(args.outfile)
  #  data_loader = torch.utils.data.DataLoader(my_dataset,\
  #                                        batch_size=1)
                                         # shuffle=True
    
    

    

    

    #add outliers
  
