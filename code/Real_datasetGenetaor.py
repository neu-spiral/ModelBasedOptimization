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



class dropLabelAddNoiseDataset(Dataset):
    def __init__(self, dataset, outliers_idx = None, threshold = 0.5, high = 1.0, low = 0.0):
        """
            Args:
                dataset: A loaded dataset
                outliers_idx: A binary array or Tensor showing the indicies corresponding to outliers
        """

        self.dataset = dataset

        self.outliers_idx = outliers_idx

        self.threshold = threshold
        self.low = low
        self.high = high

        #add noise
        for ind in range(len(self.dataset)):
            if outliers_idx[ind] == 1:
                self.noise_fn( self.dataset[ind][0] )


    def noise_fn(self, img):

        large_entrees = img >= self.threshold

        img[ large_entrees ] = self.high

        img[ torch.logical_not( large_entrees ) ] = self.low

        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
       
        if torch.is_tensor(idx):
            idx = idx.tolist()
       

        return self.dataset[idx][0]

def binaryNoise(img, threshold = 0.5, high = 1.0, low = 0.0):
    """
        Change all pixel values larger than threshold to high and all others to low.
    """
    
    large_entrees = img >= 0.5

    img[ large_entrees ] = 1.0
    
    img[ torch.logical_not( large_entrees ) ] = 0.0

    return img

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='MNIST', help="The name of the dataset to download")
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--outliers", type=float, help='A real nuber between 0 and 1, the portion of data points that are outliers.', default=0.0)
    parser.add_argument("--data_dir", type=str, default='data/', help="Directory to download data")
    parser.add_argument("--outfile", type=str, help="Outfile")
    parser.add_argument("--problem_type", choices=['labeled', 'unlabeled'], default='unlabeled')
    args = parser.parse_args()


    my_transform = transforms.Compose([transforms.ToTensor()] )   ##, transforms.Lambda(lambda img: binaryNoise(img, threshold = 0.4)) ] )#, transforms.Normalize((0.1307,), (0.3081,) )] )

    #Download data
    my_dataset_class = eval('datasets.' + args.dataset_name)
  
    dataset = my_dataset_class(args.data_dir, train=True, download=True, transform=my_transform)



    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones( len(dataset ) ) * args.outliers )

    outliers_idx = outliers_indices_distr.sample()

    
    datasetWithOutliers = dropLabelAddNoiseDataset( dataset, outliers_idx = outliers_idx )

    if args.local_rank != None:
        torch.distributed.init_process_group(backend='gloo',
                                         init_method='env://')
   
    dumpFile(args.outfile, datasetWithOutliers)

    #Sampler
    #data_sampler  = torch.utils.data.distributed.DistributedSampler(my_dataset, rank=args.local_rank)
  #  my_dataset = loadFile(args.outfile)
  #  data_loader = torch.utils.data.DataLoader(my_dataset,\
  #                                        batch_size=1)
                                         # shuffle=True
    
    

    

    

    #add outliers
  
