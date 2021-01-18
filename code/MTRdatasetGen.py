import argparse
import matplotlib
matplotlib.use('Agg')

import torchvision 
import torchvision.transforms as transforms
from torchvision import datasets

import matplotlib.pyplot as plt
from helpers import dumpFile, loadFile
import numpy as np
import sklearn.preprocessing
import torch
from torch.utils.data import Dataset
from numpy.linalg import matrix_rank
import math
torch.manual_seed(1993)



class AddNoiseDataset(Dataset):
    def __init__(self, data, targets, outliers_idx = None, threshold = 0.5, high = 1.0, low = 0.0):
        """
            Args:
                dataset: A loaded dataset
                outliers_idx: A binary array or Tensor showing the indicies corresponding to outliers
        """

        self.data = data
        self.targets = targets

        self.outliers_idx = outliers_idx


        #add noise
        for ind in range(len(self.data)):
            if outliers_idx[ind] == 1:
                self.data[ind] = self.noise_fn( self.data[ind] )
                self.targets[ind] = self.noise_fn( self.targets[ind] )

                  

    def noise_fn(self, data_i, outlier_level = 1e1):

        return data_i + randn(data_i.shape) + outlier_level

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.data[idx], self.targets[idx])

def binaryNoise(img, threshold = 0.5, high = 1.0, low = 0.0):
    """
        Change all pixel values larger than threshold to high and all others to low.
    """
    
    large_entrees = img >= 0.5

    img[ large_entrees ] = 1.0
    
    img[ torch.logical_not( large_entrees ) ] = 0.0

    return img


def readFile(file_name, target_size):
    """
        Reaf a file and return a grid-like objects containing data samples (rows), features (columns), and targets.
        The last "target_size" values in each line of the file are treated as targets.
    """

    data = []

    targets = []

    with open(file_name, 'r') as data_f:
        #read line
        for l in data_f:
            #skip over line with not containing data
            try:
                feats_targets = [eval( val ) for val in l.split( ',' )]

            except SyntaxError:
                continue


            data.append( feats_targets[:-1 * target_size] )     
    
            targets.append( feats_targets[-1 * target_size : ] )

    return torch.tensor( data, dtype=torch.float32 ), torch.tensor( targets, dtype=torch.float32 )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str, help="The name of the dataset to download")
    parser.add_argument("target_size", type = int, help = "The size of target")
    parser.add_argument("--outliers", type=float, help='A real nuber between 0 and 1, the portion of data points that are outliers.', default=0.0)
    
    parser.add_argument("--outfile", type=str, help="Outfile")
    parser.add_argument("--problem_type", choices=['labeled', 'unlabeled'], default='unlabeled')
    args = parser.parse_args()


    Data, Targets = readFile( args.datafile, args.target_size )
    

    #standardize data
    std_Data = torch.tensor( sklearn.preprocessing.scale(Data, axis = 0), dtype=torch.float32 )
    std_Targets =  torch.tensor( sklearn.preprocessing.scale(Targets, axis = 0), dtype=torch.float32 )
   


    #set outliers distribution
    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones( len( Data ) ) * args.outliers )

    #sample outlier indices
    outliers_idx = outliers_indices_distr.sample()

    #form dataset 
    datasetWithOutliers = AddNoiseDataset( std_Data, std_Targets, outliers_idx = outliers_idx )

    print(datasetWithOutliers[0])
   
    dumpFile(args.outfile, datasetWithOutliers)

    #Sampler
    #data_sampler  = torch.utils.data.distributed.DistributedSampler(my_dataset, rank=args.local_rank)
  #  my_dataset = loadFile(args.outfile)
  #  data_loader = torch.utils.data.DataLoader(my_dataset,\
  #                                        batch_size=1)
                                         # shuffle=True
    
    

    

    

    #add outliers
  
