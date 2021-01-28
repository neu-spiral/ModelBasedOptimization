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
    def __init__(self, data, targets, outliers_idx = None, outlier_bias = 0.0, outlier_var = 1.0):
        """
            Args:
                dataset: A loaded dataset
                outliers_idx: A binary array or Tensor showing the indicies corresponding to outliers
        """

        self.data = data
        self.targets = targets

        self.outlier_bias = outlier_bias
        self.outlier_var = outlier_var

        self.outliers_idx = outliers_idx


        #add noise
        if outliers_idx is not None:
            for ind in range(len(self.data)):
                if outliers_idx[ind] == 1:
                    self.data[ind] = self.noise_fn( self.data[ind] )
                    self.targets[ind] = self.noise_fn( self.targets[ind] )

                  

    def noise_fn(self, data_i):

        return data_i + torch.randn(data_i.shape) * self.outlier_var + self.outlier_bias

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
    
    parser.add_argument("--outlier_var", type=float, default=1.0, help="Variance of outliers" )
    parser.add_argument("--outlier_bias", type=float, default=0.0, help="Bias of outliers." )

    parser.add_argument("--outfile", type=str, help="Outfile")

    parser.add_argument("--outfile_test",  type=str, help="Outfile for test dataset.")
    args = parser.parse_args()


    Data, Targets = readFile( args.datafile, args.target_size )
    

    #standardize data
    std_Data = torch.tensor( sklearn.preprocessing.scale(Data, axis = 0), dtype=torch.float32 )
    std_Targets =  torch.tensor( sklearn.preprocessing.scale(Targets, axis = 0), dtype=torch.float32 )
   

    #set train/test sizes
    train_size = int(0.8 * len(std_Data) )
    test_size =  len(std_Data) - train_size

    #split data
    std_Data_train, std_Data_test = torch.split(std_Data, [train_size, test_size] )
    std_Targets_train, std_Targets_test = torch.split(std_Targets,  [train_size, test_size] )


    #set outliers distribution
    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones(train_size ) * args.outliers )

    #sample outlier indices
    outliers_idx = outliers_indices_distr.sample()


    

    #form train dataset 
    train_datasetWithOutliers = AddNoiseDataset( std_Data_train, std_Targets_train, outliers_idx = outliers_idx, outlier_bias = args.outlier_bias, outlier_var = args.outlier_var )
  
    #save train dataset
    dumpFile(args.outfile, train_datasetWithOutliers)

    if args.outliers == 0.0:
        #form test dataset
        test_datasetWithOutliers = AddNoiseDataset( std_Data_test, std_Targets_test )

        #save test dataset
        dumpFile(args.outfile_test, test_datasetWithOutliers)

    #Sampler
    #data_sampler  = torch.utils.data.distributed.DistributedSampler(my_dataset, rank=args.local_rank)
  #  my_dataset = loadFile(args.outfile)
  #  data_loader = torch.utils.data.DataLoader(my_dataset,\
  #                                        batch_size=1)
                                         # shuffle=True
    
    

    

    

    #add outliers
  
