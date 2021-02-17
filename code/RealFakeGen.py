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
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import matrix_rank
import math
from Real_datasetGenetaor import addOutliers, addWeightedOutliers
torch.manual_seed(1993)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to download")
    parser.add_argument("--outliers", type=float, help='A real nuber between 0 and 1, the portion of data points that are outliers.', default=0.0)
    parser.add_argument("--data_dir", type=str, default='data/', help="Directory to download data")
    parser.add_argument("--outfile", type=str, help="Outfile")
    parser.add_argument("--data_type", choices=['train', 'test'], default='train')

    parser.add_argument("--dev_coeff", type=float, default = 1.5, help = "Deviation coefficient, i.e., the devaition of outlires mean from original data.")

    parser.add_argument("--outlier_var", type=float, default=None, help="Variance of outliers" )
    parser.add_argument("--outlier_bias", type=float, default=None, help="Bias of outliers." )
    args = parser.parse_args()


    #transforms for dataset
    my_transform = transforms.Compose([transforms.ToTensor()] )   ##, transforms.Lambda(lambda img: binaryNoise(img, threshold = 0.4)) ] )#, transforms.Normalize((0.1307,), (0.3081,) )] )

    #get dataset class
    my_dataset_class = eval('datasets.' + args.dataset_name)
  
    #create dataset
    dataset = my_dataset_class(args.data_dir, train = args.data_type == 'train', download=True, transform=my_transform)


    #outlier dataset class
    fake_dataset_class = datasets.FakeData

    #if outliers bias and variance not given, set them in accordnace with the standard deviation of the orginial dataset
    if args.outlier_bias is None:

       #compute mean and standard deviation 
       whole_data_batch = next( iter( DataLoader(dataset, batch_size = len(dataset) ) ) ) 

       #drop labels from batch
       whole_data_batch = whole_data_batch[0]

       #mean
       dataset_mean = torch.mean(whole_data_batch, (0, 2, 3))

       #standard deviation
       dataset_stddev = torch.std(whole_data_batch, (0, 2, 3)) 

       #outliers mean
       outlier_bias = -1. * dataset_stddev * args.dev_coeff - dataset_mean

       #outliers  stddev normalization 
       outlier_stddev_norm = dataset_stddev

    else:
        outlier_bias = [args.outlier_bias]
        outlier_stddev_norm = [args.outlier_var]

        


    #create outliers dataset
    outlier_dataset = fake_dataset_class(size = len(dataset), 
                                         image_size = dataset[0][0].shape, 
                                         num_classes = 10,
                                         transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize(outlier_bias, outlier_stddev_norm)] ) 
                                         )

    #outliers distribution
    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones( len(dataset ) ) * args.outliers )

    #sample outliers indices
    outliers_idx = outliers_indices_distr.sample()

    #create dataset where data in outlier indices are corrupted by random samples from outlier dataset
    datasetWithOutliers = addWeightedOutliers( dataset, outliers_idx = outliers_idx, outliers_dataset = outlier_dataset )

   ##NOTE: DEBUGGING###############
    outlier_ind_samp  = 0
    for i in range( len(outliers_idx) ):
        if outliers_idx[i] == 1:
            outlier_ind_samp = i
    print(dataset[outlier_ind_samp][0] )
    print(datasetWithOutliers[outlier_ind_samp] - dataset[outlier_ind_samp][0] )
   #########################################

    #save dataset
    dumpFile(args.outfile, datasetWithOutliers)
