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

        #dataset with outliers
        self.datasetWithOutliers = []

        #add noise
        for ind in range(len(self.dataset)):
            if outliers_idx[ind] == 1:
                self.datasetWithOutliers.append( self.noise_fn( self.dataset[ind][0] ) )

            else:
                self.datasetWithOutliers.append(  self.dataset[ind][0] )

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
       

        return self.datasetWithOutliers[idx]


class addOutliers(dropLabelAddNoiseDataset):
    def __init__(self, dataset, outliers_idx = None, outliers_dataset = None):
        self.outliers_dataset = outliers_dataset
         
        super().__init__(dataset, outliers_idx)

    
    def noise_fn(self, img):

         outlier_data_ind = np.random.randint( low = 0, high= len( self.outliers_dataset ) - 1  )


         outlier_img = torch.mean(self.outliers_dataset[outlier_data_ind][0], dim = 0, keepdim = True  )
      
         return img + outlier_img

class addWeightedOutliers(dropLabelAddNoiseDataset):
    def __init__(self, dataset, outliers_idx = None, outliers_dataset = None, signal_ratio = 0.0):
        self.outliers_dataset = outliers_dataset

        self.signal_ratio = signal_ratio

        super().__init__(dataset, outliers_idx)


    def noise_fn(self, img):

         outlier_data_ind = np.random.randint( low = 0, high= len( self.outliers_dataset ) - 1  )


         outlier_img = torch.mean(self.outliers_dataset[outlier_data_ind][0], dim = 0, keepdim = True  )

         return img * self.signal_ratio + outlier_img * (1.0 - self.signal_ratio)
                
class contrastOutliers(dropLabelAddNoiseDataset):
    def __init__(self, dataset, outliers_idx = None):
        super().__init__(dataset, outliers_idx)

    def noise_fn(self, img):

        return torch.ones(img.shape) - img


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
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to download")
    parser.add_argument("outlier_type", choices= ['OD', 'CONT'], help="Type of the outliers to be addded to the dataset. OD adds random resized samples from outlier_dataset, while CONT replaces pixel values p with 1 - p.")

    parser.add_argument("--outlier_dataset", default='CIFAR10', help="The name of the outlier dataset to download")
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--outliers", type=float, help='A real nuber between 0 and 1, the portion of data points that are outliers.', default=0.0)
    parser.add_argument("--data_dir", type=str, default='data/', help="Directory to download data")
    parser.add_argument("--outfile", type=str, help="Outfile")
    parser.add_argument("--data_type", choices=['train', 'test'], default='train')
    args = parser.parse_args()


    #transforms for dataset
    my_transform = transforms.Compose([transforms.ToTensor()] )   ##, transforms.Lambda(lambda img: binaryNoise(img, threshold = 0.4)) ] )#, transforms.Normalize((0.1307,), (0.3081,) )] )

    #get dataset class
    my_dataset_class = eval('datasets.' + args.dataset_name)
  
    #create dataset
    dataset = my_dataset_class(args.data_dir, train = args.data_type == 'train', download=True, transform=my_transform)

    #outliers distribution
    outliers_indices_distr = torch.distributions.bernoulli.Bernoulli( torch.ones( len(dataset ) ) * args.outliers )

    #sample outliers indices
    outliers_idx = outliers_indices_distr.sample()

    if args.outlier_type  == 'OD':
        #outlier dataset class
        outlier_dataset_class = eval('datasets.' + args.outlier_dataset)

        #create outliers dataset
        outlier_dataset = outlier_dataset_class(args.data_dir,  train=True, download=True, transform = transforms.Compose([transforms.Resize(dataset[0][0].shape[-2:]), transforms.ToTensor()] ) )


        #create dataset, where data in outlier indices is corrupted by random samples from outlier dataset
        datasetWithOutliers = addOutliers( dataset, outliers_idx = outliers_idx, outliers_dataset = outlier_dataset )

    elif args.outlier_type == 'CONT':

        #create dataset, where data in outlier indices is corrupted by contrsatsing
        datasetWithOutliers = contrastOutliers( dataset, outliers_idx)
        

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
