import argparse
from Net import AEC, ConvLinSoft, ConvAEC2Soft
import torch
from torch.utils.data import Dataset, DataLoader
from ADMM import LocalSolver
from Real_datasetGenetaor import addOutliers, addWeightedOutliers, addWeightedOutliersWithLabels
from helpers import loadFile


if __name__== "__main__":
    
    ds = loadFile('data/real/MNIST_labeled_dc1.5__Fakeoutliers005') 


    net = ConvLinSoft(1, 8)

    dl_iter = iter( DataLoader(ds, batch_size  = 1) )

    data = next( dl_iter )

    out =  net(data) 

    jac, sqJac = net.getJacobian( out, quadratic = True )


    print(sqJac.shape)
