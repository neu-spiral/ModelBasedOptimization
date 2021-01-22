import argparse
import logging
from helpers import dumpFile, loadFile
from torch.utils.data import Dataset, DataLoader
from Net import AEC, DAEC, Linear, ConvAEC, ConvAEC2
import torch
from plotter import whichKey
from datasetGenetaor import labeledDataset, unlabeledDataset
from Real_datasetGenetaor import dropLabelAddNoiseDataset, addOutliers
from MTRdatasetGen import AddNoiseDataset

@torch.no_grad()
def evalObjective(model, dataloader, outliers_ind):
    """
     Compute the objective:  
                  \sum_i || F_i(Theta) ||_2 + g(theta), 
     where Theta is the model parameters.
    """
    
    stats = {}
    OBJ_tot = 0.0
    OBJ_nonOutlierstot = 0.0
   
    OBJ_originalData = 0.0
    for ind, data in enumerate( dataloader ):

        output = model(data)

        OBJ_i = torch.norm(output, p=1).item() / len( dataloader )


        OBJ_tot += OBJ_i

        if ind in outliers_ind:
            continue

        OBJ_nonOutlierstot += OBJ_i
  
    stats['totalLoss'] = OBJ_tot
    stats['non_outliersLoss'] = OBJ_nonOutlierstot
    
    return stats

@torch.no_grad()
def evalTest(model, data_loader, p):
    """
        Evaluate model on testset.
    """

    obj = 0.0

    for data in data_loader:
  
        obj += torch.squeeze( torch.norm( model(data), p = p) ) / len(data_loader)

    return obj 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='file where model parameters are stored')
    parser.add_argument('--net_model', choices=['Linear', 'AEC', 'DAEC'])
    parser.add_argument("--m_dim", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')
    parser.add_argument("--traindatafile", help="Train Dataset file.")
    parser.add_argument("--testdatafile", type=str, help="Test Dataset File.")
    parser.add_argument("--out_statsfile",  type=str, help="File to store stats.")
    args = parser.parse_args()

    #setup logging
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    model_Net = eval(args.net_model)
    model = model_Net(args.m_dim , args.m_prime)

    #load test and train datasets
    testdata = loadFile( args.testdatafile )
    traindata = loadFile( args.traindatafile )

    data_loader = DataLoader(testdata, batch_size=1)
    train_data_loader = DataLoader(traindata, batch_size=1)
    
    keywords =  {'_p1.5':1.5, '_p1':1., '_p2':2.,  '_p-2': -2.,'_p3':3.}
    keys_ordered = ['_p1.5', '_p1', '_p2', '_p-2', '_p3']

    DICS = {}
    for filename in args.filenames:
        #find out file corresponds to which alg.
        p  = whichKey(filename, keywords, keys_ordered) 

        #load model parameters
        model.loadStateDict(filename) 

        
        obj = evalTest(model, data_loader, p = float(p)) 
        print("test obj is ", obj)

      
        stats = evalObjective(model, train_data_loader, traindata.outliers_idx)

        print("Norm {}, the total loss for original input data is {}, total loss for non-outliers is {}".format(p, stats['totalLoss'], stats['non_outliersLoss']))
#        logging.info("Norm {}, the total loss for original input data is {}, total loss for non-outliers is {}".format(p, stats['totalLoss'], stats['non_outliersLoss']))

#    dumpFile(args.out_statsfile, DICS)
        



