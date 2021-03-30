import argparse
import pickle
import logging
from helpers import dumpFile, loadFile
from torch.utils.data import Dataset, DataLoader
from Net import AEC, DAEC, Linear, ConvAEC, ConvAEC2, ConvAEC2Soft, ConvLinSoft, LinearSoft
import torch
from plotter import whichKey, barPlotter
from datasetGenetaor import labeledDataset, unlabeledDataset
from Real_datasetGenetaor import dropLabelAddNoiseDataset, addOutliers, contrastOutliers, addWeightedOutliersWithLabels
from MTRdatasetGen import AddNoiseDataset

@torch.no_grad()
def evalObjective(model, dataloader, outliers_ind,  p, reg_coeff, eval_acc = False):
    """
     Compute the objective:  
                  \sum_i || F_i(Theta) ||_2 + g(theta), 
     where Theta is the model parameters.
    """
    
    stats = {}
    OBJ_tot = 0.5 * reg_coeff *  model.getParameters().frobeniusNormSq() 
    OBJ_nonOutlierstot = 0.0
   
    OBJ_nonOutlierstot_unnorm = 0.0

    outliers_size = torch.sum( outliers_ind )

    print("Train data has {} samples.".format( len(dataloader) ) )


    if eval_acc:
        correct_pred = 0.

    for ind, data in enumerate( dataloader ):

        output = model(data)


        if eval_acc:
            correct_pred += model.eval_acc( data ) * data[0].shape[0]

        if p== -2:
            OBJ_ind = torch.norm(output, p=2).item() ** 2 
        else:
            OBJ_ind = torch.norm(output, p=p).item() 

        

        OBJ_tot += OBJ_ind / len( dataloader ) 

        if outliers_ind[ind] == 1.:
            continue

        OBJ_nonOutlierstot += OBJ_ind  / (len( dataloader ) - outliers_size )


    if eval_acc:
        stats['train_acc'] = correct_pred / len(dataloader)
  
    stats['non_outliersLoss'] = OBJ_nonOutlierstot
    stats['fullObj'] = OBJ_tot


    return stats

@torch.no_grad()
def evalTest(model, data_loader, p, eval_acc = False):
    """
        Evaluate model on testset.
    """

    obj = 0.0

    if eval_acc:
        correct_pred = 0.
    print("Test data has {} samples.".format( len(data_loader) ) )

    for data in data_loader:
  
        if eval_acc:
            correct_pred +=  model.eval_acc( data ) * data[0].shape[0]
       
        if  p == -2:

            obj += torch.squeeze( torch.norm( model(data), p = 2) ).item() ** 2 / len(data_loader)
        else:
       
            obj += torch.squeeze( torch.norm( model(data), p = p) ).item() / len(data_loader)


    if eval_acc:
        return obj, correct_pred / len(data_loader)
    return obj 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str,
                   help='file where model parameters are stored')
    parser.add_argument('--net_model', choices=['Linear', 'AEC', 'DAEC', 'ConvAEC2', 'ConvAEC', 'ConvAEC2Soft', 'ConvLinSoft', 'LinearSoft'])
    parser.add_argument("--m_dim", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')
    parser.add_argument("--reg_coeff", type=float, default = 0.001, help = "Regularization coefficient")
    parser.add_argument("--traindatafile", help="Train Dataset file.")
    parser.add_argument("--testdatafile", type=str, help="Test Dataset File.")
    parser.add_argument("--out_statsfile",  type=str, help="File to store stats.")

    parser.add_argument("--plt_file", help="File to store plots")

    parser.add_argument("--eval_acc", dest='eval_acc', action='store_true', help="Pass in order to evaluate accuracies, only applicable for labeled datasest.")
    args = parser.parse_args()

    #setup logging
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    model_Net = eval(args.net_model)
    model = model_Net(args.m_dim , args.m_prime)

    #load test and train datasets
    testdata = loadFile( args.testdatafile )

    data_loader = DataLoader(testdata, batch_size=1)
    
    #setting proper keys based on patterns in filenames
    p_keywords =  {'_p1.5':1.5, '_p1':1., '_p2':2.,  '_p-2': -2.,'_p3':3.}
    p_keys_ordered = ['_p1.5', '_p1', '_p2', '_p-2', '_p3']

    alg_keywords = {'_MBO':'MBOOADM', '_SGD_': 'SGD', '_MBOSGD': 'MBOSGD'}
    alg_keywords_ordered = ['_MBOSGD', '_MBO', '_SGD_']

    alg_ord = ['MBOOADM', 'MBOSGD', 'SGD']

    outl_keywords = {'outliers00_': 0.0, 'outliers005_': 0.05 , 'outliers01_': 0.1, 'outliers02_': 0.2, 'outliers03_': 0.3}
    
    outl_ord = [0.0, 0.05, 0.1, 0.2, 0.3]

    #NOTE hard coding here!!
    train_filenames_suffix = {0.0:  '00', 0.05: '005', 0.1: '01', 0.2: '02', 0.3: '03'}
    
    
 
    #outl_keys_ord = [(p, alg) for p in [1, 1.5, 2, 'ell2Sq'] for alg in ['MBO', 'SGD', 'MBOSGD']]


    #find out file corresponds to which p, alg, and outliers count.
    p  = whichKey(filename, p_keywords, p_keys_ordered) 

    alg = whichKey(filename, alg_keywords, alg_keywords_ordered)

    outl = whichKey(filename, outl_keywords)


    #load model parameters
    model.loadStateDict(filename, device = torch.device('cpu') ) 

    #evaluate on test data 
    if args.eval_acc:
        obj_test, test_acc = evalTest(model, data_loader, p = float(p), eval_acc = args.eval_acc) 

    else:
        obj_test =  evalTest(model, data_loader, p = float(p), eval_acc = args.eval_acc)


    #get corresponding train data
    traindata = loadFile( args.traindatafile.format( train_filenames_suffix[ outl ] ) )
    train_data_loader = DataLoader(traindata, batch_size=1)

    #evaluate on train data 
    stats = evalObjective(model, train_data_loader, traindata.outliers_idx, p, reg_coeff = args.reg_coeff, eval_acc = args.eval_acc)


        
    print("Alg, outliers and p are {}, {}, and {}, respectively.".format(alg, outl, p))
    print("Non-outliers sum is {:.2f}, total loss {:.2f} and test loss is {:.2f}".format(stats['non_outliersLoss'], stats['fullObj'], obj_test) )

    stats['obj_test'] = obj_test

   if args.eval_acc:

        print("Acuuracy on train and test sets are {:.2f} and {:.2f}, respectively.".format( stats['train_acc'], test_acc ) )

        stats['test_acc'] = test_acc


   
        
    
    with open(args.out_statsfile, 'wb') as f_stat:
        pickle.dump( stats, f_stat)
