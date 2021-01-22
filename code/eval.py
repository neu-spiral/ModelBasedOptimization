import argparse
import logging
from helpers import dumpFile, loadFile
from torch.utils.data import Dataset, DataLoader
from Net import AEC, DAEC, Linear, ConvAEC, ConvAEC2
import torch
from plotter import whichKey, barPlotter
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

        if outliers_ind[ind] == 1.:
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
    parser.add_argument('--net_model', choices=['Linear', 'AEC', 'DAEC', 'ConvAEC2', 'ConvAEC'])
    parser.add_argument("--m_dim", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')
    parser.add_argument("--traindatafile", help="Train Dataset file.")
    parser.add_argument("--testdatafile", type=str, help="Test Dataset File.")
    parser.add_argument("--out_statsfile",  type=str, help="File to store stats.")

    parser.add_argument("--plt_file", help="File to store plots")
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

    alg_keywords = {'_MBO':'MBO', '_SGD_': 'SGD', '_MBOSGD': 'MBOSGD'}
    alg_keywords_ordered = ['_MBOSGD', '_MBO', '_SGD_']

    outl_keywords = {'outliers00_': 0.0, 'outliers005_': 0.05 , 'outliers01_': 0.1}
    
    #NOTE hard coding here!!
    train_filenames_suffix = {0.0:  'outliers00', 0.05: 'outliers005', 0.1: 'outliers01'}
    
    DICS_p = {}

    DICS_outl = {}

    p_kyes_ord = [(outl, alg) for outl in [0.0, 0.05, 0.1] for alg in ['MBO', 'MBOSGD', 'SGD']]
 
    outl_keys_ord = [(p, alg) for p in [1, 1.5, 2, -2] for alg in ['MBO', 'SGD']]

    for filename in args.filenames:
        #find out file corresponds to which p, alg, and outliers count.
        p  = whichKey(filename, p_keywords, p_keys_ordered) 

        alg = whichKey(filename, alg_keywords, alg_keywords_ordered)

        outl = whichKey(filename, outl_keywords)

        #load model parameters
        model.loadStateDict(filename) 

        #evaluate on test data 
        obj = evalTest(model, data_loader, p = float(p)) 

        print("test obj is ", obj)

        #get corresponding train data
        traindata = loadFile( args.traindatafile + train_filenames_suffix[ outl ] )
        train_data_loader = DataLoader(traindata, batch_size=1)

        #evaluate on train data 
        stats = evalObjective(model, train_data_loader, traindata.outliers_idx)


        print("Norm {}, the total loss for original input data is {}, total loss for non-outliers is {}".format(p, stats['totalLoss'], stats['non_outliersLoss']))
        

        #set values in dictionaries
        if p not in DICS_p:
            DICS_p[p] = {(outl, alg):  stats['non_outliersLoss']}
        else:
            DICS_p[p][(outl, alg)] = stats['non_outliersLoss']

        if outl  not in DICS_outl:
            DICS_outl[outl] = {(p, alg): obj}

        else:
            DICS_outl[outl][(p, alg)] =  obj

    #set missing values to zero
    for p in DICS_p:
        for key_p in p_kyes_ord:
            if key_p not in DICS_p[p]:
                DICS_p[p][key_p] = 0.0

    print(DICS_p)

    #plot bar
    barPlotter(DICS_p, args.plt_file + "_p", x_axis_label  ="p", y_axis_label = 'non_outliersLoss', normalize=False, lgd=True, log_bar=False, DICS_alg_keys_ordred = p_kyes_ord)

        
    #set missing values to zero
    for outl in DICS_outl:
        for outl_key in outl_keys_ord:
            if outl_key not in DICS_outl[outl]:
                DICS_outl[outl][outl_key] = 0.0

        

    #plot bar
    barPlotter(DICS_outl, args.plt_file + "_outl", x_axis_label  ="outliers", y_axis_label = 'Test Loss', normalize=False, lgd=True, log_bar=False, DICS_alg_keys_ordred = outl_keys_ord)
