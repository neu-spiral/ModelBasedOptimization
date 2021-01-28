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
def evalObjective(model, dataloader, outliers_ind,  p, reg_coeff):
    """
     Compute the objective:  
                  \sum_i || F_i(Theta) ||_2 + g(theta), 
     where Theta is the model parameters.
    """
    
    stats = {}
    OBJ_tot = 0.5 * reg_coeff *  model.getParameters().frobeniusNormSq() 
    OBJ_nonOutlierstot = 0.0
   

    outliers_size = torch.sum( outliers_ind )

    for ind, data in enumerate( dataloader ):

        output = model(data)

        if p== -2:
            OBJ_ind = torch.norm(output, p=2).item() ** 2 
        else:
            OBJ_ind = torch.norm(output, p=p).item() 

        

        OBJ_tot += OBJ_ind / len( dataloader ) 

        if outliers_ind[ind] == 1.:
            continue

        OBJ_nonOutlierstot += OBJ_ind  / (len( dataloader ) - outliers_size )
  
    stats['non_outliersLoss'] = OBJ_nonOutlierstot
    stats['fullObj'] = OBJ_tot
    
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
    parser.add_argument("--reg_coeff", type=float, default = 0.001, help = "Regularization coefficient")
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

    alg_ord = ['MBO', 'SGD', 'MBOSGD']

    outl_keywords = {'outliers00_': 0.0, 'outliers005_': 0.05 , 'outliers01_': 0.1}
    
    outl_ord = [0.0, 0.05, 0.1]

    #NOTE hard coding here!!
    train_filenames_suffix = {0.0:  '00', 0.05: '005', 0.1: '01'}
    
    DICS_p = { }

    DICS_outl = {}

    DICS_alg = {}

 
    outl_keys_ord = [(p, alg) for p in [1, 1.5, 2, 'ell2Sq'] for alg in ['MBO', 'SGD', 'MBOSGD']]

    alg_keys_ord = [(p, outl) for p in [1, 1.5, 2, 'ell2Sq'] for outl in [0.0, 0.05, 0.1] ]

    for filename in args.filenames:
        #find out file corresponds to which p, alg, and outliers count.
        p  = whichKey(filename, p_keywords, p_keys_ordered) 

        alg = whichKey(filename, alg_keywords, alg_keywords_ordered)

        outl = whichKey(filename, outl_keywords)

        #load model parameters
        model.loadStateDict(filename) 

        #evaluate on test data 
        obj = evalTest(model, data_loader, p = float(p)) 


        #get corresponding train data
        traindata = loadFile( args.traindatafile.format( train_filenames_suffix[ outl ] ) )
        train_data_loader = DataLoader(traindata, batch_size=1)

        #evaluate on train data 
        stats = evalObjective(model, train_data_loader, traindata.outliers_idx, p, reg_coeff = args.reg_coeff)


        
        print("Alg, outliers and p are {}, {}, and {}, respectively.".format(alg, outl, p))
        print("Non-outliers sum is {:.4f} total loss {:.4f} and test loss is {:.4f}".format(stats['non_outliersLoss'], stats['fullObj'], obj) )

        #change p -2 to its actual name (quadratic norm)
        if  p == -2:
            p = 'ell2Sq' 


        #set values in dictionaries
        if p not in DICS_p:
            DICS_p[p] = {}

        if alg not in DICS_p[p]:
            DICS_p[p][alg] = {outl:  stats['non_outliersLoss']}

        else:
            DICS_p[p][alg][outl] =  stats['non_outliersLoss']

      #  if outl  not in DICS_outl:
      #      DICS_outl[outl] = {(p, alg): obj}

      #  else:
      #      DICS_outl[outl][(p, alg)] =  obj
   
      #  if alg not in DICS_alg:
      #      DICS_alg[alg] = {(p, outl): stats['fullObj']}
      #  else:
      #      DICS_alg[alg][(p, outl)] = stats['fullObj']


    #set missing values to zero
    for p in DICS_p:
       for alg in alg_keywords.values():
           if p == 'ell2Sq' and alg != 'SGD':
               continue 

           if alg not in DICS_p[p]:
                DICS_p[p][alg] = dict([(outl, 0.0) for outl in outl_keywords.values()] )

           for outl in outl_keywords.values():
               if outl not in DICS_p[p][alg]:
                   DICS_p[p][alg][outl] = 0.0
  
            

    print(DICS_p)

    #plot bar for each p norm
    for p in DICS_p:
    
        barPlotter(DICS_p[p], args.plt_file + "_p{}".format(p) , x_axis_label  ="Outliers", y_axis_label = 'Non Outliers Loss', normalize=False, lgd=True, log_bar=False, DICS_alg_keys_ordred = outl_ord)

        
    #set missing values to zero
#    for outl in DICS_outl:
#        for outl_key in outl_keys_ord:
#            if outl_key not in DICS_outl[outl]:
#                DICS_outl[outl][outl_key] = 0.0

        
#    print(DICS_outl)

    #plot bar
 #   barPlotter(DICS_outl, args.plt_file + "_outl", x_axis_label  ="(p, Alg.)", y_axis_label = 'Test Loss', normalize=False, lgd=True, log_bar=True, DICS_alg_keys_ordred = outl_keys_ord)

    #set missing values to zero
 #   for alg in DICS_alg:
 #       for key_alg in alg_keys_ord:
 #           if key_alg not in DICS_alg[alg]:
 #               DICS_alg[alg][key_alg] = 0.0

 #   barPlotter(DICS_alg, args.plt_file + "_alg",  x_axis_label  = "(p, outliers)", y_axis_label = 'Objective ', normalize=False, lgd=True, log_bar=True, DICS_alg_keys_ordred  = alg_keys_ord)
