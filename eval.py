import argparse
import logging
from helpers import dumpFile
from torch.utils.data import Dataset, DataLoader
from Net import AEC, DAEC 
import torch
from plotter import whichKey

@torch.no_grad()
def evalObjective(model, dataloader, outliers_ind):
    """
     Compute the objective:  
                  \sum_i || F_i(Theta) ||_2 + g(theta), 
     where Theta is the model parameters.
    """
    
    stats = {}
    OBJ_tot = 0.0
    OBJ_dict = {}
    OBJ_nonOutlierstot = 0.0
    OBJ_originalData = 0.0
    for ind, data in enumerate( dataloader ):
        output = model(data)
        OBJ_i = torch.norm(output, p=1).item()
        OBJ_dict[ind] = OBJ_i
        OBJ_tot += OBJ_i
        if ind in outliers_ind:
            continue
        OBJ_nonOutlierstot += OBJ_i
  
    stats['totalLoss'] = OBJ_tot
    stats['non_outliersLoss'] = OBJ_nonOutlierstot
    
    return OBJ_dict, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='file where model parameters are stored')
    parser.add_argument('--Net', choices=['AEC', 'DAEC'])
    parser.add_argument("--m", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')
    parser.add_argument("--datafile", type=str, help="File storng dataset with no outliers.")
    parser.add_argument("--outliers_ind_file", type=str, help="File strig outliers locations.")
    parser.add_argument("--out_statsfile",  type=str, help="File to store stats.")
    args = parser.parse_args()

    #setup logging
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    model_Net = eval(args.Net)
    model = model_Net(args.m , args.m_prime)

    outliers_ind =  list(torch.load(args.outliers_ind_file).view(-1) )
    original_data = torch.load(args.datafile)
    data_loader = DataLoader(original_data, batch_size=1)
    
    keywords =  {'_1.5':'p=1.5', '_1':'p=1', '_2':'p=2',  '_-2':'ell 2 squared','_3':'p=3', 'SGD':'ell 2 squared (SGD)'}
    keys_ordered = ['_1.5', '_1', '_2', '_-2', '_3']

    DICS = {}
    for filename in args.filenames:
        #find out file corresponds to which alg.
        p  = whichKey(filename, keywords, keys_ordered) 
        #load model parameters
        model.loadStateDict(filename) 
        obj_dict, stats = evalObjective(model, data_loader, outliers_ind) 
        logging.info("Norm {}, the total loss for original input data is {}, total loss for non-outliers is {}".format(p, stats['totalLoss'], stats['non_outliersLoss']))

        print(stats)
        DICS[p] = stats
    dumpFile(args.out_statsfile, DICS)
        



