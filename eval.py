import argparse
from torch.utils.data import Dataset, DataLoader
from Net import AEC, DAEC 
import torch
from plotter import whichKey

@torch.no_grad()
def evalObjective(model, dataloader):
    """
     Compute the objective:  
                  \sum_i || F_i(Theta) ||_2 + g(theta), 
     where Theta is the model parameters.
    """
    
    OBJ_tot = 0.0
   
    OBJ_dict = {}
    for ind, data in enumerate( dataloader ):
        print (ind, data)
        output = model(data)
        OBJ_i = torch.norm(output, p=1) 
        OBJ_dict[ind] = OBJ_i
        OBJ_tot += OBJ_i
  
    return OBJ_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='file where model parameters are stored')
    parser.add_argument('--Net', choices=['AEC', 'DAEC'])
    parser.add_argument("--m", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')
    parser.add_argument("--datafile", type=str, help="File storng dataset with no outliers.")
    parser.add_argument("--outliers_ind_file", type=str, help="File strig outliers locations.")
    args = parser.parse_args()

   
    model_Net = eval(args.Net)
    model = model_Net(args.m , args.m_prime)

    outliers_ind = torch.load(args.outliers_ind_file)
    print(outliers_ind)
    original_data = torch.load(args.datafile)
    data_loader = DataLoader(original_data, batch_size=1)
    
    keywords =  {'_1':'p=1', '_2':'p=2',  '_-2':'ell 2 squared','_3':'p=3', 'SGD':'ell 2 squared (SGD)'}
    for filename in args.filenames:
        #find out file corresponds to which alg.
        Alg  = whichKey(filename, keywords) 
        #load model parameters
        model.loadStateDict(filename) 
        obj_dict = evalObjective(model, data_loader) 
        #print(Alg, obj_dict)
        



