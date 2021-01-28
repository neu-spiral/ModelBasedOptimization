import argparse
import logging
from helpers import dumpFile, loadFile, clearFile
from torch.utils.data import Dataset, DataLoader
from Net import AEC, DAEC, Linear, ConvAEC, ConvAEC2
import torch
from plotter import whichKey, barPlotter
from datasetGenetaor import labeledDataset, unlabeledDataset
from Real_datasetGenetaor import dropLabelAddNoiseDataset, addOutliers
from MTRdatasetGen import AddNoiseDataset
import sklearn.linear_model
import torchvision.transforms as transforms
from torchvision import datasets

@torch.no_grad()
def evalObjective(model, dataloader, outliers_ind,  p):
    """
     Compute the objective:  
                  \sum_i || F_i(Theta) ||_2 + g(theta), 
     where Theta is the model parameters.
    """
    
    stats = {}
    OBJ_tot = 0.0
    OBJ_nonOutlierstot = 0.0
   
    OBJ_originalData = 0.0
    OBJ_p =  0.5 * model.getParameters().frobeniusNormSq()

    for ind, data in enumerate( dataloader ):

        output = model(data)

        if p== -2:
            OBJ_p += torch.norm(output, p=2).item() ** 2 / len( dataloader )
        else:
            OBJ_p += torch.norm(output, p=p).item() / len( dataloader )

        OBJ_i = torch.norm(output, p=1).item() / len( dataloader )


        OBJ_tot += OBJ_i

        if outliers_ind[ind] == 1.:
            continue

        OBJ_nonOutlierstot += OBJ_i
  
    stats['totalLoss'] = OBJ_tot + 0.5 * model.getParameters().frobeniusNormSq() 
    stats['non_outliersLoss'] = OBJ_nonOutlierstot
    stats['fullObj'] = OBJ_p
    
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

@torch.no_grad()
def extractFeatures(model, data):
    """
        Given an auto-encoder model extract features from data.
    """


    extractedFeat = []
   
    labels = []

    for data_i in DataLoader(data, batch_size = 1):
        #hidden layer
        H = torch.nn.functional.sigmoid( model.conv1(data_i[0]) )

        #second conv-layer
        H = torch.nn.functional.sigmoid( model.conv2(H) )  

        #flatten
        H = torch.flatten( H, start_dim = 0)

        #add batch dim
        H = torch.unsqueeze(H, 0)
        
        extractedFeat.append(H)

          
        labels.append( data_i[1] )

    return torch.cat(extractedFeat, 0) , torch.cat(labels, 0)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    parser.add_argument("dataset_name", default='MNIST', help="Dataset to train classifier on.")
    parser.add_argument("--logfile", help="Logfile", default = 'logfiles/log')
    parser.add_argument('--net_model', choices=['Linear', 'AEC', 'DAEC', 'ConvAEC2', 'ConvAEC'])
    parser.add_argument("--m_dim", type=int, help='dimension of eacg point.')
    parser.add_argument("--m_prime", type=int, help='dimension of the embedding.')

    parser.add_argument("--data_dir", type=str, default='data/', help="Directory to download data")
    parser.add_argument("--out_statsfile",  type=str, help="File to store stats.")

    parser.add_argument("--plt_file", help="File to store plots")
    args = parser.parse_args()



    #Setup logger
    logger = logging.getLogger()

    logFile = args.logfile

    clearFile(logFile)
    fh = logging.FileHandler(logFile)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info("Starting with arguments: "+str(args))


    model_Net = eval(args.net_model)
    model = model_Net(args.m_dim , args.m_prime)


    my_transform = transforms.Compose([transforms.ToTensor()] )

    #get dataset 
    my_dataset_class = eval('datasets.' + args.dataset_name)

    #load train dataset
    dataset_train = my_dataset_class(args.data_dir, train = True, download=True, transform=my_transform)
   
    #load test dataset 
    dataset_test = my_dataset_class(args.data_dir, train = False, download=True, transform=my_transform)
    


    
    #setting proper keys based on patterns in filenames
    p_keywords =  {'_p1.5':1.5, '_p1':1., '_p2':2.,  '_p-2': -2.,'_p3':3.}
    p_keys_ordered = ['_p1.5', '_p1', '_p2', '_p-2', '_p3']

    alg_keywords = {'_MBO':'MBO', '_SGD_': 'SGD', '_MBOSGD': 'MBOSGD'}
    alg_keywords_ordered = ['_MBOSGD', '_MBO', '_SGD_']

    outl_keywords = {'outliers00_': 0.0, 'outliers005_': 0.05 , 'outliers01_': 0.1}
    
    #NOTE hard coding here!!
    train_filenames_suffix = {0.0:  '00', 0.05: '005', 0.1: '01'}
    
    DICS_p = {}

   #########



    #find out file corresponds to which p, alg, and outliers count.
    p  = whichKey(filename, p_keywords, p_keys_ordered) 

    alg = whichKey(filename, alg_keywords, alg_keywords_ordered)

    outl = whichKey(filename, outl_keywords)

    #load model parameters
    model.loadStateDict(filename) 


    best_acc = 0.0
    
    for C in [1e-2, 1e-1, 1]:
        logger.info("setting regularization to ", C)

        #iniizlie classifier
        clf = sklearn.linear_model.LogisticRegression(penalty='l2', solver = 'sag', multi_class = 'auto', C = 1e-1, random_state = 0, verbose = 1)

        #extract features and also output labels to be fed to classifer 
        X, Y = extractFeatures( model,  dataset_train)

    
        X_test, Y_test = extractFeatures( model,  dataset_test)
         
        clf.fit( X, Y)
        
        train_acc = clf.score(X, Y)
        acc = clf.score(X_test, Y_test)

        if acc > best_acc:
            best_acc = acc
            best_C = C
 
        logger.info("For C {}, test accuracy is {:.3f}.".format(C, best_acc))

        

    logger.info("For p, alg, and outeliers, {}, {}, and {}respectively, the accuracy is {:.4f}".format(p, alg, outl, acc) )

    stats = {'train_acc': train_acc, 'test_acc': best_acc , 'C': best_C}

    with open(args.out_statsfile, 'wb') as f:
        f.dump( stats )


