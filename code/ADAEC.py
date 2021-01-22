import argparse
import time
import numpy as np
import pickle
from torch import distributed, nn
import os
import  torch.utils
from torchvision import datasets, transforms
from Net import AEC, DAEC, Linear, ConvAEC, ConvAEC2
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from helpers import clearFile, dumpFile, estimate_gFunction, loadFile, pNormProxOp
import logging
import torch.optim as optim
from datasetGenetaor import labeledDataset, unlabeledDataset
from Real_datasetGenetaor import dropLabelAddNoiseDataset, addOutliers
from MTRdatasetGen import AddNoiseDataset


class reduceAnomaly(Dataset):
    """
        Return a dataset instance, where the current S matrix is subtracted from orginal values
    """
    def __init__(self, original_dataset, anomaly):
        #set attributes
        self.original_dataset = original_dataset
        self.anomaly = anomaly
   
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if torch.is_tensor( self.original_dataset[idx] ):
            return self.original_dataset[idx] - self.anomaly[idx]
        else:
            return self.original_dataset[idx][0] - self.anomaly[idx], self.original_dataset[idx][1]


class addAnomaly(reduceAnomaly):

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if torch.is_tensor( self.original_dataset[idx] ):
            return self.original_dataset[idx] + self.anomaly[idx]
        else:
            return self.original_dataset[idx][0] + self.anomaly[idx], self.original_dataset[idx][1]
    


       

class ADAEC:
    """
        A generic class for solving anomaly detection problems of the following form via ADMM, based on the work by Zhou and Paffenroth, KDD 2017:

             Minimize ∑_i ||L_i - F(θ, L_i) ||_2  + g(S)

                 Subj. to: X_i = L_i + S_i ∀i∈[n].

    """
    @torch.no_grad()
    def __init__(self, dataset, model, batch_size = 4, lr = 1e-3, momentum = 0.9, regularizerCoeff = 1.0, p = 2, regularizer= 'ell1'):

        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.regularizerCoeff = regularizerCoeff
        self.p = p

        self.lr = lr
        self.momentum = momentum

        #set problem dimensions
        self.dataset_size = len( self.dataset )

        #NOTE: handle labeled data
        if torch.is_tensor( self.dataset[0] ):
            self.data_shape = self.dataset[0].shape
        else:
            self.data_shape = self.dataset[0][0].shape

        self.N = np.prod( self.data_shape )

        #create anomaly variables
        self.S = [torch.zeros( self.N )] * self.dataset_size

        #compute norm2 squared of the dataset
        self.dataset_normSq = 0.0
        for ind in range(len(dataset)):

            self.dataset_normSq += torch.sum( dataset[ind] ** 2 )
    

    @torch.no_grad()
    def removeAnomaly(self, dataset):
        """
            Subtract current anomalies (stored in S) and return a Dataset instance.
        """
        #current anomaly
        anomaly =  [torch.reshape(S_i, self.data_shape) for S_i in self.S] 

        #return dataset by subtracting anomalies
        return  reduceAnomaly(dataset, anomaly)
        

    @torch.no_grad()
    def addAnomaly(self, dataset):
        """
            Add current anomalies (stored in S) and return a Dataset instance.
        """
        #current anomaly
        anomaly =  [torch.reshape(S_i, self.data_shape) for S_i in self.S]

        #return dataset by subtracting anomalies
        return  addAnomaly(dataset, anomaly)

    @torch.no_grad()
    def getObjective(self):
        """
            Compute and return the full objective function.
        """

        obj = self.regularizerCoeff * self.theta.frobeniusNormSq()

        for data in DataLoader(self.anomalyReducedDataset, batch_size = self.batch_size):
            obj += torch.sum( torch.norm( self.model(data_batch), p = self.p) ) / len( anomalyReducedDataset )

        return obj


    def updateTheta(self, iterations = 100, logger = logging.getLogger('SGD'), debug = False):
        """
            Update and set Theta via running SGD.
        """

        self.theta = self.model.getParameters(trackgrad = True)

        #set optimzier 
        self.optimizer = torch.optim.SGD(self.theta, lr = self.lr, momentum = self.momentum)

        #keep track of trajectories
        trace = {'OBJ': [], 'time': []}
        t_start = time.time()

        DL =  DataLoader(self.anomalyReducedDataset, batch_size = self.batch_size)

        #iterable dataloader
        iterableData  = iter( DL)

        for k in range(iterations):
            #reset gradients
            self.optimizer.zero_grad()
 
            #load new batch
            try:
                data_batch = next( iterableData )
            except StopIteration:
                iterableData  = iter( DL)


            
            #forward pass
            loss = torch.sum( torch.norm( self.model(data_batch), p = self.p) ) / len( self.dataset ) + self.regularizerCoeff * self.theta.frobeniusNormSq()

            loss.backward()

            self.optimizer.step() 

       
            if k % 1 == 0:
                logger.info("{}-th iteration of SGD, loss is {:.4f}.".format(k, loss.item() ) )

            if k == 0 or loss < BEST_loss:
                BEST_loss = loss
                BEST_var = self.theta * 1



        
        #reset parameters
        self.model.setParameters( BEST_var )

    def updateS(self):
        """
            Update S by doing the following:
                1) set S_i :=  X_i - F(θ, L_i)
                2) Update via proximal operators S := Prox(S)
        """
        reconstructedInst = []       
        S = []

        for ind in range( len (self.S) ):
            #reconstructed sample for anomaly reduced dataset
            reconstructedInst_ind = torch.flatten( self.anomalyReducedDataset[ind], start_dim = 0 ) - self.model( torch.unsqueeze(self.anomalyReducedDataset[ind], 0) )

            reconstructedInst.append( reconstructedInst_ind )
          
            #set S (sparsity) to the difference between 
            S.append( torch.flatten(self.dataset[ind], start_dim = 0 )  - reconstructedInst_ind )



        deltaS = []

        #apply proximal opartor
        for ind in range( len (self.S) ):
            if self.regularizer == 'ell1':
                newS_ind = pNormProxOp(S[ind], self.regularizerCoeff, p = 1)

                

            elif self.regularizer == 'ell21':
                newS_ind = pNormProxOp(S[ind], self.regularizerCoeff, p = 2)
           
            print(newS_ind.shape)
            deltaS.append( self.S[ind] - newS_ind )                

            #set updated S values
            self.S[ind] = newS_ind

        return deltaS, reconstructedInst

    def run(self, iterations = 100, innerIterations = 500, logger = logging.getLogger('ADAEC')):
    
        for k in range(iterations):
            t_st = time.time()

            #subtract anomaly
            self.anomalyReducedDataset = self.removeAnomaly(self.dataset)

            #upate Theta via SGD
            self.updateTheta(iterations = innerIterations, logger = logger)
             
            #update S via prox-op.
            deltaS, reconstructedInst = self.updateS()

            #add anomaly 
            self.anomalyAddedDataset = self.addAnomaly( self.dataset )

            #compute c1, c2 for checking convergence
            c1 = 0
            c2 = 0

            for ind in range(len(self.dataset)):

                c1 += torch.sum( deltaS[ind] ** 2) / self.dataset_normSq
        
                c2 += torch.sum( (torch.squeeze(self.anomalyAddedDataset[ind]) - reconstructedInst[ind] - self.S[ind]) ** 2 ) / self.dataset_normSq

            logger.info("Iteration {} done in {}(s).".format(k, time.time() - t_st) )
            logger.info("Convergence parameters, c1 and c2 are {:.4f} and {:.4f}, respectively.".format(c1, c2) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)

    parser.add_argument("--m_dim", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=8)
    parser.add_argument("--logfile", type=str,default="logfiles/proc")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for processing dataset, when None it is set equal to the dataset size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used for running off-the-shelf solvers." )
    parser.add_argument("--iterations", type=int,  default=50)
    parser.add_argument("--inner_iterations", type=int,  default=500)
    parser.add_argument("--rho", type=float, default=1.0, help="rho parameter in ADMM")
    parser.add_argument("--regularizer", choices = ['ell1', 'ell2', 'ell21'], default = 'ell1', help="Regularizer function.")
    parser.add_argument("--regularizerCoeff", type=float, default=1.0, help = "regularization coefficient")
    parser.add_argument("--p", type=float, default=2, help="p in lp-norm")
    parser.add_argument("--tracefile", type=str, help="File to store traces.")
    parser.add_argument("--statfile", help = "File to store statistics.")
    parser.add_argument("--modelfile", type=str, help="File to store model parameters.")
    parser.add_argument("--logLevel", type=str, choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument("--net_model", choices=['Linear', 'AEC', 'DAEC', 'ConvAEC', 'ConvAEC2'], default='ConvAEC')
    args = parser.parse_args()

    #Setup logger
    logger = logging.getLogger()
    logger.setLevel(eval("logging."+args.logLevel))

    logFile = args.logfile

    clearFile(logFile)
    fh = logging.FileHandler(logFile)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info("Starting with arguments: "+str(args))



    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu")

    #initialize model
    new_model = eval(args.net_model)
    model = new_model(args.m_dim, args.m_prime, device=device)


    #move model
    model = model.to(device) 


    #load dataset 
    dataset = loadFile(args.input_file)

    
    ADAEC_obj = ADAEC(dataset = dataset, 
    model = model, 
    batch_size = args.batch_size, 
    lr = args.lr, 
    momentum = 0.9, 
    regularizerCoeff = args.regularizerCoeff,
    p = args.p, 
    regularizer = args.regularizer) 

    #run alg. 
    ADAEC_obj.run( 
        iterations = args.iterations,
        innerIterations = args.inner_iterations,
        logger = logger
         )
    

