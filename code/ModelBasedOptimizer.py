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
from ADMM import LocalSolver, solveConvex, solveWoodbury, solveQuadratic, OADM, InnerADMM, ParInnerADMM
from torch.nn.parallel import DistributedDataParallel as DDP
from helpers import clearFile, dumpFile, estimate_gFunction, loadFile
import logging
import torch.optim as optim
from datasetGenetaor import labeledDataset, unlabeledDataset
from Real_datasetGenetaor import dropLabelAddNoiseDataset
from MTRdatasetGen import AddNoiseDataset


#torch.manual_seed(1993)

class ModelBasedOptimzier:
    """
       This is a generic class for solving problems of the form
       Minimize \sum_i || F_i(Theta) ||_p + g(theta) 
       via the model based method proposed by Ochs et al. in 2018. 
    """

    def __init__(self, dataset, model, rho=1.0, rho_inner = 1.0,  p=2, h=1.0, gamma=1.0, regularizerCoeff = 0.5, batch_size = 4, g_est=None, rank = None):
        #If rank is None the execution is serial. 
        self.rank = rank

        self.dataset = dataset 

        #load dataset
        if rank != None:
           data_sampler  = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
        else:
           data_sampler = None

        self.data_loader = DataLoader(dataset, sampler=data_sampler, batch_size = batch_size)


        #estimator for g function (used in runADMM)
        self.g_est = g_est

        #Check if GPU is available 
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
        else:
            device = torch.device("cpu")
      

        #Create the model (for loss function)
        self.model = model
        #**
        self.model = self.model.to(device)
        #Synch model parameters across processes
        self._synchParameters()
         
        
        if p == -2:
            logging.warning("Setting p to -2, it means that l2-norm squared is used.")
            logging.warning("l2-norm squared is not implemented for the model based solver, it is only for debugging the ADMM solver.")

        self.p = p
        #regularizerCoeff is the squared regularizer`s coefficient
        self.regularizerCoeff = regularizerCoeff
        self.h = h
        self.gamma = gamma
        self.rho = rho
        self.rho_inner = rho_inner


        #initialize ADMM solver 
        self.oadm_obj = OADM(data = dataset, model = self.model, rho = self.rho, p = self.p, h = self.h, gamma = self.gamma, regularizerCoeff = self.regularizerCoeff, rho_inner = self.rho_inner,  batch_size = batch_size)

    @torch.no_grad()
    def _synchParameters(self):
        """
            Synchronize model parameters across processes. 
        """
        #Synch model parameters
        parameters = self.model.getParameters()
        if self.rank != None:
            for parameter in parameters:
                torch.distributed.broadcast(parameter, 0)
            self.model.setParameters( parameters )
            logger.info("Synchronized model parameters across processes.")



    def MSE(self, epochs=1, batch_size=8):
        """
            Minimize Mean Squared Error via PyTorch optimziers.
        """
        t_start = time.time()

        #Define optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        #DatalLoader
        data_loader = DataLoader(self.dataset,  batch_size=batch_size)

        #Keep track of progress in trace
        trace = {}
        trace[0] = {}
        init_loss = 0.0
        for ind, solver_i in enumerate(self.ADMMsolvers):
            #loss due to the i-th datapoint
            init_loss += torch.norm(self.model( solver_i.data), p=2) ** 2

        trace[0]['OBJ'] = init_loss
        trace[0]['time'] = time.time() - t_start
        logger.info("Epoch {0}, loss is {1:.4f}".format(0, init_loss) )
        for i in range(1, epochs+1):
             
            #Keep track of loss throught the iterations 
            running_loss = 0.0
            loss = 0.0
            for ind, data in enumerate(data_loader):
                #loss due to the i-th datapoint
                loss = torch.norm(self.model(data), p=2) ** 2
                 
                #zero the parameter gradients
                optimizer.zero_grad()

                #backprop 
                loss.backward(retain_graph=False)

                #SGD step 
                optimizer.step()
                #update running loss
                running_loss += loss.item()

            trace[i] = {}
            trace[i]['OBJ'] = running_loss
            trace[i]['time'] = time.time() - t_start

            logger.info("Epoch {0}, loss is {1:.4f}".format(i, running_loss) )    
        return trace        


    def runSGD(self, lr, iterations = 200, logger =  logging.getLogger('LSBBM'), debug = False):
        """
            Solve the following problem via SGD:
                      Minimize 1/n ∑_i ||F_i(θ)||_p + G(θ).
        """

        #get current model vraibales
        theta_VAR = self.model.getParameters( trackgrad=True )

        optimizer = torch.optim.Adam(theta_VAR, lr = lr)#, momentum=0.9)

        t_start = time.time()

        #keep track of trajectories
        trace = {'OBJ': [self.getObjective()], 'time': [time.time() - t_start]}

        for _ in range(iterations):

            #add regulrization
            loss = self.oadm_obj.getGfunc( theta_VAR )

            #load batch
            data_i = next( iter( self.data_loader ) )

            output = self.model( data_i )
                
             #evaluate loss for the batch
            if self.p == -2:
                loss += torch.sum( torch.norm(output, p = 2, dim = 1) ** 2 ) / data_i.shape[0] 
            else:
                loss +=  torch.sum( torch.norm(output, p = self.p, dim = 1)  ) / data_i.shape[0]

            #backward pass
            loss.backward()

            #optimizer step
            optimizer.step()

            if debug:
                #evaluate objective
                OBJ = self.getObjective( theta_VAR )

                logger.info("Objective is {:.4f}.".format( OBJ ) )

                trace['OBJ'].append( OBJ.item() )
                trace['time'].append( time.time() - t_start)
        return trace
    
 
    @torch.no_grad()
    def getObjective(self, theta=None):
        """
         Compute the objective:  
                  \sum_i || F_i(Theta) ||_p + g(theta) 
         for a given Theta.
         NOTE: The current implmentation chnages the model parameters to theta.
        """

        #regularization term
        if theta is None:
            OBJ_tot = self.oadm_obj.getGfunc( self.model.getParameters() * 1 )
        else:
            OBJ_tot = self.oadm_obj.getGfunc( theta )

        if theta != None:
            self.model.setParameters( theta )

        #pass through the dataset
        for data_i in self.data_loader:

            output = self.model( data_i )

            if self.p == -2:
                OBJ_tot += torch.sum( torch.norm(output, p = 2, dim = 1) ** 2 ) / len(self.dataset)

            else:
                OBJ_tot += torch.sum( torch.norm(output, p = self.p, dim = 1) ) / len(self.dataset)

        if self.rank != None:
            torch.distributed.all_reduce(OBJ_tot) 

        return OBJ_tot 

    @torch.no_grad()
    def getModelDiscrepancy(self, theta):
        """
         Evaluate the discrepancy between the model function and the original function at the given point theta.
                | \sum_i   || F_i(theta) ||_p  -  || F_i(theta_k) +  ∇F_i(theta_k)(theta - theta_k) ||_p |
        """

        obj = self.getObjective(theta)
        modelObj = 0.0 
        for ADMMsolver in self.ADMMsolvers:
            modelObj += ADMMsolver.evalModelLoss( theta )
        return  abs(obj - modelObj)


    @torch.no_grad()
    def updateVariable(self, s_k, DELTA, maxiters=100, delta =0.9, gamma = 0.1, eta = 1.0):

        last = time.time()

   
        #mult by 1 to make sure that theta_k is not pointing to the paramaters of the model (o.w. modifying model parameters, would modify theta_k too!)
        theta_k = self.model.getParameters() * 1

        obj_k = self.getObjective( theta_k )

        #initial step-size
        stp_size = eta * delta 

        #iteratively make the stepsize smaller
        for pow_ind in range(maxiters):

            #new point a convex comb. of theta_k and new point (s_k)
            new_theta = (1.0 - stp_size) * theta_k + stp_size *  s_k

            #eval objective for new_theta
            obj_new = self.getObjective( new_theta )

 
            #check if objective for new_theta is within the desired boundaries r
            if obj_new <= obj_k + gamma * DELTA:
                break 
          
            #shrink step-size
            stp_size *= delta

        now = time.time()
        logger.debug('New step-size found and parameter updated in {0:.2f}(s).'.format(now - last) )

        return obj_new             
        
         
    
        

    def run(self, stopping_eps = 1.e-4, iterations = 20, innerIterations = 100, world_size = 1, l2SquaredSolver='MBO', logger = logging.getLogger('LSBBM'), debug = False):
        """
            Run the Line Search Baed Bregman Minmization Algorithm, where at each iteration the new desecnet direction found via calling the run method of oadm. Then step size is set via Armijo line search.
        """

        #for ell2 squared solve problem via SGD
        if self.p == -2 and l2SquaredSolver != 'MBO':
            logging.info('Solving ell2 norm squared via {}'.format(l2SquaredSolver) )
            return self.MSE(epochs=iterations)

        t_start = time.time()

        #initialize the trace dict
        trace = {}
        trace['OBJ'] = [ self.getObjective() ]
        trace['time'] = [time.time() - t_start]
        

        #main iterations
        for i in range(iterations):
            
            #find a dsecnt direction
            oadm_trace, model_delta = self.oadm_obj.run(iterations = innerIterations, world_size = world_size, logger = logger, debug = debug)             
     
            #update variable via Armijo line search
            OBJ = self.updateVariable(s_k = self.oadm_obj.theta_bar1, DELTA = model_delta)
               
     
            #log stats
            if logger is not None:
                logger.info("{}-th iteration of LSBBM, objective function is {:.4f}.".format(i, OBJ) )

            #add to trace
            trace['OBJ'].append( OBJ )
            trace['time'].append( time.time() - t_start )

        return trace

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--world_size", type=int, default=1, help="Number of processes to spawn for parallel computations, defaults to 1 (no parallelism).")
    parser.add_argument("--m_dim", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=8)
    parser.add_argument("--logfile", type=str,default="logfiles/proc")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for processing dataset, when None it is set equal to the dataset size.")
    parser.add_argument("--inner_iterations", type=int,  default=40)
    parser.add_argument("--iterations", type=int,  default=50)
    parser.add_argument("--rho", type=float, default=1.0, help="rho parameter in ADMM")
    parser.add_argument("--rho_inner", type=float, default=1.0, help="rho parameter in InnerADMM.")
    parser.add_argument("--gamma", type=float, default=1.0, help= "gamma parameter in OADM")
    parser.add_argument("--h", type=float, default=1.0, help= "h parameter in LSBB")
    parser.add_argument("--regularizerCoeff", type=float, default=1.0, help = "regularization coefficient")
    parser.add_argument("--p", type=float, default=2, help="p in lp-norm")
    parser.add_argument("--tracefile", type=str, help="File to store traces.")
    parser.add_argument("--outfile", type=str, help="File to store model parameters.")
    parser.add_argument("--logLevel", type=str, choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument("--net_model", choices=['Linear', 'AEC', 'DAEC', 'ConvAEC', 'ConvAEC2'], default='ConvAEC')
    parser.add_argument("--l2SquaredSolver", type=str, choices=['SGD', 'MBO'], help='Solver to use for ell 2 norm squared.')
    args = parser.parse_args()


    
    if args.local_rank != None:
        torch.manual_seed(1993 + args.local_rank)
        torch.distributed.init_process_group(backend='gloo',
                                         init_method='env://')
    else:
        torch.manual_seed(1993)

    #Setup logger
    logger = logging.getLogger()
    logger.setLevel(eval("logging."+args.logLevel))

    if args.local_rank is not None:
        logFile = args.logfile + "_process" + str(args.local_rank)
    else:
        logFile = args.logfile

    print(logFile)
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

    #model = Linear(args.m, args.m_prime)

    model = model.to(device)

    #load dataset 
   # dataset =  torch.load(args.input_file)
    dataset = loadFile(args.input_file)

 
    #estimate g function
    if args.p not in [1, 2, -2]:
        g_est = estimate_gFunction(args.p)
    else:
        g_est = None


    #OADM
    #oadm_solver = OADM(dataset, rho=5.0, p=2, h=1.0, gamma=1.0, regularizerCoeff = 0.5, batch_size = 12, model = model)


    #oadm_solver.run( iterations = 1, world_size =3 )
    
    #NOTE
    #return
    #print(ss)
   ###########MBO############# 

    MBO = ModelBasedOptimzier(dataset = dataset, model = model, rho = args.rho, rho_inner = args.rho_inner,  p = args.p, h = args.h, gamma = args.gamma, regularizerCoeff = args.regularizerCoeff, batch_size = args.batch_size, g_est = g_est)

    #run the model based optimaizer
    trace = MBO.run( iterations  = args.iterations, innerIterations  = args.inner_iterations, world_size = args.world_size, logger = logger) 

   #run sgd 
    trace_SGD_lr3 = MBO.runSGD(lr = 1e-3, debug = True, logger = logger) 

    trace_SGD_lr4 = MBO.runSGD(lr = 1e-4, debug = True, logger = logger)

    trace_SGD_lr5 = MBO.runSGD(lr = 1e-5, debug = True, logger = logger)

    #save model parameters
 #   model.saveStateDict( args.outfile )
   
    

    with open(args.tracefile + "_sgd3",'wb') as f:
        pickle.dump(trace_SGD_lr3,  f)

    with open(args.tracefile + "_sgd4",'wb') as f:
        pickle.dump(trace_SGD_lr4,  f)

    with open(args.tracefile + "_sgd5",'wb') as f:
        pickle.dump(trace_SGD_lr5,  f)

    with open(args.tracefile + "_mbo",'wb') as f:
        pickle.dump(trace,  f)




