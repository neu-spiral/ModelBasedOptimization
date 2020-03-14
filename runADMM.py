import argparse
import time
from torch import distributed, nn
import os
import  torch.utils
from torchvision import datasets, transforms
from LossFunctions import AEC
from torch.utils.data import Dataset, DataLoader
from ADMM import ADMM
from torch.nn.parallel import DistributedDataParallel as DDP
from helpers import clearFile
import logging

#torch.manual_seed(1993)

class ModelBasedOptimzier:
    """
       This is a generic class for solving problems of the form
       Minimize \sum_i || F_i(Theta) ||_p + g(theta) 
       via the model based method proposed by Ochs et al. in 2018. 
    """

    def __init__(self, dataset, rank=None, model_spec=None, regularizerCoeff=0.0):
        #load dataset
        if rank != None:
           data_sampler  = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
        else:
           data_sampler = None
        data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=1)

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
        else:
            device = torch.device("cpu")
      

        
    
        
        #initialize ADMM solvers
        self.ADMMsolvers = []
        for ind, data in enumerate( data_loader ):
            data = data.to(device)
            if ind == 0:
                ADMMsolver = ADMM(data=data, model_spec=model_spec, regularizerCoeff=regularizerCoeff)
                model = ADMMsolver.model
            else:
                ADMMsolver = ADMM(data=data, model_spec=model_spec, regularizerCoeff=regularizerCoeff, model=model)
            self.ADMMsolvers.append( ADMMsolver )
        logging.info("Initialized {} ADMMsolvers".format( ind +1 )) 



         #synchronize the parameters
        self._synchParameters()
        logging.info("Synched parameters.")


        self.regularizerCoeff = regularizerCoeff
        self._setInitialPoint()

         #******************************************
        self.runADMM(rank=rank)


    def _setInitialPoint(self):
        """
          Set initial parameters (point) to a feasible point.
        """

        #initialize the optimization vars
        self.theta = self.ADMMsolvers[0].model.getParameters()

    def _synchParameters(self):
        """
            Synchronize model parameters across processes. 
        """
        parameters = self.ADMMsolvers[0].model.getParameters()
        torch.distributed.broadcast(parameters, 0)
        for ADMMsolver in self.ADMMsolvers:
            ADMMsolver.model.setParameters( parameters  )
            


    def runADMM(self, iterations=50, rank=None):
        """
          Execute the ADMM algrotihm for the current model function.
        """
  
        for k in range(iterations):
            t_start = time.time()
            PRES_TOT = 0.0
            DRES_TOT = 0.0
            OBJ_TOT = 0.5 * torch.norm(self.ADMMsolvers[0].primalTheta - self.ADMMsolvers[0].Theta_k) ** 2
            first_ord_TOT = 0.0
            second_ord_TOT = 0.0

            #Update Y and adapt duals for each solver 
            last = time.time()
            for ADMMsolver in self.ADMMsolvers:
                DRES, PRES = ADMMsolver.updateYAdaptDuals()
                first_ord, second_ord =  ADMMsolver.getCoeefficients()

                OBJ_TOT += ADMMsolver.getObjective()
                PRES_TOT += PRES ** 2
                DRES_TOT += DRES ** 2

                first_ord_TOT  += first_ord
                second_ord_TOT += second_ord

            now = time.time()
            logging.debug('Loop took {} (s)'.format(now - last) )
            #Aggregate first_ord_TOT and second_ord_TOT across processes
            now = time.time()
            torch.distributed.reduce(first_ord_TOT, 0)
            torch.distributed.reduce(second_ord_TOT, 0)
            torch.distributed.reduce(PRES_TOT, 0)
            torch.distributed.reduce(DRES_TOT, 0)
            torch.distributed.reduce(OBJ_TOT, 0)
            logging.debug('Reduction took {}(s)'.format(time.time() - now))

            #Compute Theta (proc 0 is responsible for this)
            if rank == 0:
                ADMMsolver_i = self.ADMMsolvers[0]
                DRES_theta = ADMMsolver_i.updateTheta(first_ord_TOT, second_ord_TOT)
                DRES_TOT += DRES_theta

            logging.info("Iteration {} is done in {} (s), OBJ is {} ".format(k, time.time() - t_start, OBJ_TOT ))
            logging.info("Iteration {}, PRES is {}, DRES is {}".format(k, PRES_TOT, DRES_TOT) )

            last = time.time()
            #broadcast the updated Theta 
            torch.distributed.broadcast(self.ADMMsolvers[0].primalTheta, 0)
            now = time.time()
            logging.debug('Broadcast took {}(s)'.format(now - last))

            #update Theta for the rest of the solvers across processes
            for ADMMsolver in self.ADMMsolvers[1:]:
                self.ADMMsolvers[0].updateTheta(first_ord_TOT, second_ord_TOT, self.ADMMsolvers[0].primalTheta)
        
    def getObjective(self):
        OBJ_tot = 0.5 * torch.norm(self.ADMMsolvers[0].primalTheta - self.ADMMsolvers[0].Theta_k) ** 2
        for ADMMsolver in self.ADMMsolvers:
            OBJ_tot += ADMMsolver.output
        torch.distributed.reduce(OBJ_TOT, 0) 
        return OBJ_tot
        
    def runOptimzier(self, rank, iterations=1):
        for k in range(iterations):
            #Synch parameters
            self._synchParameters()

            #run ADMM algortihm
            self.runADMM(rank=rank)
            
                  
         
       
     
        





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=5)
    parser.add_argument("--logfile", type=str,default="logfiles/proc")
    parser.add_argument("--iterations", type=int,  default=10)
    args = parser.parse_args()

    torch.manual_seed(1993 + args.local_rank)

    FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    clearFile( args.logfile + str(args.local_rank)  )
    logging.basicConfig(filename=args.logfile + str(args.local_rank), level=logging.INFO)

    torch.distributed.init_process_group(backend='gloo',
                                         init_method='env://')


    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu")
    
    #initialize model
   # model = AEC(args.m, args.m_prime, device)
   # model = model.to(device)


    dataset =  torch.load(args.input_file)
   # run_proc(args.local_rank, args, dataset, model)
    MBO = ModelBasedOptimzier(dataset=dataset, rank=args.local_rank, model_spec={'loss':AEC, 'parameter_dim':(args.m, args.m_prime)})
   
