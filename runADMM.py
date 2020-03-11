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

def run_proc(rank, args, dataset, model):
    print("Running process %d" %rank)
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu")
  #  device = torch.device("cpu")


    #load dataset
    data_sampler  = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=1)

    #synchronize the parameters
    parameters = model.getParameters()
    torch.distributed.broadcast(parameters, 0)
    model.setParameters( parameters  )



    

    

    #initialize ADMM solvers
    ADMMsolvers = []
    for data in data_loader:
        data = data.to(device) 
        ADMMsolver = ADMM(data, model)
        ADMMsolvers.append( ADMMsolver )
    logging.info("Initialized {} ADMMsolvers".format(len(ADMMsolvers)) )

    runADMM(ADMMsolvers, rank, args.iterations)
        


     
     

def runADMM(ADMMsolvers, rank, iterations=10):
    "Given a list of ADMMsolvers run the ADMM algrotihm"
    for k in range(iterations):
        t_start = time.time()
        PRES_TOT = 0.0
        DRES_TOT = 0.0
        OBJ_TOT = 0.5 * torch.norm(ADMMsolvers[0].primalTheta - ADMMsolvers[0].Theta_k) ** 2
        first_ord_TOT = 0.0
        second_ord_TOT = 0.0

        #Update Y and adapt duals for each solver 
        last = time.time()
        for ADMMsolver in ADMMsolvers:
            DRES, PRES = ADMMsolver.updateYAdaptDuals()
            first_ord, second_ord =  ADMMsolver.getCoeefficients()

            OBJ_TOT += ADMMsolver.getObjective()
            PRES_TOT += PRES ** 2
            DRES_TOT += DRES ** 2

            first_ord_TOT  += first_ord
            second_ord_TOT += second_ord

        now = time.time()
        print ('Loop took {} (s)'.format(now - last) )
        #Aggregate first_ord_TOT and second_ord_TOT across processes
        now = time.time()
        torch.distributed.reduce(first_ord_TOT, 0) 
        torch.distributed.reduce(second_ord_TOT, 0)
        torch.distributed.reduce(PRES_TOT, 0)
        torch.distributed.reduce(DRES_TOT, 0)
        torch.distributed.reduce(OBJ_TOT, 0)
        print ('Reduction took {}(s)'.format(time.time() - now))
        
        #Compute Theta (proc 0 is responsible for this)
        if rank == 0:
            ADMMsolver_i = ADMMsolvers[0]
            DRES_theta = ADMMsolver_i.updateTheta(first_ord_TOT, second_ord_TOT)
            DRES_TOT += DRES_theta

            
            logging.info("Iteration {} is done in {} (s), OBJ is {} ".format(k, time.time() - t_start, OBJ_TOT ))
            logging.info("Iteration {}, PRES is {}, DRES is {}".format(k, PRES_TOT, DRES_TOT) )

        last = time.time()
        #broadcast the updated Theta 
        torch.distributed.broadcast(ADMMsolvers[0].primalTheta, 0)
        now = time.time()
        print ('Broadcast took {}(s)'.format(now - last))

        #update Theta for the rest of the solvers across processes
        for ADMMsolver in ADMMsolvers[1:]:
             ADMMsolvers[0].updateTheta(first_ord_TOT, second_ord_TOT, ADMMsolvers[0].primalTheta)

        



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--m_prime", type=int,  default=5)
    parser.add_argument("--logfile", type=str,default="logfiles/proc")
    parser.add_argument("--iterations", type=int,  default=10)
    args = parser.parse_args()

    torch.manual_seed(1993 + args.local_rank)
    clearFile( args.logfile + str(args.local_rank)  )
    logging.basicConfig(filename=args.logfile + str(args.local_rank), level=logging.INFO)

    torch.distributed.init_process_group(backend='gloo',
                                         init_method='env://')


    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu")
    
    #initialize model
    model = AEC(args.m, args.m_prime, device)
    model = model.to(device)


    dataset =  torch.load(args.input_file)
    run_proc(args.local_rank, args, dataset, model)
   
