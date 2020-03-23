import argparse
import time
from torch import distributed, nn
import os
import  torch.utils
from torchvision import datasets, transforms
from LossFunctions import AEC, Linear
from torch.utils.data import Dataset, DataLoader
from ADMM import ADMM
from torch.nn.parallel import DistributedDataParallel as DDP
from helpers import clearFile
import logging
import torch.optim as optim


#torch.manual_seed(1993)

class ModelBasedOptimzier:
    """
       This is a generic class for solving problems of the form
       Minimize \sum_i || F_i(Theta) ||_p + g(theta) 
       via the model based method proposed by Ochs et al. in 2018. 
    """

    def __init__(self, dataset, model, rho=5.0, p=2, rank=None, regularizerCoeff=0.0):
        #If rank is None the execution is serial. 
        self.rank = rank

        self.dataset = dataset 
        #load dataset
        if rank != None:
           data_sampler  = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
        else:
           data_sampler = None
        data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=1)

        #Check if GPU is available 
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
        else:
            device = torch.device("cpu")
      

        #Create the model (for loss function)
        self.model = model
        self.model.to(device)
        #Synch model parameters across processes
        self._synchParameters()
         
        
        #initialize ADMM solvers
        self.ADMMsolvers = []
        for ind, data in  enumerate(data_loader):
            data = data.to(device)
            ADMMsolver = ADMM(data=data, rho=rho, p=p, regularizerCoeff=regularizerCoeff, model=self.model)
            self.ADMMsolvers.append( ADMMsolver )
        logging.info("Initialized {} ADMMsolvers".format( ind +1 )) 
        #p is the parameter in lp-norm
        self.p = p
        #regularizerCoeff is the squared regularizer`s coefficient
        self.regularizerCoeff = regularizerCoeff

         #******************************************

    @torch.no_grad()
    def _synchParameters(self):
        """
            Synchronize model parameters across processes. 
        """
        #Synch model parameters
        parameters = self.model.getParameters()
        if self.rank != None:
            torch.distributed.broadcast(parameters, 0)
            self.model.setParameters( parameters )
            logging.info("Synchronized model parameters across processes.")


    def runADMM(self, iterations=50):
        """
          Execute the ADMM algrotihm for the current model function.
        """

        #Initialize solver variables
        for ADMMsolver in self.ADMMsolvers:
            ADMMsolver._setVARS()
  
        for k in range(iterations):
            t_start = time.time()
            PRES_TOT = 0.0
            DRES_TOT = 0.0
            OBJ_TOT = 0.0
            model_loss = 0.0
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

                model_loss += ADMMsolver.evalModelLoss()
                first_ord_TOT  += first_ord
                
                second_ord_TOT += second_ord

            now = time.time()
            logging.debug('Loop took {} (s)'.format(now - last) )
            #Aggregate first_ord_TOT and second_ord_TOT across processes
            now = time.time()

            if self.rank != None:
                torch.distributed.all_reduce(first_ord_TOT)
                torch.distributed.all_reduce(second_ord_TOT)
                torch.distributed.all_reduce(PRES_TOT)
                torch.distributed.all_reduce(DRES_TOT)
                torch.distributed.all_reduce(OBJ_TOT)


                logging.debug('Reduction took {}(s)'.format(time.time() - now))

            #Compute Theta (proc 0 is responsible for this)
            ADMMsolver_i = self.ADMMsolvers[0]
            DRES_theta = ADMMsolver_i.updateTheta(first_ord_TOT, second_ord_TOT)
            #Update Theta for the rest of the solvers across processes
            for ADMMsolver in self.ADMMsolvers[1:]:
                ADMMsolver.updateTheta(first_ord_TOT, second_ord_TOT, self.ADMMsolvers[0].primalTheta)
            DRES_TOT += DRES_theta
            #Add the quadratic term to OBJ
            OBJ_TOT += 0.5 * torch.norm(self.ADMMsolvers[0].primalTheta - self.ADMMsolvers[0].Theta_k) ** 2
           

            logging.info("Iteration {} is done in {} (s), OBJ is {} ".format(k, time.time() - t_start, OBJ_TOT ))
            logging.info("Iteration {}, PRES is {}, DRES is {}".format(k, PRES_TOT, DRES_TOT) )


       #Evaluate delta
        delta_TOT = 0.0
        for ADMMsolver in self.ADMMsolvers:
            delta_TOT += ( ADMMsolver.evalModelLoss( ADMMsolver.primalTheta  ) - torch.norm(ADMMsolver.output, p=self.p) )
        if self.rank != None:
            torch.distributed.all_reduce(delta_TOT)
        delta_TOT += torch.norm(ADMMsolver.primalTheta - ADMMsolver.Theta_k, p=2)**2
                        
        return delta_TOT

    def runSGD(self, epochs=1):
        """
           Minimize the model function plus the squared loss:

               min_theta  ∑_i || F_i(theta_k) +  ∇F_i(theta_k)(theta - theta_k) ||_p + ||theta - theta_k||_2^2,
           via SGD.
        """

        #Set the optimization variable 
        theta = self.model.getParameters()
        theta.requires_grad = True
        theta_k = self.model.getParameters()
        theta_k.requires_grad = False

        squaredLoss = nn.MSELoss(reduction='sum') 
        #Define optimizer
        optimizer = optim.SGD([theta], lr=0.001)

        
        for i in range(epochs):
            #Proximity loss ||theta - theta_k||_2^2
            sq_loss = 0.5 * squaredLoss(theta, theta_k)
            #Keep track of loss throught the iterations 
            running_loss = 0.0
            for ind, solver_i in enumerate(self.ADMMsolvers):


                #zero the parameter gradients
                optimizer.zero_grad() 

                #loss
                loss = solver_i.evalModelLoss( theta)
                if ind == 0:
                    loss = loss + sq_loss
                #backprop
                loss.backward(retain_graph=False)

                #SGD step 
                optimizer.step()
                running_loss += loss.item()

            logging.info("Epoch {}, loss is {}".format(i, running_loss) )
                

    @torch.no_grad()
    def getObjective(self, theta=None):
        """
         Compute the objective:  
                  \sum_i || F_i(Theta) ||_p + g(theta) 
         for a given Theta.
         NOTE: The current implmentation chnages the model parameters to theta.
        """

        OBJ_tot = 0.0
        if theta != None:
            self.model.setParameters( theta )

        for ADMMsolver in self.ADMMsolvers:
            output = self.model(ADMMsolver.data)
            OBJ_tot += torch.norm(output, p=self.p) 

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
    def updateVariable(self, DELTA):

        last = time.time()
        delta =0.5
        gamma = 0.5
        eta = 2.0

        obj_k = self.getObjective()
        theta_k = self.ADMMsolvers[0].Theta_k
        s_k = self.ADMMsolvers[0].primalTheta
        stp_size = eta * delta 
        while True:
            obj_new = self.getObjective( (1.0 - stp_size) * theta_k + stp_size *  s_k )
            if obj_new <= obj_k + gamma * DELTA:
                break 
            stp_size *= delta
        now = time.time()
        logging.info('New step-size found and parameter updated in {}(s).'.format(now - last) )
        return obj_new             
        
         
    
        

    def run(self,  innerIterations=50, iterations=1):
        for k in range(iterations):
            OBJ = self.getObjective()
     
            logging.info('Outer iteration {}, OBJ is {}'.format(k, OBJ) )

            #run ADMM algortihm
            model_improvement = self.runADMM( innerIterations )
            
            self.updateVariable( model_improvement ) 
            
     
                  
               
     
        





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=5)
    parser.add_argument("--logfile", type=str,default="logfiles/proc")
    parser.add_argument("--iterations", type=int,  default=10)
    parser.add_argument("--rho", type=float, default=1.0)
    args = parser.parse_args()


    
    if args.local_rank != None:
        torch.manual_seed(1993 + args.local_rank)
        torch.distributed.init_process_group(backend='gloo',
                                         init_method='env://')
    else:
        torch.manual_seed(1993)

    #Setup logger
    FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    clearFile( args.logfile + str(args.local_rank)  )
    logging.basicConfig(filename=args.logfile + str(args.local_rank), level=logging.INFO)



    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu")
    
    #initialize model
   # model = AEC(args.m, args.m_prime, device)
   # model = model.to(device)


    dataset =  torch.load(args.input_file)
    model = Linear(args.m, args.m_prime)
   # run_proc(args.local_rank, args, dataset, model)
    MBO = ModelBasedOptimzier(dataset=dataset, model=model, rank=args.local_rank, rho=args.rho)
   # dim_theta = MBO.model.getParameters().size()  

   
    MBO.runSGD(10)
  #  theta = MBO.model.getParameters()
  #  for alpha in range(100):
  #      diff = MBO.getModelDiscrepancy( theta + alpha * torch.randn(dim_theta) / 100. )
  #      print (diff)
    MBO.runADMM()
 #   MBO.run(innerIterations=args.iterations) 




