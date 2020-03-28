import argparse
import time
import pickle
from torch import distributed, nn
import os
import  torch.utils
from torchvision import datasets, transforms
from Net import AEC, Linear
from torch.utils.data import Dataset, DataLoader
from ADMM import ADMM
from torch.nn.parallel import DistributedDataParallel as DDP
from helpers import clearFile, dumpFile, estimate_gFunction
import logging
import torch.optim as optim


#torch.manual_seed(1993)

class ModelBasedOptimzier:
    """
       This is a generic class for solving problems of the form
       Minimize \sum_i || F_i(Theta) ||_p + g(theta) 
       via the model based method proposed by Ochs et al. in 2018. 
    """

    def __init__(self, dataset, model, rho=5.0, p=2, rank=None, regularizerCoeff=0.0, g_est=None):
        #If rank is None the execution is serial. 
        self.rank = rank

        self.dataset = dataset 
        #load dataset
        if rank != None:
           data_sampler  = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
        else:
           data_sampler = None
        data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=1)


        #estimator for g function (used in runADMM)
        self.g_est = g_est
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
        logger.info("Initialized {} ADMMsolvers".format( ind +1 )) 
        #p is the parameter in lp-norm
        #NOTE: here, p = -2  means l2-norm squared
        if p == -2:
            logging.warning("Setting p to -2, it means that l2-norm squared is used.")
            logging.warning("l2-norm squared is not implemented for the model based solver, it is only for debugging the ADMM solver.")
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
            logger.info("Synchronized model parameters across processes.")


    def runADMM(self, iterations=50, eps=1e-04):
        """
          Execute the ADMM algrotihm for the current model function.
        """
       
        #Initialize solver variables
        for ADMMsolver in self.ADMMsolvers:
            ADMMsolver._setVARS()
        
        t_start = time.time()
        #trace to keep trak of progress
        trace = {}
        for k in range(iterations):
            t_start = time.time()
            PRES_TOT = 0.0
            DRES_TOT = 0.0
            OBJ_TOT = 0.0
            squaredLoss =  0.5 * torch.norm(self.ADMMsolvers[0].primalTheta - self.ADMMsolvers[0].Theta_k) ** 2
            first_ord_TOT = 0.0
            second_ord_TOT = 0.0

            #Update Y and adapt duals for each solver 
            last = time.time()
            for ADMMsolver in self.ADMMsolvers:
                #Eval objective for each term (solver)
                OBJ_TOT += ADMMsolver.evalModelLoss()
                #Eval residuals for each term (solver)
                DRES, PRES = ADMMsolver.updateYAdaptDuals( self.g_est )
                #Eval first and second order terms for each term (solver)
                first_ord, second_ord =  ADMMsolver.getCoeefficients()


                PRES_TOT += PRES ** 2
                DRES_TOT += DRES ** 2

                first_ord_TOT  += first_ord
                
                second_ord_TOT += second_ord

            now = time.time()
            logger.debug('Updated primal Y variables in {}(s)'.format(now - last) )
            #Aggregate first_ord_TOT and second_ord_TOT across processes
            now = time.time()

            if self.rank != None:
                torch.distributed.all_reduce(first_ord_TOT)
                torch.distributed.all_reduce(second_ord_TOT)
                torch.distributed.all_reduce(PRES_TOT)
                torch.distributed.all_reduce(DRES_TOT)
                torch.distributed.all_reduce(OBJ_TOT)


                logger.debug('Reduction took {}(s)'.format(time.time() - now))

            #Compute Theta (proc 0 is responsible for this)
            ADMMsolver_i = self.ADMMsolvers[0]
            DRES_theta = ADMMsolver_i.updateTheta(first_ord_TOT, second_ord_TOT)
            #Update Theta for the rest of the solvers across processes
            for ADMMsolver in self.ADMMsolvers[1:]:
                ADMMsolver.updateTheta(first_ord_TOT, second_ord_TOT, self.ADMMsolvers[0].primalTheta)
            DRES_TOT += DRES_theta ** 2

            #square root 
            PRES_TOT = PRES_TOT.item() ** 0.5
            DRES_TOT = DRES_TOT.item() ** 0.5 
            #Add the quadratic term to OBJ
            OBJ_TOT += squaredLoss 
 
            t_now = time.time()
            trace[k] = {}
            trace[k]['OBJ'] = OBJ_TOT.item()
            trace[k]['PRES'] = PRES_TOT
            trace[k]['DRES'] = DRES_TOT
            trace[k]['time'] = t_now - t_start
            logger.debug("Iteration {0} is done in {1:.2f} (s), OBJ is {2:.4f}".format(k, time.time() - t_start, OBJ_TOT ))
            logger.debug("Iteration {0}, PRES is {1:.4f}, DRES is {2:.4f}".format(k, PRES_TOT, DRES_TOT) )

            #terminate if desired convergence achieved
            if PRES_TOT <= eps and DRES_TOT <= eps:
                break
 
        #log last iteration stats
        logger.info("Iteration {0} is done in {1:.2f} (s), OBJ is {2:.4f}".format(k, time.time() - t_start, OBJ_TOT ))
        logger.info("Iteration {0}, PRES is {1:.4f}, DRES is {2:.4f}".format(k, PRES_TOT, DRES_TOT) )
        #Evaluate delta (model improvement) 
        delta_TOT = 0.0
        for ADMMsolver in self.ADMMsolvers:
            delta_TOT += ( ADMMsolver.evalModelLoss( ADMMsolver.primalTheta  ) - torch.norm(ADMMsolver.output, p=self.p) )
        if self.rank != None:
            torch.distributed.all_reduce(delta_TOT)
        delta_TOT += torch.norm(ADMMsolver.primalTheta - ADMMsolver.Theta_k, p=2) ** 2
                        
        return delta_TOT, trace

    def runSGD(self, epochs=1, batch_size=8):
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

        t_start = time.time() 
        #Keep track of progress in trace
        trace = {}
        trace[0] = {}
        init_loss = 0.0
        for ind, solver_i in enumerate(self.ADMMsolvers):
            #loss due to the i-th datapoint
            init_loss += solver_i.evalModelLoss( theta ).item()

        trace[0]['OBJ'] = init_loss
        trace[0]['time'] = time.time() - t_start

        for i in range(1, epochs+1):
            #Proximity loss ||theta - theta_k||_2^2
            sq_loss = 0.5 * squaredLoss(theta, theta_k)
            #Keep track of loss throught the iterations 
            running_loss = sq_loss.item()
            for ind, solver_i in enumerate(self.ADMMsolvers):
                #loss due to the i-th datapoint
                loss_i = solver_i.evalModelLoss( theta)


                if ind == 0:
                    loss = loss_i + sq_loss

                elif ind == len(self.ADMMsolvers) - 1:
                    #Increment the loss
                    loss = loss + loss_i
                    #zero the parameter gradients
                    optimizer.zero_grad()

                    #backprop 
                    loss.backward(retain_graph=False)
                    #SGD step 
                    optimizer.step()
                    
                elif ind > 0 and ind % batch_size == 0:
                    #zero the parameter gradients
                    optimizer.zero_grad()

                    #backprop 
                    loss.backward(retain_graph=False)
                    #SGD step 
                    optimizer.step()
                    #Compute the loss 
                    loss = loss_i
                else:
                    #Increment the loss
                    loss = loss + loss_i              

                running_loss += loss_i.item()
                   
            t_now = time.time()
            trace[i] = {}
            trace[i]['OBJ'] = running_loss
            trace[i]['time'] = t_now - t_start
            logger.info("Epoch {0}, loss is {1:.4f}".format(i, running_loss) )
        return theta.data, trace
                

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
            logger.warning("Evaluting the objective, this will modify model parameters.")
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
        delta =0.85
        gamma = 0.1
        eta = 1.0

        obj_k = self.getObjective()
        theta_k = self.ADMMsolvers[0].Theta_k
        s_k = self.ADMMsolvers[0].primalTheta
        stp_size = eta * delta 
        while True:
            new_theta = (1.0 - stp_size) * theta_k + stp_size *  s_k
            obj_new = self.getObjective( new_theta )
            print(obj_new, obj_k + gamma * DELTA)
            if obj_new <= obj_k + gamma * DELTA:
                break 
            stp_size *= delta
        now = time.time()
        logger.debug('New step-size found and parameter updated in {0:.2f}(s).'.format(now - last) )
        return obj_new             
        
         
    
        

    def run(self, iterations=20, innerIterations=50):
        if self.p < 1:
            raise Exception("Please enter a value for p that is greater than or eqaul to 1.")
        trace = {}
        t_start = time.time()
        trace[0] = {}
        trace[0]['OBJ'] =  self.getObjective()
        trace[0]['time'] = time.time() - t_start 
        trace[0]['DELTA'] = 0.0
        logger.info('Outer iteration 0, OBJ is {0:.4f}'.format(trace[0]['OBJ'] ) )
        for k in range(1, iterations+1):
            #run ADMM algortihm
            model_improvement, admm_trace = self.runADMM( innerIterations )
            
            #update optimization variable via Armijo rule and set model parameters
            self.updateVariable( model_improvement ) 

            #log and keep track of stats
            OBJ = self.getObjective()
            logger.info('Outer iteration {0}, OBJ is {1:.4f}, model improvement {2:.4f}'.format(k, OBJ, model_improvement) )
            trace[k] = {}
            trace[k]['OBJ'] = OBJ
            trace[k]['DELTA'] = model_improvement
            trace[k]['time'] = time.time() - t_start
        return trace 
           
            
     
                  
               
     
        





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=8)
    parser.add_argument("--logfile", type=str,default="logfiles/proc")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--inner_iterations", type=int,  default=40)
    parser.add_argument("--iterations", type=int,  default=50)
    parser.add_argument("--rho", type=float, default=1.0, help="rho parameter in ADMM")
    parser.add_argument("--p", type=float, default=2, help="p in lp-norm")
    parser.add_argument("--tracefile", type=str, help="File to store traces.")
    parser.add_argument("--logLevel", type=str, choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'], default='INFO')
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
    logFile = args.logfile + str(args.local_rank)
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
    model = AEC(args.m, args.m_prime, device=device)
    #model = Linear(args.m, args.m_prime)
    model = model.to(device)

    #data 
    dataset =  torch.load(args.input_file)
    
    #estimate g function
    if args.p not in [1, 2]:
        g_est = estimate_gFunction(args.p)
  
    #initialize a model based solver
    MBO = ModelBasedOptimzier(dataset=dataset, model=model, rank=args.local_rank, rho=args.rho, p=args.p, g_est=g_est)
    trace =  MBO.run(iterations = args.iterations, innerIterations=args.inner_iterations)
    dumpFile(args.tracefile, trace)
   
 #   theta, trace_SGD =  MBO.runSGD(args.inner_iterations, args.batch_size)
 #   delta, trace_ADMM = MBO.runADMM(args.inner_iterations )
    

 #   with open(args.tracefile + "_sgd{}".format(args.batch_size),'wb') as f:
 #       pickle.dump(trace_SGD,  f)

 #   with open(args.tracefile + "_admm",'wb') as f:
 #       pickle.dump(trace_ADMM,  f)




