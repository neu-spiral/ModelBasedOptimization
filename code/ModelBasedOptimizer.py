import argparse
import time
import numpy as np
import pickle
from torch import distributed, nn
import os
import  torch.utils
from torchvision import datasets, transforms
from Net import AEC, DAEC, Linear, ConvAEC
from torch.utils.data import Dataset, DataLoader
from ADMM import LocalSolver, solveConvex, solveWoodbury, solveQuadratic, OADM, InnerADMM
from torch.nn.parallel import DistributedDataParallel as DDP
from helpers import clearFile, dumpFile, estimate_gFunction, loadFile
import logging
import torch.optim as optim
from datasetGenetaor import labeledDataset, unlabeledDataset
from Real_datasetGenetaor import dropLabelDataset


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
        #**
        self.model = self.model.to(device)
        #Synch model parameters across processes
        self._synchParameters()
         
        
        #initialize ADMM solvers
        self.ADMMsolvers = []
        for ind, data in  enumerate(data_loader):
            ADMMsolver = LocalSolver(data=data, rho=rho, p=p, regularizerCoeff=regularizerCoeff, model=self.model)
            self.ADMMsolvers.append( ADMMsolver )

        self.dataset_size = len(self.ADMMsolvers)
        logger.info("Initialized {} ADMMsolvers".format( self.dataset_size )) 


        #Instantiate (global) convex solvers
        #self.globalSolver = solveConvex(model, rank)
        #logger.info("Initialized the global solver for updating Theta.")

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
            for parameter in parameters:
                torch.distributed.broadcast(parameter, 0)
            self.model.setParameters( parameters )
            logger.info("Synchronized model parameters across processes.")


    def runADMM(self, ADMMsolvers, iterations=50, eps=1e-04):
        """
          Execute the ADMM algrotihm for the current model function.
        """

        #Instantiate (global) convex solvers
        globalSolver = solveQuadratic(model, self.rank)
       
        #Initialize solver variables
        for ADMMsolver in ADMMsolvers:
            print("setting up")
            ADMMsolver.setVARS(primalTheta = globalSolver.primalTheta, Theta_k = globalSolver.Theta_k, quadratic = True)
        
        t_start = time.time()
        #trace to keep trak of progress
        trace = {}
        for k in range(iterations):
            t_start_k = time.time()
            PRES_TOT = 0.0
            DRES_TOT = 0.0
            OBJ_TOT = 0.0
            squaredLoss =  0.5 * (globalSolver.primalTheta - globalSolver.Theta_k).frobeniusNormSq() 

            #Update Y and adapt duals for each solver 
            last = time.time()
            for solver_ind, ADMMsolver in enumerate(ADMMsolvers):

                #Eval objective for each term (solver)
                OBJ_TOT += ADMMsolver.getObjective()

                #Eval residuals for each term (solver)
                DRES, PRES = ADMMsolver.updateYAdaptDuals( self.g_est )

                PRES_TOT += PRES ** 2
                DRES_TOT += DRES ** 2

                print(solver_ind)

            now = time.time()
            logger.debug('Updated primal Y variables in {}(s)'.format(now - last) )

            #Aggregate first_ord_TOT and second_ord_TOT across processes
            if self.rank != None:
                torch.distributed.all_reduce(PRES_TOT)
                torch.distributed.all_reduce(DRES_TOT)
                torch.distributed.all_reduce(OBJ_TOT)
                #log information 
                logger.debug('Reduction took {}(s)'.format(time.time() - now))

            last = time.time()
            #Update theta via convexSolvers
            DRES_theta = globalSolver.updateTheta( ADMMsolvers)

            logger.debug("Updates Theta variables in {}(s)".format(time.time() - last) )

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
            
            logger.debug("Iteration {0} is done in {1:.2f} (s), OBJ is {2:.4f}".format(k, time.time() - t_start_k, OBJ_TOT ))
            logger.debug("Iteration {0}, PRES is {1:.4f}, DRES is {2:.4f}".format(k, PRES_TOT, DRES_TOT) )

            #terminate if desired convergence achieved
            if k ==  iterations - 1 or (PRES_TOT <= eps and DRES_TOT <= eps):
                #Evaluate delta (model improvement) 
                delta_TOT = 0.0
                for ADMMsolver in ADMMsolvers:
                    if self.p == -2:
                        delta_TOT += ( ADMMsolver.evalModelLoss( ADMMsolver.primalTheta  ) - torch.norm(ADMMsolver.output, p=2) ** 2 )
                    else:
                        delta_TOT += ( ADMMsolver.evalModelLoss( ADMMsolver.primalTheta  ) - torch.norm(ADMMsolver.output, p=self.p) )

                #if running multiple processes sum up accross processes
                if self.rank != None:
                    torch.distributed.all_reduce(delta_TOT)
                delta_TOT += (globalSolver.primalTheta - globalSolver.Theta_k).frobeniusNormSq() 

                if delta_TOT < 0:
                    break
 
        #log last iteration stats
        logger.info("Inner ADMM done in {0} iterations, took  {1:.2f}(s), final objective is {2:.4f}".format(k, time.time() - t_start, OBJ_TOT ))
        logger.info("PRES is {1:.4f}, DRES is {2:.4f}".format(k, PRES_TOT, DRES_TOT) )
                        
        return globalSolver.primalTheta, delta_TOT, trace

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
            if self.p == -2:
                OBJ_tot += torch.norm(output, p=2) ** 2
            else:
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
    def updateVariable(self, s_k, DELTA, maxiters=100):

        last = time.time()
        delta =0.85
        gamma = 0.1
        eta = 1.0

        obj_k = self.getObjective()
   
        #mult by 1 to make sure that theta_k is not pointing to the paramaters of the model (o.w. modifying model parameters, would modify theta_k too!)
        theta_k = self.model.getParameters() * 1

       # s_k = self.ADMMsolvers[0].primalTheta

        #initial step-size
        stp_size = eta * delta 
        for pow_ind in range(maxiters):
            #new point a convex comb. of theta_k and new point (s_k)
            new_theta = (1.0 - stp_size) * theta_k + stp_size *  s_k

            #eval objective for new_theta
            obj_new = self.getObjective( new_theta )
            #print(obj_new, obj_k + gamma * DELTA)
 
            #check if objective for new_theta is within the desired boundaries r
            if obj_new <= obj_k + gamma * DELTA:
                break 
          
            #shrink step-size
            stp_size *= delta

        now = time.time()
        logger.debug('New step-size found and parameter updated in {0:.2f}(s).'.format(now - last) )
        return obj_new             
        
         
    
        

    def run(self, stopping_eps=1.e-4, iterations=20, innerIterations=50, batch_size=None, l2SquaredSolver='MBO'):
        if self.p == -2 and l2SquaredSolver != 'MBO':
            logging.info('Solving ell2 norm squared via {}'.format(l2SquaredSolver) )
            return self.MSE(epochs=iterations)


        #setup batchs
        if batch_size == None:
            solver_indicies = [range( self.dataset_size )]
        else:
            intervals = list( np.arange(0, self.dataset_size, batch_size) )
            if self.dataset_size - 1 not in intervals:
                intervals.append(self.dataset_size - 1 )
            solver_indicies = [range(intervals[int_i], intervals[int_i + 1] ) for int_i in range(len(intervals) -1 )]


        #Accuracy for inner problem 
        eps_init = 1.e0
        #Factor with which increase the accuracy at each iter
        eps_factor = 0.9

        #Keep track of progress in trace
        trace = {}
        t_start = time.time()
        trace[0] = {}
        trace[0]['OBJ'] =  self.getObjective()
        trace[0]['time'] = time.time() - t_start 
        trace[0]['DELTA'] = 0.0
        logger.info('Outer iteration 0, OBJ is {0:.4f}'.format(trace[0]['OBJ'] ) )

        #main loop
        for k in range(1, iterations+1):
            #set the required accuracy for ADMM 
            eps = eps_init * eps_factor ** (k-1)

            #initialize model improvment 
            model_improvement_TOT = 0.0

            #time 
            t_start = time.time()

           

            #run ADMM for batches
            for interval_index, indicies in enumerate(solver_indicies): 


                #run ADMM algortihm
                newTheta, model_improvement, admm_trace = self.runADMM(ADMMsolvers=self.ADMMsolvers[indicies[0]: indicies[-1]], iterations=innerIterations, eps=eps )

                #increment/initialize newTheta
                if interval_index == 0:
                    newTheta_AVG = newTheta / self.dataset_size
                else:
                    newTheta_AVG += newTheta / self.dataset_size

                #increment model improvement
                model_improvement_TOT += model_improvement
            
                #log info
                logger.info('Batch number {0}, time spent in this iteration is {1:.1f}, model improvement {2:.4f}'.format(interval_index, time.time() - t_start,  model_improvement_TOT) )

            #update optimization variable via Armijo rule and set model parameters
            self.updateVariable(s_k=newTheta_AVG, DELTA=model_improvement_TOT )

            #log and keep track of stats
            OBJ = self.getObjective()
            logger.info('Outer iteration {0}, OBJ is {1:.4f}, model improvement {2:.4f}'.format(k, OBJ, model_improvement) )
            trace[k] = {}
            trace[k]['OBJ'] = OBJ
            trace[k]['DELTA'] = model_improvement
            trace[k]['time'] = time.time() - t_start

            #terminate if model improvement (a negative number) is within stopping_eps
            if abs(model_improvement) < stopping_eps:
                break
        return trace 
           
            
     
                  
               
     
        





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--m_dim", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=8)
    parser.add_argument("--logfile", type=str,default="logfiles/proc")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size for processing dataset, when None it is set equal to the dataset size.")
    parser.add_argument("--inner_iterations", type=int,  default=40)
    parser.add_argument("--iterations", type=int,  default=50)
    parser.add_argument("--rho", type=float, default=1.0, help="rho parameter in ADMM")
    parser.add_argument("--p", type=float, default=2, help="p in lp-norm")
    parser.add_argument("--tracefile", type=str, help="File to store traces.")
    parser.add_argument("--outfile", type=str, help="File to store model parameters.")
    parser.add_argument("--logLevel", type=str, choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument("--net_model", choices=['Linear', 'AEC', 'DAEC', 'ConvAEC'], default='AEC')
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
    logFile = args.logfile + "_process" + str(args.local_rank)
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
    oadm_solver = OADM(dataset, rho=5.0, p=2, h=1.0, gamma=1.0, regularizerCoeff = 0.5, batch_size = 10, model = model, theta_k = model.getParameters() )



    #oadm_solver.updateTheta1viaSGD(A, b, c, coeff)

    #oadm_solver.run(iterations = 100, logger = logger)

    oadm_solver.runSGD()


    

  
    #initialize a model based solver
#    MBO = ModelBasedOptimzier(dataset=dataset, model=model, rank=args.local_rank, rho=args.rho, p=args.p, g_est=g_est)
    #trace =  MBO.run(iterations = args.iterations, innerIterations=args.inner_iterations, batch_size=args.batch_size, l2SquaredSolver=args.l2SquaredSolver)
 #   dumpFile(args.tracefile, trace)

    #save model parameters
 #   model.saveStateDict( args.outfile )
   
 #   theta, trace_SGD =  MBO.runSGD(args.inner_iterations, args.batch_size)
 #   delta, trace_ADMM = MBO.runADMM(ADMMsolvers=MBO.ADMMsolvers, iterations=args.inner_iterations )
    

 #   with open(args.tracefile + "_sgd{}".format(args.batch_size),'wb') as f:
 #       pickle.dump(trace_SGD,  f)

 #   with open(args.tracefile + "_admm",'wb') as f:
 #       pickle.dump(trace_ADMM,  f)




