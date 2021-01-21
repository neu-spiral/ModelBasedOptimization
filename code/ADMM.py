import argparse 
import time
import logging
import numpy as np
import logging
from Net import AEC, DAEC, TensorList
from torch.utils.data import Dataset, DataLoader
import torch
from helpers import pNormProxOp, clearFile, estimate_gFunction, loadFile, _testOpt
from  datasetGenetaor import unlabeledDataset
import torch.nn as nn
from torch.multiprocessing import Process
import torch.distributed as dist
import os
import math

#torch.manual_seed(1993)

#@torch.no_grad()
class InnerADMM():
    def __init__(self, A, sqA, b, c, coeff, p, model, init_solution = None, rho_inner = 1.0):
        """
            A class that implements ADMM for generic problems of the form:

                Minimize ∑_i ||A_ix + b_i||_p + coeff * ||x - c||_2^2.
            Args:
                A: a list of matrices A_i, where each element (A_i) is a TensorList
                sqA: the sum of squared A_i matrices, a d by d Tensor
                b: a list of vectors b_i, where each element (b_i) is a Tensor.
                c: a TensorList

                p: a float number, showing the p-norm.
                coeff: a positive float number
                model: neural network model

                init_solution (optional): initial solution given as a TensorList
                rho_inner (optional): a psoitive float number that shows the rho parameters in ADMM
        """

        self.A = A
        self.sqA = sqA
        self.b = b
        self.c = c
        self.coeff = coeff

        self.model = model

        self.rho_inner = rho_inner
        self.p = p

        #set primal and dual variables
        if init_solution is None:
            self.x = self.c * 0.0
        else:
            self.x = init_solution

        #estimate g function
        if self.p not in [1, 2, -2]:
            self.g_est = estimate_gFunction(self.p)
        else:
            self.g_est = None
    
        self.y = []
        self.z = []

        for ind, A_i in enumerate(self.A):

            y_i = model.vecMult(self.x, Jacobian = A_i) + self.b[ ind ]

            self.y.append( y_i )

            z_i = y_i * 0.0
    
            self.z.append( z_i )
    
    @torch.no_grad()
    def run(self, iterations = 100, eps = 1.e-4, debug = True, logger = logging.getLogger('Inner ADMM')):

        #compute the second order term in computing x 
        seqcon_ord_term = 2 * self.coeff * torch.eye( self.sqA.shape[0] )  + self.rho_inner * self.sqA 

        #compute the inverse of the second order term
        seqcon_ord_term_inv = torch.inverse( seqcon_ord_term )
        
    
        for k in range(iterations):
            t_st = time.time() 

            for ind, A_i in enumerate(self.A):
                #compute the affine function
                Ax_plus_b_i = self.model.vecMult(self.x, Jacobian = A_i) + self.b[ ind ]



                self.z[ind] = self.z[ind] + self.y[ ind ] - Ax_plus_b_i

                
                #old y
                old_y_i = self.y[ ind ]

      
                #update y via prox. operator for p-norms                   
                self.y[ ind ] = pNormProxOp( Ax_plus_b_i - self.z[ ind ], self.rho_inner, g_est = self.g_est, p = self.p) 

                #sum up first order terms
                if ind == 0:

                    first_ord_term = self.model.vecMult( self.b[ ind ] - self.z[ ind ] - self.y[ ind ], Jacobian = A_i, left = True )
 
                    #in debug mode compute objective and residuals
                    if debug:
                    
                        PRES = torch.norm(self.y[ ind ] - Ax_plus_b_i, p=2) ** 2

                        DRES = torch.norm(self.y[ ind ] - old_y_i, p=2) ** 2

                        OBJ = torch.norm(self.y[ ind ], p = self.p)


                else:
                    first_ord_term += self.model.vecMult( self.b[ ind ] - self.z[ ind ] - self.y[ ind ], Jacobian = A_i, left = True )

                    #in debug mode compute objective and residuals
                    if debug:

                        PRES += torch.norm(self.y[ ind ] - Ax_plus_b_i, p=2) ** 2

                        DRES += torch.norm(self.y[ ind ] - old_y_i, p=2) ** 2

                        OBJ += torch.norm(self.y[ ind ], p = self.p)


            #update x 
            first_ord_tot =  - self.rho_inner * first_ord_term + 2 * self.coeff * self.c

            first_ord_tot = first_ord_tot.getTensor()

            self.x = torch.matmul(seqcon_ord_term_inv, first_ord_tot )

            self.x = TensorList.formFromTensor(self.x, self.c.TLShape() )

            OBJ += self.coeff * (self.x - self.c).frobeniusNormSq()

            if debug:
                logger.info("{}-th iterations of Inner ADMM done in {:.2f}.".format(k, time.time() - t_st) )
                logger.info("Objective is {} and primal and dual residuals are {} and {}, respectively.".format(OBJ, PRES, DRES) )

            if PRES <= eps and DRES <= eps:
                break

        return self.x 
                 
 
                


                                 
class ParInnerADMM(InnerADMM):

    def init_processes(self, rank, world_size, iterations, eps, debug, logger, backend = 'gloo'):

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        dist.init_process_group(backend, rank=rank, world_size = world_size)

        self._runADMM(rank, world_size, iterations = iterations, eps = eps, debug = debug, logger = logger)

    def _runADMM(self, rank, world_size, iterations = 100, eps = 1.e-4, debug = True, logger = logging.getLogger('Inner ADMM')):

        logger.info("Starting Parallel Inner ADMM")

        #partition indices to be processed by each rank
        partitions = {}
        
        st_ind = 0

        #number of indices to be processed by each rank
        partition_size = math.ceil(len(self.A) / world_size)

        for r in range(world_size):
           
            end_ind = min( partition_size + st_ind, len(self.A) )

            partitions[r] = range(st_ind, end_ind)
   
            st_ind =  end_ind



        #main iterations 
        for k in range(iterations):


            #go over the corresponding indices 
            for ind in partitions[rank]:

                A_i = self.A[ind]

                #compute the affine function
                Ax_plus_b_i = self.model.vecMult(self.x, Jacobian = A_i) + self.b[ ind ]



                self.z[ind] = self.z[ind] + self.y[ ind ] - Ax_plus_b_i


                #old y
                old_y_i = self.y[ ind ]


                #update y via prox. operator for p-norms                   
                self.y[ ind ] = pNormProxOp( Ax_plus_b_i - self.z[ ind ], self.rho_inner, g_est = self.g_est, p = self.p)

                #sum up first order terms
                if ind == partitions[rank][0]:

                    first_ord_term = self.model.vecMult( self.b[ ind ] - self.z[ ind ] - self.y[ ind ], Jacobian = A_i, left = True )

                    #in debug mode compute objective and residuals
                    if debug:

                        PRES = torch.norm(self.y[ ind ] - Ax_plus_b_i, p=2) ** 2

                        DRES = torch.norm(self.y[ ind ] - old_y_i, p=2) ** 2

                        OBJ = torch.norm(self.y[ ind ], p = self.p)


                else:
                    first_ord_term += self.model.vecMult( self.b[ ind ] - self.z[ ind ] - self.y[ ind ], Jacobian = A_i, left = True )

                    #in debug mode compute objective and residuals
                    if debug:

                        PRES += torch.norm(self.y[ ind ] - Ax_plus_b_i, p=2) ** 2

                        DRES += torch.norm(self.y[ ind ] - old_y_i, p=2) ** 2

                        OBJ += torch.norm(self.y[ ind ], p = self.p)


            #sum up first-order terms and residuals across processes
            for first_ord_tens in first_ord_term:
                dist.all_reduce(first_ord_tens, op=dist.reduce_op.SUM)
 
            if debug:
                dist.all_reduce(PRES,  op=dist.reduce_op.SUM)
                dist.all_reduce(DRES,  op=dist.reduce_op.SUM)
                dist.all_reduce(OBJ,  op=dist.reduce_op.SUM)

                if PRES <= eps and DRES <= eps:
                    break

            #rank 0 updates x 
       
            if rank == 0:
                first_ord_tot =  - self.rho_inner * first_ord_term + 2 * self.coeff * self.c

                first_ord_tot = first_ord_tot.getTensor()

                new_x = torch.matmul(self.seqcon_ord_term_inv, first_ord_tot )

                self.x = TensorList.formFromTensor(new_x, self.c.TLShape() )

                OBJ += self.coeff * (self.x - self.c).frobeniusNormSq()

                if debug:
                    logger.info("Objective is {} and primal and dual residuals are {} and {}, respectively.".format(OBJ, PRES, DRES) )
   
            #synch x (updated by rank 0)
            for x_tens in self.x:
                torch.distributed.broadcast(x_tens, src = 0)
                        


    def run(self, world_size = 1, iterations = 100, eps = 1.e-4, debug = True, logger = logging.getLogger('Inner ADMM')):
            

        #compute the second order term in computing x 
        seqcon_ord_term = 2 * self.coeff * torch.eye( self.sqA.shape[0] )  + self.rho_inner * self.sqA

        #compute the inverse of the second order term
        self.seqcon_ord_term_inv = torch.inverse( seqcon_ord_term )


        processes = []


        #spawn processes
        for rank in range(world_size):
            p = Process(target = self.init_processes, args = (rank, world_size, iterations, eps, debug, logger))

            p.start()
            processes.append(p)

        for p in processes:
            p.join()


        return self.x

class OADM():
    def __init__(self, data, model, rho=1.0, p=2, h=1.0, gamma=1.0, regularizerCoeff=1.0, rho_inner = 1.0,  batch_size = 8):
        """
             A class that implements OADM algrotithm for solving problems of the form:

                 Minimize ∑_i ||F_i(θ^k) + D_Fi(θ^k)(θ1 - θ^k)||_p + h/2 ||θ1 - θ^k||_2^2 + g(θ2) 

                 Subj. to: θ2 ∈ C
                           θ1 = θ2,

             where in the base class g(θ2) = regularizerCoeff * ||θ2||_2^2 and C is all real numbers. Render getGfunc and updateTheta2 to for other cases. 

             Args:
                 data: dataset
                 rho: rho parameter for used in OADM algrotihm
                 gamma: coefficient for squared term used in OADM
                 h: coefficient for the quadratic term around the current solution
                 regularizerCoeff: coefficient for regularization
                 model: torch model 
                 p: p-norm to use
                 theta_k: current parameters of model
        """

        self.data = data
        self.model = model

        self.rho = rho
        self.p = p
        self.h = h
        self.gamma = gamma
        self.rho_inner = rho_inner
        self.batch_size = batch_size
        self.regularizerCoeff = regularizerCoeff


        self.data_loader = DataLoader(data, batch_size = batch_size, shuffle = True)

 
        #initialize optimization variables 
        self.theta1 = self.model.getParameters() * 0
        self.theta2 =  self.model.getParameters() * 0
        self.dual = self.model.getParameters() * 0


    def _getInnerADMMCoefficients(self, data_batch, quadratic = True):
        """
            Compute the matrices and vectors for the problem to be solved via ADMM, i.e., 
            Minimize 1/B ∑_i ||A_ix + b_i||_p + \lambda ||x - c||_2^2
        """


        #initialize A and b 
        b = []
        A = []

            

        #forward pass over the loaded batch
        for ind in range(self.batch_size):

            #NOTE: labeld data
            if type( data_batch ) == list:
                data_i = torch.unsqueeze(data_batch[0][ ind ], 0), torch.unsqueeze(data_batch[1][ ind ], 0)
                              

            else:
                data_i = data_batch[ ind ]
                #add batch dimension
                data_i = torch.unsqueeze(data_i, 0)

             
            #forward pass
            F_i = self.model( data_i)

            #compute Jacobian and its square
            if quadratic:
                D_i, sqD_i = self.model.getJacobian(F_i, quadratic = quadratic)

            else:
                D_i = self.model.getJacobian(F_i, quadratic = quadratic)

            #compute A and b
            with torch.no_grad():

                b.append(  F_i - self.model.vecMult( vec = self.theta_k, Jacobian = D_i ) )
                
                A.append( D_i )

                if quadratic:
                    if ind == 0:
                        sqA_sum = sqD_i
                    else:
                        sqA_sum += sqD_i

                else:
                    sqA_sum = None

                

        with torch.no_grad():


            c = self.rho / (self.rho + self.gamma + self.h) * (self.theta2 - self.dual) + \
                self.gamma / (self.rho + self.gamma + self.h)  * self.theta1 + \
                self.h / (self.rho + self.gamma + self.h) * self.theta_k
            
            Blambda = self.batch_size * (self.rho + self.gamma + self.h) / 2


            return A, sqA_sum, b, c, Blambda, self.theta1 * 1.0

    def _getInnerADMMCoefficientsPAR(self, world_size, data_batch, quadratic = True):
        
        with torch.no_grad():
            c = self.rho / (self.rho + self.gamma + self.h) * (self.theta2 - self.dual) + \
                self.gamma / (self.rho + self.gamma + self.h)  * self.theta1 + \
                self.h / (self.rho + self.gamma + self.h) * self.theta_k

            Blambda = self.batch_size * (self.rho + self.gamma + self.h) / 2


        processes = []


        #spawn processes
        for rank in range(world_size):
            p = Process(target = self.init_processes_and_getJac, args = (rank, world_size) )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()


    def init_processes_and_getJac(self, rank, world_size, backend='gloo'):

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        dist.init_process_group(backend, rank=rank, world_size = world_size)

         
        


    def getObj(self, theta1, theta2, data_batch = None):
        """
            Compute the obejective 
                  Minimize 1/n ∑_i ||A_i * θ1 + b_i||_p + h/2 * ||θ1 - θ_k||_2^2 + G(θ2),
            where the first summation is over all points.
        """

        if data_batch is None:
            #load whole data
            data_batch = next( iter( DataLoader(self.data, batch_size = len( self.data ) ) ) )
     
        #NOTE: labeled datasets
        if type( data_batch ) == list:
            data_batch_size = data_batch[0].shape[0]
        else:
            data_batch_size = data_batch.shape[0]

        #compute all A and b matrices and vectors
        A, sqA_sum, b, c, coeff, init_sol = self._getInnerADMMCoefficients(data_batch, quadratic = False )        


        #backward propagation is not needed for the rest of computations
        with torch.no_grad():
            #add the anchoring and the regulrization terms
            OBJ = self.h / 2 * (theta1 - self.theta_k). frobeniusNormSq() + self.getGfunc( theta2 )

            for ind, A_i in enumerate(A):
                #compute model functions for each datapoint
                OBJ += torch.norm( self.model.vecMult( vec = theta1 , Jacobian = A_i ) + b[ ind ], p = self.p) / data_batch_size

            return OBJ
   

    @torch.no_grad()
    def updtateTheta2(self):
        """
           Update theta2 via solving:
                Minimize G(theta2) + ρ||theta1^t + dual^t - theta2||_2^2,
           
           where in this base class we set G(θ) = regularizerCoeff/2 * ||θ||_2^2.
        """

        return self.rho / (self.rho + self.regularizerCoeff) * (self.theta1 + self.dual)

    def getGfunc(self, theta):
        """
            Compute the G function, i.e., terms in objective that corresponds to θ2.

                
        """
        return self.regularizerCoeff / 2 * theta.frobeniusNormSq()

    def _updateTheta1viaSGD(self, A, b, c, coeff):

        #get current model vraibales
        theta_VAR = self.model.getParameters(trackgrad=True)


        #initialize thet_VAR to primalTheta
        for var_ind, var in enumerate( self.theta1 ):
            theta_VAR[var_ind].data.copy_( var )

 
        optimizer = torch.optim.SGD(theta_VAR, lr=0.01, momentum=0.9)
        

        for _ in range(100):
            optimizer.zero_grad()

             
            loss = coeff *  (theta_VAR - c).frobeniusNormSq()



            for ind, A_i in enumerate(A):
                loss += torch.norm(self.model.vecMult(vec = theta_VAR,  Jacobian = A_i ) + b[ind] , p = self.p ) 
    

            loss.backward()                    
            
            optimizer.step()


    def runSGD(self, iterations = 100, logger = logging.getLogger('SGD'), lr = 1e-2, momentum = 0.0, debug = False):

        logger.info("Starting SGD iterations.")

        #get current model vraibales
        theta_VAR = self.model.getParameters(trackgrad=True) 

        

        #get current theta_k (i.e., model parameters)
        self.theta_k = self.model.getParameters() * 1


        #compute the objective around the anchor point theta_k
        self.OBJ_theta_k = self.getGfunc( self.theta_k )

        with torch.no_grad():
            for data_i in self.data_loader:
                self.OBJ_theta_k += torch.sum( torch.norm( self.model( data_i ), p = self.p, dim = 1)  ) / len( self.data )


        optimizer = torch.optim.SGD(theta_VAR, lr = lr, momentum = momentum)


        #keep track of trajectories
        trace = {'OBJ': [], 'time': []}
        t_start = time.time()

        #get iterable dataloader 
        iterableData = iter( self.data_loader )

        for k in range(iterations):

            optimizer.zero_grad()

            loss = self.h / 2 * (theta_VAR - self.theta_k).frobeniusNormSq() +  self.getGfunc( theta_VAR )

            #load a new data batch
            try:
                data_batch = next( iterableData )
            except StopIteration:
                iterableData = iter( self.data_loader )

            #get terms for running Inner ADMM 
            A, sqA_sum, b, c, coeff, init_sol = self._getInnerADMMCoefficients( data_batch, quadratic = False )
     
            for ind, A_i in enumerate(A):
                loss += torch.norm(self.model.vecMult(vec = theta_VAR,  Jacobian = A_i ) + b[ind] , p = self.p ) / self.batch_size

            loss.backward()

            optimizer.step()
            
            if k % 10 == 0:
                logger.info("{}-th iterations, loss is {:.4f}.".format(k, loss.item()))
 
            if debug:
                OBJ = self.getObj(theta_VAR, theta_VAR)

                logger.info("Full objective is {}".format( OBJ ) )


                #keep trace
                trace['OBJ'].append( OBJ )
                trace['time'].append( time.time() - t_start )

                if k == 0 or OBJ < BEST_loss:
                    BEST_loss = OBJ
                    BEST_var  = theta_VAR * 1 
 

                #compute model improvement
                model_improvement  = self.getModelImprovement( BEST_var )

            else:

                if k == 0 or loss.item() < BEST_loss:
                    BEST_loss = loss.item()
                    BEST_var  = self.model.getParameters() * 1

                    #evaluate an estimation of model improvement 
                    model_improvement = self.OBJ_theta_k - BEST_loss
                

        logger.info("Best computed objective is {:.4f}.".format( BEST_loss ) )


        #NOTE: reset model variables (optimizer modifies model parameters)
        self.model.setParameters( self.theta_k )

        #set variables
        self.theta_bar1 = BEST_var 
       

        return trace, model_improvement
        

    def run(self, iterations = 100, eps = 1.e-3, eval_full_model_freq = 0.05, inner_iterations = 500, inner_eps = 1e-3, logger = logging.getLogger('OADM'), adapt_parameters = True, debug = False, world_size = 1):
        """
            Run the iterations of the OADM algrotihm.
        """ 

        logger.info("Starting to run OADM iterations.")

        eval_full_model_iters = int( 1. / eval_full_model_freq )

        #get current theta_k (i.e., model parameters)
        self.theta_k = self.model.getParameters() * 1

        #initialize primal and dual variables
        self.theta1 = self.model.getParameters() * 0
        self.theta2 = self.model.getParameters() * 0
        self.dual =  self.model.getParameters() * 0

        #initialize average variables
        self.theta_bar1 = self.theta1 * 1
        self.theta_bar2 = self.theta2 * 1


        #compute the objective around the anchor point theta_k
        self.OBJ_theta_k = self.getGfunc( self.theta_k )

        for data_i in self.data_loader:
            self.OBJ_theta_k += torch.sum( torch.norm( self.model( data_i ), p = self.p, dim = 1)  ) / len( self.data )

        #keep track of trajectories
        trace = {'OBJ': [], 'time': [], 'RES': []}

        t_start = time.time()

        #initalize stats   
        k = 0
        model_improvement = 0
        PRES = eps + 1
        DERS = eps + 1
        stoch_OBJ = 0.0   

        #get iterable dataloader 
        iterableData = iter( self.data_loader )

        #OADM iterations
        while k < iterations and (PRES > eps or DRES > eps):
            #start of iteration
            t_start_it = time.time()

            if adapt_parameters:
                #adapt parameters (proportional to strong convexity coefficient)
                self.rho = self.h * (k + 1) 
                self.gamma = self.regularizerCoeff * (k + 1)

            #load a new batch 
            try:
                data_batch = next( iterableData )

            except StopIteration:
                iterableData = iter( self.data_loader )


            #get terms for running Inner ADMM 
            A, sqA_sum, b, c, coeff, init_sol = self._getInnerADMMCoefficients( data_batch )


            #log pre-computation time
            logger.info("Computed Jacobians and cosnatnt terms in {:.1f} (s)".format(time.time() - t_start_it) )            

            with torch.no_grad():


                #inner admm initialization 
                if world_size == 1:
                    InnerADMM_obj = InnerADMM(A = A, sqA = sqA_sum, b = b, c = c, coeff = coeff, p = self.p, model = self.model, init_solution = init_sol, rho_inner = self.rho_inner)
 
                    #update theta1 via InnerADMM
                    self.theta1 = InnerADMM_obj.run( iterations = inner_iterations, eps = inner_eps, logger = logger)

                else:
                    #if number of processes is larger than 1 (parallel setting) initialize the appropriate InnerADMM class
                    InnerADMM_obj = ParInnerADMM(A = A, sqA = sqA_sum, b = b, c = c, coeff = coeff, p = self.p, model = self.model, init_solution = init_sol, rho_inner = self.rho_inner)

                    #update theta1 via InnerADMM
                    self.theta1 = InnerADMM_obj.run( iterations = inner_iterations, eps = inner_eps, world_size = world_size, logger = logger)


                #keep currennt (old) theta2
                old_theta2 = self.theta2 * 1

                #update theta2
                self.theta2 = self.updtateTheta2()

                #adapt dual variables
                self.dual += self.theta1 - self.theta2


                #primal residual
                PRES = (self.theta1 - self.theta2).frobeniusNormSq()

                #dual residual
                DRES = (self.theta2 - old_theta2).frobeniusNormSq()

                #update average values
                self.theta_bar1 = (self.theta_bar1 * k + self.theta1) / (k + 1)
                self.theta_bar2 = (self.theta_bar2 * k + self.theta2) / (k + 1)

                #compute the infeasibility (residual)
                FEAS = (self.theta_bar1 - self.theta_bar2).frobeniusNormSq()

                


            #evaluate objective for sampled data
            stoch_OBJ += self.getObj(self.theta1, self.theta2, data_batch)

            #evaluate an estimation of model improvement 
            model_improvement = self.OBJ_theta_k - stoch_OBJ / (k + 1)

            logger.info("{}-th iterations of OADM is done in {:.1f} (s), PRES and DRES are {:.4f} and {:.4f}, respectively.".format(k, time.time() - t_start_it, PRES, DRES ) )
            logger.info("Infeasibility for average thetas is {:.4f}, avaraged objective value is {:.4f}".format( FEAS, stoch_OBJ / (k + 1) ) )
            logger.info("Model improvement is {:.4f}.".format( model_improvement ) )
  
            if debug:

                #compute the full objective
                OBJ = self.getObj(self.theta_bar1, self.theta_bar1)

                #compute model improvement
                model_improvement  = self.getModelImprovement( self.theta_bar1 )

                #log objective and residual values
                logger.info("OADM, primal and dual residuals are {:.4f} and {:.4f}, respectively.".format(PRES, DRES) )
                logger.info("Full objective for average thetas is {:.4f}".format( OBJ ) )

                #append to trace
                trace['OBJ'].append( OBJ )
                trace['RES'].append( RES )
                trace['time'].append( time.time() - t_start ) 

      
            
            #increment k
            k += 1    

        return trace, model_improvement
    
    def getModelImprovement(self, theta_var):
        """
            Compute the improvement of the objective function (i.e., the model function) w.r.t. anchor point theta_k.
        """  

        OBJ = self.getObj(theta_var, theta_var) 

        return self.OBJ_theta_k - OBJ



            

class LocalSolver():
    "LocalSolver for executing steps of ADMM."
    def __init__(self, data, rho=5.0, p=2, squaredConst=1.0, regularizerCoeff=0.0, model=None):
        self.rho = rho
        self.p = p
        self.squaredConst  = squaredConst  
        self.regularizerCoeff =  regularizerCoeff
        self.model = model


        #Outputs is the functions evaluated after a fowrard pass. 
        #* NOTE: data has the batch dimenion equal to one. 
      #  b_size = data.size()[0]
      #  if b_size !=1 :
      #      raise Exception("batch dimenion is not one, aborting the execution.")
        self.data = data
        
        #self.convexSolver = solveQuadratic( self.regularizerCoeff )

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            #Move model to GPU
            self.model = self.model.cuda()
            #** Move data to GPU
            if torch.is_tensor(data):
                data = data.cuda()
            elif type(data) == list:
                data_x, data_y = data
                data = [data_x.cuda(), data_y.cuda()] 
       


        #Initialize variables.
       # self._setVARS() 
        
    def setVARS(self, primalTheta, Theta_k,  quadratic=False):
        """
           Initialize primal, dual and auxiliary variables and compute Jacobian. 
        """
        #Theta is the current model parameter
        self.Theta_k = Theta_k

        #Froward pass for data
        output =  self.model( self.data )

        #Compute Jacobian
        with torch.no_grad():
            if quadratic:
                Jac, sqJac = self.model.getJacobian(output, quadratic=quadratic)
                self.squaredJac = sqJac
            else:
                Jac = self.model.getJacobian(output, quadratic=quadratic)
            #Get tensor data from the output, the computational graph is not needed here. 
            self.output = output.data
            self.Jac = Jac

            #Initialize Y
            self.primalY = self.output
            #set dimensions
            self.dim_d = self.Theta_k.size()
      
            self.primalTheta = primalTheta 
            #Initialize dual vars U
            self.dual = torch.zeros( self.primalY.size() )
            if self.use_cuda:
                self.dual = self.dual.cuda()
        
        
    @torch.no_grad()
    def updateYAdaptDuals(self, g_est=None):
        """
            Update the primal Y variable via prox. operator for the p-norm.
        """
        vec = self.primalTheta - self.Theta_k
        vecJacobMult_j = self.model.vecMult(vec, Jacobian=self.Jac)
       # vecJacobMult_j = torch.matmul(vec, self.Jac.T)


        #Primal residual
        PRES = self.primalY - self.output - vecJacobMult_j
        #Adapt duals

        
        self.dual += PRES 
        oldPrimalY =  self.primalY
        #Update Y 
        self.primalY = pNormProxOp(vecJacobMult_j + self.output - self.dual, self.rho, p=self.p, g_est=g_est) 


        #logging.debug("Optimality of the proximal operator solution is:")
        #logging.debug( _testOpt(self.primalY, vecJacobMult_j + self.output - self.dual, rho=self.rho, p=self.p) )
         
        if self.use_cuda:
            self.primalY = self.primalY.cuda()

        return  torch.norm(oldPrimalY - self.primalY, p=2), torch.norm(PRES, p=2)


    @torch.no_grad()   
    def getCoeefficients(self, quadratic=False):
        """
            Return the coefficientes for the first order and the second order terms for updating primalTheta
        """ 

        U_hat = self.dual + self.primalY - self.output + self.model.vecMult(vec=self.Theta_k, Jacobian=self.Jac)

        first_ord = self.rho * self.model.vecMult(vec=U_hat, Jacobian=self.Jac, left=True) 
 

        if quadratic:
            second_ord = self.rho * self.squaredJac 
            return first_ord, second_ord

        return first_ord

    def getLocalLoss(self, Theta_var):
        """
            Return the loss for the current solver (data).
   
                 rho/2 ||D_i Theta_var - U_hat||_2^2
        """

  
        U_hat = self.dual + self.primalY - self.output + self.model.vecMult(vec=self.Theta_k, Jacobian=self.Jac)

        DTheta =  self.model.vecMult(vec=Theta_var, Jacobian=self.Jac, trackgrad=True)


        MSELoss = 0.
        for i in range(len(DTheta)):
            MSELoss += ( DTheta[i] - U_hat[0,i] ) ** 2


        return self.rho/2. * MSELoss
        




    @torch.no_grad()
    def getObjective(self):
        """
           Compute the objective for ADMM iterations, i.e.,

                ||Y_i||_p. 
        """
        if self.p == -2:
            return  torch.norm( self.primalY, p=2) ** 2

        return torch.norm( self.primalY, p=self.p ) 

    def evalModelLoss(self, Theta=None):
        """
         Compute the model loss function, around Theta_k, i.e.,

               ||F_i(theta_k) + D_i(theta - theta_k)||_p. 
        """

        if Theta == None:
            Theta = self.primalTheta 
        vec = Theta - self.Theta_k
        vecJacobMult_j = self.model.vecMult(vec=vec, Jacobian=self.Jac)
       # vecJacobMult_j = torch.matmul(vec, self.Jac.T)

        if self.p == -2:
            return torch.norm(vecJacobMult_j + self.output, p=2) ** 2
        return torch.norm(vecJacobMult_j + self.output, p=self.p) 
  

class GlobalSolvers:
    "Class of solvers for problems that require aggregated information over all data."
    def __init__(self, model, rank=None):
        self.model  = model
        self.rank = rank

        #create variables

        #current solution
        self.Theta_k = self.model.getParameters()

        #primal variable Theta (note that mult. by 1 makes sure that primalTheta and Theta_k are not pointers to the same tensor)      
        self.primalTheta = self.model.getParameters() * 1
        
    def _getVec(self, ADMMsolvers, quadratic=False):
        """
            Return first and second order terms for updating Theta, which is a quadratic problem:

            first_ord = rho \Sum_i D_i hat{u}_i + squaredConst * Theta_k
            second_ord = rho \Sum_i D_i D_i^T + (squaredConst + regularizationCoeff) * I
        """

        for solver_ind, ADMMsolver in enumerate(ADMMsolvers):
            if quadratic:
                first_ord, second_ord = ADMMsolver.getCoeefficients(quadratic = quadratic)
            else:
                first_ord =  ADMMsolver.getCoeefficients(quadratic = quadratic)


            if solver_ind == 0:
                first_ord_TOT = first_ord

                if quadratic:
                    second_ord_TOT = second_ord
            else:
                first_ord_TOT  += first_ord

                if quadratic:
                    second_ord_TOT += second_ord

            if self.rank != None:
                torch.distributed.all_reduce(first_ord_TOT)

                if quadratic:
                    torch.distributed.all_reduce(second_ord_TOT)

        first_ord_TOT += ADMMsolvers[0].squaredConst * self.Theta_k


        if quadratic:
            second_ord_TOT += (ADMMsolvers[0].squaredConst + ADMMsolvers[0].regularizerCoeff) * torch.eye( second_ord.size()[1]  )    

            return first_ord_TOT, second_ord_TOT
              
        return first_ord_TOT

    def solve(self, ADMMsolvers):
        """
            Return the optimal Theta by solving the qudartic programming problem. Given are the LocalSolvers (ADMMSolver), which determine the first and second order terms in the problem. 
        """
        pass

    def updateTheta(self, ADMMsolvers):
        #compute new Theta
        newTheta = self.solve(ADMMsolvers)

        #check optimilaity
        print("Optimilaity is ", self.debug(newTheta, ADMMsolvers) )


        #compute dual residual 
        DRES = (newTheta - self.primalTheta).frobeniusNormSq() ** 0.5



        #update Theta 
        for var_ind, var in enumerate( newTheta ):
            self.primalTheta[var_ind].copy_( var.data )
        return DRES

    def debug(self, sol, ADMMsolvers):
       """
           Check the optimilaity of the solution.
       """

       first_ord, second_ord = self._getVec(ADMMsolvers, quadratic = True)

       sol_tensor = sol.getTensor()

       return torch.matmul(second_ord, sol_tensor)  - first_ord.getTensor()

class solveConvex(GlobalSolvers):
    """
           Solve the following convex problem:
                 Minimize: quadCoeff * ||theta||_2^2 + 0.5 * theta^T * A * theta  - squaredConst *  <theta, b>
                 Subj. to:  theta \in Reals,
           via SGD.
    """ 

    def solve(self, ADMMsolvers, epochs=1000, batch_size=None, learning_rate=0.001):
        #default batch_size is all samples
        if batch_size == None:
            batch_size = len(ADMMsolvers)

        #get current model vraibales
        theta_VAR = self.model.getParameters(trackgrad=True)        

        #initialize thet_VAR to primalTheta
        for var_ind, var in enumerate( self.primalTheta ):
            theta_VAR[var_ind].data.copy_( var )

        logging.info("Solving convex quadratic problem")

        #get optimizer
        optimizer = torch.optim.SGD(theta_VAR, lr=learning_rate)


        for epoch in range(epochs):
            ts = time.time()


            #compute the quadratic proximity loss       
            proximty_loss =  0.5 * (ADMMsolvers[0].primalTheta - theta_VAR).frobeniusNormSq()

            loss = proximty_loss
            running_loss = proximty_loss.item()
            running_grad_norm = 0.

            for ind, ADMMsolver in enumerate( ADMMsolvers ):

                loss_i = ADMMsolver.getLocalLoss(theta_VAR )

                #increment loss
                loss += loss_i
                running_loss += loss_i.item()

                if ind == len(ADMMsolvers) -1 or (ind % batch_size == 0 and ind>0):
                    #zero out gradients
                    optimizer.zero_grad()

                    #backprop
                    loss.backward()

                    for var in theta_VAR:
                        running_grad_norm += torch.norm(var.grad, p=1)

                        #add up gradients across processes if multiple processes are running
                        if self.rank != None:
                           torch.distributed.all_reduce(var.grad) 

                    #update parameters 
                    optimizer.step()                  

                    loss = 0.0
            if epoch % 10 == 0:
                logging.info("Epoch {} done in {}(s), total loss is {}".format(epoch, time.time() - ts, running_loss))                    
                logging.info("Total gradient is {}".format(running_grad_norm))

        return theta_VAR        

   # @torch.no_grad()
    def updateTheta(self, ADMMsolvers):
        """
            Update theta by in all local solver instances. 
        """

        #update theta by solving the quadratic convex problem via gradient descent
        newTheta = self.solve(ADMMsolvers)

        oldPrimalTheta = self.Theta_k 
        #set new primalTheta 
        for var_ind, var in enumerate( newTheta ):
            self.primalTheta[var_ind].copy_( var.data )

        #return dual residual 
        DRES =  (oldPrimalTheta - self.primalTheta).frobeniusNormSq() ** 0.5
        return  DRES 


@torch.no_grad()
class solveWoodbury(GlobalSolvers):

    def __init__(self, A, b, rho):

        self.A = A
        self.b = b
        self.rho

    def _setMat(self, A):
        "Compute the matrix ∑_i A_i A_iT."

        self.dim_N = b[0].shape[-1]
        self.dim_n = len(b) 

        AAT = torch.eye(self.dim_n * self.dim_N)

       
        for solver_ind_i, ADMMsolver_i in enumerate(ADMMsolvers):
            for solver_ind_j, ADMMsolver_j in enumerate(ADMMsolvers):     
                for row_ind_i, row_i in enumerate(ADMMsolver_i.Jac):
                    for row_ind_j, row_j in   enumerate(ADMMsolver_j.Jac):
                        A_row_ind = solver_ind_i * self.dim_N + row_ind_i
                        A_col_ind = solver_ind_j * self.dim_N + row_ind_j

                        AAT[A_row_ind, A_col_ind] += row_i * row_j * ADMMsolvers[0].rho

        self.Amat = A
 



    def _debug(self, ADMMsolvers, b):
        MAT = []
        b_vec = b.getTensor()
        for ADMMsolver in ADMMsolvers:
            out = self.model(ADMMsolver.data)
            Jac_Mat = self.model.getJacobian_old(out)
            MAT.append(Jac_Mat)
        Jac_Cat = torch.cat(MAT, 0) 

        D_transpose_b = torch.matmul(Jac_Cat, b_vec)

        Amat = torch.eye(self.dim_N * self.dim_n) + torch.matmul(Jac_Cat, Jac_Cat.T)
        
        matInv_D_transpose_b, decom = torch.solve(D_transpose_b.unsqueeze(1), Amat)

        D_matInv_D_transpose_b = torch.matmul(Jac_Cat.T, matInv_D_transpose_b)
        return D_matInv_D_transpose_b

    def _multMatVec(self, ADMMsolvers, b, debug_val=None):
        """Compute the matrix inversion and product:

               D(I_{nN} + D^T D)^-1 D^T b,

            where D is the concatenation of the Jacobians D_i, i=1,...,n.
        """
    
        #initalize
        D_transpose_b = torch.zeros(self.dim_N * self.dim_n, 1)
        
        #populate values in D_transpose_b
        for solver_ind_i, ADMMsolver_i in enumerate(ADMMsolvers):
            for row_ind_i, row_i in enumerate(ADMMsolver_i.Jac):
                ind = solver_ind_i * self.dim_N + row_ind_i
                D_transpose_b[ind] = row_i * b

   
        matInv_D_transpose_b, decom = torch.solve(D_transpose_b, self.Amat)


        for solver_ind_i, ADMMsolver_i in enumerate(ADMMsolvers):

            #compute D_i times matInv_D_transpose_b for each i (ADMMsolver)
            D_matInv_D_transpose_b_i = self.model.vecMult(vec=matInv_D_transpose_b[solver_ind_i * self.dim_N: (solver_ind_i + 1) * self.dim_N].T, left=True, Jacobian=ADMMsolver_i.Jac)
       

            if solver_ind_i  == 0:
                D_matInv_D_transpose_b = D_matInv_D_transpose_b_i
            else:
                D_matInv_D_transpose_b += D_matInv_D_transpose_b_i

        return D_matInv_D_transpose_b 

    def solve(self, ADMMsolvers):
        b = self._getVec(ADMMsolvers)     

        try:
            self.Amat
        except AttributeError:
            self._setMat(ADMMsolvers)
      
        #debugging 
        #debug_val = self._debug(ADMMsolvers, b)

        D_matInv_D_transpose_b = self._multMatVec(ADMMsolvers, b)
        #print(debug_val, D_matInv_D_transpose_b)
        sol = ADMMsolvers[0].squaredConst * b - ADMMsolvers[0].rho * D_matInv_D_transpose_b 

        #test solution
        #self.testSolution(ADMMsolvers, b, sol)
     
        return sol 

        
    def testSolution(self, ADMMsolvers, b, sol):

        b_solved = sol * ADMMsolvers[0].squaredConst
        for ADMMsolver in ADMMsolvers:
            D_i_sol = self.model.vecMult(vec=sol, Jacobian=ADMMsolver.Jac)
            b_solved +=  self.model.vecMult(vec=D_i_sol, left=True, Jacobian=ADMMsolver.Jac) * ADMMsolver.rho

        print( b_solved - b)
       
            
    def updateTheta(self, ADMMsolvers):
        #compute new Theta
        newTheta = self.solve(ADMMsolvers)


        #compute dual residual 
        DRES = (newTheta - self.primalTheta).frobeniusNormSq() ** 0.5

         

        #update Theta 
        for var_ind, var in enumerate( newTheta ):
            self.primalTheta[var_ind].copy_( var.data )
        return DRES
                
                

class solveQuadratic(GlobalSolvers):

    @torch.no_grad()
    def solve(self, ADMMsolvers):

        try:
            self.Ainv
            first_trem =  self._getVec(ADMMsolvers)

        except AttributeError:
            first_trem, second_term =  self._getVec(ADMMsolvers, quadratic=True)

            self.Ainv = torch.inverse( second_term )
 
        #convert the TensorList to Tensor to matrix-vector product
        first_trem_tensor = first_trem.getTensor()

        sol_tensor = torch.matmul(self.Ainv, first_trem_tensor)

        return first_trem.formFromTensor( sol_tensor, first_trem.TLShape() )


       
        

        
         


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=8)
    parser.add_argument("--iterations", type=int,  default=100)
    parser.add_argument("--logfile", type=str,default="serial.log")
    parser.add_argument("--logLevel", type=str, choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=2, help="p in lp-norm")
    args = parser.parse_args()


    clearFile( args.logfile  )
    logging.basicConfig(filename=args.logfile, level=eval("logging." + args.logLevel) )

    #Load dataset

    dataset =   loadFile(args.input_file)
    data_loader = DataLoader(dataset, batch_size=1) 

    #Instansiate model
    model = AEC(args.m, args.m_prime)

    if args.p not in [1, 2]:
        g_est = estimate_gFunction(args.p)
    else:
        g_est = None

    #Instansiate ADMMsolvres
    ADMMsolvers = []
    if torch.cuda.is_available():
        model = model.cuda()

    #load data 
    for ind, data in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        ADMMsolver = LocalSolver(data, model=model, rho=args.rho, p=args.p)
        ADMMsolvers.append(ADMMsolver)
        print(ind)
        if ind == 10:
            break
    logging.info("Initialized the ADMMsolvers for {} datapoints".format(len(ADMMsolvers)) )

    #Instantiate convex solvers
    globalSolver = solveConvex(model= model)

    
    #ADMM iterations 
    for k in range(args.iterations):
        t_start = time.time()

        #initialize residuals 
        PRES_TOT = 0.0
        DRES_TOT = 0.0        

        #compute the quadratic proximity loss       
        proximty_loss =  0.5 * (ADMMsolvers[0].primalTheta - ADMMsolvers[0].Theta_k).frobeniusNormSq()

        #initialize objective for current iteration
        OBJ_TOT = proximty_loss.item()
        loss_TOT = proximty_loss

        ind = 0
        for ADMMsolver in ADMMsolvers:
            #Update Y and adapt duals for each solver 
            DRES, PRES = ADMMsolver.updateYAdaptDuals(g_est)

            #increment current objective value and residuals            
            OBJ_TOT += ADMMsolver.getObjective()
            PRES_TOT += PRES
            DRES_TOT += DRES        
            print(ind)
            ind += 1

     
        #Update theta via convexSolvers
        DRES_theta = globalSolver.updateTheta(ADMMsolvers) 
   
        DRES_TOT += DRES_theta 

        
        t_end = time.time()
        logging.info("Iteration {} is done in {} (s), OBJ is {} ".format(k, t_end - t_start, OBJ_TOT ))
        logging.info("Iteration {}, PRES is {}, DRES is {}".format(k, PRES_TOT, DRES_TOT) )
         


