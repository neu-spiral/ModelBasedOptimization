import argparse 
import time
import logging
import numpy as np
import logging
from Net import AEC, DAEC, Embed
from torch.utils.data import Dataset, DataLoader
import torch
from helpers import pNormProxOp, clearFile, estimate_gFunction, loadFile, _testOpt
from  datasetGenetaor import unlabeledDataset
import torch.nn as nn

#torch.manual_seed(1993)

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
        self._setVARS() 
        
    def _setVARS(self, quadratic=False):
        """
           Initialize primal, dual and auxiliary variables and compute Jacobian. 
        """
        #Theta is the current model parameter
        self.Theta_k = self.model.getParameters()

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
      
            self.primalTheta = self.Theta_k
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


        logging.debug("Optimality of the proximal operator solution is:")
        logging.debug( _testOpt(self.primalY, vecJacobMult_j + self.output - self.dual, rho=self.rho, p=self.p) )
         
        if self.use_cuda:
            self.primalY = self.primalY.cuda()

        return  torch.norm(oldPrimalY - self.primalY, p=2), torch.norm(PRES, p=2)


    @torch.no_grad()   
    def getCoeefficients(self):
        """
            Return the coefficientes for the first order and the second order terms for updating primalTheta
        """ 

        U_hat = self.dual + self.primalY - self.output + self.model.vecMult(vec=self.Theta_k, Jacobian=self.Jac)
        first_ord = self.rho * torch.matmul(U_hat, self.Jac) 
        second_ord = self.rho * self.squaredJac 

        return first_ord, second_ord

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
    def updateTheta(self, first_ord=None, second_ord=None, newTheta=None):
        """
            Update theta by solving the convex problem
        """
        oldPrimalTheta = self.primalTheta        
        if newTheta is not None:
            self.primalTheta = newTheta
            return (oldPrimalTheta - self.primalTheta).frobeniusNormSq() ** 0.5

        b = first_ord + self.squaredConst * self.Theta_k

        scaledI = self.squaredConst * torch.eye( self.dim_d ).unsqueeze(0)
        if self.use_cuda:
            scaledI = scaledI.cuda()

        A = second_ord.unsqueeze(0) + scaledI

        #Solve the convex problem  
        self.primalTheta = self.convexSolver.solve(A=A, b=b)
       # print ( torch.matmul(A, self.primalTheta.T) - b.T )
        return torch.norm(oldPrimalTheta - self.primalTheta, p=2)



    @torch.no_grad()
    def getObjective(self):
        """
           Compute the objective for ADMM iterations. 
        """
        if self.p == -2:
            return  torch.norm( self.primalY, p=2) ** 2

        return torch.norm( self.primalY, p=self.p ) 

    def evalModelLoss(self, Theta=None):
        """
         Compute the model loss function, around Theta_k.
        """

        if Theta == None:
            Theta = self.primalTheta 
        vec = Theta - self.Theta_k
        #vecJacobMult_j = self.model.vecMult(vec, Jacobian=self.Jac)
        vecJacobMult_j = torch.matmul(vec, self.Jac.T)
        if self.p == -2:
            return torch.norm(vecJacobMult_j + self.output, p=2) ** 2
        return torch.norm(vecJacobMult_j + self.output, p=self.p) 
  

class GlobalSolvers:
    "Class of solvers for problems that require aggregated information over all data."
    def __init__(self, ADMMsolvers, model, rank=None):
        self.ADMMsolvers = ADMMsolvers
        self.model  = model
        self.rank = rank
 
        
       

class solveConvex(GlobalSolvers):
    """
           Solve the following convex problem:
                 Minimize: quadCoeff * ||theta||_2^2 + 0.5 * theta^T * A * theta  - squaredConst *  <theta, b>
                 Subj. to:  theta \in Reals,
           via SGD.
    """ 

    def solve(self, epochs=100, batch_size=None, learning_rate=0.001):
        #default batch_size is all samples
        if batch_size == None:
            batch_size = len(self.ADMMsolvers)

        #get current model vraibales
        theta_VAR = self.model.getParameters(trackgrad=True)        

        #initialize thet_VAR to primalTheta
        for var_ind, var in enumerate( self.ADMMsolvers[0].primalTheta ):
            theta_VAR[var_ind].data.copy_( var )

        logging.info("Solving convex quadratic problem")

        #get optimizer
        optimizer = torch.optim.SGD(theta_VAR, lr=learning_rate)


        for epoch in range(epochs):
            ts = time.time()


            #compute the quadratic proximity loss       
            proximty_loss =  0.5 * (self.ADMMsolvers[0].primalTheta - theta_VAR).frobeniusNormSq()

            loss = proximty_loss
            running_loss = proximty_loss.item()
            running_grad_norm = 0.

            for ind, ADMMsolver in enumerate( self.ADMMsolvers ):

                loss_i = ADMMsolver.getLocalLoss(theta_VAR )

                #increment loss
                loss += loss_i
                running_loss += loss_i.item()

                if ind == len(self.ADMMsolvers) -1 or (ind % batch_size == 0 and ind>0):
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

            logging.info("Epoch {} done in {}(s), total loss is {}".format(epoch, time.time() - ts, running_loss))                    
            logging.info("Total gradient is {}".format(running_grad_norm))

        return theta_VAR        

   # @torch.no_grad()
    def updateTheta(self):
        """
            Update theta by in all local solver instances.
        """
        #update theta by solving the quadratic convex problem via gradient descent
        newTheta = self.solve()

        oldPrimalTheta = self.ADMMsolvers[0].primalTheta
        for ADMMsolver in self.ADMMsolvers:
            for var_ind, var in enumerate( newTheta ):
                ADMMsolver.primalTheta[var_ind].copy_( var.data )
        #return dual residual 
        return (oldPrimalTheta - self.ADMMsolvers[0].primalTheta).frobeniusNormSq() ** 0.5
        
        

class solveQuadratic(GlobalSolvers):
    def __init__(self,ADMMsolvers, model, rank=None, quadCoeff=0.0):
        """
           Solve the following convex problem:
                 Minimize: quadCoeff * ||theta||_2^2 + 0.5 * theta^T * A * theta  - squaredConst *  <theta, b>
                 Subj. to:  theta \in Reals,
        """
        super(solveQuadratic, self).__init__(ADMMsolvers, model, rank)
        self.quadCoeff = quadCoeff 


    @torch.no_grad()
    def _getCoeff(self):
        first_ord_TOT = 0.
        second_ord_TOT = 0.

        for ADMMsolver in self.ADMMsolvers:
            first_ord, second_ord =  ADMMsolver.getCoeefficients()        
            #add first order and second order coeffocients
            first_ord_TOT  += first_ord
            second_ord_TOT += second_ord

            if self.rank != None:
                torch.distributed.all_reduce(first_ord_TOT)
                torch.distributed.all_reduce(second_ord_TOT) 

    @torch.no_grad()
    def solve(self, A, b):
        #Qudartic term due to the norm2 squared regularizer        
        A_regulrizer = self.quadCoeff * torch.eye( A.size()[1]  )
        if torch.cuda.is_available():
            A_regulrizer = A_regulrizer.cuda()

        A += A_regulrizer
        b = b.T.unsqueeze(0)
        sol, LUdecomp = torch.solve(b, A)
        
        #test
       # A = A.squeeze(0)
       # b = b.squeeze(0)
       # sol = sol.squeeze(0)
        #print (torch.matmul(A, sol) - b)
  
        return sol.squeeze(2)
        
         


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
    globalSolver = solveConvex(ADMMsolvers, model)

    
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
        print(k)

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
        DRES_theta = globalSolver.updateTheta() 
   
        DRES_TOT += DRES_theta 

        
        t_end = time.time()
        logging.info("Iteration {} is done in {} (s), OBJ is {} ".format(k, t_end - t_start, OBJ_TOT ))
        logging.info("Iteration {}, PRES is {}, DRES is {}".format(k, PRES_TOT, DRES_TOT) )
         


