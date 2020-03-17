import argparse 
import time
import logging
import numpy as np
import logging
from LossFunctions import AEC
from torch.utils.data import Dataset, DataLoader
import torch
from helpers import pNormProxOp, clearFile

#torch.manual_seed(1993)

class ADMM():
    "ADMM Solver"
    def __init__(self, data, model_spec=None, rho=5.0, p=2, squaredConst=1.0, regularizerCoeff=0.0, model=None):
        self.rho = rho
        self.p = p
        self.squaredConst  = squaredConst  
        self.regularizerCoeff =  regularizerCoeff

        if model is None:
            #model_spec is a dictionary with 'loss' dtermines the loss function and 'parameter_dim' determines the dimensions. 
            self.model = model_spec['loss']( model_spec['parameter_dim'][0],  model_spec['parameter_dim'][1] )
        else:
            self.model = model


        #Outputs is the functions evaluated after a fowrard pass. 
        #* NOTE: data has the batch dimenion equal to one. 
        b_size = data.size()[0]
        if b_size !=1 :
            raise Exception("batch dimenion is not one, aborting the execution.")
        self.data = data
        
        self.convexSolver = solveQuadratic( self.regularizerCoeff )

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model.cuda()
       


        #Initialize variables.
       # self._setVARS() 

        
   # @torch.no_grad()    
    def _setVARS(self):
        """
           Initialize primal, dual and auxiliary variables and compute Jacobian. 
        """

        self.output = self.model( self.data )
        
        #Initialize Y
        self.primalY = self.output
        #Initialize theta
        self.Theta_k = self.model.getParameters()
        #set dimensions
        self.dim_d = self.Theta_k.size()[-1]
        self.dim_N = self.output.size()[1]
      
        self.primalTheta = self.Theta_k
        #Initialize dual vars U
        self.dual = torch.zeros( self.primalY.size() )
        if self.use_cuda:
            self.dual = self.dual.cuda()
      
       
        #Compute Jacobian
        Jac, sqJac = self.model.getJacobian(self.output, quadratic=True)
       
        self.Jac = Jac
        self.squaredJac = sqJac
        
        
    @torch.no_grad()
    def updateYAdaptDuals(self):
        """
            Update the primal Y variable via prox. operator for the p-norm.
        """
        vec = self.primalTheta - self.Theta_k
        #vecJacobMult_j = self.model.vecMult(vec, Jacobian=self.Jac)
        vecJacobMult_j = torch.matmul(vec, self.Jac.T)


        #Primal residual
        PRES = self.primalY - self.output - vecJacobMult_j
        #Adapt duals
        self.dual += PRES 
        oldPrimalY =  self.primalY
        #Update Y 
        self.primalY = pNormProxOp(vecJacobMult_j + self.output - self.dual, self.rho, p=self.p) 
        if self.use_cuda:
            self.primalY = self.primalY.cuda()

        return  torch.norm(oldPrimalY - self.primalY), torch.norm(PRES)


    @torch.no_grad()   
    def getCoeefficients(self):
        """
            Return the coefficientes for the first order and the second order terms for updating primalTheta
        """ 
        
        self.U_hat = self.dual + self.primalY - self.output + torch.matmul(self.Theta_k, self.Jac.T)
        first_ord = self.rho * torch.matmul(self.U_hat, self.Jac) 
        second_ord = self.rho * self.squaredJac 

        return first_ord, second_ord

    @torch.no_grad()
    def updateTheta(self, first_ord, second_ord, newTheta=None):
        """
            Update theta by solving the convex problem
        """
        
        if newTheta is not None:
            self.primalTheta = newTheta
            return 0.0
        oldPrimalTheta = self.primalTheta
        b = first_ord + self.squaredConst * self.Theta_k

        scaledI = self.squaredConst * torch.eye( self.dim_d ).unsqueeze(0)
        if self.use_cuda:
            scaledI = scaledI.cuda()

        A = second_ord.unsqueeze(0) + scaledI

        #Solve the convex problem  
        self.primalTheta = self.convexSolver.solve(A=A, b=b)
       # print ( torch.matmul(A, self.primalTheta.T) - b.T )
        return torch.norm(oldPrimalTheta - self.primalTheta)

    def setModelParameters(self):
        """
           Set model paramaters.
        """
        self.model.setParameters( self.primalTheta  ) 


    @torch.no_grad()
    def getObjective(self):
        """
           Compute the objective.
        """
   
        return torch.norm( self.primalY, p=self.p )
    
        
        

class solveQuadratic():
    def __init__(self, quadCoeff=0.0):
        """
           Solve the following convex problem:
                 Minimize: quadCoeff * ||theta||_2^2 + 0.5 * theta^T * A * theta  - squaredConst *  <theta, b>
                 Subj. to:  theta \in Reals,
        """
        self.quadCoeff = quadCoeff 
    def solve(self, A, b):
        
        A += self.quadCoeff * torch.eye( A.size()[1]  )
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
    parser.add_argument("--m_prime", type=int,  default=2)
    parser.add_argument("--iterations", type=int,  default=10)
    parser.add_argument("--logfile", type=str,default="serial.log")
    parser.add_argument("--rho", type=float, default=1.0)
    args = parser.parse_args()


    clearFile( args.logfile  )
    logging.basicConfig(filename=args.logfile, level=logging.INFO)

    #Load dataset

    dataset =  torch.load(args.input_file)
    data_loader = DataLoader(dataset, batch_size=1) 
    #Instansiate model
    model = AEC(args.m, args.m_prime)
    #Instansiate ADMMsolvres
    ADMMsolvers = []
    for data in data_loader:
        ADMMsolver = ADMM(data, model, rho=args.rho)
        ADMMsolvers.append(ADMMsolver)
    logging.info("Initialized the ADMMsolvers for {} datapoints".format(len(ADMMsolvers)) )

    #ADMM iterations 
    for k in range(args.iterations):
        t_start = time.time()
        PRES_TOT = 0.0
        DRES_TOT = 0.0              
        OBJ_TOT = 0.5 * torch.norm(ADMMsolvers[0].primalTheta - ADMMsolvers[0].Theta_k) ** 2 
        first_ord_TOT = 0.0
        second_ord_TOT = 0.0 
        #Update Y and adapt duals for each solver 
        for ADMMsolver in ADMMsolvers:
            DRES, PRES = ADMMsolver.updateYAdaptDuals()
            first_ord, second_ord =  ADMMsolver.getCoeefficients()

            OBJ_TOT += ADMMsolver.getObjective()
            PRES_TOT += PRES
            DRES_TOT += DRES        

            first_ord_TOT  += first_ord
            second_ord_TOT += second_ord
           

        #Aggregate first order and the scond order terms and solve the convex problem for theta
        ADMMsolver_i = ADMMsolvers[0]
        DRES_theta = ADMMsolver_i.updateTheta(first_ord_TOT, second_ord_TOT)
        DRES_TOT += DRES_theta
        for ADMMsolver in ADMMsolvers[1:]:
            ADMMsolver_i.updateTheta( first_ord_TOT, second_ord_TOT, ADMMsolvers[0].primalTheta) 
     
        t_end = time.time()
        logging.info("Iteration {} is done in {} (s), OBJ is {} ".format(k, t_end - t_start, OBJ_TOT ))
        logging.info("Iteration {}, PRES is {}, DRES is {}".format(k, PRES_TOT, DRES_TOT) )
         


