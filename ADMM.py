import argparse 
import logging
import numpy as np
import logging
from LossFunctions import AEC
from torch.utils.data import Dataset, DataLoader
import torch
from helpers import pNormProxOp

#torch.manual_seed(1993)

class ADMM():
    "ADMM Solver"
    def __init__(self, data, model=None, rho=1.0, squaredConst=1.0):
        self.rho = rho
        self.squaredConst  = squaredConst  
        self.regularizerCoeff =  0.0
        self.model = model 
        #Outputs is the functions evaluated after a fowrard pass. 
        #* NOTE: data has the batch dimenion equal to one. 
        b_size = data.size()[0]
        if b_size !=1 :
            raise Exception("batch dimenion is not one. Aborting the algorithm.")
        self.data = data
        
        self.output = self.model( self.data )
        self.convexSolver = solveConvex()

        #Initialize variables.
        self._setVARS() 

        
    @torch.no_grad()    
    def _setVARS(self):
        """
           Initialize primal, dual and auxiliary variables.
        """
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
        self.primalY = pNormProxOp(vecJacobMult_j + self.output - self.dual, self.rho) 
        return  torch.norm(oldPrimalY - self.primalY), PRES


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
    def updateTheta(self, first_ord, second_ord):
        """
            Update theta by solving the convex problem
        """
        
        oldPrimalTheta = self.primalTheta
        b = first_ord + self.squaredConst * self.Theta_k
        A = second_ord.unsqueeze(0) + (self.squaredConst + self.regularizerCoeff) * torch.eye( self.dim_d ).unsqueeze(0)
        #Solve the convex problem  
        self.primalTheta = self.convexSolver.solve(A=A, b=b)
        return torch.norm(oldPrimalTheta - self.primalTheta)
   
        
        

class solveConvex():
    def __init__(self):
        """
           Solve the following convex problem:
                 Minimize:  g(theta) + 0.5 * theta^T * A * theta  - squaredConst *  <theta, b>
                 Subj. to:  theta \in C,
           here in the basic version the function g is zero (non-existenet) and the set C is alli the space of all real vectors. 
        """
        pass 
         
    def solve(self, A, b):
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
    args = parser.parse_args()


    logging.basicConfig(filename=args.logfile, level=logging.INFO)

    #Load dataset

    dataset =  torch.load(args.input_file)
    data_loader = DataLoader(dataset, batch_size=1) 
    #Instansiate model
    model = AEC(args.m, args.m_prime)
    #Instansiate ADMMsolvres
    ADMMsolvers = []
    for data in data_loader:
        ADMMsolver = ADMM(data, model)
        ADMMsolvers.append(ADMMsolver)

    logging.info("Initialized the ADMMsolvers for {} datapoints".format(len(ADMMsolvers)) )


    for k in range(args.iterations):
             
        #Update Y and adapt duals for each solver 
        for ADMMsolver in ADMMsolvers:
            DRES, PRES = ADMMsolver.updateYAdaptDuals()
            first_ord, second_ord =  ADMMsolver.getCoeefficients()
            
            try:
                first_ord_TOT  += first_ord
                second_ord_TOT += second_ord
                PRES_TOT += PRES
                DRES_TOT += DRES
           
            except NameError:
                first_ord_TOT = first_ord
                second_ord_TOT = second_ord
                PRES_TOT = PRES
                DRES_TOT = DRES
        print ("Iter ", k, " PRES ", PRES_TOT, " DRES ", DRES_TOT)
        #Aggregate first order and the scond order terms and solve the convex problem for theta
        ADMMsolver_i = ADMMsolvers[0]
        ADMMsolver_i.updateTheta(first_ord_TOT, second_ord_TOT)
         


