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
    def __init__(self, data, model=None, rho=1.0):
        self.rho = rho
        self.model = model 
        #Outputs is the functions evaluated after a fowrard pass. 
        #* NOTE: data has the batch dimenion equal to one. 
        b_size = data.size()[0]
        if b_size !=1 :
            raise Exception("batch dimenion is not one. Aborting the algorithm.")
        self.data = data
        
        self.output = self.model( self.data )
        self.convexSolvers = solveConvex()

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
        self.primalTheta = self.Theta_k
        #Initialize dual vars U
        self.dual = torch.zeros( self.primalY.size() )
       
        #Compute Jacobian
        Jac, sqJac = self.model.getJacobian(self.output, quadratic=True)
       
        self.Jac = Jac
        self.second_ord = sqJac
        
    @torch.no_grad()
    def updateYAdaptDuals(self):
        """
            Update the primal Y variable via prox. operator for the p-norm.
        """
        vec = self.primalTheta - self.Theta_k
        #vecJacobMult_j = self.model.vecMult(vec, Jacobian=self.Jac)
        vecJacobMult_j = torch.matmul(vec, self.Jac.T)

        #Adapt duals
        self.dual += ( self.primalY - self.output - vecJacobMult_j )
        #Update Y 
        self.primalY = pNormProxOp(vecJacobMult_j + self.output - self.dual, self.rho) 


    @torch.no_grad()   
    def getCoeefficients(self):
        """
            Return the coefficientes for the first order and the second order terms for updating primalTheta
        """ 
        
        self.U_hat = self.dual + self.primalY - self.output + torch.matmul(self.Theta_k, self.Jac.T)
        first_ord = torch.matmul(self.U_hat, self.Jac) 
        
        return self.rho * first_ord, self.rho * self.second_ord

    @torch.no_grad()
    def updateTheta(self, first_ord, second_ord):
        """
            Update theta by solving the convex problem
        """
   
        first_ord.detach()
        second_ord.detach()
        self.primalTheta = self.convexSolvers.solve(first_ord, second_ord, self.Theta_k)

        

class solveConvex():
    def __init__(self):
        """
           Solve the following convex problem:
                 Minimize:      0.5 * theta^T * second_ord * theta  -  theta^T * u_hat + 0.5 \|theta - theta_k \|_2^2 
                 Subj. to: theta \in R^d
        """

         
    def solve(self, first_ord, second_ord, theta_k):
        A = second_ord + torch.eye( theta_k.size()[0] )
        b = theta_k + first_ord
        np.linalg.solve(A,b)
        
         


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=2)
    parser.add_argument("--iterations", type=int,  default=20)
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
        i = 0
             
        #Update Y and adapt duals for each solver 
        for ADMMsolver in ADMMsolvers:
            ADMMsolver.updateYAdaptDuals()
            first_ord, second_ord =  ADMMsolver.getCoeefficients()
            
            if i==0:
                first_ord_TOT = first_ord
                second_ord_TOT = second_ord
            else:
                first_ord_TOT  += first_ord
                second_ord_TOT += second_ord
        #Aggregate first order and the scond order terms and solve the convex problem for theta
      #  ADMMsolver_i = ADMMsolvers[0]
      #  ADMMsolver_i.updateTheta(first_ord_TOT, second_ord_TOT)
        break
