import torch
from helpers import pNormProxOp
class ADMM():
    "ADMM Solver"
    def __init__(self, rho, lossFunc, Input):
        self.rho = rho
        self.lossFunc = lossFunc
        self.Input = Input
    def getVARS(self):
        #Outputs is the functions evaluated after a fowrard pass. 
        self.Outputs = self.lossFunc( self.Input )
        #Initialize Y
        self.Y = self.Outputs
        #Initialize theta
        self.Theta_k = self.lossFunc.getParameters()
        self.Theta = self.Theta_k
        #Initialize dual vars U
        self.U = torhc.zeros( self.Y.size )
        
    def updateY(self):
        vec = self.Theta - self.current_parameters
        (batch_size, input_size) = self.Input.size
        for j in range(batch_size):
            vecJacobMult_j = self.lossFunc.vecMult(vec, self.Outputs[j,:])
            Y_j = pNormProxOp(vecJacobMult_j + self.Outputs[l,:] - self.U) 
         
    def solve(self):
        self.Y = self.updateY()
