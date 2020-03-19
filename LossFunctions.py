import torch
import logging
from torch.multiprocessing import Process
import os
import torch.distributed as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as FUNC



class LossFunction(nn.Module):
    "A generic class written for definiing a loss function F(θ; X) where X is the input data and θ is the parametr."
    def __init__(self, device):
        super(LossFunction, self).__init__()
        self.device = device

    @torch.no_grad()
    def getParameters(self):
        """
            Return the model parameters as a single vectorized tensor.
        """
        return torch.cat( [parameter.view(-1) for parameter in self.parameters()] ).unsqueeze(0)
       # return [param for param in self.parameters()]


    @torch.no_grad()
    def getGrads(self):
        """
            Return gradient w.r.t. model parameters of all model layers  as a single vectorized tensor.
        """
        #print (torch.cat( [parameter.grad.view(-1) for parameter in self.parameters()], 0 ).unsqueeze(0).size())
        return torch.cat( [parameter.grad.view(-1) for parameter in self.parameters()], 0 ).unsqueeze(0)
       # return [parameter.grad for parameter in self.parameters()] 

    @torch.no_grad()
    def setParameters(self, params):
        """Set model parameters."""
   
        new_parameters = params.squeeze(0)
        last_Index = 0
        for name, param in self.named_parameters():
            param_size_tot = 1
            for parm_size in param.size():
                param_size_tot *= parm_size
            param.copy_(new_parameters[last_Index: param_size_tot + last_Index].view( param.size()  ) )
            last_Index += param_size_tot
        logging.warning("Model parameters modified.")
            

    @torch.no_grad()
    def _getJacobian_aux(self, output, i):
        """
          Return the i-th row of the Jacobian, i.e., the gradient of the i-th node in the output layer, w.r.t. network paraneters

          evaluated for output
        """
        #Compute the i-th column in the Jacobian matrix, i.e., the grdaient of the i-th neuron in the output layer w.r.t. model parameters
        selctor = np.zeros( output.size()[-1]  )
        selctor[i] = 1
        selector = torch.tensor( selctor  )
        selector =  selector.view(1, -1)
        selector = selector.to(self.device) 
        #Reset gradients to zero
        self.zero_grad()

        #Do a backward pass to compute the Jacobian
        #retain_graph is set to True as the auxiliary function is called multiple times.
        output.backward(selector, retain_graph=True )

        #Get grads
        return  self.getGrads()

            
    @torch.no_grad()
    def getJacobian(self, output, quadratic=False):
        """
            Return the Jacobian matrix evaluauted at output, if quadratic is True, the  it also return the suared of the Jacobian. 
        """

        bath_size, output_size = output.size()    
        if bath_size != 1:
             raise Exception('Batch dimension is not equal to one.')
        for i in range(output_size):
            Jacobian_i_row = self._getJacobian_aux(output, i)
            if  i == 0:
                Jacobian = Jacobian_i_row
                if quadratic:
                    SquaredJacobian = torch.ger(Jacobian_i_row.squeeze(0), Jacobian_i_row.squeeze(0))
            else:
                Jacobian = torch.cat((Jacobian, Jacobian_i_row), dim=0) 
                if quadratic:
                     SquaredJacobian += torch.ger(Jacobian_i_row.squeeze(0), Jacobian_i_row.squeeze(0))
        if not quadratic:
            return Jacobian
        return Jacobian, SquaredJacobian
    
                    
             
            
            
    @torch.no_grad()
    def vecMult(self, vec, output=None, Jacobian=None, left=False):
        """
           Multiply the Jacobian and the vector vec. Note that output must have batch dimension of 1.
        """



        if Jacobian == None:
            if output == None:
                raise Exception('Neither output nor Jacobian is passed.')
           
            (bath_size, output_size) = output.size()
            if bath_size != 1:
                raise Exception('Batch dimension is not equal to one.')
            Jacobian = self.getJacobian(output)

        if left:
            return torch.matmul(vec, Jacobian)
        else:
            return torch.matmul(Jacobian, vec)

    def forward(self, X):
        """Given an input X execute a forward pass."""
        pass

class AEC(LossFunction):
    "A class for Autoencoders; the input size is m anbd the encoded size is m_prime."
    def __init__(self, m , m_prime, device=torch.device("cpu")):
        super(AEC, self).__init__(device)
        self.m = m
        self.m_prime = m_prime
        self.fc1 = nn.Linear(m, m_prime) 
        self.fc2 = nn.Linear(m_prime, m) 
    def forward(self, X):
        "Given an input X execute a forward pass."
        Y = torch.sigmoid( self.fc1(X) )
        Y = self.fc2(Y)
        return Y - X


       
        
     
    
        
    
 
        

if __name__ == "__main__":
    AE = AEC(4, 2)
    sample_input = torch.randn(1, 4)
    theta = []
  #  theta.append( torch.ones(2, 4) )
  #  theta.append (torch.randn(2) )
  #  theta.append ( torch.randn(4, 2))
  #  theta.append ( torch.randn(4))
  #  AE.set(theta)

    model_parameters = AE.getParameters()
    print (model_parameters )

    AE.setParameters( model_parameters )
    model_parameters = AE.getParameters()
    print (model_parameters )   #, dict(AE.named_parameters()) )
   # Jac, sqJac = AE.getJacobian(sample_output, True)
    

    
    
    
    
    
    
    
   
    


