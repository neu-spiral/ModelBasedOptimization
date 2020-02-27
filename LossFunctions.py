import torch
from torch.multiprocessing import Process
import os
import torch.distributed as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as FUNC



class LossFunction(nn.Module):
    "A generic class written for definiing a loss function F(θ; X) where X is the input data and θ is the parametr."
    def __init__(self):
        print("Initializing LossFunction.")
        super(LossFunction, self).__init__()

    def getParameters(self):
        "Return the model parameters as a single vectorized tensor."
        return torch.cat( [parameter.view(-1) for parameter in self.parameters()] ) 
    def getGrads(self):
        "Return gradient w.r.t. model parameters of all model layers  as a single vectorized tensor."
        return torch.cat( [parameter.grad.view(-1) for parameter in self.parameters()] ).unsqueeze(0)   
    def setParameters(self, params):
        """Set model parameters."""

        ind = 0
        for parameter in self.parameters():
            if parameter.size() != params[ind].size():
                raise Exception('Dimensions do not match')
            parameter.data =  params[ind].data
            ind += 1


    def _getJacobian_aux(self, output, i, device=torch.device("cpu")):
        """
          Return the i-th row of the Jacobian, i.e., the gradient of the i-th node in the output layer, w.r.t. network paraneters

          evaluated for output
        """
        #Compute the i-th column in the Jacobian matrix, i.e., the grdaient of the i-th neuron in the output layer w.r.t. model parameters
        selctor = np.zeros( output.size()[-1]  )
        selctor[i] = 1
        selector = torch.tensor( selctor  )
        selector =  selector.view(1, -1)
        selector = selector.to(device) 
        #Reset gradients to zero
        self.zero_grad()
        #Do a backward pass to compute the Jacobian
        #retain_graph is set to True as the auxiliary function is called multiple times.
        
        output.backward(selector, retain_graph=True )
        #Get grads
        return  self.getGrads()

            

    def getJacobian(self, output, quadratic=False, device=torch.device("cpu")):
        """
            Return the Jacobian matrix evaluauted at output, if quadratic is True, the  it also return the suared of the Jacobian. 
        """

        bath_size, output_size = output.size()    
        if bath_size != 1:
             raise Exception('Batch dimension is not equal to one.')
        for i in range(output_size):
            Jacobian_i_row = self._getJacobian_aux(output, i, device)
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
    
 ###NOT IMPLEMENTED 
    def getParallelJacobian(self, Input, size=10):
        """Return the Jacobian, computed in parallel."""
        def fn(rank, size):
            print (f"rank and size are {rank} and {size}")
            r = 9 

        def init_process(rank, size, fn, backend='gloo'):
            """ Initialize the distributed environment. """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(backend, rank=rank, world_size=size)
            fn(rank, size)
    
        processes = []
        for rank in range(size):
            p = Process(target=init_process, args=(rank, size, fn))
            p.start()
            processes.append(p)

        for prorcess in processes:
            prorcess.join()
###END NOT IMPLEMENTED             
                    
             
            
            
        
    def vecMult(self, vec, output=None, Jacobian=None):
        """
           Multiply the Jacobian and the vector vec. Note that output must have batch dimension of 1.
        """

        bath_size, output_size = output.size()
        if bath_size != 1:
             raise Exception('Batch dimension is not equal to one.')


        if Jacobian == None:
            if output == None:
                raise Exception('Neither output nor Jacobian is passed.')
            Jacobian = self.getJacobian(output)

        return torch.ger(Jacobian, vec)

    def forward(self, X):
        """Given an input X execute a forward pass."""
        pass

class AEC(LossFunction):
    "A class for Autoencoders; the input size is m anbd the encoded size is m_prime."
    def __init__(self, m , m_prime):
        super(AEC, self).__init__()
        self.m = m
        self.m_prime = m_prime
        self.fc1 = nn.Linear(m, m_prime) 
    #    self.fc2 = nn.Linear(m_prime, m) 
    def forward(self, X):
        "Given an input X execute a forward pass."
        X = torch.sigmoid( self.fc1(X) )
    #    X = self.fc2(X)
        return X


       
        
     
    
        
    
 
        

if __name__ == "__main__":
    AE = AEC(4, 2)
    sample_input = torch.randn(1, 4)
    theta = []
  #  theta.append( torch.ones(2, 4) )
  #  theta.append (torch.randn(2) )
  #  theta.append ( torch.randn(4, 2))
  #  theta.append ( torch.randn(4))
  #  AE.set(theta)

    selector = torch.tensor([1.0, 0.0], dtype=torch.float)
    selector =  selector.view(1, -1)

    AE.zero_grad()
    sample_output = AE(sample_input)

    
    model_parameters = AE.getParameters()
    vec = torch.randn(model_parameters.size() )
    
    AE.getParallelJacobian(sample_input, size=4)
   # Jac, sqJac = AE.getJacobian(sample_output, True)
    

    
    
    
    
    
    
    
   
    


