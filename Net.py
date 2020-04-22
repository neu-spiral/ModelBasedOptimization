import torch
import argparse
from helpers import loadFile
import logging
from torch.multiprocessing import Process
import os
import torch.distributed as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as FUNC
import time
from datasetGenetaor import unlabeledDataset
from torch.utils.data import Dataset, DataLoader

class TensorList(list):
    "A generic class for linear algebra operations on lists of tensors."

   
    def __add__(self, tensorList_other):
        outTL = []
        for ind, tensor in enumerate(self):
            outTL.append( tensor.add( tensorList_other[ind]  )  )
        return TensorList(outTL)

    def __sub__(self, tensorList_other):
        outTL = []
        for ind, tensor in enumerate(self):
            outTL.append( tensor.sub( tensorList_other[ind]  )  )
        return TensorList(outTL)

    def __mul__(self, other):
        "return the inner product"
        if type(other) == TensorList:
            out = 0
            for ind, tensor in enumerate(self):
               out += torch.dot(tensor.view(-1), other[ind].view(-1) )
            return out

        elif type(other) in [float, int]:
           out = [] 
           for ind, tensor in enumerate(self):
               out.append( tensor * other  )
           return TensorList(out)
  
    def __rmul__(self, other):
        if type(other) in [float, int]:
            return self.__mul__( other  )        

    def frobeniusNormSq(self):
        out = 0.0
        for tensor in self:
            out += torch.norm(tensor, p=2) ** 2
        return out     

    def size(self):
        TL_size = 0
        for tensor in  self:
            tensor_size = 1
            for size_i in tensor.size():
                tensor_size *= size_i
            TL_size += tensor_size
        return TL_size 

class Network(nn.Module):
    "A generic class written for definiing a loss function F(θ; X) where X is the input data and θ is the parametr."
    def __init__(self, device):
        super(Network, self).__init__()
        self.device = device

    @torch.no_grad()
    def getParameters(self):
        """
            Return the model parameters as a single vectorized tensor.
        """
        #return torch.cat( [parameter.view(-1) for parameter in self.parameters()] ).unsqueeze(0)
        return TensorList([param.data for param in self.parameters()])


    @torch.no_grad()
    def getGrads(self):
        """
            Return gradient w.r.t. model parameters of all model layers  as a single vectorized tensor.
        """
        #print (torch.cat( [parameter.grad.view(-1) for parameter in self.parameters()], 0 ).unsqueeze(0).size())
     #   return torch.cat( [parameter.grad.view(-1) for parameter in self.parameters()], 0 ).unsqueeze(0)
        return  TensorList( [parameter.grad.clone()  for parameter in self.parameters()]  )

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
    def getJacobian_old(self, output, quadratic=False):
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
    def getJacobian(self, output, quadratic=False):
        """
            Return the Jacobian matrix evaluauted at output, if quadratic is True, the  it also return the suared of the Jacobian. 
        """

        bath_size, output_size = output.size()
        if bath_size != 1:
             raise Exception('Batch dimension is not equal to one.')

        Jacobian = ()
        for i in range(output_size):
            Jacobian_i_row = self._getJacobian_aux(output, i)
            if quadratic:
                SquaredJacobian = torch.ger(Jacobian_i_row.squeeze(0), Jacobian_i_row.squeeze(0))

            Jacobian += (Jacobian_i_row,)
        if not quadratic:
            return Jacobian
        return Jacobian, SquaredJacobian
    
                    
             
            
            
    @torch.no_grad()
    def vecMult(self, vec, output=None, Jacobian=None, left=False):
        """
           Multiply the Jacobian and the vector vec. Note that output must have batch dimension of 1.
           Jacobian is a list, where each element is a list of parameter grdaients.
           vec must be a TensorList object.
        """


        if Jacobian == None:
            if output == None:
                raise Exception('Neither output nor Jacobian is passed.')
           
            (bath_size, output_size) = output.size()
            if bath_size != 1:
                raise Exception('Batch dimension is not equal to one.')
            Jacobian = self.getJacobian(output)

        out = []
        for Jacobian_i in Jacobian:
            out.append( Jacobian_i * vec )
        return torch.tensor( out ).unsqueeze(0)
            
    @torch.no_grad()
    def saveStateDict(self, PATH):
        """
            Save the model parameters in path. This method saves the state_dict not the entire model.
        """
        torch.save(self.state_dict(), PATH)

    
    @torch.no_grad()
    def loadStateDict(self, PATH):
        """
            Load the model, where state_dict is stored in path.
        """
        self.load_state_dict(torch.load(PATH))
        self.eval()

    def forward(self, X):
        """Given an input X execute a forward pass."""
        pass

class Linear(Network):
    "A class for shallow Linear models; the input size is m anbd the output size is m_prime."
    def __init__(self, m , m_prime, device=torch.device("cpu")):
        super(Linear, self).__init__(device)
        self.m = m
        self.m_prime = m_prime
        self.fc1 = nn.Linear(m, m_prime)

    def forward(self, data):
        "Given an input X execute a forward pass."
        X, Y = data
        return Y - self.fc1(X) 

class AEC(Network):
    "A class for Autoencoders; the input size is m anbd the encoded size is m_prime."
    def __init__(self, m , m_prime, device=torch.device("cpu")):
        super(AEC, self).__init__(device)
        self.m = m
        self.m_prime = m_prime
        self.fc1 = nn.Linear(m, m_prime) 
        self.fc2 = nn.Linear(m_prime, m) 
    def forward(self, X):
        "Given an input X execute a forward pass."
        if type(X) == list:
            X = X[0]
        X = X.view(X.size(0), -1)
        Y = torch.sigmoid( self.fc1(X) )
        Y = self.fc2(Y)
        return Y - X 


class DAEC(Network):
    """
        A class for Denoising Autoencoders; the input size is m anbd the encoded size is m_prime.
    """
    def __init__(self, m , m_prime, noise='Gaussian', noise_level=1.e-3, device=torch.device("cpu")):
        super(DAEC, self).__init__(device)
        self.m = m
        self.m_prime = m_prime
        self.fc1 = nn.Linear(m, m_prime)
        self.fc2 = nn.Linear(m_prime, m)
        self.noise = noise
        if noise == 'Gaussian':
            self.noise_distr = torch.distributions.multivariate_normal.MultivariateNormal( torch.zeros(self.m), noise_level * torch.eye(self.m) )
        elif noise == 'Bernoulli':
            self.noise_distr = torch.distributions.bernoulli.Bernoulli( (1.0 - noise_level) * torch.ones(self.m)   )
    def forward(self, X):
        "Given an input X execute a forward pass."
       
        N = self.noise_distr.sample()
        if noise == 'Gaussian':
            X_corrupted = X + N
        elif noise == 'Bernoulli':
            X_corrupted = X * N
        Y = torch.sigmoid( self.fc1(X_corrupted) )
        Y = self.fc2(Y)
        return Y - X

class Embed(Network):
    """
        A class for Non-negative Matrix Factorization; the input size is m and the encoded size is m_prime.
    """
    def __init__(self, m , m_prime, device=torch.device("cpu")):
        super(Embed, self).__init__(device)
        #Dictionary size
        self.m = m
        #Embedding size
        self.m_prime = m_prime
        self.embedding = nn.Embedding(self.m, self.m_prime)
    def forward(self, X):
        return self.embedding( X )
        

       
        
     
    
        
    
 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    args = parser.parse_args() 
    model  = AEC(2, 1)

    theta = model.getParameters()
    theta_2 =  2* theta 
   
    x = torch.randn(1, 2)
    y = model(x)
    Jac = model.getJacobian(y)

    print(model.vecMult(vec = theta , Jacobian=Jac).size())
    


    #dataset = loadFile( args.input_file )
    #ds_loader = DataLoader(dataset, batch_size=1)
    #model.getJacobian(output)
   #print('took {}'.format(time.time() - tS))
    #for i, data in enumerate(ds_loader):
        
    #    img, lbl = data
    #    output = model(img)
    #    tS = time.time()
    #    jac = model.getJacobian(output)
    #    print(jac[0])
    #    print('data {} Jacobian computation now has taken {}'.format(i, time.time() - tS))
    
    
    
    
    
   
    


