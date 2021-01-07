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

    def __iadd__(self, tensorList_other):
        outTL = []
        for ind, tensor in enumerate(self):
            outTL.append( tensor.add_( tensorList_other[ind]  )  )
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

    def __truediv__(self, other):
        assert type(other) in [float, int], 'Type is not valid'
 
        other = 1./other
        return self.__mul__(other)
  
    def __rmul__(self, other):
        if type(other) in [float, int]:
            return self.__mul__( other  )        

    def __pow__(self, other):
        out = []
        for tensor in self:
            out.append(tensor ** other)
        return TensorList(out)

    def outProd(self, other=None):
        """
            Given another TensorList, compute the outer prodcut between them. Here each of the TensorLists is trated as a long 1-d tensor (vector). Then outer-product 
            of these vectors is returned. 
            If other is None, return ouuterproduct of the TensorList with itself.
        """

        if other is  not None:
            assert self.size() == other.size(), "TensorLists must be of the same size"

        vec_TL = self.getTensor()
     
        if other is None:
            return torch.ger(vec_TL, vec_TL)

        else:
            other_vec_TL = other.self.getTensor()
            return torch.ger(vec_TL, other_vec_TL)

        

    def getTensor(self):
        "Return a tneosr that is the vectorization and concatenation of th tensors in the TensorList."
        list_of_tens = []
        for tensor in self:
            list_of_tens.append(tensor.view(-1))
        return torch.cat(list_of_tens, 0)

    def frobeniusNormSq(self):
        out = 0.0
        for tensor in self:
            out += torch.norm(tensor, p=2) ** 2
        return out     

    def TLShape(self):
        """
          Return the shapes for tensors in TensorList.
        """
        return [tensor.shape for tensor in self]

    @staticmethod
    def formFromTensor(tensor, TL_shape):
        """
         Given a Tensor and a list of shapes, convert tensor into a TensorList.
        """

        tensor = tensor.view(-1)

        st_ind = 0

        newL = []

        for shape_i in TL_shape:
            newL.append(  torch.reshape(tensor[st_ind: st_ind + np.prod(shape_i)], shape_i ) )

            st_ind  += np.prod(shape_i)

        return TensorList( newL ) 

         
            


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
    def getParameters(self, trackgrad=False):
        """
            Return the model parameters as a single vectorized tensor.
        """
        #return torch.cat( [parameter.view(-1) for parameter in self.parameters()] ).unsqueeze(0)
        if trackgrad:
            outL = []
            for param in self.parameters():
                new_param = torch.zeros(param.size(), requires_grad=True)
                new_param.copy_( param.data )
                outL.append( new_param )
            return TensorList( outL  )
        else:
            return TensorList([param.data  for param in self.parameters()])

    


    @torch.no_grad()
    def getGrads(self):
        """
            Return gradient w.r.t. model parameters of all model layers  as a TensorList object.
        """
        return  TensorList( [parameter.grad.clone()  for parameter in self.parameters()]  )

    @torch.no_grad()
    def getGrads_old(self):
        """
            Return gradient w.r.t. model parameters of all model layers  as a vectorized tensor.
            NOTE: This this method is only for debugging and is inefficient in practice.
        """        
        return torch.cat( [parameter.grad.view(-1) for parameter in self.parameters()], 0 ).unsqueeze(0)

    @torch.no_grad()
    def setParameters(self, new_params):
        """Set model parameters."""
   
        for param_ind, param in enumerate(self.parameters()):
            param.copy_( new_params[param_ind].data  )
        logging.warning("Model parameters modified.")
            


   #*NOTE
    @torch.no_grad()
    def _getJacobian_aux(self, output, i, debug=False):
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
        if debug:
             return self.getGrads_old()
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
            Jacobian_i_row = self._getJacobian_aux(output, i, debug=True)
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
            Jacobian_i_row = self._getJacobian_aux(output=output, i=i)


            #compute the outer-product of Jacobian with itself transpose 
            if quadratic:
                if i == 0:
                    SquaredJacobian = Jacobian_i_row.outProd()
                else:
                    SquaredJacobian += Jacobian_i_row.outProd()

            Jacobian += (Jacobian_i_row,)

        if not quadratic:
            return Jacobian

        return Jacobian, SquaredJacobian
    
                    
             
            
            
    #@torch.no_grad()
    def vecMult(self, vec, output=None, left=False, Jacobian=None, trackgrad=False):
        """
           Multiply the Jacobian and the vector vec:

                   D_i \times vec, D_i \in R^{N \times d}, vec \in R^d,
            vec must be a TensorList object.

           if left:
   
                   D_i^T \times vec, D_i \in R^{N \times d}, vec \in R^N.
           vec must be a pytorch tensor object.

           Note that output must have batch dimension of 1.
           Jacobian is a list, where each element is a list of parameter grdaients.
        """


        if Jacobian == None:
            if output == None:
                raise Exception('Neither output nor Jacobian is passed.')
           
            (bath_size, output_size) = output.size()
            if bath_size != 1:
                raise Exception('Batch dimension is not equal to one.')
            Jacobian = self.getJacobian(output)

        #left multiplication
        if left:
            #assert that vec has the right dimensions
            #batch_size must be 1
            assert vec.size()[0] == 1, 'vec must have a batch dimension of 1.' 
            #the number of rows in vec must be equal to the number of elements (rows) in Jacobian
            assert vec.size()[1] == len(Jacobian), 'number of rows do not match' 

            for row_ind, Jacobian_i_row in enumerate(Jacobian):
                if row_ind ==0:
                    out = Jacobian_i_row * float(vec[0, row_ind])
                else:
                    out = out + Jacobian_i_row * float(vec[0, row_ind])

            return out 

        #right multiplication 
        assert type(vec) == TensorList, "pass a TensorList as vec"        

        out = []
        for Jacobian_i in Jacobian:
            out.append( Jacobian_i * vec )
        if trackgrad:
           return out
        
        out =  torch.tensor( out).unsqueeze(0)
        return out

                    

     
            
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

        #flatten input
        X = X.view(X.size(0), -1)

        #apply the linear encoder followed by activation function 
        Y = torch.sigmoid( self.fc1(X) )

        #apply linear decoder 
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
        

class ConvAEC(Network):
    def __init__(self, k_in, k_h, kernel_x = 3, kernel_y = 3, padding=0, strides = 1, device=torch.device("cpu")):        
        super(ConvAEC, self).__init__(device)
 
        self.k_in = k_in
        self.k_h = k_h
        self.conv_lyr = torch.nn.Conv2d(k_in, 
                                        k_h,
                                        (kernel_x, kernel_y),
                                        stride = strides,
                                        padding = padding
                                       )

        self.deconv_lyr = torch.nn.ConvTranspose2d(k_h,
                                        k_in,
                                        (kernel_x, kernel_y),
                                        stride = strides,
                                        padding = padding
                                       )

    def forward(self, X):
        H = self.conv_lyr( X )
        #H = torch.nn.ReLU( H )
     
        Y = self.deconv_lyr(H)

        #Y = torch.nn.ReLU( Y )
        return torch.flatten(Y, start_dim = 1) - torch.flatten(X, start_dim = 1)
        



 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    args = parser.parse_args() 
    model  = ConvAEC(1, 8)



    x = torch.randn(1, 1, 10, 5)

    out = model(x)


    Jac, sqJac = model.getJacobian(out, quadratic = True)

    Jac_old = model.getJacobian_old(out, quadratic = False)

    print( torch.matmul(torch.transpose(Jac_old, 0, 1), Jac_old)  - sqJac)
 #   print(sqJac.shape)

    #print("Jacobian is", torch.tensor([tl.getTensor() for tl in Jac]) )

    #print("old Jacobian is", Jac_old)

    #print("Suqared Jac is ", sqJac)


     

    

   # ds_loader = DataLoader(dataset, batch_size=1)
    #model.getJacobian(output)
   #print('took {}'.format(time.time() - tS))
   # for i, data in enumerate(ds_loader):
        
   #     img, lbl = data
   #     output = model(img)
   #     tS = time.time()
   #     jac = model.getJacobian(output)
    #    print(jac[0])
   #     print('data {} Jacobian computation now has taken {}'.format(i, time.time() - tS))
    
    
    
    
    
   
    


