import torch
import argparse
from helpers import loadFile
import logging
from torch.multiprocessing import Process
import os
import random
import torch.distributed as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, device = torch.device("cpu")):
        super(Network, self).__init__()
        self.device = device


    @torch.no_grad()
    def getParameters(self, trackgrad=False):
        """
            Return the model parameters as a single vectorized tensor.
        """
        #return torch.cat( [parameter.view(-1) for parameter in self.parameters()] ).unsqueeze(0)
        if trackgrad:
            #return self.parameters()

            return TensorList([param  for param in self.parameters()])
            #outL = []
            #for param in self.parameters():
            #    new_param = torch.zeros(param.size(), requires_grad=True, device = self.device)
#
             #   new_param.copy_( param.data )
             #   outL.append( new_param )

            #return TensorList( outL  )
        else:
            return TensorList([param.data  for param in self.parameters()])

    


    @torch.no_grad()
    def getGrads(self):
        """
            Return gradient w.r.t. model parameters of all model layers  as a TensorList object.
        """
        return  TensorList( [parameter.grad.clone().to(self.device)  for parameter in self.parameters()]  )

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
        selector = torch.zeros( output.size()[-1]  )
        selector[i] = 1
        #selector = torch.tensor( selctor  )
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

        #NOTE
        if trackgrad:
           return out
        
        out =  torch.tensor( out).unsqueeze(0).to( self.device )
        return out

                    

     
            
    @torch.no_grad()
    def saveStateDict(self, PATH):
        """
            Save the model parameters in path. This method saves the state_dict not the entire model.
        """
        torch.save(self.state_dict(), PATH)

    
    @torch.no_grad()
    def loadStateDict(self, PATH, device = None):
        """
            Load the model, where state_dict is stored in path.
        """

        if device is None:
            loaded_path = torch.load(PATH)
        else:
            loaded_path = torch.load(PATH, map_location = device)
        
        self.load_state_dict(loaded_path )
        self.eval()

    def forward(self, X):
        """Given an input X execute a forward pass."""
        pass

class Linear(Network):
    "A class for shallow Linear models; the input size is m anbd the output size is m_prime."
    def __init__(self, m , m_prime, hidden = 16, device=torch.device("cpu")):
        super(Linear, self).__init__(device)
        self.m = m
        self.m_prime = m_prime

        #fully connected layers 
        self.fc1 = nn.Linear(m, hidden)
        self.fc2 = nn.Linear(hidden, m_prime)

    def forward(self, data):
        "Given an input X execute a forward pass."
        X, Y = data

        #flatten input batch 
        X = torch.flatten(X, start_dim = 1)

        #first fully-connected layer
        h = F.softplus( self.fc1(X) )

        #second fully connected layer
        Y_model = F.softmax( self.fc2(h), dim = 1 )

        return Y - Y_model

    def eval_acc(self, data):

        X, Y = data

        pred_soft_lbl = Y - self.forward(data)


        pred_lbl = torch.argmax(pred_soft_lbl, 1)

        correct_pred = torch.sum(pred_lbl == Y)

        acc = correct_pred / Y.shape[0]

        return acc

class LinearSoft( Linear ):
    def forward(self, data):
        "Given an input X execute a forward pass."
        X, Y = data

        #flatten input batch 
        X = torch.flatten(X, start_dim = 1)

        #first fully-connected layer
        h = F.softplus( self.fc1(X) )

        #second fully connected layer
        Y_model = F.softplus( self.fc2(h) )

        return Y - Y_model

        

class Linear3(Network):
    "A class for shallow Linear models; the input size is m anbd the output size is m_prime."
    def __init__(self, m , m_prime, hidden2 = 64, hidden1 = 32, device=torch.device("cpu")):
        super(Linear3, self).__init__(device)
        self.m = m
        self.m_prime = m_prime
        self.fc1 = nn.Linear(m, hidden2)
        self.fc2 = nn.Linear(hidden2, hidden1)

        self.fc3 = nn.Linear(hidden1, m_prime)


    def forward(self, data):
        "Given an input X execute a forward pass."
        X, Y = data

        h = F.softmax( self.fc1(X) )

        h = F.softmax( self.fc2(h) )
          

        Y_model = F.softmax( self.fc3(h) )

        return Y - Y_model

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
        Y = F.sigmoid( self.fc1(X) )

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

class MF(Network):
    """
        A class for Non-negative Matrix Factorization; the input size is m and the encoded size is k.
    """
    def __init__(self, m , m_prime, k,  device=torch.device("cpu")):
        super(MF, self).__init__(device)

        #Dictionary size
        self.m = m

        #Original size
        self.m_prime = m_prime

        #Embedding size
        self.k = k

        self.embedding = nn.Embedding(self.m, self.k)

        self.embeddingCol = nn.Embedding(self.m_prime, self.k)


        #compute all columns embedding
        self.col_embedding = self.embeddingCol( torch.tensor( range(self.m_prime) ) )
        

    def forward(self, data):
        matrix, ind = data

        #embeddings for given rows
        emb_rows = self.embedding( ind )


        reconstructed_matrix = torch.matmul(emb_rows, torch.transpose(self.col_embedding, 0, 1) )
        

        return reconstructed_matrix - matrix
        

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
        

class ConvAEC2(Network):
    def __init__(self, k_in, k_h = 16, k_h2 = 4, kernel_x = 3, kernel_x2 = 3, kernel_y = 3, kernel_y2 = 3, kernel_pool = 2, device=torch.device("cpu")):
        super(ConvAEC2, self).__init__(device)
       
        #Encoder
        self.conv1 = nn.Conv2d(k_in, k_h, kernel_x)  
        self.conv2 = nn.Conv2d(k_h, k_h2, kernel_x2)
       # self.pool = nn.MaxPool2d(kernel_pool, kernel_pool)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(k_h2, k_h, kernel_y, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(k_h, k_in, kernel_y, stride=1)


    def forward(self, X):
        H = F.sigmoid(self.conv1(X))
       # H = self.pool(H)
        H = F.sigmoid(self.conv2(H))
       # H = self.pool(H)
        H = F.sigmoid(self.t_conv1(H))
        Y = F.sigmoid(self.t_conv2(H))
              
        return torch.flatten(Y, start_dim = 1) - torch.flatten(X, start_dim = 1)


class ConvAEC2Soft(Network):
    def __init__(self, k_in, k_h = 16, k_h2 = 4, kernel_x = 3, kernel_x2 = 3, kernel_y = 3, kernel_y2 = 3, kernel_pool = 2, device=torch.device("cpu")):
        super(ConvAEC2Soft, self).__init__(device)

        #Encoder
        self.conv1 = nn.Conv2d(k_in, k_h, kernel_x)
        self.conv2 = nn.Conv2d(k_h, k_h2, kernel_x2)
       # self.pool = nn.MaxPool2d(kernel_pool, kernel_pool)

        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(k_h2, k_h, kernel_y, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(k_h, k_in, kernel_y, stride=1)


    def forward(self, X):
        H = F.softplus(self.conv1(X))

        H = F.softplus(self.conv2(H))

        H = F.softplus(self.t_conv1(H))

        Y = F.softplus(self.t_conv2(H))

        return torch.flatten(Y, start_dim = 1) - torch.flatten(X, start_dim = 1)

class ConvLin(Network):
    def __init__(self, k_in, k_h, k_h2 = 4, kernel_x = 3, kernel_x2 = 3, h_in = 28, w_in = 28, device=torch.device("cpu")):

        super(ConvLin, self).__init__(device)

        #encoder
        self.conv1 = nn.Conv2d(k_in, k_h, kernel_x)
        self.conv2 =  nn.Conv2d(k_h, k_h2, kernel_x2)
   
        #dimensions first hidden layer
        h1 = (h_in - kernel_x) + 1
        w1 = (w_in - kernel_x) + 1

        #dimensions second hidden layer
        h2 = (h1 - kernel_x2) + 1
        w2 = (w1 - kernel_x2) + 1
         
        hidden_size = k_h2 * h2 * w2

        self.FC =  nn.Linear(hidden_size, h_in * w_in)

    def forward(self, X):

        #encoder
        H = F.sigmoid(self.conv1(X))
        H = F.sigmoid(self.conv2(H))

        #flatten
        H = torch.flatten(H, start_dim = 1)

        #decoder
        X_recon = self.FC(H)

        return X_recon - torch.flatten(X, start_dim = 1)

class ConvLinSoft(Network):
    def __init__(self, k_in, k_h, k_h2 = 4, k_h3 = 1,  kernel_x = 3, kernel_x2 = 3, kernel_x3 = 8,  h_in = 28, w_in = 28, dim_out = 10, device=torch.device("cpu")):

        super(ConvLinSoft, self).__init__(device)

        #conv layers
        self.conv1 = nn.Conv2d(k_in, k_h, kernel_x)
        self.conv2 =  nn.Conv2d(k_h, k_h2, kernel_x2)
        self.conv3 = nn.Conv2d(k_h2, k_h3, kernel_x3)

        #dimensions first hidden layer
        h1 = (h_in - kernel_x) + 1
        w1 = (w_in - kernel_x) + 1

        #dimensions second hidden layer
        h2 = (h1 - kernel_x2) + 1
        w2 = (w1 - kernel_x2) + 1

        #dimensions third hidden layer
        h3 = (h2 - kernel_x3) + 1
        w3 = (w2 - kernel_x3) + 1
    

        hidden_size = k_h3 * h3 * w3

        self.FC =  nn.Linear(hidden_size, dim_out)

    def forward(self, X):
        X_data, X_lbl = X


        #conv layers
        H = F.softplus(self.conv1(X_data))

        H = F.softplus( self.conv2(H) )

        H = F.softplus( self.conv3(H) )

        #flatten
        H = torch.flatten(H, start_dim = 1)
       
        #prediction
        pred_lbl = F.softmax( self.FC( H ) )

        

        return X_lbl - pred_lbl

    def eval_acc(self, X):
        X_data, X_lbl = X

        pred_soft_lbl = X_lbl - self.forward(X) 


        print(pred_soft_lbl.shape, pred_soft_lbl, X_lbl)
        pred_lbl = torch.argmax(pred_soft_lbl, 1)

        print(pred_lbl)
        correct_pred = torch.sum(pred_lbl == X_lbl)

        acc = correct_pred / X_lbl.shape[0]

        return acc 

class Conv1dLineSoft(Network):
    def __init__(self, dim_in, dim_out, c_numb = 1, kernel_x = 3, device=torch.device("cpu")):

        super(Conv1dLineSoft, self).__init__(device)

        self.conv1d = nn.Conv1d(1, c_numb, kernel_x)

        dim_h = ( (dim_in - kernel_x) + 1 ) * c_numb

        self.FC = nn.Linear(dim_h, dim_out)

    def forward(self, X):

        X_data, X_lbl = X


        if len(X_data.shape) == 2:
            X_data = torch.unsqueeze(X_data, 1)

        elif len(X_data.shape == 3) and X_data.shape[1] == 1:
            X_data = X_data
  
        else:
            print("Input must be either 2-d or 3-d with the channel dimension of 1 (the first dimension is batch)")
            raise ValueError 
        
        H = F.softplus( self.conv1d(X_data) )

        H = torch.flatten(H, start_dim = 1)

        H = self.FC( H )

        X_lbl_pred = F.softplus( H )

        return X_lbl - X_lbl_pred
        
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    args = parser.parse_args() 

     

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  

    x = torch.randn(1, 280)

    y = torch.randn(1, 16)


    model = Conv1dLineSoft( 280, 16 )

    out =  model( (x, y) ) 

    jac, sqJac = model.getJacobian( out, quadratic = True )

    print(sqJac.shape)

    #10 rows, 6 cols, embed size 3
#    model  = MF(100, 6, 3)

#    matrix = torch.randn(100, 6)

#    ind = torch.randint(low = 0, high = 99, size = (1,))


#    mat_ind = matrix[ ind ]

#    data = (mat_ind, ind)

#    out = model( data ) 

   # jac = model.getJacobian(out)

   # print(jac)

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
    
    
    
    
    
   
    


