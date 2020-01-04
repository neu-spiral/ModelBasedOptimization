import torch
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
        return torch.cat( [parameter.grad.view(-1) for parameter in self.parameters()] )   
    def set(self, θ):
        "Set model parameters."
        ind = 0
        for parameter in self.parameters():
            if parameter.size() != θ[ind].size():
                raise Exception('Dimensions do not match')
            parameter.data =  θ[ind].data
            ind += 1

    def _vecMult_aux(self, vec, output, i, quadratic=False):
        """Multiply the Jacobian and the vector (1-d tensor) vec, if qudratic is True also compute the outer product of the Jacobian with itself. 
          The Jacobian is computed  ONLY w.r.t. i-th neuron in the output."""

        
        #Compute the i-th column in the Jacobian matrix, i.e., the grdaient of the i-th neuron in the output layer w.r.t. model parameters
        selctor = np.zeros( output.size()[-1]  )
        selctor[i] = 1
        selector = torch.tensor( selctor  )
        selector =  selector.view(1, -1)
        #Reset gradients to zero
        self.zero_grad()
        #Do a backward pass to compute the Jacobian
        #retain_graph is set to True as the auxiliary function is called multiple times.
        output.backward(selector, retain_graph=True ) 

        #Get grads
        allParameterGrads_i = self.getGrads()
        #Do inner prod 
        vecMult_i = torch.dot(allParameterGrads_i, vec)

        if not quadratic:
            return vecMult_i
        else:
            #Concatenate all gradients 
            return vecMult_i, torch.ger(allParameterGrads_i, allParameterGrads_i)
    def vecMult(self, vec, output, quadratic=False):
        "Multiply the Jacobian and the vector vec, if qudratic is True also compute the outer product of the Jacobian with itself."
        bath_size, output_size = output.size()
        prod = torch.zeros(output_size)
        

        for i in range(output_size):
            if not quadratic:
                prod[i] = self._vecMult_aux(vec, output, i, False)
            else:
                prod[i], outerProd_i = self._vecMult_aux(vec, output, i, True)
                if i == 0:
                    outerProd = outerProd_i
                else:
                    outerProd.add_( outerProd_i  )

        if not quadratic:
            return prod
        else:
            return prod, outerProd
    def forward(self, X):
        "Given an input X execute a forward pass."
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
    for parm in AE.parameters():
        print (parm)
    
    prod, outerProdAllParameters_tot = AE.vecMult(vec , sample_output, True)

    
    
    
    
    
    
    
   
    


