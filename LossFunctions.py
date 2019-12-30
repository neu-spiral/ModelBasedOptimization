import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as FUNC



class LossFunction(nn.Module):
    "A generic class written for definiing a loss function F(θ; X) where X is the input data and θ is the parametr."
    def __init__(self):
        print("Initializing LossFunction.")
        super(LossFunction, self).__init__()
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
        self.fc2 = nn.Linear(m_prime, m) 
    def forward(self, X):
        "Given an input X execute a forward pass."
        X = torch.sigmoid( self.fc1(X) )
        X = self.fc2(X)
        return X
    def set(self, θ):
        "Set model parameters."
        ind = 0
        for parameter in self.parameters():
            if parameter.size() != θ[ind].size():
                print(ind)
                print (parameter.size(), θ[ind].size())
                raise Exception('Dimensions do not match')
            parameter.data =  θ[ind].data
            ind += 1
    def _vecMult_aux(self, vec, output, i, quadratic=False):
        """Multiply the Jacobian and the vector vec, if qudratic is True also compute the outer product of the Jacobian with itself. 
          The Jacobian is computed  ONLY w.r.t. i-th neuron in the output."""
        selctor = np.zeros(self.m)
        selctor[i] = 1
        selector = torch.tensor( selctor  )
        selector =  selector.view(1, -1)
        #Reset gradients to zero
        self.zero_grad()
        #Do a backward pass to compute the Jacobian
        #retain_graph is set to True as the auxiliary function is called multiple times.
        output.backward( selector ,retain_graph=True ) 
        #Compute grad w.r.t. parameters
        grad_i = 0.0
        if quadratic:
            outerProdAllParameters_i = []
        index = 0
        for parameter in self.parameters():
            grad_i += torch.dot(parameter.grad.view(-1), vec[index].view(-1) )
            index += 1
            if quadratic:
                #Compute the outer product.
                outerProdAllParameters_i.append( torch.ger( parameter.grad.view(-1), parameter.grad.view(-1) )  )
                
        if not quadratic:
            return grad_i
        else:
            return grad_i, outerProdAllParameters_i
    def vecMult(self, vec, output, quadratic=False):
        "Multiply the Jacobian and the vector vec, if qudratic is True also compute the outer product of the Jacobian with itself."
        bath_size, output_size = output.size()
        prod = torch.zeros(output_size)
        

        for i in range(output_size):
            if not quadratic:
                prod[i] = self._vecMult_aux(vec, output, i, False)
            else:
                prod[i], outerProdAllParameters_i = self._vecMult_aux(vec, output, i, True)
                if i == 0:
                    outerProdAllParameters_tot = outerProdAllParameters_i
                else:
                    for j  in range(len( outerProdAllParameters_i   )):
                        outerProdAllParameters_tot[j].add_( outerProdAllParameters_i[j]   )

        if not quadratic:
            return prod
        else:
            return prod, outerProdAllParameters_tot


       
        
     
    
        
    
 
        

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
    prod, outerProdAllParameters_tot = AE.vecMult( list(AE.parameters() ), sample_output, True)
    print(outerProdAllParameters_tot)

    
    
    
    
    
    
    
   
    


