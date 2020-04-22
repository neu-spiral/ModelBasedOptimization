import torch 
import torch.optim as optim


torch.manual_seed(1993)
a = torch.randn(2, requires_grad = True)

print(a.requires_grad)

b = a**2
optimizer = optim.SGD([a], lr=0.001)
for i in range(3):
    #print(a, a.grad)
    optimizer.zero_grad()
    print(a) 
    z = b.mean()
    Y = z**2
    Y.backward(retain_graph=True)
    optimizer.step()

