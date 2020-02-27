import argparse
from torch import distributed, nn
import os
import  torch.utils
from torchvision import datasets, transforms
from LossFunctions import AEC
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(1993)

def run_process(rank, args, dataset):
    print("Running process %d" %rank)
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu")
    #device = torch.device("cpu")
    data_sampler  = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=1)


    model = AEC(args.m, args.m_prime)
    model = model.to(device) 

    #Share the parameters 
    params = []
    for param in model.parameters():
        if rank != 0:
            param = torch.zeros( param.size() )
            param = param.to(device) 
            params.append( param )
        torch.distributed.broadcast(param, src=0)  
    if rank != 0:
        model.setParameters(params) 
    print (model.getParameters())

   

    for data in data_loader:
        data = data.to(device)
        output = model(data)     
        output = output.to(device)
        jacb = model.getJacobian(output, device=device)
        break
    print("At last line in run_process")

     




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--m_prime", type=int,  default=2)
    args = parser.parse_args()

    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    

    dataset =  torch.load(args.input_file)
    run_process(args.local_rank, args, dataset)
   
