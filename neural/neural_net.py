from Value import Value
import numpy as np
import matplotlib.pyplot as plt
from Neuron import MLP
from GD import GD


def train(model,epoch,labels,lr=0.01):
    
    for e in range(epoch):
        ypreds = [model(x) for x in xs]
        l = [(ypred-label)**2 for label,ypred in zip(labels,ypreds)]
        Loss = sum(l,Value(0.0))
        Loss.backward()
        GD(model.parameters(),lr)
        print(f"epoch: {e+1} , Loss: {Loss.data}")
    
def children(root,nodes=[]):
    
    
    if root not in nodes:
        nodes.append(root)
        
    for n in root._prev:
        children(n,nodes)
    
    return nodes
        
        


model = MLP(3,[4,4,1])

xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]

ys = [-1.0,-1.0,1,1.0]

train(model,200,ys)

ypreds = [model(x).data for x in xs]

print(ypreds)

        
