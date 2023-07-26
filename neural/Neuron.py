import random
from Value import Value

class Neuron:
    
    def __init__(self,nin):
        
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
   
    
    def __call__(self,x):
        res = Value(0.0)
        r = [self.w[i]*x[i] for i in range(len(self.w))]
        for i in r:
            res += i
        res2 = res + self.b
        ans = res2.tanh()
        
        return ans
    
     
    def parameters(self):
        return self.w + [self.b]

class Layer:
    
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self,x):
        
        outs = [n(x) for n in self.neurons]
        if len(outs)==1:
            outs = outs[0]
        return outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    
    def __init__(self,nin,nouts):
        
        sz = [nin] + nouts
        
        self.Layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self,x):
        
        for layer in self.Layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for l in self.Layers for p in l.parameters()]