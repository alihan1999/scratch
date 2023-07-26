import math

class Value:
    
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
        self._backward= lambda:None
        
        self.grad = 0.0
        
    def __repr__(self):
        
        return f"Value(data={self.data})"
    
    def __add__(self,second):
        
        second = second if isinstance(second,Value) else Value(second)
        
        out = Value(self.data+second.data,(self,second),'+')
        
        def _backward():
            self.grad += out.grad
            second.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self,second):
        
        
        second = second if isinstance(second,Value) else Value(second)
        
        out = Value(self.data*second.data,(self,second),'*')
        def _backward():
            self.grad += out.grad * second.data
            second.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    
    def tanh(self):
        
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        
        def _backward():
            self.grad += out.grad * (1-t**2)
        out._backward = _backward
        
        return out
    
    def expression(self):
        
        if len(self._prev)==0:
            return f"{self.label}({self.data})"
        nodes = [d for d in self._prev]
        return f"{nodes[0].expression()} {self._op} {nodes[1].expression()}"
    
    def backward(self):
        topo = []

        visited = set()

        def Build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    Build(c)
                topo.append(v)
        Build(self)
        
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()
    
    def exp(self):
        x = self.data
        
        out = Value(math.exp(x),(self,),'exp')
        
        def _backward():
            self.grad += out.grad * out.data
        
        out._backward = _backward
        
        return out
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        
        out = Value(self.data**other,(self, ),f'**{other}')
        
        def _backward():
            self.grad += out.grad * other*(self.data**(other-1))
        out._backward = _backward
        
        return out
    
    
    def __neg__(self):
        return self*-1
    
    def __sub__(self,other):
        return self+(-other)
    
    def __truediv__(self,other):
        
        return self*other**-1
    
    def __rmul__(self,other):
        
        return self*other