import numpy as np

class Bandit:
    
    
    def __init__(self,m,i,es=0):
        
        self.m=m
        
        self.m_estimate = es
        
        self.trials = 0
        
        self.i = i
        
        
    
    def play(self):
        
        return np.random.randn() + self.m
    
    def update(self,x):
        
        self.trials+=1
        
        self.m_estimate = (self.m_estimate*(self.trials-1)+x) * (1/self.trials)
   