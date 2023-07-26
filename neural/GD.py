def GD(params,rate):
    
    for p in params:
        p.data += -1*rate*p.grad
        