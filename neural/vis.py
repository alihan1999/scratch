from graphviz import Digraph


def trace(root):
    
    nodes,edges = set(),set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes,edges

def draw_graph(root):
    
    g = Digraph(format='svg',graph_attr={'rankdir':'LR'})
    
    nodes,edges = trace(root)
    
    for n in nodes:
        
        nid = str(id(n))
        
        g.node(name=nid,label="{%s | data %.4f | grad %.4f}" % (n.label,n.data,n.grad),shape = 'record')
        
        if n._op:
            g.node(name = nid + n._op,label = n._op)
            g.edge(nid+n._op,nid)
    for u,v in edges:
        g.edge(str(id(u)),str(id(v))+v._op)
    
    return g
