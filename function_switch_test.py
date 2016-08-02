# test the returned value from a function to be stacked.
# edited by yyq, 2016-4-27

import theano

def build_a_function(n, outside_tensor):
    a = theano.tensor.vector()
    b = theano.tensor.vector()
    c = theano.tensor.vector()
    d = theano.tensor.vector()
    
    p = theano.tensor.switch(n, a, outside_tensor)
    q = theano.tensor.switch(n, b, outside_tensor)
    r = theano.tensor.switch(n, c, outside_tensor)
    s = theano.tensor.switch(n, d, outside_tensor)
    
    return a, b, c,d, [p,q,r, s]

n = theano.shared(0.)
outside_tensor = theano.tensor.vector()
a,b,c,d,e = build_a_function(n,outside_tensor)
m = theano.tensor.stack(e)
    
f = theano.function([a,b,c,d,outside_tensor], m)

n.set_value(1.)
print f([1,1],[2,2],[3,3],[4,4],[5,5])
n.set_value(0.)
print f([1,1],[2,2],[3,3],[4,4],[5,5])

