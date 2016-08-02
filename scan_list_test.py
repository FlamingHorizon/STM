# test whether theano.scan can take step function which returns a list.
# edited by yyq, 2016-4-27

import theano
import theano.tensor as T
import numpy

def step(a, b):
    c = 2 * (a + b)
    d = a + b
    e = a - b
    f = a[1] * b[1]
    q = [d,e,f]
    return c, d,e,f
m = T.matrix()
n = T.matrix()

rval, updates = theano.scan(step, sequences=[m,n])

f = theano.function([m, n], rval)

print 'function created~'

m = numpy.array([[1,2,3,4,5,6],[2,3,4,5,6,7]]).astype('float32')
n = numpy.array([[0,1,2,3,4,5],[0,1,2,3,4,5]]).astype('float32')

print type(f(m,n))
print f(m, n)
