# test feed_forward function building.
# edited by yyq, 2016-04-29


import theano
import theano.tensor as tensor
import numpy
from collections import OrderedDict

def feed_forward_stm(use_noise, pf0, pf1, m, proj, q, mask):
    nsteps = q.shape[0]
    def ff_step(single_q, single_m, ev1, ev2, single_proj):
        medium = single_q + single_proj
        # medium_ = tensor.switch(use_noise, medium, ev1)
        out = m['params'] * medium * single_m
        # out_ = tensor.switch(use_noise, out, ev2)
        return medium, out
    rval, updates = theano.scan(ff_step, sequences=[q, mask, pf0,pf1], name='feed_forward_layers', n_steps=nsteps, non_sequences=[proj])
    return rval[0], rval[1] 


use_noise = theano.shared(0.)
params_ = numpy.random.randn(5).astype('float32')
params = theano.shared(params_, name='params')
m = OrderedDict()
m['params'] = params
q = tensor.alloc(0., 3,5)
proj = tensor.alloc(0., 5)
pf0 = tensor.alloc(0., 3, 5)
pf1 = tensor.alloc(0., 3, 5)
mask = tensor.alloc(0.,3,5)
use_noise.set_value(1.)

mediums, outs = feed_forward_stm(use_noise, pf0, pf1, m, proj, q,mask)

print mediums.ndim, outs.ndim
grads = tensor.grad(mediums.sum().sum() + outs.sum().sum(),wrt=m.values())
f_grad = theano.function([pf0,pf1,proj,q,mask], grads,name='f_grad')


