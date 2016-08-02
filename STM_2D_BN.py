'''
Build a STM with two-grid-LSTM and FeedForward Network
Use tied weight for every dimension
Use the Time Average of [h1,c1,h2,c2] as data representation
Added batch nomalization according to paper : 'recurrent batch nomalization'
edited by yyq at 2016-04-20
'''
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import scipy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import stm_prepare as imdb

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)} # imdb.xxx are just function names to be called.

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    #lstm
    params = param_init_grid3(options,
                              params,
                              prefix=options['encoder'])
    # feedforward
    params = param_init_ff(options, params, prefix=options['encoder'])

    # batch nomalization
    params = param_init_bn(options, params, prefix=options['encoder'])


    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def ortho_weight_NS(mdim, ndim): # for m != n, non-square. return a matrix the same shape as (m,n)
    W = numpy.random.randn(mdim, ndim)
    u,s,v = scipy.linalg.svd(W, full_matrices=False)
    if mdim >= ndim:
        return u.astype(config.floatX)
    else:
        return v.astype(config.floatX)

def random_weight_NS(mdim, ndim):
    W = numpy.random.rand(mdim,ndim) * 2 - 1
    return W
    

# initialize the params of 3-grid lstm network
def param_init_grid3(options, params, prefix='lstm'):

    # W converts data into the same dim as hidden layer. Only used once in the input grid side.
    W = random_weight_NS(options['dim_proj'], options['dim_hidden'])
    params[_p(prefix, 'W')] = W
    
    # U1 is the LSTM params in each grid side. They are shared in this experiment by all blocks.
    U1 = numpy.concatenate([ortho_weight_NS(2 * options['dim_hidden'], options['dim_hidden']),
                            ortho_weight_NS(2 * options['dim_hidden'], options['dim_hidden']),
                            ortho_weight_NS(2 * options['dim_hidden'], options['dim_hidden']),
                            ortho_weight_NS(2 * options['dim_hidden'], options['dim_hidden'])], axis=1)
    params[_p(prefix, 'U1')] = U1
    print 'U1.shape:' 
    print U1.shape
    
    b1 = numpy.zeros((4 * options['dim_hidden'],))
    params[_p(prefix, 'b1')] = b1.astype(config.floatX)
    
    return params

# initialize the params of ff network
def param_init_ff(options, params, prefix="lstm"):
    """
    Init FF parameter:
    :see: init_params
    """
    W_ff_h1 = 0.1 * numpy.random.randn(4 * options['dim_hidden'],
                                options['dim_ff_hidden1']).astype(config.floatX)
    W_ff_h2 =  0.1 * numpy.random.randn(options['dim_ff_hidden1'],
                                options['dim_ff_hidden2']).astype(config.floatX)
                                
    W_ff_h3 = 0.1 * numpy.random.randn(options['dim_ff_hidden2'],
                                options['dim_ff_hidden3']).astype(config.floatX)

    ff_q_emb = numpy.random.randn(options['dim_q'],
                                options['dim_q_emb']).astype(config.floatX)
    W_ff_q = 0.1 * numpy.random.randn(options['dim_q_emb'],
                                options['dim_ff_hidden1']).astype(config.floatX)

    W_ff_o = numpy.random.randn(options['dim_ff_hidden3'],
                                options['dim_proj']).astype(config.floatX)
    b_ff_h1 = numpy.zeros(options['dim_ff_hidden1']).astype(config.floatX)
    b_ff_o = numpy.zeros(options['dim_proj']).astype(config.floatX)

    params['W_ff_h1'] = W_ff_h1
    params['ff_q_emb'] = ff_q_emb
    params['W_ff_q'] = W_ff_q
    params['W_ff_h2'] = W_ff_h2
    params['W_ff_h3'] = W_ff_h3
    params['W_ff_o'] = W_ff_o
    params['b_ff_h1'] = b_ff_h1
    params['b_ff_o'] = b_ff_o
    return params

def param_init_bn(options, params, prefix="lstm"):
    # scale parameters at each layer
    gamma_h1 = options['gamma_init_bn'] * numpy.ones(4 * options['dim_hidden']).astype(config.floatX)
    gamma_h2 = options['gamma_init_bn'] * numpy.ones(4 * options['dim_hidden']).astype(config.floatX)
    gamma_l1 = options['gamma_init_bn'] * numpy.ones(options['dim_ff_hidden1']).astype(config.floatX)
    gamma_l2 = options['gamma_init_bn'] * numpy.ones(options['dim_ff_hidden2']).astype(config.floatX)
    gamma_l3 = options['gamma_init_bn'] * numpy.ones(options['dim_ff_hidden3']).astype(config.floatX)
    gamma_o = options['gamma_init_bn'] * numpy.ones(options['dim_proj']).astype(config.floatX)
    
    b_l2 = options['b_init_bn'] * numpy.ones(options['dim_ff_hidden2']).astype(config.floatX)
    b_l3 = options['b_init_bn'] * numpy.ones(options['dim_ff_hidden3']).astype(config.floatX)
    
        
    
    params['gamma_h1'] = gamma_h1
    params['gamma_h2'] = gamma_h2
    params['gamma_l1'] = gamma_l1
    params['gamma_l2'] = gamma_l2
    params['gamma_l3'] = gamma_l3
    params['gamma_o'] = gamma_o
    
    params['b_l2'] = b_l2
    params['b_l3'] = b_l3
    
    return params
    
    
def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

def grid_lstm_block(use_noise, population, i, j, tparams, h_1, h_2, c_1, c_2, options, prefix='lstm', m=None): # m is the mask value of each sample in this time step
    print 'h_1:%d h_2:%d ' %(h_1.ndim, h_2.ndim)
    # c1 is the new cell value while c_1 is the input cell value from the previous time step
    h_1_emb = tensor.dot(h_1, tparams[_p(prefix, 'U1')][0:options['dim_hidden'], :])
    h_2_emb = tensor.dot(h_2, tparams[_p(prefix, 'U1')][options['dim_hidden']:, :])
    e_h_1_ = h_1_emb.mean(axis=0) # batch expectation, shall be returned and saved.
    v_h_1_ = ((h_1_emb - e_h_1_) ** 2).mean(axis=0) # batch variation, shall be returned and saved.
    e_h_1 = tensor.switch(use_noise, e_h_1_, population[i][j][0])
    v_h_1 = tensor.switch(use_noise, v_h_1_, population[i][j][1])
    e_h_2_ = h_2_emb.mean(axis=0) # batch expectation, shall be returned and saved.
    v_h_2_= ((h_2_emb - e_h_2_) ** 2).mean(axis=0) # batch variation, shall be returned and saved.
    e_h_2 = tensor.switch(use_noise, e_h_2_, population[i][j][2])
    v_h_2 = tensor.switch(use_noise, v_h_2_, population[i][j][3])
    h_1_hat = (h_1_emb - e_h_1) / (v_h_1 + 0.0001) ** 0.5
    h_2_hat = (h_2_emb - e_h_2) / (v_h_2 + 0.0001) ** 0.5
    preact1 = h_1_hat * tparams['gamma_h1'] + h_2_hat * tparams['gamma_h2'] + tparams[_p(prefix, 'b1')]
    i1 = tensor.nnet.sigmoid(_slice(preact1, 0, options['dim_hidden']))
    f1 = tensor.nnet.sigmoid(_slice(preact1, 1, options['dim_hidden']))
    o1 = tensor.nnet.sigmoid(_slice(preact1, 2, options['dim_hidden']))
    c1 = tensor.tanh(_slice(preact1, 3, options['dim_hidden']))
    c1 = f1 * c_1 + i1 * c1 
    c1 = m[:, None] * c1 + (1. - m)[:, None] * c_1
    h1 = o1 * tensor.tanh(c1)
    h1 = m[:, None] * h1 + (1. - m)[:, None] * h_1

    # preact2 = tensor.dot(h_in, tparams[_p(prefix, 'U1')]) + tparams[_p(prefix, 'b1')]
    # i2 = tensor.nnet.sigmoid(_slice(preact2, 0, options['dim_hidden']))
    # f2 = tensor.nnet.sigmoid(_slice(preact2, 1, options['dim_hidden']))
    # o2 = tensor.nnet.sigmoid(_slice(preact2, 2, options['dim_hidden']))
    # c2 = tensor.tanh(_slice(preact2, 3, options['dim_hidden']))
    # c2 = f2 * c_2 + i2 * c2 
    # c2 = m[:, None] * c2 + (1. - m)[:, None] * c_2
    # h2 = o2 * tensor.tanh(c2)
    # h2 = m[:, None] * h2 + (1. - m)[:, None] * h_2

    # take side 3 as output direction
    
    h1_emb_2 = tensor.dot(h1, tparams[_p(prefix, 'U1')][0:options['dim_hidden'], :])
    h_2_emb_2 = tensor.dot(h_2, tparams[_p(prefix, 'U1')][options['dim_hidden']:, :])
    e_h1_2_ = h1_emb_2.mean(axis=0) # batch expectation, shall be returned and saved.
    v_h1_2_ = ((h1_emb_2 - e_h1_2_) ** 2).mean(axis=0) # batch variation, shall be returned and saved.
    e_h1_2 = tensor.switch(use_noise, e_h1_2_, population[i][j][4])
    v_h1_2 = tensor.switch(use_noise, v_h1_2_, population[i][j][5])
    e_h_2_2_ = h_2_emb_2.mean(axis=0) # batch expectation, shall be returned and saved.
    v_h_2_2_ = ((h_2_emb_2 - e_h_2_2_) ** 2).mean(axis=0) # batch variation, shall be returned and saved.
    e_h_2_2 = tensor.switch(use_noise, e_h_2_2_, population[i][j][6])
    v_h_2_2 = tensor.switch(use_noise, v_h_2_2_, population[i][j][7])
    h1_hat_2 = (h1_emb_2 - e_h1_2) / (v_h1_2 + 0.0001) ** 0.5
    h_2_hat_2 = (h_2_emb_2 - e_h_2_2) / (v_h_2_2 + 0.0001) ** 0.5
    preact2 = h1_hat_2 * tparams['gamma_h1'] + h_2_hat_2 * tparams['gamma_h2'] + tparams[_p(prefix, 'b1')]
    i2 = tensor.nnet.sigmoid(_slice(preact2, 0, options['dim_hidden']))
    f2 = tensor.nnet.sigmoid(_slice(preact2, 1, options['dim_hidden']))
    o2 = tensor.nnet.sigmoid(_slice(preact2, 2, options['dim_hidden']))
    c2 = tensor.tanh(_slice(preact2, 3, options['dim_hidden']))
    c2 = f2 * c_2 + i2 * c2 
    c2 = m[:, None] * c2 + (1. - m)[:, None] * c_2 # m: 1 * n_samples; m[:,None]: n_samples * 1. then broadcast to n_samples * dim_hidden to multiply elementwise with c
    h2 = o2 * tensor.tanh(c2)
    h2 = m[:, None] * h2 + (1. - m)[:, None] * h_2

    print 'h1:%d h2:%d ' %(h1.ndim, h2.ndim)
    print 'h_2_emb_2.ndim is %d' %(h_2_emb_2.ndim)
    print 'e_h1_2_.ndim is %d' %(e_h1_2_.ndim)
    print 'e_h1_2.ndim is %d' %(e_h1_2.ndim)
    print 'pop[%d][%d][4].ndim is %d' %(i,j,population[i][j][4].ndim)

    return h1, h2, c1, c2, [e_h_1, v_h_1, e_h_2, v_h_2, e_h1_2, v_h1_2, e_h_2_2, v_h_2_2]

def grid_lstm_cube(use_noise, population, tparams, origin_data, options, prefix='lstm', mask=None):
    # size_1 = origin_data.shape[0]
    size_1 = options['grid_depth_1']
    size_2 = options['grid_depth_2']
    # size_3 = options['grid_depth_3']
    dim_hidden = options['dim_hidden']
    if origin_data.ndim == 3:
        n_samples = origin_data.shape[1]
    else:
        n_samples = 1
    assert mask is not None
    
    
    input_data = tensor.dot(origin_data, tparams[_p(prefix, 'W')])

    h_list_all = [] # four dim tensor of hidden states
    c_list_all = []
    h_input_all = []
    bnstates_all = []
    for i in range(size_1):
        h_list_all.append([])
        c_list_all.append([])
        bnstates_all.append([])
        for j in range(size_2):
            if i < 1:
                h_1 = tensor.alloc(numpy_floatX(0.), n_samples, dim_hidden)
                c_1 = tensor.alloc(numpy_floatX(0.), n_samples, dim_hidden)
            else:
                h_1 = h_list_all[i-1][j][0]
                c_1 = c_list_all[i-1][j][0]

            if j < 1:
                c_2 = tensor.alloc(numpy_floatX(0.), n_samples, dim_hidden)
                    #if k >= 1:
                    #    h_2 = tensor.alloc(numpy_floatX(0.), n_samples, dim_hidden)
                    #else:
                    #    h_2 = input_data[i]
                    #    h_input_all.append(h_2)
                    #    #h_2 = tensor.alloc(numpy_floatX(0.), n_samples, dim_hidden)
                h_2 = input_data[i]
                h_input_all.append(h_2)
            else:
                h_2 = h_list_all[i][j-1][1]
                c_2 = c_list_all[i][j-1][1]

            h1, h2, c1, c2, bnstates = grid_lstm_block(use_noise, population, i, j, tparams, h_1, h_2, c_1, c_2, options, prefix, mask[i, :])
                #print h1.ndim, h2.ndim, h3.ndim
            h_list_sides = tensor.stacklists([h1, h2])
                #print h_list_sides.ndim
            h_list_all[i].append(h_list_sides)
            c_list_sides = tensor.stacklists([c1, c2])
            c_list_all[i].append(c_list_sides)
            print type(bnstates)
            print 'bnstates[1].ndim is %d' %(bnstates[1].ndim)
            bnstates_ = tensor.stacklists(bnstates)
            bnstates_all[i].append(bnstates_)
    
    out_list_1 = [h_list_all[i][-1][1] for i in range(size_1)] # h_list_all: first three are cube index. last is the output dimension of that block, from 0 to 1.
    out_list_0 = [h_list_all[i][-1][0] for i in range(size_1)]
    out_list_c1 = [c_list_all[i][-1][1] for i in range(size_1)]
    out_list_c0 = [c_list_all[i][-1][0] for i in range(size_1)]
    print 'every h to stacklists is in dim: %d' %(h_list_all[-1][-1][1].ndim)
    proj_h1 = tensor.stacklists(out_list_1)
    proj_h0 = tensor.stacklists(out_list_0)
    proj_c1 = tensor.stacklists(out_list_c1)
    proj_c0 = tensor.stacklists(out_list_c0)
    proj = tensor.concatenate([proj_h1, proj_h0, proj_c1, proj_c0], axis=2)
    print 'proj.ndim is %d' %(proj.ndim)
    all_medium_states = tensor.stacklists(h_list_all)
    all_bn_states = tensor.stacklists(bnstates_all)
    print 'all_medium_states.ndim is %d' %(all_medium_states.ndim)
    h_input_all = tensor.stacklists(h_input_all)
    return proj, all_medium_states, h_input_all, all_bn_states

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_hidden']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_hidden']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_hidden']))
        c = tensor.tanh(_slice(preact, 3, options['dim_hidden']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_hidden = options['dim_hidden']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_hidden),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_hidden)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

#feed forward layer
def feed_forward_stm(use_noise, pf0, pf1, pf2, pf3, tparams, proj, q, options, prefix='ff',
                            mask=None):
    nsteps = q.shape[0]
    def ff_step(single_q, single_m, ev1, ev2, ev3, evo, single_proj):
    
        qemb_t = tensor.dot(single_q, tparams['ff_q_emb'])
        l1_t_linear = tensor.dot(single_proj, tparams['W_ff_h1']) + tensor.dot(qemb_t, tparams['W_ff_q'])
        print 'l1_t_linear.ndim: %d' %(l1_t_linear.ndim)
        e_l1_t_ = l1_t_linear.mean(axis=0)
        print 'e_l1_t_.ndim: %d' %(e_l1_t_.ndim)
        v_l1_t_ = ((l1_t_linear - e_l1_t_) ** 2).mean(axis=0)
        print 'v_l1_t_.ndim: %d' %(v_l1_t_.ndim)
        e_l1_t = tensor.switch(use_noise, e_l1_t_, ev1[0])
        print 'ev1[0].ndim: %d' %(ev1[0].ndim)
        print 'e_l1_t.ndim: %d' %(e_l1_t.ndim)
        v_l1_t = tensor.switch(use_noise, v_l1_t_, ev1[1])
        print 'ev1[1].ndim: %d' %(ev1[1].ndim)
        print 'v_l1_t.ndim: %d' %(v_l1_t.ndim)
        l1_t_hat = tparams['gamma_l1'] * ((l1_t_linear - e_l1_t) / (v_l1_t + 0.0001) ** 0.5) + tparams['b_ff_h1']
        print 'l1_t_hat.ndim: %d' %(l1_t_hat.ndim)
        h1_t = tensor.nnet.sigmoid(l1_t_hat)
        print 'h1_t.ndim: %d' %(h1_t.ndim)
        
        l2_t_linear = tensor.dot(h1_t, tparams['W_ff_h2'])
        e_l2_t_ = l2_t_linear.mean(axis=0)
        v_l2_t_ = ((l2_t_linear - e_l2_t_) ** 2).mean(axis=0)
        e_l2_t = tensor.switch(use_noise, e_l2_t_, ev2[0])
        v_l2_t = tensor.switch(use_noise, v_l2_t_, ev2[1])
        l2_t_hat = tparams['gamma_l2'] * ((l2_t_linear - e_l2_t) / (v_l2_t + 0.0001) ** 0.5) + tparams['b_l2']
        h2_t = tensor.nnet.sigmoid(l2_t_hat)

        l3_t_linear = tensor.dot(h2_t, tparams['W_ff_h3'])
        e_l3_t_ = l3_t_linear.mean(axis=0)
        v_l3_t_ = ((l3_t_linear - e_l3_t_) ** 2).mean(axis=0)
        e_l3_t = tensor.switch(use_noise, e_l3_t_, ev3[0])
        v_l3_t = tensor.switch(use_noise, v_l3_t_, ev3[1])
        l3_t_hat = tparams['gamma_l3'] * ((l3_t_linear - e_l3_t) / (v_l3_t + 0.0001) ** 0.5) + tparams['b_l3']
        h3_t = tensor.nnet.softplus(l3_t_hat)
        
        
        o_t_linear = tensor.dot(h3_t, tparams['W_ff_o'])
        e_o_t_ = o_t_linear.mean(axis=0)
        v_o_t_ = ((o_t_linear - e_o_t_) ** 2).mean(axis=0)
        e_o_t = tensor.switch(use_noise, e_o_t_, evo[0])
        v_o_t = tensor.switch(use_noise, v_o_t_, evo[1])
        o_t_hat = tparams['gamma_o'] * ((o_t_linear - e_o_t) / (v_o_t + 0.0001) ** 0.5) + tparams['b_ff_o']
        o_t = o_t_hat * single_m[:,None]
        
        return o_t, qemb_t, single_proj, h1_t, h2_t, h3_t, tensor.stacklists([e_l1_t, v_l1_t]), tensor.stacklists([e_l2_t, v_l2_t]), tensor.stacklists([e_l3_t, v_l3_t]), tensor.stacklists([e_o_t, v_o_t])

    
    rval, updates = theano.scan(ff_step, sequences=[q, mask, pf0,pf1,pf2,pf3], name='feed_forward_layers', n_steps=nsteps, non_sequences=[proj])
    return rval[0], rval[1], rval[2], rval[3], rval[4], rval[5], rval[6:]
    

# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_grid3, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y,cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, q, population, pf0, pf1, pf2, pf3, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()] # grad values wrt each weight. shared variables inited by all zeros.
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()] # accumulated delta-x E(delta-x ^ 2) from t=1 to now. shared varialbes inited by all zeros.
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()] # accumulated grad value E(g ^ 2)

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2,g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([x, mask, y, q, population, pf0, pf1, pf2, pf3], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared') 
                                    # f_grad_shared input x,mask,y,q, output cost, and execute the two updates above with real values.

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)] # this is the delta-x result
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)] # update E(x ^ 2) according to x.
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)] # update tparams, counting the new weights.

    f_update = theano.function([lr], [], updates=ru2up + param_up, # f_update take as input the learning rate but ignore it. then execute the above two updates.
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model_STM(tparams, options):
    # this function aims at building STM with one LSTM and one FF network.
    
    trng = RandomStreams(SEED)

    # Used for dropout.
    # Also used for batch nomalization
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype=config.floatX) # x is n_steps * n_samples * dimproj
    mask = tensor.matrix('mask', dtype=config.floatX) # n_steps * n_samples
    y = tensor.tensor3('y', dtype=config.floatX) # y is n_questions * n_samples * dimproj
    q = tensor.tensor3('q', dtype=config.floatX) # q is n_questions * n_samples * dimq
    # population is a tensor while population_ff is a list of tensors, each representing the e and v of a ff layer.
    population = tensor.alloc(numpy_floatX(0.), options['grid_depth_1'], options['grid_depth_2'], 8, 4 * options['dim_hidden'])
    pf0 = tensor.alloc(numpy_floatX(0.), options['dim_q'], 2, options['dim_ff_hidden1']) 
    pf1 = tensor.alloc(numpy_floatX(0.), options['dim_q'], 2, options['dim_ff_hidden2'])
    pf2 = tensor.alloc(numpy_floatX(0.), options['dim_q'], 2, options['dim_ff_hidden3'])
    pf3 = tensor.alloc(numpy_floatX(0.), options['dim_q'], 2, options['dim_proj'])

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    proj, all_medium_states, h_input_all, all_bn_states = grid_lstm_cube(use_noise, population, tparams, x, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    print 'proj.ndim is: %d' %(proj.ndim)
    print 'proj_0.ndim is: %d' %(proj[0].ndim)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
        # proj is n_samples * dim_hidden
    print 'proj.ndim is: %d' %(proj.ndim)
    
    # pred may be n_questions * n_samples * dimproj, same shape as y 
    # bn_states_ff is a list. each member is a time sequence of e and v.
    pred, qemb_seq, proj_seq, h1_seq, h2_seq, h3_seq, bn_states_ff = feed_forward_stm(use_noise, pf0, pf1, pf2, pf3, tparams, proj, q, options, prefix=options['encoder'],
                            mask=mask)
    print 'pred.ndim is: %d' %(pred.ndim)
    
    f_debug = theano.function([x, mask, y, q, population, pf0, pf1, pf2, pf3], [pred, qemb_seq, proj_seq, x, mask, y, q, h1_seq, h2_seq, h3_seq, all_medium_states, h_input_all], name='f_debug')
    print 'f_debug created~'

    f_retrieve = theano.function([x, mask, q, population, pf0, pf1, pf2, pf3], pred, name='f_retrieve')
    print 'f_retrieve created~'
    
    f_bn_store = theano.function([x, mask, q, population, pf0, pf1, pf2, pf3], [all_bn_states, bn_states_ff[0],bn_states_ff[1],
                                                                            bn_states_ff[2], bn_states_ff[3]], name='f_bn_store')
    print 'f_bn_store created~'

    cost = ((pred - y) ** 2).sum(axis=2)
    print 'cost.ndim is: %d' %(cost.ndim)
    cost = (cost * mask).sum(axis=0)
    print 'cost.ndim is: %d' %(cost.ndim)
    cost = cost / mask.sum(axis=0)
    print 'cost.ndim is: %d' %(cost.ndim)
    cost = cost.mean(axis=0)
    print 'cost.ndim is: %d' %(cost.ndim)

    f_prediction = theano.function([x, mask, y, q, population, pf0, pf1, pf2, pf3], cost, name='f_prediction')

    print 'f_prediction created~'

    return use_noise, x, mask, y, q, population, pf0, pf1, pf2, pf3, f_retrieve, f_prediction, cost, f_debug, f_bn_store, proj, pred


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, po, po_ff, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0.
    # compute the error of whole data by accumulating one batch a time.
    for _, valid_index in iterator:
        x, mask, y, q = prepare_data([data[0][t] for t in valid_index],
                                  [data[1][t] for t in valid_index],
                                  [data[2][t] for t in valid_index],
                                  maxlen=None)
        valid_err += f_pred(x, mask, y, q, po, po_ff[0], po_ff[1], po_ff[2], po_ff[3])
    valid_err = valid_err / len(iterator)
    return valid_err

def debug_print(f_pred, f_grad, prepare_data, data, iterator, verbose=False):
    # print some hidden values to confirm the rightness of model.
    _, valid_index = iterator[0]
    x, mask, y, q = prepare_data([data[0][t] for t in valid_index],
                              [data[1][t] for t in valid_index],
                              [data[2][t] for t in valid_index],
                              maxlen=None)
    [pred, qemb_seq, proj_seq, data_x, data_mask, data_y, data_q, h1_seq, h2_seq, h3_seq, all_medium_states, h_input_all] = f_pred(x, mask, y, q)
    #temp_grads = f_grad(x,mask,y,q)
    #print len(temp_grads)
    #for t in temp_grads:
    #    print t
    #    print '\n'
    #print 'the input data x:'
    #print data_x[:,0,:]   
    #print data_x[:,0,:].shape   
    #print 'the input mask :'
    #print data_mask[:,0]   
    #print data_mask[:,0].shape   
    #print 'the query q:'
    #print data_q[:,0,:]   
    #print data_q[:,0,:].shape  
    #print 'the lstm hidden layer after mean pooling:'
    #print proj_seq[:,0,:]   
    #print proj_seq[:,0,:].shape  
    #print 'the embedding sequence of q:'
    #print qemb_seq[:,0,:]   
    #print qemb_seq[:,0,:].shape   
    #print 'the h1 sequence:'
    #print h1_seq[:,0,:]   
    #print h1_seq[:,0,:].shape   
    #print 'the h2 sequence:'
    #print h2_seq[:,0,:]   
    #print h2_seq[:,0,:].shape   
    
    # show the medium states:
    #print 'h1 seq at highest layer:'
    #print all_medium_states[:,-1,-1,0,0,:]
    #print 'h2 seq at highest layer:'
    #print all_medium_states[:,-1,-1,1,0,:]
    #print 'h3 seq at highest layer:'
    #print all_medium_states[:,-1,-1,2,0,:]

    #print 'h1 seq at lowest layer:'
    #print all_medium_states[:,0,0,0,0,:]
    #print 'h2 seq at lowest layer:'
    #print all_medium_states[:,0,0,1,0,:]
    #print 'h3 seq at lowest layer:'
    #print all_medium_states[:,0,0,2,0,:]
    #
    #
    #print 'h1 seq at vertical dim:'
    #print all_medium_states[-1,1,:,0,0,:]
    #print 'h2 seq at vertical dim:'
    #print all_medium_states[-1,1,:,1,0,:]
    #print 'h3 seq at vertical dim:'
    #print all_medium_states[-1,1,:,2,0,:]

    #print 'the embedded input:'
    #print h_input_all[:,0,:]

    print 'the prediction:'
    print pred[:,0,:]   
    #print pred[:,0,:].shape   
    print 'the target:'
    print data_y[:,0,:]   
    #print data_y[:,0,:].shape

def e_v_update(f_bn_store, po_z, po_ff_z, prepare_data, data, iterator, batch_size):

    ev = numpy.zeros(options['grid_depth_1'], options['grid_depth_2'], 8, options[4 * 'dim_hidden'])
    ev_ff = [numpy.zeros(options['dim_q'], 2, options['dim_ff_hidden1']), 
                        numpy.zeros(options['dim_q'], 2, options['dim_ff_hidden2']),
                        numpy.zeros(options['dim_q'], 2, options['dim_ff_hidden3']),
                        numpy.zeros(options['dim_q'], 2, options['dim_proj'])]

    idx = 0
    for _, valid_index in iterator:
        idx = idx + 1
        x, mask, y, q = prepare_data([data[0][t] for t in valid_index],
                                  [data[1][t] for t in valid_index],
                                  [data[2][t] for t in valid_index],
                                  maxlen=None)
        po, po_ff_0, po_ff_1, po_ff_2, po_ff_3 = f_bn_store(x, mask, q, po_z, po_ff_z[0], po_ff_z[1], po_ff_z[2], po_ff_z[3])
        po_ff = [po_ff_0, po_ff_1, po_ff_2, po_ff_3]
        # update e:
        ev[:,:,0,:] = (ev[:,:,0,:] * (idx - 1) + po[:,:,0,:]) / idx
        ev[:,:,2,:] = (ev[:,:,2,:] * (idx - 1) + po[:,:,2,:]) / idx
        ev[:,:,4,:] = (ev[:,:,4,:] * (idx - 1) + po[:,:,4,:]) / idx
        ev[:,:,6,:] = (ev[:,:,6,:] * (idx - 1) + po[:,:,6,:]) / idx
        ev[:,:,1,:] = (ev[:,:,1,:] * (idx - 1) + (batch_size / batch_size - 1) * po[:,:,1,:]) / idx
        ev[:,:,3,:] = (ev[:,:,3,:] * (idx - 1) + (batch_size / batch_size - 1) * po[:,:,3,:]) / idx
        ev[:,:,5,:] = (ev[:,:,5,:] * (idx - 1) + (batch_size / batch_size - 1) * po[:,:,5,:]) / idx
        ev[:,:,7,:] = (ev[:,:,7,:] * (idx - 1) + (batch_size / batch_size - 1) * po[:,:,7,:]) / idx
        # update v:
        for j in range(len(ev_ff)):
            ev_ff[j][:,0,:] = (ev_ff[j][:,0,:] * (idx - 1) + po_ff[j][:,0,:]) / idx
            ev_ff[j][:,1,:] = (ev_ff[j][:,1,:] * (idx - 1) + (batch_size / batch_size - 1) * po_ff[j][:,1,:]) / idx
            
    return ev, ev_ff


def train_lstm(
    dim_proj=10,  # data dimension
    dim_hidden=100, # lstm unit number
    dim_ff_hidden1=150, # ff hidden layer size
    dim_ff_hidden2=150,
    dim_ff_hidden3=150,
    grid_depth_1=5, # time step of input data. to be removed because scan() can handle a tensorVariable as iterator.
    grid_depth_2=3, # number of grid lstm layers on certain sides. grid_depth_1 == time steps of data for default case.
    #grid_depth_3=2,
    dim_q=5, # dim of question data
    dim_q_emb=50, # dim of question embedding
    gamma_init_bn=0.1, # initial gamma for bn layers
    b_init_bn=0.1, # initial b for all bn layers
    patience=30,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=50,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=500,  # Compute the validation error after this number of update.
    saveFreq=1500,  # Save the parameters after every saveFreq updates
    batch_size=32,  # The batch size during training.
    valid_batch_size=50,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset) # return two functions from stm_prepare.py, named load_data and prepare_data

    print 'Loading data'
    train, valid, test = load_data(valid_portion=0.05) # train[xyq][data_id][time_step][data_value]
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx], [test[2][n] for n in idx])

    # import pdb
    # pdb.set_trace()

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, q, population, pf0, pf1, pf2, pf3, f_retrieve, f_prediction, cost, f_debug, f_bn_store, proj, pred) = build_model_STM(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['W_ff_o'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    
    # f_prediction returns cost without L2 normolization, which is used during validation and test; f_cost returns the whole cost.

    f_cost = theano.function([x, mask, y, q, population, pf0, pf1, pf2, pf3], cost, name='f_cost')
    print 'f_cost created~'

    print tparams.keys()
    print len(tparams)
    grads = tensor.grad(pred.sum().sum().sum(), wrt=tparams.values(), disconnected_inputs='ignore')
    f_grad = theano.function([x, mask, q,  population, pf0, pf1, pf2, pf3], grads, name='f_grad')
    # grads = tensor.grad(cost, wrt=tparams.values())
    # f_grad = theano.function([x, mask, y, q, population, pf0, pf1, pf2, pf3], grads, name='f_grad')
    print 'f_grad created~'

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, q, population, pf0, pf1, pf2, pf3, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)  # kf_valid: a list of tuples. [(id, batch_index)]
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)  # these two include all batch indices.

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None   # the best parameter ever acquired.
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size
    
    po_z = numpy.zeros(options['grid_depth_1'], options['grid_depth_2'], 8, options[4 * 'dim_hidden'])
    po_ff_z = [numpy.zeros(options['dim_q'], 2, options['dim_ff_hidden1']), 
                        numpy.zeros(options['dim_q'], 2, options['dim_ff_hidden2']),
                        numpy.zeros(options['dim_q'], 2, options['dim_ff_hidden3']),
                        numpy.zeros(options['dim_q'], 2, options['dim_proj'])]



    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    

    
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.) # use dropout for training update

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index] # x[data_id][timestep][value]
                q = [train[2][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y, q = prepare_data(x, y, q) # x[timestep][data_id][value]
                # print x[1], mask[1], y[1], q[1]
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y, q, po_z, po_ff_z[0], po_ff_z[1], po_ff_z[2], po_ff_z[3])
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost # eidx: idx of epoch; uidx: idx of this batch in this epoch; cost: train set cost

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'


            # after each epoch of multiple updates, do validation.
            # save all medium state e and v's.
            po, po_ff = e_v_update(f_bn_store, po_z, po_ff_z, prepare_data, train, kf, batch_size)
            use_noise.set_value(0.)  # dont use dropout for test
            train_err = pred_error(f_prediction, po, po_ff, prepare_data, train, kf)
            valid_err = pred_error(f_prediction, po, po_ff, prepare_data, valid,
                                   kf_valid)
            test_err = pred_error(f_prediction, po, po_ff, prepare_data, test, kf_test)
                    

            history_errs.append([valid_err, test_err])

            if (uidx == 0 or
                valid_err <= numpy.array(history_errs)[:,
                                                       0].min()):

                best_p = unzip(tparams)
                bad_counter = 0

            print ('Train ', train_err, 'Valid ', valid_err,
                   'Test ', test_err)
                    
            # debug_print(f_debug, f_grad,  prepare_data, train,kf)

            if (len(history_errs) > patience and
                valid_err >= numpy.array(history_errs)[:-patience,
                                                       0].min()):
                bad_counter += 1
                if bad_counter > patience:
                    print 'Early Stop!'
                    estop = True

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    use_noise.set_value(1.)
    po, po_ff = e_v_update(f_bn_store, po_z, po_ff_z, prepare_data, train, kf, batch_size)
    use_noise.set_value(0.)  # dont use dropout for test
    train_err = pred_error(f_prediction, po, po_ff, prepare_data, train, kf)
    valid_err = pred_error(f_prediction, po, po_ff, prepare_data, valid,
                                   kf_valid)
    test_err = pred_error(f_prediction, po, po_ff, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=50000,
        test_size=500,
    )
