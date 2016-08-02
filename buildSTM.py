# build a model of linear STM and train 
# yyq, 20160114

import numpy as np
import cPickle
import theano
import theano.tensor as T

# create np random
rng = np.random

# hyper-parameters
epoches = 100
lrate = 0.05
maxLen = 100
dataDim = 50
memCols = 200


# initialize parameters
def ParamInit(maxLen, dataDim, memCols):
	params = {}
	# create W1
	params['W1'] = theano.shared(rng.randn(maxLen,memCols), name='W1')
	# create W2
	params['W2'] = theano.shared(rng.randn(maxLen,memCols), name='W2')
	return params

# use scan to compute prediction and cost
def STMLayer(h_in, h_key, params, maxLen, dataDim, memCols):
	nsteps = maxLen
	def OneStep(h_t, m_t):
		m_t_ = m_t + T.dot(h_t, params['W1'])
		return m_t_
	m_t_, updates = theano.scan(OneStep, 
								sequences=[h_in],
								outputs_info=[T.alloc(np.zeros(dataDim,memCols))]
								n_steps = nsteps)
	
	
	

# build theano graph
def BuildSTM(params, maxLen, dataDim, memCols)
	h_in = T.matrix('h_in', dtype=config.floatX)
	h_key = T.matrix('h_key', dtype=config.floatX)
	h_pre, cost = STMLayer(h_in, h_key, params, maxLen, dataDim, memCols)

