# prepare data format for LSTM_STM.py
# edited by yyq, 2016-01-19
import cPickle
import gzip
import os

import numpy
import theano


def prepare_data(seqs, targets, queries, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swap the axis!
    """
    # x: a list of arrays with dim * different sequence lengths
    lengths = [len(s[1]) for s in seqs]
    dim_x = len(seqs[0])
    dim_q = len(queries[0])

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples, dim_x)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    y = numpy.zeros((maxlen, n_samples, dim_x)).astype(theano.config.floatX)
    q = numpy.zeros((maxlen, n_samples, dim_q)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx, :] = s.T
        x_mask[:lengths[idx], idx] = 1.
    
    for idx, s in enumerate(targets):
        y[:lengths[idx], idx, :] = s.T

    for idx, s in enumerate(queries):
        q[:lengths[idx], idx, :] = s.T

    return x, x_mask, y, q


def get_dataset_file(dataset, default_dataset, origin):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present

    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        import urllib
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    return dataset


def load_data(valid_portion=0.1, 
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    f = open('trainData1000.pkl', 'rb')
    g = open('testData1000.pkl', 'rb')

    train_x = cPickle.load(f)
    train_q = cPickle.load(f)
    train_y = cPickle.load(f)
    test_x = cPickle.load(g)
    test_q = cPickle.load(g)
    test_y = cPickle.load(g)
    f.close()
    g.close()

    # split training set into validation set
    n_samples = len(train_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_x = [train_x[s] for s in sidx[n_train:]]
    valid_y = [train_y[s] for s in sidx[n_train:]]
    valid_q = [train_q[s] for s in sidx[n_train:]]
    train_x = [train_x[s] for s in sidx[:n_train]]
    train_y = [train_y[s] for s in sidx[:n_train]]
    train_q = [train_q[s] for s in sidx[:n_train]]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x][1]))

    if sort_by_len:
        sorted_index = len_argsort(test_x)
        test_x = [test_x[i] for i in sorted_index]
        test_y = [test_y[i] for i in sorted_index]
        test_q = [test_q[i] for i in sorted_index]

        sorted_index = len_argsort(valid_x)
        valid_x = [valid_x[i] for i in sorted_index]
        valid_y = [valid_y[i] for i in sorted_index]
        valid_q = [valid_q[i] for i in sorted_index]

        sorted_index = len_argsort(train_x)
        train_x = [train_x[i] for i in sorted_index]
        train_y = [train_y[i] for i in sorted_index]
        train_q = [train_q[i] for i in sorted_index]

    train = (train_x, train_y, train_q)
    valid = (valid_x, valid_y, valid_q)
    test = (test_x, test_y, test_q)

    return train, valid, test
