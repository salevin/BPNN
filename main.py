"""
In terms of moving forward, I think the next step project-wise is to implement the algorithm in Zeiler's paper.  We will
also need to implement basic ideas of a backpropagation neural network.  This might be a good exercise for you:
   a)  See if you can train a backpropagation network with two inputs nodes (plus a bias node), four hidden units, and
   one output unit to learn XOR.
   b)  See if you can train a backpropagation network with four input nodes (plus a bias node), ten hidden units, and
   seven output units to learn the 4-bit Hamming code.  The second thing is just the result of multiplying any four bits
    (as a row vector) by the matrix

\mathbf{G} := \begin{pmatrix}
1 & 0 & 0 & 0 & 1 & 1 & 0 \\
0 & 1 & 0 & 0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1 & 1 & 1 & 1 \\
\end{pmatrix}_{4,7}

So to encode (1,0,1,1) you'd multiply (1, 0, 1, 1) G and get (1,0,1,1,0,1,0)  [all the operations are "mod 2", so you
take remainder of dividing by 2 after you do anything.]
"""

# Huge Credit to honnibal
# Made off of https://gist.github.com/honnibal/6a9e5ef2921c0214eeeb


import timeit

import numpy
import theano
import theano.tensor as T
import random

__author__ = 'slevin'


def make_data():
    train_set = _make_xor(900)
    valid_set = _make_xor(10000)
    test_set = _make_xor(500)

    return _make_array(train_set), _make_array(valid_set), _make_array(test_set)


def _make_array(xy):
    data_x, data_y = xy
    return zip(
        numpy.asarray(data_x, dtype=theano.config.floatX),
        numpy.asarray(data_y, dtype='int32'))


def _make_xor(size):
    x = []
    y = []
    for i in range(size):
        x.append([random.randint(0, 1), random.randint(0, 1)])
        y.append(x[i][0] ^ x[i][1])
    return x, y


def _init_logreg_weights(n_hidden, n_out):
    weights = numpy.zeros((n_hidden, n_out), dtype=theano.config.floatX)
    bias = numpy.zeros((10,), dtype=theano.config.floatX)
    return (
        theano.shared(name='W', borrow=True, value=weights),
        theano.shared(name='b', borrow=True, value=bias)
    )


def _init_hidden_weights(n_in, n_out):
    rng = numpy.random.RandomState(1234)
    weights = numpy.asarray(
        rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    bias = numpy.zeros((n_out,), dtype=theano.config.floatX)
    return (
        theano.shared(value=weights, name='W', borrow=True),
        theano.shared(value=bias, name='b', borrow=True)
    )


# Define how an input is fed through a layer of the network, and how a step of
# the stochastic gradient descent is computed.

# Note that these are *symbolic expressions* --- we are just compiling code here.
# These functions are only called during compile_model.  The *actual* feed-forward
# and SGD update procedures, which happen iteratively on each example, are
# Theano-internal.
def feed_forward(activation, weights, bias, input_):
    return activation(T.dot(input_, weights) + bias)


def sgd_step(param, cost, learning_rate):
    return param - (learning_rate * T.grad(cost, param))


# These are also symbolic.
def L1(L1_reg, w1, w2):
    return L1_reg * (abs(w1).sum() + abs(w2).sum())


def L2(L2_reg, w1, w2):
    return L2_reg * ((w1 ** 2).sum() + (w2 ** 2).sum())


def compile_model(n_in, n_classes, n_hidden, learning_rate, L1_reg, L2_reg):
    '''Compile train and evaluation functions, which we'll then call iteratively
    to train the parameters.  This function is called exactly once --- think of
    it like a compiler.  We declare variables, allocate memory, and define some
    computation.
    '''
    # allocate symbolic variables for the data
    x = T.vector('x')  # Features
    y = T.iscalar('y')  # (Gold) Label

    # Allocate and initialize weights.  These are stored internally, and updated.
    hidden_W, hidden_b = _init_hidden_weights(n_in, n_hidden)
    logreg_W, logreg_b = _init_logreg_weights(n_hidden, n_classes)

    # Estimate P(y | x) given the current weights
    p_y_given_x = feed_forward(
        T.nnet.softmax,
        logreg_W,
        logreg_b,
        feed_forward(
            T.tanh,
            hidden_W,
            hidden_b,
            x))  # <--- Our input variable (the features)

    cost = (
        -T.log(p_y_given_x[0, y])  # <-- Negative log likelihood of gold label
        + L1(L1_reg, logreg_W, hidden_W)
        + L2(L2_reg, logreg_W, hidden_W)
    )

    # Compile the training function.  Successive calls to this update the weights.
    # Internal state is maintained.
    # The output is "cost", which requires the computation of p_y_given_x.  We
    # also define how to update the weights based on the input label.
    train_model = theano.function(
        inputs=[x, y],
        outputs=cost,  # <-- Output depends on cost, which depends on P(y | x)
        updates=[
            (logreg_W, sgd_step(logreg_W, cost, learning_rate)),
            (logreg_b, sgd_step(logreg_b, cost, learning_rate)),
            (hidden_W, sgd_step(hidden_W, cost, learning_rate)),
            (hidden_b, sgd_step(hidden_b, cost, learning_rate)),
        ]
    )

    # Compile the evaluation function, which returns a 0/1 loss wrt the true
    # label.  Note that the output depends on p_y_given_x, so the program must
    # compute it.
    evaluate_model = theano.function(
        inputs=[x, y],
        outputs=T.neq(y, T.argmax(p_y_given_x[0])),
    )
    return train_model, evaluate_model


def main(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=30,
         n_hidden=500):
    train_examples, dev_examples, test_examples = make_data()
    print '... building the model'
    train_model, evaluate_model = compile_model(2 * 1, 10, n_hidden, learning_rate, L1_reg, L2_reg)
    print '... training'
    start = timeit.default_timer()
    for epoch in range(1, n_epochs + 1):
        for x, y in train_examples:
            train_model(x, y)
        # compute zero-one loss on validation set
        error = numpy.mean([evaluate_model(x, y) for x, y in dev_examples])
        print('epoch %i, validation error %f %%' % (epoch, error * 100))
    end = timeit.default_timer()
    print("Finished Ran for %i for seconds and average of %f seconds per epoch" % ((end - start), ((end - start)/n_epochs)))

    go_on = True
    while go_on:
        x1 = input("First binary input: ")
        x2 = input("Second binary input: ")
        if evaluate_model([x1, x2], 1) == 0:
            ans = 1
        else:
            ans = 0
        print(ans)

        want = raw_input("Want to continue? (y/n): ")

        chosen = False
        while not chosen:
            if want.lower() == 'n':
                go_on = False
                chosen = True
            elif want.lower() != 'y':
                want = raw_input("What was that? (y/n): ")
            else:
                chosen = True


if __name__ == '__main__':
    main()
