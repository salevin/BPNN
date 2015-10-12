__author__ = 'slevin'

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
