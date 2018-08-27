'''
Emotion classification with LSTM
@author: Omar U. Florez
'''

import tensorflow as tf

# Model:
#   Data transformation:
#       - [Batch size, seq_max_len] -> [Batch size, seq_max_len, vocab_size]
#       - [Batch size, seq_max_len, vocab_size] -> [Batch size, seq_max_len * vocab_size]
#       - [Batch size, seq_max_len, vocab_size] -> length of input sequence x [batch size, vocabulary size]
#   Neural Network architecture:
#       - outputs, final_state = length of input sequence x BasicLSTMCell(state_size)
#       - output = outputs[-1]
#       - hidden = [Batch size, state_size] x [state_size, 256]
#       - hidden = tanh(hidden)
#       - logits = [state_size, 256] x [256, num_classes]
#       - probabilities = Softmax(logits)
class LSTM:
    def __init__(self, seq_max_len, state_size, vocab_size, num_classes):
        # Maximum number of units in the input (e.g., words)
        self.seq_max_len = seq_max_len
        # Number of dimensions of the internal memory
        self.state_size = state_size
        # Number of unique characters defining input data
        self.vocab_size = vocab_size
        # Number of output classes
        self.num_classes = num_classes

    def build_model(self):
        self.x = tf.placeholder(shape=[None, self.seq_max_len], dtype=tf.int32)
        # Convert word ids from the input into ortogonal vectors
        x_one_hot = tf.one_hot(self.x, self.vocab_size)

        # Forming the input representation for LSTM
        #   - length of input sequence x [batch size, vocabulary size]
        x_one_hot = tf.cast(x_one_hot, tf.float32)
        rnn_input = tf.unstack(x_one_hot, axis=1)

        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y_onehot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        # Define weights
        weights = {
            'layer_0': tf.Variable(tf.random_normal([self.state_size, 256])),
            'layer_1': tf.Variable(tf.random_normal([256, self.num_classes]))
        }
        # Define bias weights
        biases = {
            'layer_0': tf.Variable(tf.random_normal([256])),
            'layer_1': tf.Variable(tf.random_normal([self.num_classes]))
        }

        init_state = tf.zeros([self.batch_size, self.state_size])
        # BasicLSTMCell(): this function considers an zero init_state tensor by default
        cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        self.outputs, self.final_state = tf.contrib.rnn.static_rnn(cell, rnn_input, dtype=tf.float32)

        output = self.outputs[-1]
        hidden = tf.matmul(output, weights['layer_0']) + biases['layer_0']
        hidden = tf.nn.tanh(hidden)
        self.logits = tf.matmul(hidden, weights['layer_1']) + biases['layer_1']
        self.probs = tf.nn.softmax(self.logits, axis=1)

        self.correct_preds = tf.equal(tf.argmax(self.probs, axis=1), tf.argmax(self.y_onehot, axis=1))
        self.precision = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))
        return

    def step_training(self, learning_rate=0.01):
        # softmax_cross_entropy_with_logits(): This function computes the probabilistic distance
        # between two distributions: the actual classes (y_onehot) and the predicted ones as log-probabilities
        # (self.logits). While the classes are mutually exclusive, their probabilities need not be.
        # All that is required is that each row of labels is a valid probability distribution.
        # If they are not, the computation of the gradient will be incorrect.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_onehot,
                                                                             logits=self.logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return loss, optimizer






