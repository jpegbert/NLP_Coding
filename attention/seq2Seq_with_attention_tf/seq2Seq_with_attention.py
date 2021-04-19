import numpy as np
import tensorflow as tf


tf.reset_default_graph()

sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)


n_step = 5
n_hidden = 128


def make_batch(sentences):
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    return input_batch, output_batch, target_batch


enc_inputs = tf.placeholder(tf.float32, [None, None, n_class])
dec_inputs = tf.placeholder(tf.float32, [None, None, n_class])
targets = tf.placeholder(tf.int64, [1, n_step])

attn = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))


def get_att_score(dec_output, enc_output):
    score = tf.squeeze(tf.matmul(enc_output, attn), 0)
    dec_output = tf.squeeze(dec_output, [0, 1])
    return tf.tensordot(dec_output, score, 1)


def get_att_weight(dec_output, enc_outputs):
    attn_scores = []
    enc_outputs = tf.transpose(enc_outputs, [1, 0, 2])
    for i in range(n_step):
        attn_scores.append(get_att_score(dec_output, enc_outputs[i]))
    return tf.reshape(tf.nn.softmax(attn_scores), [1, 1, -1])


model = []

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    enc_outputs, enc_hidden = tf.nn.dynamic_rnn(enc_cell, enc_inputs, dtype=tf.float32)


with tf.variable_scope('decoder'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    inputs = tf.transpose(dec_inputs, [1, 0, 2])
    hidden = enc_hidden
    for i in range(n_step):
        dec_output, hidden = tf.nn.dynamic_rnn(dec_cell,
                                               tf.expand_dims(inputs[i], 1),
                                               initial_state=hidden,
                                               dtype=tf.float32,
                                               time_major=True)
        attn_weights = get_att_weight(dec_output, enc_outputs)
        # Attention.append(tf.squeeze(attn_weights))
        context = tf.matmul(attn_weights, enc_outputs)
        dec_output = tf.squeeze(dec_output, 0)
        context = tf.squeeze(context, 1)
        model.append(tf.matmul(tf.concat((dec_output, context), 1), out))


# trained_attn = tf.stack([Attention[0], Attention[1], Attention[2], Attention[3], )
model = tf.transpose(model, [1, 0, 2])
prediction = tf.argmax(model, 2)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(2000):
        input_batch, output_batch, target_batch = make_batch(sentences)
        # _, loss, attention = sess.run([optimizer, cost, trained_attn],
        _, loss = sess.run([optimizer, cost], feed_dict={enc_inputs: input_batch, dec_inputs: output_batch, targets: target_batch})
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.6f}'.format(loss))

    predict_batch = [np.eye(n_class)[[word_dict[n] for n in 'P P P P P'.split()]]]
    result = sess.run(prediction, feed_dict={enc_inputs: input_batch, dec_inputs: predict_batch})
    print(sentences[0].split(), '->', [number_dict[n] for n in result[0]])


