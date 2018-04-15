import pickle
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
import tensorflow as tf


# import data
# frequencies, times, spectogram, label
with open('training_v2.pickle', 'rb') as handle:
    parsed_dict = pickle.load(handle)


# parameters
label_count = 2 # number of classes
batch_size = 100 # size for the batch
sample_rate = 16000 # Number of audio samples per second
clip_duration_ms = 1 # Length of each audio clip to be analyzed
window_size_ms  = 30 # Duration of frequency analysis window
window_stride_ms = 10 # How far to move in time between frequency windows
dct_coefficient_count = 40 # Number of frequency bins to use for analysis
training_ratio = 0.9
frequencies, times, spectogram, label = parsed_dict[list(parsed_dict.keys())[0]]
shapes = spectogram.shape
hm_epochs = 500

# training and testing indices
perm = np.random.permutation(len(parsed_dict.keys()))
training_ids = perm[0:int(training_ratio*len(parsed_dict.keys()))]
testing_ids = [i for i in perm if i not in training_ids]

# placeholder
fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, shapes[0], shapes[1]])
y = tf.placeholder(dtype=tf.int32, shape=[None,label_count])
dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

def dict_cleaning(parsed_dict,shapes):
    arrays = np.zeros((len(parsed_dict.keys()),shapes[0],shapes[1]))
    y_array = []
    counter = 0
    for i in parsed_dict.keys():
        frequencies, times, spectogram, label = parsed_dict[i]
        y_array.append(label)
        if spectogram.shape != shapes:
            spectogram_padded = np.zeros(shapes, dtype=np.float32)
            spectogram_padded[:spectogram.shape[0], :spectogram.shape[1]] = spectogram
            arrays[counter] = spectogram_padded
        else:
            arrays[counter] = spectogram
        counter += 1
    return arrays,y_array

def one_hot_encoder(inputs):
    values = set(inputs)
    outputs = np.zeros([len(inputs),len(values)])
    for iter_value, value in enumerate(values):
        indices = [i for i, s in enumerate(inputs) if value in s]
        outputs[indices,iter_value] = 1
    return outputs


# CNN single input
def conv_model(fingerprint_input, is_training, dropout_prob):
    input_frequency_size = shapes[0] # 40
    input_time_size = shapes[1] # 0
    fingerprint_4d = tf.reshape(fingerprint_input,[-1, input_time_size, input_frequency_size, 1]) # [batch, in_height, in_width, in_channels]
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal([first_filter_height, first_filter_width, 1, first_filter_count], stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                              'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    # if is_training:
    #     first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    # else:
    #     first_dropout = first_relu
    first_dropout = tf.where(is_training,tf.nn.dropout(first_relu, dropout_prob),first_relu)
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                               'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    # if is_training:
    #     second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    # else:
    #     second_dropout = second_relu
    second_dropout = tf.where(is_training, tf.nn.dropout(second_relu, dropout_prob), second_relu)
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout,
                                       [-1, second_conv_element_count])
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    return final_fc, dropout_prob


def train_neural_network(hm_epochs, parsed_dict, is_training):
    prediction = conv_model(fingerprint_input, is_training, dropout_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction[0], labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    correct = tf.equal(tf.argmax(prediction[0], 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            print(epoch)
            epoch_loss = 0
            for iter_batch in range(int(len(training_ids) / batch_size)):
                epoch_x = training_arrays[iter_batch*batch_size:(iter_batch+1)*batch_size,]
                epoch_y = training_y_array[iter_batch*batch_size:(iter_batch+1)*batch_size,]
                epoch_x = np.nan_to_num(epoch_x)

                _, c = sess.run([optimizer, cost], feed_dict={fingerprint_input: epoch_x, y: epoch_y, dropout_prob:0.5, is_training: True})
                epoch_loss += c

            print('Training epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            print('Validation accuracy:',accuracy.eval({fingerprint_input: np.nan_to_num(testing_arrays), y:testing_y_array, dropout_prob:0.5, is_training: False}))





arrays,y_array = dict_cleaning(parsed_dict,shapes)
training_arrays = arrays[training_ids]
training_y_array = one_hot_encoder([y_array[i] for i in training_ids])
testing_arrays = arrays[testing_ids]
testing_y_array = one_hot_encoder([y_array[i] for i in testing_ids])