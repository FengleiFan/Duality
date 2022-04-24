import matplotlib.pyplot as plt

from sklearn import datasets

import numpy as np

import tensorflow as tf
# import some data to play with
breast_cancer = datasets.load_breast_cancer()
data = np.float32(breast_cancer.data)  # we only take the first two features.
target = breast_cancer.target

#input_data = data[:,0:1]/np.max(data[:,0:1])
input_data = data[:,0:5]

output_data = np.int32(np.zeros((569,2)))
output_data[:,0] = target
output_data[:,1] = 1-target

output_data = 1-output_data

#%%
tf.reset_default_graph()
# Network Parameters

variance = 0.8

num_input = 5

wide_hidden_1 = 2 
wide_hidden_2 = 2 
wide_hidden_3 = 2 

num_output = 2

Arm1_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, wide_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([wide_hidden_1, wide_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([wide_hidden_2, wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm1_biases = {
    'b1': tf.Variable(tf.random_normal([wide_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([wide_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm2_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, wide_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([wide_hidden_1, wide_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([wide_hidden_2, wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm2_biases = {
    'b1': tf.Variable(tf.random_normal([wide_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([wide_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm3_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, wide_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([wide_hidden_1, wide_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([wide_hidden_2, wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm3_biases = {
    'b1': tf.Variable(tf.random_normal([wide_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([wide_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm4_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, wide_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([wide_hidden_1, wide_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([wide_hidden_2, wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm4_biases = {
    'b1': tf.Variable(tf.random_normal([wide_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([wide_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm5_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, wide_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([wide_hidden_1, wide_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([wide_hidden_2, wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm5_biases = {
    'b1': tf.Variable(tf.random_normal([wide_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([wide_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm6_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, wide_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([wide_hidden_1, wide_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([wide_hidden_2, wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm6_biases = {
    'b1': tf.Variable(tf.random_normal([wide_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([wide_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm7_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, wide_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([wide_hidden_1, wide_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([wide_hidden_2, wide_hidden_3],mean=0.0, stddev=variance)),
}

Arm7_biases = {
    'b1': tf.Variable(tf.random_normal([wide_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([wide_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([wide_hidden_3],mean=0.0, stddev=variance)),
}

deep_hidden_1 = 3
deep_hidden_2 = 3
deep_hidden_3 = 3


DeepArm_Input_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, deep_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([deep_hidden_2, deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_Input_biases = {
    'b1': tf.Variable(tf.random_normal([deep_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([deep_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_1_weights = {
    'h1': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([deep_hidden_2, deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_1_biases = {
    'b1': tf.Variable(tf.random_normal([deep_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([deep_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_2_weights = {
    'h1': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([deep_hidden_2, deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_2_biases = {
    'b1': tf.Variable(tf.random_normal([deep_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([deep_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_3_weights = {
    'h1': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([deep_hidden_2, deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_3_biases = {
    'b1': tf.Variable(tf.random_normal([deep_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([deep_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_4_weights = {
    'h1': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([deep_hidden_2, deep_hidden_3],mean=0.0, stddev=variance)),
}

DeepArm_4_biases = {
    'b1': tf.Variable(tf.random_normal([deep_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([deep_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([deep_hidden_3],mean=0.0, stddev=variance)),
}



DeepArm_Output_weights = {
    'h1': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_1],mean=0.0, stddev=variance)),
    'h2': tf.Variable(tf.random_normal([deep_hidden_1, deep_hidden_2],mean=0.0, stddev=variance)),
    'h3': tf.Variable(tf.random_normal([deep_hidden_2, num_output],mean=0.0, stddev=variance)),
}

DeepArm_Output_biases = {
    'b1': tf.Variable(tf.random_normal([deep_hidden_1],mean=0.0, stddev=variance)),
    'b2': tf.Variable(tf.random_normal([deep_hidden_2],mean=0.0, stddev=variance)),
    'b3': tf.Variable(tf.random_normal([num_output],mean=0.0, stddev=variance)),
}


def feedforward(x, weights_r, biases_r):
    
    output_0 = tf.matmul(x, weights_r) + biases_r
    output = tf.nn.relu(output_0)
    
    return output

def twin(x, Arm_weights, Arm_biases):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = feedforward(x, Arm_weights['h1'], Arm_biases['b1'])
    
    layer_2 = feedforward(layer_1, Arm_weights['h2'], Arm_biases['b2'])
    
    layer_3 = feedforward(layer_2, Arm_weights['h3'], Arm_biases['b3'])

    return layer_3

final_bias = tf.Variable(tf.random_normal([2],mean=3, stddev=0.1))

X = tf.placeholder(tf.float32, shape = (None, 5))
Y = tf.placeholder(tf.float32, shape = (None, 2))


Wide_sub1 = twin(X, Arm1_weights, Arm1_biases)





Deep_sub1  = twin(X, DeepArm_Input_weights, DeepArm_Input_biases)
Deep_sub2  = twin(Deep_sub1, DeepArm_1_weights, DeepArm_1_biases)
Deep_sub3  = twin(Deep_sub2, DeepArm_2_weights, DeepArm_2_biases)
Deep_sub4  = twin(Deep_sub2, DeepArm_3_weights, DeepArm_3_biases)
Deep_sub5  = twin(Deep_sub4, DeepArm_4_weights, DeepArm_4_biases)
Deep_sub6  = twin(Deep_sub5, DeepArm_Output_weights, DeepArm_Output_biases)

output_layer_1 = -Wide_sub1  + final_bias

output_layer_2 = Deep_sub6

output_layer = output_layer_1 + output_layer_2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))

optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()

num_steps = 100000


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data


        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, accuracy], feed_dict={X: input_data, Y: output_data})
        # Display logs per step
        if i % 1000 == 0:
            print('Step %i: Minibatch Loss: %f' % (i, l))

            Prediction = sess.run(output_layer, feed_dict={X: input_data, Y: output_data})

            
    Prediction = sess.run(output_layer, feed_dict={X: input_data, Y: output_data})
            
            


#%%

# =============================================================================
# Prediction_ = Prediction.argmax(axis=1)
# 
# 
# 
# plt.figure()
# 
# plt.scatter(input_data, target)
# plt.scatter(input_data, Prediction_)
# 
# plt.xlabel('MEAN RADIUS')
# plt.yticks([0,1])
# =============================================================================





