from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

nx, ny  = (100, 100)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

xv, yv  = np.meshgrid(x, y)

mu = 3
sigma = 1

np.random.seed(seed=19)

Coefficient_x = np.random.normal(mu, sigma, 9)
Coefficient_y = np.random.normal(mu, sigma, 9)


output_data_x  = Coefficient_x[0] + Coefficient_x[1]*xv + Coefficient_x[2]*xv**2 + Coefficient_x[3]*xv**3 + Coefficient_x[4]*xv**4 + Coefficient_x[5]*xv**5 + Coefficient_x[6]*xv**6 + Coefficient_x[7]*xv**7 + Coefficient_x[8]*xv**8     
output_data_y  = Coefficient_y[0] + Coefficient_y[1]*yv + Coefficient_y[2]*yv**2 + Coefficient_y[3]*yv**3 + Coefficient_y[4]*yv**4 + Coefficient_y[5]*yv**5 + Coefficient_y[6]*yv**6 + Coefficient_y[7]*yv**7 + Coefficient_y[8]*yv**8

output_data  = output_data_x + output_data_y
output_data  = np.reshape(output_data,(1,-1)).T
#%%
xv = np.reshape(xv,(1,-1))
yv = np.reshape(yv,(1,-1))


input_data = np.array((xv[0,:], yv[0,:]))
input_data = input_data.T

input_data_copy = input_data
output_data_copy = output_data

index = [i for i in range(10000)]  
np.random.shuffle(index) 
input_data = input_data[index]
output_data = output_data[index]


# Network Parameters

num_input = 1 # MNIST data input (img shape: 28*28)

num_hidden_1 = 2 # 1st layer num features
num_hidden_2 = 1 # 2nd layer num features (the latent dim)
num_hidden_3 = 1 


twin1_weights_r = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}
twin1_biases_r = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}

twin1_weights_g = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}
twin1_biases_g = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}

twin1_weights_b = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}
twin1_c = {
    'encoder_c1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_c2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_c3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}

twin2_weights_r = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}
twin2_biases_r = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}

twin2_weights_g = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}
twin2_biases_g = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}

twin2_weights_b = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}
twin2_c = {
    'encoder_c1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'encoder_c2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'encoder_c3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}

def quadratic_operation(x, weights_r, biases_r, weights_g, biases_g, weights_b, c):
    
    output_0 = (tf.matmul(x, weights_r) + biases_r)*(tf.matmul(x, weights_g) + biases_g)+tf.matmul(tf.pow(x,2), weights_b) + c
    output = tf.nn.relu(output_0)
    
    return output


# Building the encoder
def twin1(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = quadratic_operation(x,twin1_weights_r['encoder_h1'], twin1_biases_r['encoder_b1'], twin1_weights_g['encoder_h1'], 
                                  twin1_biases_g['encoder_b1'], twin1_weights_b['encoder_h1'], twin1_c['encoder_c1'])
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = quadratic_operation(layer_1,twin1_weights_r['encoder_h2'], twin1_biases_r['encoder_b2'], twin1_weights_g['encoder_h2'], 
                                  twin1_biases_g['encoder_b2'], twin1_weights_b['encoder_h2'], twin1_c['encoder_c2'])

    # Decoder Hidden layer with sigmoid activation #1
    layer_3 = quadratic_operation(layer_2,twin1_weights_r['encoder_h3'], twin1_biases_r['encoder_b3'], twin1_weights_g['encoder_h3'], 
                                  twin1_biases_g['encoder_b3'], twin1_weights_b['encoder_h3'], twin1_c['encoder_c3'])

    return layer_3

def twin2(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = quadratic_operation(x,twin2_weights_r['encoder_h1'], twin2_biases_r['encoder_b1'], twin2_weights_g['encoder_h1'], 
                                  twin2_biases_g['encoder_b1'], twin2_weights_b['encoder_h1'], twin2_c['encoder_c1'])
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = quadratic_operation(layer_1,twin2_weights_r['encoder_h2'], twin2_biases_r['encoder_b2'], twin2_weights_g['encoder_h2'], 
                                  twin2_biases_g['encoder_b2'], twin2_weights_b['encoder_h2'], twin2_c['encoder_c2'])

    # Decoder Hidden layer with sigmoid activation #1
    layer_3 = quadratic_operation(layer_2,twin2_weights_r['encoder_h3'], twin2_biases_r['encoder_b3'], twin2_weights_g['encoder_h3'], 
                                  twin2_biases_g['encoder_b3'], twin2_weights_b['encoder_h3'], twin2_c['encoder_c3'])

    return layer_3


# Training Parameters
learning_rate = 0.002
num_steps = 10000


# Construct model
X = tf.placeholder(tf.float32, shape = (None, 2))
Y = tf.placeholder(tf.float32, shape = (None, 1))
    
encoder_1 = twin1(X[:,0:1])
encoder_2 = twin2(X[:,1:2])


# Prediction
y_pred = encoder_1 + encoder_2
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data


        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: input_data, Y: output_data})
        # Display logs per step
        if i % 2000 == 0:
            print('Step %i: Minibatch Loss: %f' % (i, l))

            Prediction = sess.run(y_pred, feed_dict={X: input_data_copy})


                   
#%%

csfont = {'fontname':'Times New Roman'}
plt.figure()

t = np.arange(1,10001)


plt.plot(t.T, output_data_copy,label='Original Curve')
plt.plot(t.T, Prediction,label='Network Fitting')

plt.legend()

plt.axis('off')
            
















