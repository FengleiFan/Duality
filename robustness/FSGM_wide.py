import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from sklearn import preprocessing

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


Train_image = mnist.train.images
y_train = mnist.train.labels

Test_image = mnist.test.images
y_test = mnist.test.labels

All_image = np.load('All_image_DR_skewed.npy')

x_max = np.max(All_image)
x_min = np.min(All_image)

All_image = (1/16*x_max*(All_image-x_min))/(x_max-x_min)

x_train = All_image[0:55000]
x_test  = All_image[55000:65000]

_, M_train_label = np.where(y_train==1)

_, M_test_label = np.where(y_test==1)


# =============================================================================
# plt.figure()
# plt.title('Activation')
# 
# vis_x = x_train[:, 0]
# vis_y = x_train[:, 1]
# plt.scatter(vis_x, vis_y, c=M_train_label, marker = '.', cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.show()
# plt.title('Local Algorithm')
# 
# plt.figure()
# plt.title('Activation')
# 
# vis_x = x_test[:, 0]
# vis_y = x_test[:, 1]
# plt.scatter(vis_x, vis_y, c=M_test_label, marker = '.', cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.show()
# plt.title('Local Algorithm')
# =============================================================================


#%%
tf.reset_default_graph()
# Network Parameters

num_input = 1 # MNIST data input (img shape: 28*28)

num_hidden_1 = 4 # 1st layer num features
num_hidden_2 = 2 # 2nd layer num features (the latent dim)
num_hidden_3 = 1 


twin1_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}


twin1_biases = {
    'b1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'b3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}


twin1_c = {
    'c1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'c2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'c3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}


twin2_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],mean=0.0, stddev=0.1)),
    'h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],mean=0.0, stddev=0.1)),

}
twin2_biases = {
    'b1': tf.Variable(tf.random_normal([num_input, num_hidden_1],mean=0.0, stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'b3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}


twin2_c = {
    'c1': tf.Variable(tf.random_normal([num_hidden_1],mean=0.0, stddev=0.1)),
    'c2': tf.Variable(tf.random_normal([num_hidden_2],mean=0.0, stddev=0.1)),
    'c3': tf.Variable(tf.random_normal([num_hidden_3],mean=0.0, stddev=0.1)),

}

def quadratic_operation(x, weight, bias, c):
    
    output = tf.matmul(tf.pow(x,2), weight) + tf.matmul(x, bias) + c

    
    return output


# Building the encoder
def twin1(x):

    layer = quadratic_operation(x,twin1_weights['h1'], twin1_biases['b1'], twin1_c['c1'])

    return tf.reduce_prod(layer, axis = 1, keepdims = True)

def twin2(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer = quadratic_operation(x,twin2_weights['h1'], twin2_biases['b1'], twin2_c['c1'])
    
    return tf.reduce_prod(layer, axis = 1, keepdims = True)


# Training Parameters
learning_rate = 0.002
num_steps = 10000



# Construct model
X = tf.placeholder(tf.float32, shape = (None, 2))
y = tf.placeholder(tf.float32, shape = (None, 10))
    

def model(X):
    
    
    encoder_1 = twin1(X[:,0:1])
    encoder_2 = twin2(X[:,1:2])
    
    feature = tf.concat([encoder_1, encoder_2], axis = 1)
    
    #feature = X
    
    
    FC1_weights = tf.get_variable('W1', dtype=tf.float32, shape=[2,300], initializer=tf.truncated_normal_initializer(stddev=0.01))
    FC1_biases =  tf.get_variable('b1', dtype=tf.float32, shape = [300], initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    FC2_weights = tf.get_variable('W2', dtype=tf.float32, shape=[300,200], initializer=tf.truncated_normal_initializer(stddev=0.01))
    FC2_biases =  tf.get_variable('b2', dtype=tf.float32, shape = [200], initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    FC3_weights = tf.get_variable('W3', dtype=tf.float32, shape=[200,10], initializer=tf.truncated_normal_initializer(stddev=0.01))
    FC3_biases =  tf.get_variable('b3', dtype=tf.float32, shape = [10], initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    
    FC_layer1 = tf.nn.relu(tf.matmul(feature, FC1_weights) + FC1_biases)
    
    FC_layer2 = tf.nn.relu(tf.matmul(FC_layer1, FC2_weights) + FC2_biases)
    
    output_logits = tf.matmul(FC_layer2, FC3_weights) + FC3_biases
    
    return output_logits

output_logits = model(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate = 0.002, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

grads = tf.gradients(loss, X)
# Model predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

init = tf. global_variables_initializer()
batch_size = 1000
epochs = 200
train_accuracy = np.zeros((1,epochs))
# Create an interactive session (to keep the session in the other cells)
sess = tf.InteractiveSession()
# Initialize all variables
sess.run(init)
# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    # Randomly shuffle the training data at the beginning of each epoch 

    index = [i for i in range(55000)]  
    np.random.shuffle(index) 
    x_train = x_train[index]
    y_train = y_train[index]
    

    
    for iteration in range(num_tr_iter):


        x_batch = x_train[batch_size*iteration:batch_size*(iteration+1)]
        y_batch = y_train[batch_size*iteration:batch_size*(iteration+1)] 
        # Run optimization op (backprop)

        sess.run(optimizer, feed_dict={X: x_batch, y: y_batch})

        if iteration % 2 == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict={X: x_batch, y: y_batch})

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))
    
    train_accuracy[0,epoch] = sess.run(accuracy, feed_dict = {X: x_train, y: y_train})
    print(train_accuracy[0,epoch])
        
         



#%%

print(sess.run(accuracy, feed_dict = {X: x_test, y: y_test}))

# attack generation # FGSM

perturbation = sess.run(grads, feed_dict = {X: x_test, y: y_test})

#grad = 1000*perturbation[0]

sign = perturbation[0]

sign[sign<0] = -1
sign[sign>0] = 1

grad = 62*(1/1000)*sign

print(sess.run(accuracy, feed_dict = {X: x_test, y: y_test})-sess.run(accuracy, feed_dict = {X: x_test+grad, y: y_test}))

# =============================================================================
# grad = (2/10)*sign
# 
# print(sess.run(accuracy, feed_dict = {X: x_test+grad, y: y_test}))
# 
# grad = (3/10)*sign
# 
# print(sess.run(accuracy, feed_dict = {X: x_test+grad, y: y_test}))
# 
# grad = (5/10)*sign
# 
# print(sess.run(accuracy, feed_dict = {X: x_test+grad, y: y_test}))
# 
# grad = (10/10)*sign
# 
# print(sess.run(accuracy, feed_dict = {X: x_test+grad, y: y_test}))
# =============================================================================

#%%
# =============================================================================
# plt.figure()
# plt.plot(np.arange(epochs), train_accuracy[0,:])
# =============================================================================

