# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:58:44 2018

@author: shrinidhi
"""

import cv2
import numpy as np
import tensorflow as tf
import os

def image_resize(image,size,inter=cv2.INTER_AREA): # function to resize the images--images have different dimensions
    dim=(size,size)
    resize_img=cv2.resize(image,dim,interpolation=inter)
    return resize_img
# Change path to directory of training images 
train_path="/home/shrilock/Downloads/train_set"
  

#calculate number of training examples 
num_train=0
for dir in os.listdir(train_path):
    num_train+=len(os.listdir(train_path+'/'+dir))
    
#calculate number of testing examples
num_test=len(os.listdir(test_path))    
    
##############################################################################
# define training and testing parameters
##############################################################################
train_x=np.zeros((num_train,1080000))
train_y=np.zeros((num_train,2))
test_x=np.zeros((num_test,1080000))
test_y=np.zeros((num_test,2))

##############################################################################
# This block constructs training and testing sets
##############################################################################
index=0

for dir in os.listdir(train_path):
    f_path=train_path+'/'+dir
    for filename in os.listdir(f_path):
        image_path=train_path+'/'+dir+'/'+filename
        img=cv2.imread(image_path)
        resize_img=image_resize(img,600)
        arr=np.array(resize_img)
        f_arr=arr.flatten()
        train_x[index]=f_arr
        if 'mountain_bike' in filename:
            train_y[index]=[0,1]
        else:
            train_y[index]=[1,0]
        index+=1    
    
            
 # Change data type and normalize to feed into the model        
train_array_x=train_x.astype('float32')
train_array_y=train_y.astype('float32') 
train_x_norm=train_array_x/255

     
##############################################################################
#define image specs        
##############################################################################

img_size=600
img_flat=1080000
num_classes=2
num_channels=3

##############################################################################
#TensorFlow helper functions
##############################################################################

def new_weights(shape): # Function to initialize random weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
def new_biases(length): # Function to initialize Bias
    return tf.Variable(tf.constant(0.05, shape=[length]))    

# Function to declare a new convolutional layer 
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights   

# Function to flatten the output from the convolution layer in order to feed it into the fully connected layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features     


# Function to declare a new fully connected layer
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
    
##############################################################################
# Configure layers in the model
##############################################################################    

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.    
    

##############################################################################
# TensorFlow Grapgh
##############################################################################        

x = tf.placeholder(tf.float32, shape=[None, img_flat], name='x') # Placeholder for image data

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels]) # Reshape image data into lenght X width X channels 

y_true = tf.placeholder(tf.float32 , shape=[None, num_classes], name='y_true') # Placeholder for image target

y_true_cls = tf.argmax(y_true, axis=1) # Identify the class


# Stacking Layers
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
                   
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)               
                   
layer_flat, num_features = flatten_layer(layer_conv2)                   



layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
           

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
                         
# Applying softmax to the output for classification                         
y_pred = tf.nn.softmax(layer_fc2,name='predicted_prob')
# Identify the Class
y_pred_cls = tf.argmax(y_pred, axis=1,name='predicted_class')

# Calculate the cross-entropy between true and predicted values
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
# Deduce loss for these training examples
cost = tf.reduce_mean(cross_entropy)

# Reduce the cost using Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##############################################################################
# TensorFlow Run
##############################################################################

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 20
num_iterations=10

for i in range(num_iterations):
    j=0
    while j<len(train_x):
        start=j
        end=j+train_batch_size 
        batch_x,batch_y=train_x_norm[start:end],train_array_y[start:end]
        feed_dict_train = {x: batch_x, y_true: batch_y}
        _,loss=session.run([optimizer,cost], feed_dict=feed_dict_train)
        
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        j+=train_batch_size
            
    print('Loss at epoch',i,'is',loss,'and accuracy is',acc)

Writer=tf.summary.FileWriter('/home/shrilock/Downloads/logs/mog',session.graph)    

# Saving the trained model
saver = tf.train.Saver()

saver.save(session, './my_cnn_model')   



    

            
        
        
        






                         

              
                         






        




