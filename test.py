# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:48:50 2018

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

#Change path to directory of where your test images are stored
test_path='/home/shrilock/Downloads/test_set' 


#calculate number of testing examples
num_test=len(os.listdir(test_path)) 

# Initialize test data and test target
test_x=np.zeros((num_test,1080000))
test_y=np.zeros((num_test,2))

image_array=[] # arrat to save images for displaying later
index=0

# Constructing test set
for filename in os.listdir(test_path):
    file_path=test_path+'/'+filename
    img=cv2.imread(file_path)
    image_array.append(img)
    resize_img=image_resize(img,600)
    arr=np.array(resize_img)
    f_arr=arr.flatten()
    test_x[index]=f_arr
    if 'mountain_bike' in filename:
        test_y[index]=[0,1]
    else:
        test_y[index]=[1,0]
    index+=1   

# Change data type and normalize to feed into the model    
test_array_x=test_x.astype('float32')
test_array_y=test_y.astype('float32') 
test_x_norm=test_array_x/255    

# Starting TensorFlow Session
sess=tf.Session() 
saver = tf.train.import_meta_graph('./my_cnn_model.meta') # Reloading meta data from saved model
saver.restore(sess,tf.train.latest_checkpoint('./')) # Restoring the saved model

graph = tf.get_default_graph()  # get graph
# Get the placeholders
x = graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0")

feed_dict_test={x:test_x_norm,y_true:test_array_y} #construct feed dictionary for testing

prediction= graph.get_tensor_by_name("predicted_prob:0") # Get the prediction operation from the saved graph
predicted_class=graph.get_tensor_by_name("predicted_class:0") #Get the prediction class operation from the saved graph

y_hat,y_cls=sess.run([prediction,predicted_class],feed_dict=feed_dict_test) # Run prediction and prediction class operation

##############################################################################
# The following block displays the confidence results over respective images
##############################################################################

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2

for i in range(num_test):
    text=''
    confidence_value=y_hat[i][y_cls[i]]*100
    if y_cls[i]==0:
        bike='Road Bike'
    else:
        bike='Mountain Bike'
    text='Confidene, '+bike+'='+str(confidence_value)+'%' 
    cv2.putText(image_array[i],text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    cv2.imshow("img",image_array[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    







