# For the maths used in Convonets please refer to the Andrej Karpathy blog and this link > http://cs231n.github.io/convolutional-networks/#conv
# IF you want to go in detail please read this -> https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

import argparse
import sys
#importing datset from tensorflow 
from tensorflow.examples.tutorials.mnist import input_data
#importing tensorflow for computation
import tensorflow as tf 

FLAGS = None

def mnist_convonet(x):
	# x will recieve a tensor of dimension (N_examples, 784), 784 dimension after faltnning th e image. 
	# 28,28 is the size of the image; the dimansion will reduce when you read further code. 
	x_image  = tf.reshape(x, [-1,28,28,1])

	# Frist Layer, this will be a Conv Layer -> Max pooling layer
	# W_conv1 ->Weight of first conv Layer
	# 5,5     -> Patch size
	# 1       -> Input channel (our image is grascale; if there is RGB then channel -> 3)
	# 32      -> Number of output Channel. 
	# After this step our image will change from 1 input channel to 32 input channel
	W_conv1 = weight_variable([5, 5, 1, 32])
	#bias 
	b_conv1 = bias_variable([32])
	#Relu is used as activation Layer, we don't use sigmoid because Relu is Faster and reduce liklehood of Vanishing Gradient 
	#Relu is f = max(0,a); f = max(0,a);  where a = Wx + b
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	# Pooling layer - downsamples by 2X.
	# Pooling Layer are useful to reduce the size of your input parameters(successively reduce the dimension after convolution operations)
	# control model overfitting
	# This method will reduce the image size to 14x14(image size reduced; The formula is given on the website mnetioned above)
	h_pool1 = max_pool_2x2(h_conv1)
  	# Second convolutional layer -- maps 32 feature maps to 64.
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	# Second pooling layer.
    #This method will reduce the image size to 7x7(image size reduced again, )
	h_pool2 = max_pool_2x2(h_conv2)
	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    # Why we use FC Layer, Please read this answer on quora. 
    #https://www.quora.com/Why-are-fully-connected-layers-used-at-the-very-end-output-side-of-convolutional-NNs-Why-not-earlier
	W_fc1 = weight_variable([7*7*64,1024]) # weight Matrix
	b_fc1 = bias_variable([1024]) 		   # Bias Matrix

    #Same procedure as aboouve layeres
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # Image Flattening
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) 			# apply relu actvation.
	# Dropout - controls the complexity of the model, prevents co-adaptation of features
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	# Now Mapping the 1024 features to 10 classes.
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10]) 
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	return y_conv, keep_prob

def conv2d(x, W):
	# Stride    -> factor by with which we slide the filter
	# When the stride is 1 then we move the filters one pixel at a time
	# Padding   ->  allow us to control the spatial size of the output volumes
	
	""" returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding ='SAME') 

def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
 	"""weight_variable generates a weight variable of a given shape."""
 	initial = tf.truncated_normal(shape, stddev=0.1)
 	return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  	# Data Import
  	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

  	# Creating placeholder for input image
  	# placeholder - a value that we'll input when we ask TensorFlow to run a computation.
  	# This Placeholder will holder the 28 x 28 image flattened into 1 x 728 dimension 
  	x = tf.placeholder(tf.float32, [None, 784])
  	# None ->  indicates that the first dimension, corresponding to the batch size, can be of any size
  	
  	# creating placeholder holding one-hot 10-dimensional target label 
  	y_ = tf.placeholder(tf.float32, [None, 10])
  	
  	# calling our deepnn function which return convolution layer and keep_probablity
  	y_conv, keep_prob = deepnn(x)

  	# Computes softmax cross entropy between logits and labels
  	# logits are unscaled log probabilities.
  	# Cross-entropy can be used as an error measure when a network's outputs can be thought of as representing independent hypotheses
  	cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

  	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

  	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

  	with tf.Session() as sess:
  		sess.run(tf.global_varaible_initializer())

  		for i in range(1000):
  			# fetching data in batches
  			batch = mnist.train.next_batch(100)

  			if i % 100 == 0:
  				train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
  				print('step %d, training accuracy %g' % (i, train_accuracy))
  				# We train on keep_probabilty of 50 % and dpoping 50 % connection 
  				# The keep_prob value is used to control the dropout rate used when training the neural network. 
  				train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  		# We evaluate our accuracy on 0 % dropout or we can say 100% keep_prob.		
  		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  		if __name__ == '__main__':
  			parser = argparse.ArgumentParser()
  			parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  			FLAGS, unparsed = parser.parse_known_args()
  			tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


