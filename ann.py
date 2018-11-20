import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from essentials import plot_confusion_matrix
# from tensorflow.examples.tutorials.mnist import input_data #imports mnist input data from tensorflow examples. 
# mnist = input_data.read_data_sets("MNIST/data", one_hot = True) #using input data call read data sets from a folder MNIST/data and store in mnist.
# from matplotlib import pyplot as plt

#hyperparameters
learning_rate = 3*1e-3 
training_epochs = 150 
display_step = 1 #after how many epochs we want to output our desired results on screen
batch_size = 150 

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
# n_input = 784 # MNIST data input (img shape: 28*28)
# n_classes = 10 # MNIST total classes (0-9 digits)
n_input = 24 # ECG data input (shape: 1*24)
n_classes = 5 # ECG total classes 

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

#all activation functions
activations = {
	'relu':tf.nn.relu,
	'softmax':tf.nn.softmax,
	'sigmoid':tf.nn.sigmoid,
	'tanh':tf.nn.tanh
}

ys = np.loadtxt('y.csv')
xs = np.loadtxt('Data.csv',delimiter=',')
class_names = ['Normal','P','LBBB','RBBB','PVC'] # 5 classes

# Feature normalization
mu, std, x_normalized = np.mean(xs, axis = 0), np.std(xs, axis = 0), []
for i,j,k in zip(xs.T,mu,std):
	x_normalized.append((i-j)/k)
xs = np.asarray(x_normalized).T

#splitting data into mini-batch
def next_batch(x_data, y_data, batch_no, batch_size):
	try:
		x_batch = x_data[batch_no*batch_size:(batch_no+1)*batch_size]
		y_batch = y_data[batch_no*batch_size:(batch_no+1)*batch_size]	
	except:
		x_batch = x_data[batch_no*batch_size:len(x_data)]
		y_batch = y_data[batch_no*batch_size:len(y_data)]
	return x_batch, y_batch

#converting one column vector to one-hot vectors
def to_onehot(data):
	onehot_data = np.zeros((len(data),n_classes))
	for i,j in enumerate(data):
		onehot_data[i][int(j)] = int(j) 
	return onehot_data

def multilayer_perceptron(x, weights, biases, act_fun):
	# Hidden layer 1
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = activations[act_fun](layer_1)
	# Hidden layer 2
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = activations[act_fun](layer_2)
	# Output layer 
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	out_layer = activations['act_fun'](out_layer)
	return out_layer

# Construct model and call the multilayer_perceptron function by passing in x, weights and biases.
pred = multilayer_perceptron(x, weights, biases,'tanh')

# Define loss and optimizer.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#for optimization we use Adam optimizer, we can use Gradient Descent as well.
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

ys = to_onehot(ys)
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.2, random_state = 5) # 20% as test

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph.
with tf.Session() as sess:
	sess.run(init) 
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0. #initialize avg_cost to zero.
		# total_batch = int(mnist.train.num_examples/batch_size) 
		total_batch = int(len(x_train)/batch_size)
		
		for i in range(total_batch):   
			# batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			batch_xs, batch_ys = next_batch(x_train, y_train, i, batch_size) 
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,	   
														  y: batch_ys})
			# Compute average loss											
			avg_cost += c / total_batch  

		# Display logs per epoch step
		if (epoch+1) % display_step == 0:
			print ("Epoch:", '%04d'%(epoch+1), "cost=", "{:.9f}".format(avg_cost))

	print ("Optimization Finished!")	

	# Test model and calculate accuracy using reduce_mean
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
	print ("Accuracy:", accuracy.eval({x: x_test, y: y_test}))

	#calculate confusion matrix using sklearn function
	confusion = confusion_matrix(tf.argmax(y, 1).eval({x: x_test, y: y_test}),
	 								tf.argmax(pred, 1).eval({x: x_test, y: y_test}))
	# print and plot confusion matrix
	plot_confusion_matrix(confusion, classes=class_names, title='Confusion matrix')