#######################################HEADER FILES#####################################################
import numpy as np
import pandas as pd
import math, os
import datetime
from sklearn.metrics import accuracy_score, f1_score
import argparse,pickle
import tarfile
# from dataAugmentation import *
#import matplotlib.pyplot as plt, matplotlib.image as mpimg

np.random.seed(1234)

#################################DEAFULT PARAMETER VALUES#####################################################
mini_batch_size = 20
anneal = True
eta = 0.001
lambda_reg = 0.1


#######################################ACTIVATION FUNCTIONS#####################################################
def sigmoid(a):
	return np.where(a > 0, 1. / (1. + np.exp(-a)), np.exp(a) / 1+(np.exp(a)))

def tanh(a):
	return np.tanh(a)

def relu(a):
	return np.where(a > 0, a, 0.0)

def softmax(a):
	sum_exp_a = np.sum(np.exp(a), axis=1).reshape(a.shape[0], 1)
	return np.exp(a)/sum_exp_a

def linear(a):
	sum_a = np.sum(a, axis=1).reshape(a.shape[0], 1)
	return a/sum_a


#######################################LOSS FUNCTIONS#####################################################
def mean_squared(fx, y):
	loss = 0
	if(fx.shape[0] > 10000):
		for i in range(0, fx.shape[0], mini_batch_size):
			fx_mini = fx[i:i + mini_batch_size]
			y_mini = y[i:i + mini_batch_size]
			diff = fx_mini - y_mini
			loss += np.sum(diff * diff)			
	
		return loss

	diff = fx - y
	return np.sum(diff * diff)


def cross_entropy(fx, y):
	loss = 0
	if(fx.shape[0] > 10000):
		for i in range(0, fx.shape[0], mini_batch_size):
			fx_mini = fx[i:i + mini_batch_size]
			y_mini = y[i:i + mini_batch_size]
			cross = np.matmul(y_mini, np.transpose(fx_mini))
			loss +=  -np.trace(np.log(cross))
		return loss

	return -np.trace(np.log(np.matmul(y, np.transpose(fx))))

#######################################DERIVATIVE OF LOSS FUNCTIONS#####################################################
def sq(fx, y, a):#	
	der = np.zeros(fx.shape)
	
	for i in range(fx.shape[0]):
		diff = (fx[i] - y[i])
		mat1 = fx[i]*np.identity(k)
		mat2 = np.matmul(np.reshape(fx[i],(k,-1)),np.reshape(fx[i],(-1,k)))
		mat = mat1 - mat2
		der[i] = np.matmul(diff.transpose(),mat).transpose()

	return der


def ce(fx, y, a):
	return -(y - fx)
#######################################FORWARD PROPAGATION#####################################################

def forward_prop(x, W, b):
	a = []
	h = []
	# layer_bias = np.tile(np.array(layer_bias), (5,1))
	input_cur = x

	for (layer_weights, layer_bias) in zip(W[:-1], b[:-1]):
		#layer_bias = layer_bias.reshape(1, layer_bias.shape[0])	
		cur_a = np.matmul(input_cur, np.transpose(layer_weights)) + layer_bias		
		cur_h = activation(cur_a)
		a.append(cur_a)
		h.append(cur_h)
		input_cur = cur_h		

	cur_a = np.matmul(input_cur, np.transpose(W[-1])) + b[-1]
	cur_h = last_layer_activation(cur_a)
	a.append(cur_a)
	h.append(cur_h)
	fx = h[-1]
	
	return a, h, fx


#######################################DERIVATIVES W.R.T HIDDEN LAYER ACTIVATION#####################################################
def grad_h(w, g_a):
	return np.matmul(g_a, w)

#######################################DERIVATIVES W.R.T HIDDEN LAYER AGGREGATOR#####################################################

def grad_a_tanh(h, g_h):
	return g_h * (1 - h*h)

def grad_a_sigmoid(h, g_h):
	return g_h * (h * (1.0 - h))	

def grad_a_relu(h, g_h):
	return np.where(h > 0, g_h * h, 0.0)

#######################################BACK PROPAGATION#####################################################

def back_prop(x, y, W, b, a, h, fx):
	g_W = [np.zeros(p.shape) for p in W]
	g_b = [np.zeros(p.shape) for p in b]

	h.insert(0, x)
	loss_grad = error(fx, y, a[-1])
	
	for layer in range(len(W)-1, -1, -1): 		
		g_W[layer] = np.matmul(np.transpose(loss_grad), h[layer])
		g_b[layer] = np.sum(loss_grad, axis=0)
		
		var = grad_h(W[layer],loss_grad)
		loss_grad = grad_a(h[layer],var)
		
	return g_W, g_b





#######################################MINI BATCH GRADIENT DESCENT#####################################################
def gradient_descent(X_train, Y_train, X_val, Y_val, W, b, eta):
	etaL = eta
	prev_loss = np.inf
	for e in range(epochs):
		loss = 0
		loss_val = 0
		steps = 0

		randomize = np.arange(X_train.shape[0])
		np.random.shuffle(randomize)
		X_train = X_train[randomize]
		Y_train = Y_train[randomize]
		
		

		for i in range(0, X_train.shape[0], mini_batch_size):
			X_train_mini = X_train[i:i + mini_batch_size]
			Y_train_mini = Y_train[i:i + mini_batch_size]

			a_mini, h_mini, fx_mini = forward_prop(X_train_mini, W, b)
			g_W_mini, g_b_mini = back_prop(X_train_mini, Y_train_mini, W, b, a_mini, h_mini, fx_mini)

			dW = [np.zeros(p.shape) for p in W]
			db = [np.zeros(p.shape) for p in b]
			for i in range(len(dW)):
				dW[i] = g_W_mini[i]
			for i in range(len(db)):
				db[i] = g_b_mini[i]

			etadW = [np.zeros(p.shape) for p in W]
			etadb = [np.zeros(p.shape) for p in b]
			for i in range(len(dW)):
				etadW[i] = etaL*dW[i]/float(mini_batch_size)				
			for i in range(len(W)):
				W[i]=W[i]-etadW[i]	


			for i in range(len(db)):
				etadb[i] = etaL*db[i]/float(mini_batch_size)				
			for i in range(len(b)):
				b[i]=b[i]-etadb[i]	

				loss += loss_function(fx_mini, Y_train_mini)


			#Log file writing. Uncomment 
			steps += 1
			if(logging == True and steps%100 == 0):				
				a_train, h_train, fx_train = forward_prop(X_train, W, b)
				loss_train_log = loss_function(fx_train, Y_train)
				y_cap_train = (np.argmax(fx_train, axis = 1))
				error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

				a_val, h_val, fx_val = forward_prop(X_val, W, b)
				loss_val_log = loss_function(fx_val, Y_val)
				y_cap_val = np.argmax(fx_val, axis = 1)	
				error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

				log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
				log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		if(logging_per_epoch == True):
			a_train, h_train, fx_train = forward_prop(X_train, W, b)
			loss_train_log = loss_function(fx_train, Y_train)
			y_cap_train = (np.argmax(fx_train, axis = 1))
			error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

			a_val, h_val, fx_val = forward_prop(X_val, W, b)
			loss_val_log = loss_function(fx_val, Y_val)
			y_cap_val = np.argmax(fx_val, axis = 1)	
			error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

			log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
			log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		

		#Check for annealing
		if(anneal == True):	
			a, h, fx_val = forward_prop(X_val, W, b)
			loss_val += loss_function(fx_val, Y_val)

			if(loss_val > prev_loss):
				etaL = etaL/2.0
				for i in range(len(W)):
					W[i]=W[i]+etadW[i]	

				for i in range(len(b)):
					b[i]=b[i]+etadb[i]
			else:
				prev_loss =loss_val	
		print(loss)		

	return W, b








#######################################MOMENTUM BASED GRADIENT DESCENT#####################################################

def momentum_gradient_descent(X, Y, X_val, Y_val, W, b, eta):
	prev_loss = np.inf
	etaL = eta
	for e in range(epochs):
		loss = 0
		loss_val = 0
		steps = 0

		randomize = np.arange(X.shape[0])
		np.random.shuffle(randomize)
		X = X[randomize]
		Y = Y[randomize]

		
		num_points = 0
		dW = [np.zeros(p.shape) for p in W]
		db = [np.zeros(p.shape) for p in b]

		prev_dW = [np.zeros(p.shape) for p in W]
		prev_db = [np.zeros(p.shape) for p in b]

		for i in range(0, X_train.shape[0], mini_batch_size):
			X_train_mini = X_train[i:i + mini_batch_size]
			Y_train_mini = Y_train[i:i + mini_batch_size]

			a_mini, h_mini, fx_mini = forward_prop(X_train_mini, W, b)
			g_W_mini, g_b_mini = back_prop(X_train_mini, Y_train_mini, W, b, a_mini, h_mini, fx_mini)

			dW = [np.zeros(p.shape) for p in W]
			db = [np.zeros(p.shape) for p in b]
			for i in range(len(dW)):
				dW[i] = g_W_mini[i]
			for i in range(len(db)):
				db[i] = g_b_mini[i]

			etadW = [np.zeros(p.shape) for p in W]
			etadb = [np.zeros(p.shape) for p in b]

			for i in range(len(dW)):
				etadW[i] = etaL*(dW[i]/float(mini_batch_size)) + gamma*prev_dW[i]			
			for i in range(len(W)):
				W[i]=W[i]-etadW[i]	

			for i in range(len(db)):
				etadb[i] = etaL*(db[i]/float(mini_batch_size)) + gamma*prev_db[i]				
			for i in range(len(b)):
				b[i]=b[i]-etadb[i]	


			prev_dW = [np.zeros(p.shape) for p in W]
			prev_db = [np.zeros(p.shape) for p in b]

			for i in range(len(etadW)):
				prev_dW[i] = etadW[i]
			for i in range(len(etadb)):
				prev_db[i] = etadb[i]

			loss += loss_function(fx_mini, Y_train_mini)

			steps += 1
			if(logging == True and steps%100 == 0):				
				a_train, h_train, fx_train = forward_prop(X_train, W, b)
				loss_train_log = loss_function(fx_train, Y_train)
				y_cap_train = (np.argmax(fx_train, axis = 1))
				error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

				a_val, h_val, fx_val = forward_prop(X_val, W, b)
				loss_val_log = loss_function(fx_val, Y_val)
				y_cap_val = np.argmax(fx_val, axis = 1)	
				error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

				log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
				log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		
		if(logging_per_epoch == True):
			a_train, h_train, fx_train = forward_prop(X_train, W, b)
			loss_train_log = loss_function(fx_train, Y_train)
			y_cap_train = (np.argmax(fx_train, axis = 1))
			error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

			a_val, h_val, fx_val = forward_prop(X_val, W, b)
			loss_val_log = loss_function(fx_val, Y_val)
			y_cap_val = np.argmax(fx_val, axis = 1)	
			error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

			log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
			log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		#Check for annealing
		if(anneal == True):	
			a, h, fx_val = forward_prop(X_val, W, b)
			loss_val += loss_function(fx_val, Y_val)

			if(loss_val > prev_loss):
				etaL = etaL/2.0
				for i in range(len(W)):
					W[i]=W[i]+etadW[i]	

				for i in range(len(b)):
					b[i]=b[i]+etadb[i]
			else:
				prev_loss =loss_val	
		print(loss)			
		

	return W, b



#######################################NAG DESCENT#####################################################

def nag(X, Y, X_val, Y_val, W, b, eta):
	prev_loss = np.inf
	etaL =eta
	for e in range(epochs):		

		randomize = np.arange(X.shape[0])
		np.random.shuffle(randomize)
		X = X[randomize]
		Y = Y[randomize]
		
		num_points = 0
		dW = [np.zeros(p.shape) for p in W]
		db = [np.zeros(p.shape) for p in b]

		prev_dW = [np.zeros(p.shape) for p in W]
		prev_db = [np.zeros(p.shape) for p in b]

		W_la = []
		b_la = []
		for i in range(len(W)):
			W_la.append(W[i] - gamma*prev_dW[i])
		for i in range(len(b)):
			b_la.append(b[i] - gamma*prev_db[i])	
		
		loss = 0
		loss_val = 0		
		steps = 0

		for i in range(0, X_train.shape[0], mini_batch_size):
			X_train_mini = X_train[i:i + mini_batch_size]
			Y_train_mini = Y_train[i:i + mini_batch_size]

			a_mini, h_mini, fx_mini = forward_prop(X_train_mini, W, b)
			g_W_mini, g_b_mini = back_prop(X_train_mini, Y_train_mini, W_la, b_la, a_mini, h_mini, fx_mini)

			dW = [np.zeros(p.shape) for p in W]
			db = [np.zeros(p.shape) for p in b]
			for i in range(len(dW)):
				dW[i] = g_W_mini[i]
			for i in range(len(db)):
				db[i] = g_b_mini[i]



			etadW = [np.zeros(p.shape) for p in W]
			etadb = [np.zeros(p.shape) for p in b]

			for i in range(len(dW)):
				etadW[i] = etaL*dW[i]/float(mini_batch_size) + gamma*prev_dW[i]				
			for i in range(len(W)):
				W[i]=W[i]-etadW[i]	

			for i in range(len(db)):
				etadb[i] = etaL*db[i]/float(mini_batch_size) + gamma*prev_db[i]					
			for i in range(len(b)):
				b[i]=b[i]-etadb[i]	
			
			prev_dW = []
			prev_db = []
			for i in range(len(etadW)):
				prev_dW.append(etadW[i])
			for i in range(len(etadb)):
				prev_db.append(etadb[i])

			W_la = []
			b_la = []
			for i in range(len(W)):
				W_la.append(W[i] - gamma*etadW[i])
			for i in range(len(b)):
				b_la.append(b[i] - gamma*etadb[i])

			loss += loss_function(fx_mini, Y_train_mini)

			steps += 1
			if(logging == True and steps%100 == 0):				
				a_train, h_train, fx_train = forward_prop(X_train, W, b)
				loss_train_log = loss_function(fx_train, Y_train)
				y_cap_train = (np.argmax(fx_train, axis = 1))
				error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

				a_val, h_val, fx_val = forward_prop(X_val, W, b)
				loss_val_log = loss_function(fx_val, Y_val)
				y_cap_val = np.argmax(fx_val, axis = 1)	
				error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

				log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
				log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		
		if(logging_per_epoch == True):
			a_train, h_train, fx_train = forward_prop(X_train, W, b)
			loss_train_log = loss_function(fx_train, Y_train)
			y_cap_train = (np.argmax(fx_train, axis = 1))
			error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

			a_val, h_val, fx_val = forward_prop(X_val, W, b)
			loss_val_log = loss_function(fx_val, Y_val)
			y_cap_val = np.argmax(fx_val, axis = 1)	
			error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

			log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
			log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		#Check for annealing
		if(anneal == True):	
			a, h, fx_val = forward_prop(X_val, W, b)
			loss_val += loss_function(fx_val, Y_val)

			if(loss_val > prev_loss):
				etaL = etaL/2.0
				for i in range(len(W)):
					W[i]=W[i]+etadW[i]	

				for i in range(len(b)):
					b[i]=b[i]+etadb[i]
			else:
				prev_loss =loss_val	
		print(loss)		

	return W, b



#######################################ADAM GRADIENT DESCENT#####################################################

def adam(X, Y, X_val, Y_val, W, b, eta):
	prev_loss = np.inf
	etaL =eta
	m_w = [np.zeros(p.shape) for p in W]
	v_w = [np.zeros(p.shape) for p in W]
	m_b = [np.zeros(p.shape) for p in b]
	v_b = [np.zeros(p.shape) for p in b]
	t = 0
	epsilon = 1e-8

	for e in range(epochs):
		loss = 0
		loss_val = 0
		steps = 0

		randomize = np.arange(X.shape[0])
		np.random.shuffle(randomize)
		X = X[randomize]
		Y = Y[randomize]

		
		num_points = 0
		dW = [np.zeros(p.shape) for p in W]
		db = [np.zeros(p.shape) for p in b]

		for i in range(0, X_train.shape[0], mini_batch_size):
			X_train_mini = X_train[i:i + mini_batch_size]
			Y_train_mini = Y_train[i:i + mini_batch_size]

			a_mini, h_mini, fx_mini = forward_prop(X_train_mini, W, b)
			g_W_mini, g_b_mini = back_prop(X_train_mini, Y_train_mini, W, b, a_mini, h_mini, fx_mini)

			dW = [np.zeros(p.shape) for p in W]
			db = [np.zeros(p.shape) for p in b]
			for i in range(len(dW)):
				dW[i] = g_W_mini[i]
			for i in range(len(db)):
				db[i] = g_b_mini[i]

			t += 1.0

			for i in range(len(dW)):
				m_w[i] = (beta1*m_w[i] + (1.0-beta1)*dW[i])/(1.0 - math.pow(beta1,t))
			for i in range(len(db)):
				m_b[i] = (beta1*m_b[i] + (1.0-beta1)*db[i])/(1.0 - math.pow(beta1,t))

			for i in range(len(dW)):
				#print (beta2*v_w[i][0] + (1.0-beta2)*((dW[i])**2))/(1.0 - math.pow(beta2,t))
				v_w[i] = (beta2*v_w[i] + (1.0-beta2)*((dW[i])**2))/(1.0 - math.pow(beta2,t))
			for i in range(len(db)):
				v_b[i] = (beta2*v_b[i] + (1.0-beta2)*((db[i])**2))/(1.0 - math.pow(beta2,t))
				
			etadW = [np.zeros(p.shape) for p in W]
			etadb = [np.zeros(p.shape) for p in b]

			for i in range(len(dW)):
				etadW[i] = (eta * m_w[i])/(np.sqrt(v_w[i] + epsilon))				
			for i in range(len(W)):
				W[i]=W[i]-etadW[i]	


			for i in range(len(db)):
				etadb[i] = (eta * m_b[i])/(np.sqrt(v_b[i] + epsilon))					
			for i in range(len(b)):
				b[i]=b[i]-etadb[i]	

			loss += loss_function(fx_mini, Y_train_mini)

			#Log file writing
			steps += 1
			if(logging == True and steps%100 == 0):
				
				
				a_train, h_train, fx_train = forward_prop(X_train, W, b)
				loss_train_log = loss_function(fx_train, Y_train)
				y_cap_train = (np.argmax(fx_train, axis = 1))
				error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

				a_val, h_val, fx_val = forward_prop(X_val, W, b)
				loss_val_log = loss_function(fx_val, Y_val)
				y_cap_val = np.argmax(fx_val, axis = 1)	
				error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

				log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
				log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		
		if(logging_per_epoch == True):
			a_train, h_train, fx_train = forward_prop(X_train, W, b)
			loss_train_log = loss_function(fx_train, Y_train)
			y_cap_train = (np.argmax(fx_train, axis = 1))
			error_train = 1.0 - accuracy_score(np.argmax(Y_train, axis = 1), y_cap_train)

			a_val, h_val, fx_val = forward_prop(X_val, W, b)
			loss_val_log = loss_function(fx_val, Y_val)
			y_cap_val = np.argmax(fx_val, axis = 1)	
			error_val = 1.0 - accuracy_score(np.argmax(Y_val, axis = 1), y_cap_val)

			log_train_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_train_log)+", Error: "+str(error_train)+", lr: "+str(etaL)+"\n")
			log_val_file.write("Epoch "+str(e+1)+", Step "+str(steps)+", Loss: "+str(loss_val_log)+", Error: "+str(error_val)+", lr: "+str(etaL)+"\n")

		if(anneal == True):	
			a, h, fx_val = forward_prop(X_val, W, b)
			loss_val += loss_function(fx_val, Y_val)

			if(loss_val > prev_loss):
				etaL = etaL/2.0
				for i in range(len(W)):
					W[i]=W[i]+etadW[i]	
				for i in range(len(b)):
					b[i]=b[i]+etadb[i]

				for i in range(len(m_w)):
					m_w[i] = m_w[i]*(1 - math.pow(beta1,t))
				for i in range(len(m_b)):
					m_b[i] = m_b[i]*(1 - math.pow(beta1,t))
					
				for i in range(len(v_w)):
					v_w[i] = v_w[i]*(1 - math.pow(beta2,t))
				for i in range(len(v_b)):
					v_b[i] = v_b[i]*(1 - math.pow(beta2,t))	
	

				for i in range(len(dW)):
					m_w[i] = m_w[i] - (1-beta1)*dW[i]
					m_w[i] = m_w[i]/beta1
				for i in range(len(db)):
					m_b[i] = m_b[i] - (1-beta1)*db[i]
					m_b[i] = m_b[i]/beta1

				for i in range(len(dW)):
					v_w[i] = v_w[i] - (1-beta2)*((dW[i])**2)
					v_w[i] = v_w[i]/beta2
				for i in range(len(db)):
					v_b[i] = v_b[i] - (1-beta2)*((db[i])**2)	
					v_b[i] = v_b[i]/beta2

			else:
				prev_loss = loss_val

		print(loss)		

	return W, b







#######################################TRAINING NETWORK#####################################################
def train_network(X, Y, X_val, Y_val, N, k, eta, activation):
	Nshift = list(N)
	Nshift.insert(0, len(X[0]))
	newN = list(N)
	newN.append(k)

	W = [np.random.uniform(-0.25, 0.25, (p[1], p[0])) for p in zip(Nshift, newN)]
	b = [np.random.uniform(-0.25, 0.25, (1, p)) for p in newN]

	# #Xavier initialisation
	# #794 = 784(ip dimension) + 10(output dimension)
	# if (activation == 'sigmoid'):
	# 	W = [np.random.uniform(-np.sqrt(6.0/794.0), np.sqrt(6.0/794.0), (p[1], p[0])) for p in zip(Nshift, newN)]
	# 	b = [np.random.uniform(-np.sqrt(6.0/794.0), np.sqrt(6.0/794.0), (1, p)) for p in newN]

	# elif (activation == 'tanh'):
	# 	W = [np.random.uniform(-4.0*np.sqrt(6.0/794.0), 4.0*np.sqrt(6.0/794.0), (p[1], p[0])) for p in zip(Nshift, newN)]
	# 	b = [np.random.uniform(-4.0*np.sqrt(6.0/794.0), 4.0*np.sqrt(6.0/794.0), (1, p)) for p in newN]
	

	(W,b) = gd_algo(X, Y, X_val, Y_val, W, b, eta)
	 
	return W,b



#######################################TEST FUNCTION#####################################################
def test_func():

	X = np.array([[1, 2, 1, 2], [2, 4, 2, 4], [1, 2, 3, 1], [40, 50, 60, 100], [50, 60, 70, 30], [60,40,30,90]])
	Y = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])
	# X = np.array([[1,], [5], [2], [6]])
	# Y = np.array([[[0], [1]], [[1], [0]], [[0], [1]], [[1], [0]]])
	# X = np.array([[1,0], [0, 1], [1,1], [0, 0]])
	# Y = np.array([[[1], [0]], [[1], [0]], [[0], [1]], [[0], [1]]])


	N = [10]
	k = 2
	# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	(W, b) = train_network(X, Y, X, Y, N, 2, 0.01)
	# y_cap = []
	
	a_val, h_val, fx_val = forward_prop(X, W, b)

	print(fx_val)
	print(np.argmax(fx_val, axis = 1))
	# y_cap.append(np.argmax(fx_val))

	print([a for a in zip(np.argmax(Y, axis = 1),np.argmax(fx_val, axis = 1))])







#######################################ARGUMENT PARSING#####################################################

parser = argparse.ArgumentParser(description='Neural Network')

parser.add_argument('--lr', action="store", dest = 'lr', type=float)
parser.add_argument('--momentum', action="store", dest="momentum", type=float)
parser.add_argument('--num_hidden', action="store", dest="num_hidden", type=int)
parser.add_argument('--sizes', action="store", dest="sizes")
parser.add_argument('--activation', action="store", dest="activation")
parser.add_argument('--loss', action="store", dest="loss")
parser.add_argument('--opt', action="store", dest="opt")
parser.add_argument('--batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('--anneal', action="store", dest="anneal")
parser.add_argument('--save_dir', action="store", dest="save_dir")
parser.add_argument('--expt_dir', action="store", dest="expt_dir")
parser.add_argument('--train', action="store", dest="train")
parser.add_argument('--val', action="store", dest="val")
parser.add_argument('--test', action="store", dest="test")


args = parser.parse_args()
print(args)


#######################################SETTING HYPERPARAMETERS#####################################################
options = {'sq':sq, 'ce':ce, 'mean_squared' :mean_squared,'cross_entropy' : cross_entropy,
'sigmoid':sigmoid, 'tanh':tanh, 'relu' : relu, 'linear':linear, 'softmax':softmax,
'gd':gradient_descent, 'momentum':momentum_gradient_descent, 'nag':nag, 'adam':adam,
}

error = options[args.loss]
gd_algo =  options[args.opt]
activation =  options[args.activation]

eta = args.lr
mini_batch_size = args.batch_size

gamma = args.momentum
if(args.anneal in ['true', 'True', 't']):
	anneal = True

save_dir = args.save_dir
expt_dir = args.expt_dir
os.system("mkdir " + expt_dir)
os.system("mkdir " + save_dir)
log_train_file = open(expt_dir+"log_train.txt", 'w')
log_train_file = open(expt_dir+"log_train.txt", 'a')
log_val_file = open(expt_dir+"log_val.txt", 'w')
log_val_file = open(expt_dir+"log_val.txt", 'a')

train_file = args.train
val_file = args.val
test_file = args.test

N = [int(neurons.strip()) for neurons in args.sizes.strip().split(',')]

# if(not(mini_batch_size == 1 or mini_batch_size%5 == 0)):
# 	raise ValueError('Valid values for batch_size are 1 and multiples of 5 only')

if(args.loss == 'sq'):
	loss_function = options['mean_squared']
	last_layer_activation =  options['linear']
else:	
	loss_function = options['cross_entropy']
	last_layer_activation = options['softmax']

if(args.activation == 'sigmoid'):
	grad_a = grad_a_sigmoid
elif(args.activation == 'tanh'):
	grad_a = grad_a_tanh
else:
	grad_a = grad_a_relu


#Parameters for Adam
beta1 = 0.9
beta2 = 0.99
#Number of epochs
epochs = 20
#Set number of classes
k = 10


logging = False
logging_per_epoch = True





#######################################TRAIN AND TEST#######################################################


#Load training data
train = pd.read_csv(train_file)
X_train = (train.ix[:,1:-1].values).astype('float32')
labels_train = train.ix[:,-1].values.astype('int32')
#Convert to one-hot
Y_train = np.zeros((labels_train.shape[0], 10))
Y_train[np.arange(labels_train.shape[0]), labels_train] = 1


#Moved here cuz has to be passed for annealing
#Load validation data
val = pd.read_csv(val_file)
X_val = (val.ix[:,1:-1].values).astype('float32')
labels_val = val.ix[:,-1].values.astype('int32')
#Convert to one-hot
Y_val = np.zeros((labels_val.shape[0], 10))
Y_val[np.arange(labels_val.shape[0]), labels_val] = 1


#Train
mean_X = X_train.mean().astype(np.float32)
std_X = X_train.std().astype(np.float32)
X_train = (X_train-mean_X)/std_X

mean_X = X_val.mean().astype(np.float32)
std_X = X_val.std().astype(np.float32)
X_val = (X_val-mean_X)/std_X

print("Begin Training at " + str(datetime.datetime.time(datetime.datetime.now())))
N  = [100,100,100,100,100]
W1, b1 = train_network(X_train, Y_train, X_val, Y_val, N, k, eta, activation)
N  = [100, 100, 50, 50, 50]
W2, b2 = train_network(X_train, Y_train, X_val, Y_val, N, k, eta, activation)
N  = [200, 200, 200, 100, 100,50]
W3, b3 = train_network(X_train, Y_train, X_val, Y_val, N, k, eta, activation)
N  = [200,200,200,100,100,100]
W4, b4 = train_network(X_train, Y_train, X_val, Y_val, N, k, eta, activation)
N  = [300,300,200,100,50]
W5, b5 = train_network(X_train, Y_train, X_val, Y_val, N, k, eta, activation)



#Done training
log_val_file.close()
log_train_file.close()

# #Pickle and save W and b
# f = open(save_dir+"W_and_b",'wb')
# pickle.dump([W, b], f)
# f.close()

#Check pickle
# f2 = open(save_dir+"Wandb", 'rb')
# s = pickle.load(f2)
# print(s[0], s[1])
# f2.close()


#Validate
print("Begin Validation at " + str(datetime.datetime.time(datetime.datetime.now())))

(a_val, h_val, fx_val1) = forward_prop(X_val, W1, b1)
y_cap1 = (np.argmax(fx_val1, axis = 1))
(a_val, h_val, fx_val2) = forward_prop(X_val, W2, b2)
y_cap2 = (np.argmax(fx_val2, axis = 1))
(a_val, h_val, fx_val3) = forward_prop(X_val, W3, b3)
y_cap3 = (np.argmax(fx_val3, axis = 1))
(a_val, h_val, fx_val4) = forward_prop(X_val, W4, b4)
y_cap4 = (np.argmax(fx_val4, axis = 1))
(a_val, h_val, fx_val5) = forward_prop(X_val, W5, b5)
y_cap5 = (np.argmax(fx_val5, axis = 1))

y_cap = []
for i in range(len(y_cap1)):
	y_cap_i_list = [y_cap1[i], y_cap2[i], y_cap3[i], y_cap4[i], y_cap5[i]]
	decision = max(y_cap_i_list,key=y_cap_i_list.count)
	y_cap.append(decision)
	

print("Accuracy Score : " + str(accuracy_score(labels_val, y_cap)))
print("F-score : " + str(f1_score(labels_val, y_cap, average = 'macro')))



if(test_file != None):
	#Load test data
	test = pd.read_csv(test_file)
	X_test = (test.ix[:,1:].values).astype('float32')
	test_ids = (test.ix[:,0].values).astype('int32')

	#Standardising test data
	mean_X = X_test.mean().astype(np.float32)
	std_X = X_test.std().astype(np.float32)
	X_test = (X_test-mean_X)/std_X

	#Begin test
	print("Begin Test at " + str(datetime.datetime.time(datetime.datetime.now())))
	(a_val, h_val, fx_val1) = forward_prop(X_test, W1, b1)
	y_cap1 = (np.argmax(fx_val1, axis = 1))
	(a_val, h_val, fx_val2) = forward_prop(X_test, W2, b2)
	y_cap2 = (np.argmax(fx_val2, axis = 1))
	(a_val, h_val, fx_val3) = forward_prop(X_test, W3, b3)
	y_cap3 = (np.argmax(fx_val3, axis = 1))
	(a_val, h_val, fx_val4) = forward_prop(X_test, W4, b4)
	y_cap4 = (np.argmax(fx_val4, axis = 1))
	(a_val, h_val, fx_val5) = forward_prop(X_test, W5, b5)
	y_cap5 = (np.argmax(fx_val5, axis = 1))

	y_cap = []
	for i in range(len(y_cap1)):
		y_cap_i_list = [y_cap1[i], y_cap2[i], y_cap3[i], y_cap4[i], y_cap5[i]]
		decision = max(y_cap_i_list,key=y_cap_i_list.count)
		y_cap.append(decision)

	
	#Write submission
	submissions=pd.DataFrame({"id": test_ids, "label": y_cap})
	submissions.to_csv("test_submission_ensemble_5.csv", index=False, header=True)





os.system('spd-say "Finished Executing"')


