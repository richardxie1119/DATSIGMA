import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import ipywidgets as widgets
from ipywidgets import Box, IntSlider

import xgboost
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import torch

class scMSModel():

	def __init__(self, intens_mtx):

		self.model = {}
		self.intens_mtx = intens_mtx
		self.names = list(intens_mtx.index)



	def get_kfold_cv(self, k):
    
		sample_names = self.intens_mtx.index.values
		sample_names_shuffled = shuffle(sample_names,random_state=19)
		sample_names_kfold_test = np.array_split(sample_idx_shuffled,k)

		self.cv = sample_names_kfold_test

		return sample_names_kfold_test

	#def cross_val(self, model):




	def get_gbt_model(self, learning_rate=0.1,min_split_loss=0,
	              max_depth=6, n_estimators=300,
	              reg_lambda=1,reg_alpha=1,subsample=1):


		model = xgboost.XGBClassifier(learning_rate=learning_rate,
		                              min_split_loss=min_split_loss,max_depth=max_depth,
		                              n_estimators=n_estimators,
		                              reg_lambda=reg_lambda,reg_alpha=reg_alpha,
		                              subsample=subsample,use_label_encoder=False,n_jobs=8)
		self.model['gbt'] = model
		return model


	def get_rf_model(self, n_estimators=300, criterion='gini',
	             max_depth=None, min_samples_split=2,
	             min_samples_leaf=1, min_weight_fraction_leaf=0.0,
	             max_features='auto', max_leaf_nodes=None,max_samples=None):

		model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
		                               max_depth=max_depth, min_samples_split=min_samples_split,
		                               min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
		                               max_features=max_features, max_leaf_nodes=max_leaf_nodes,
		                               max_samples=None, random_state=19, n_jobs=8)
		self.model['rf'] = model
		return model


	def get_svm_model(self, kernel, C):

		model = SVC(kernel=kernel,C=C, random_state=19)

		self.model['svm'] = model
		return model


	def get_LR_model(self, penalty='l2',tol=0.0001,
	             C=1.0, max_iter=100):


		model = LogisticRegression(penalty=penalty,tol=tol,
		                           C=C,max_iter=max_iter, random_state=19, n_jobs=8)

		self.model['LR'] = model
		return model


	def get_LDA_model(self,):

		model = LinearDiscriminantAnalysis()

		self.model['LDA'] = model
		return model


	def get_KNN_model(self, n_neighbors=5, algorithm='auto',
	              leaf_size=30, metric='minkowski'):

		model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm,
		              leaf_size=leaf_size, metric=metric, n_jobs=8)

		self.model['KNN'] = model
		return model



	def get_DNN_model(self, layer_shapes):

		model = Net(layer_shapes=layer_shapes)

		self.model['DNN'] = model
		self.layer_shapes = layer_shapes
		return model



	def train_DNN(self, X_train, y_train, learning_rate=0.001, epochs=30, batch_size=32):

		trainset = dataset(X_train,y_train)
		#DataLoader
		trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

		optimizer = torch.optim.Adam(self.model['DNN'].parameters(),lr=learning_rate)

		if self.layer_shapes[-1] >1:
			loss_fn = nn.CrossEntropyLoss()
		else:
			loss_fn = nn.BCELoss()

		losses = []
		accuracy = []
		for i in range(epochs):
			for j,(x_batch,y_batch) in enumerate(trainloader):

				#calculate output
				output = self.model['DNN'](x_batch)

				#calculate loss
				if self.layer_shapes[-1] >1:
					loss = loss_fn(output, Variable(y_batch).long())
				else:
					loss = loss_fn(output,y_batch.reshape(-1,1))

				#accuracy
				predicted = self.model['DNN'](torch.tensor(X_train, dtype=torch.float32))
				if self.layer_shapes[-1] >1:
					__, predicted = torch.max(predicted, dim = 1)

				acc = (predicted.reshape(-1).detach().numpy().round() == y_train).mean()
				#backprop
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			losses.append(loss)
			accuracy.append(acc)

			print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

		return losses, accuracy



class dataset(Dataset):

	def __init__(self,x,y):
	    self.x = torch.tensor(x,dtype=torch.float32)
	    self.y = torch.tensor(y,dtype=torch.float32)
	    self.length = self.x.shape[0]

	def __getitem__(self,idx):
	    return self.x[idx],self.y[idx]

	def __len__(self):
	    return self.length



class Net(nn.Module):

	def __init__(self,layer_shapes):

		super(Net,self).__init__()
		self.layers = nn.ModuleList()
		self.layer_shapes = layer_shapes

		for i in range(len(layer_shapes)-1):
			self.layers.append(nn.Linear(layer_shapes[i],layer_shapes[i+1]))

	def forward(self,x):
	    for i in range(len(self.layers)-1):
	    	x = torch.relu(self.layers[i](x))

	    if self.layer_shapes[-1] >1:
	    	x = self.layers[-1](x)
	    else:
	    	x = torch.sigmoid(self.layers[-1](x))


	    return x
