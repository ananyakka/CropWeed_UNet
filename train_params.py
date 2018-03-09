'''
Custom metrics and losses
So far contains working versions of:
dice_coef, dice_coef_loss
jaccard_coef, jaccard_coef_loss
weighted_dice_coef, weighted_dice_coef_loss
mean_cross_entropy
f_score_weighted, f_score_weighted_loss
precision_score_class
recall_score_class
sensitivity
'''


from keras import backend as K

from keras.losses import categorical_crossentropy
from keras.layers import multiply

from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf


def weighted_dice_coef(y_true, y_pred, smooth=0.0):
	#Average dice coefficient per batch.

	# Weighted by number of instances in each image
	num_samples = K.sum(K.round(y_true), axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels
	total_samples = K.sum(num_samples, axis=-1,keepdims=True)
	prop_samples=num_samples/total_samples
	print(prop_samples.shape)
	weights = 1.0 - prop_samples

	# weight order: weeds, background, plants
	# # Same weight for all images
	# weights = np.array([20,1,1])
	# weights=np.array([10,1,1])

	intersection = K.sum(y_true * y_pred, axis=[-2,-3])
	intersection_weighted = K.sum(intersection*weights)
	
	summation = K.sum(y_true, axis=[-2,-3]) + K.sum(y_pred, axis=[-2,-3])
	summation_weighted = K.sum(summation*weights)
	
	dice = K.mean((2.0 * intersection + smooth) / (summation+ smooth))

	return dice 

def weighted_dice_coef_loss(y_true, y_pred):
	return  -weighted_dice_coef(y_true, y_pred, smooth=0.0)

def mean_cross_entropy(y_true, y_pred):
	"""
	A weighted version of keras.objectives.categorical_crossentropy

	Variables:
	weights: numpy array of shape (C,) where C is the number of classes

	Usage:
	weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
	loss = weighted_categorical_crossentropy(weights)
	model.compile(loss=loss,optimizer='adam')
	"""
	# weight order: weeds, background, plants

	# if len(K.int_shape(y_pred))!= 3:
	if K.int_shape(y_true)[0] != None:
		# scale predictions so that the class probas of each sample sum to 1
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

		# # Same weight for all images
		# loss = y_true * K.log(y_pred) * weights
		# loss = -K.sum(loss, -1)

		# Weighted individually by number of instances
		num_samples = K.sum(y_true, axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels

		total_samples = K.sum(num_samples, axis=-1,keepdims=True)
		prop_samples = num_samples/total_samples
		# weights = 1.0 - prop_samples

		weights = [0.9, 0.05, 0.1]
		# weights= [0.9, 0.05, 0.5]

		print(y_pred.shape)
		batch_size = 32
		# batch_size=K.int_shape(y_pred)[0]
		loss = []
		for iImage in range(batch_size):
			loss_temp = y_true[iImage, ...] * K.log(y_pred[iImage, ...])*weights
			loss_sum = -K.mean(loss_temp, -1)
			loss.append(loss_sum)
		# loss = -K.sum(loss, -1)
		loss = K.stack(loss)	

	# elif len(K.int_shape(y_pred))== 3:
	else:
		# scale predictions so that the class probas of each sample sum to 1
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

		# Weighted individually by number of instances
		num_samples = K.sum(y_true, axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels

		total_samples = K.sum(num_samples, axis=-1,keepdims=True)
		prop_samples = num_samples/total_samples
		# weights = 1.0 - prop_samples
		weights = [0.9, 0.05, 0.2]

		loss_temp = y_true * K.log(y_pred)*weights
		loss = -K.mean(loss_temp, -1)

	return loss

def f_score_weighted(y_true, y_pred):
	precision = precision_score_class(y_true, y_pred)
	# print(precision.shape)
	recall = recall_score_class(y_true, y_pred)

	# weighted by number of instances
	num_samples = K.sum(y_true, axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels
	print(num_samples)
	total_samples = K.sum(num_samples, axis=-1,keepdims=True)
	print(total_samples)
	prop_samples=num_samples/total_samples
	print(prop_samples)
	# weight order: weeds, background, plants
	weights = np.array([0.8,0.1,0.1])
	# weights = 1 - prop_samples
	precision_weighted = precision*weights
	recall_weighted =recall*weights

	f1_score = 2*(precision_weighted*recall_weighted)/(precision_weighted+recall_weighted) # this value is blowing up; why??
	# print(f1_score.shape)
	f1_score_weighted = K.mean(f1_score, axis=-1) # weighted f1 score
	# print(f1_score_weighted)
	# print(f1_score_weighted.shape)
	
	return f1_score_weighted


def f_score_weighted_loss(y_true, y_pred):
	return -f_score_weighted(y_true, y_pred)

def precision_score_class(y_true, y_pred):

	# true_positives = K.sum(K.round(y_true * y_pred), axis=[1,2])
	# predicted_positives = K.sum(K.round(y_pred), axis=[1,2])
	true_positives = K.sum((y_true * y_pred), axis=[-2,-3])
	predicted_positives = K.sum(y_pred, axis=[-2,-3])
	precision = true_positives/predicted_positives

	return precision

def recall_score_class(y_true, y_pred):
	# true_positives = K.sum(K.round(y_true * y_pred), axis=[1,2])
	# predicted_positives = K.sum(K.round(y_pred), axis=[1,2])
	# possible_positives =K.sum(K.round(y_true), axis=[1,2])

	# if y_true.shape[-1] == 
	# print(y_pred.shape[-2])
	# print(y_pred.shape[-3])
	true_positives = K.sum((y_true * y_pred), axis=[-2,-3])
	predicted_positives = K.sum(y_pred, axis=[-2,-3])
	possible_positives =K.sum(y_true, axis=[-2,-3])
	recall = predicted_positives/possible_positives
	# print('recall shape')
	# print(recall.shape)

	return recall

def sensitivity(y_true, y_pred):
	return recall_score_class(y_true, y_pred)
