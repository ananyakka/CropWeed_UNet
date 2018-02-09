
from keras import backend as K

from keras.losses import categorical_crossentropy

from sklearn.metrics import f1_score
import numpy as np




'''
def dice_coeff(y_true, y_pred):
    y_true_vec = K.flatten(y_true)
    y_pred_vec = K.flatten(y_pred)
    intersection = K.sum(y_true_vec * y_pred_vec)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_vec) + K.sum(y_pred_vec) + 1.0)


def jaccard_coef(y_true, y_pred):
    y_true_vec = K.flatten(y_true)
    y_pred_vec = K.flatten(y_pred)
    intersection = K.sum(y_true_vec * y_pred_vec)
    return (intersection + 1.0) / (K.sum(y_true_vec) + K.sum(y_pred_vec) - intersection + 1.0)


def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
return -dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=0.0):
	#Average dice coefficient per batch.
	intersection = K.sum(y_true * y_pred, axis=[0,-1,-2])
	summation = K.sum(y_true, axis=[0,-1,-2]) + K.sum(y_pred, axis=[0,-1,-2])

	return K.mean((2.0 * intersection + smooth) / (summation + smooth),axis=0)

'''
def dice_coef(y_true, y_pred, smooth=0.0):
	#Average dice coefficient per batch.
	print('true shape')

	print(y_true.shape)
	print('pred shape')
	print(y_pred.shape)
	print('mult shape')
	print((y_true*y_pred).shape)
	intersection = K.sum(y_true * y_pred, axis=[1,2])
	print(intersection.shape)
	print('intersection.shape')
	summation = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
	print(summation.shape)

	dice = K.mean((2.0 * intersection + smooth) / (summation + smooth))
	print(dice.shape)
	cce = categorical_crossentropy(y_true, y_pred)
	print('cce shape')
	print(cce.shape)
	return dice 

def dice_coef_loss(y_true, y_pred):
	return  -dice_coef(y_true, y_pred, smooth=0.0)


def jaccard_coef(y_true, y_pred, smooth=0.0):
	#Average jaccard coefficient per batch.
	intersection = K.sum(y_true * y_pred, axis=[0,-1,-2])
	union = K.sum(y_true, axis=[0,-1,-2]) + K.sum(y_pred, axis=[0,-1,-2]) - intersection
	return K.mean( (intersection + smooth) / (union + smooth))

def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

def jaccard_cross_entropy_loss(y_true, y_pred):
	return jaccard_coef_loss(y_true, y_pred)+ categorical_crossentropy(y_true, y_pred)

def dice_cross_entropy_loss(y_true, y_pred):
	return dice_coef_loss(y_true, y_pred) + categorical_crossentropy(y_true, y_pred)


def f_score_weighted(y_true, y_pred):
	print('y_true.shape')
	print(y_true.shape)
	y_true_class = K.max(y_true,axis=-1)
	print(y_true_class.shape)
	# y_true_flat = K.expand_dims(y_true_class, -1)
	# y_true_flat = K.reshape(y_true_class, K.shape(y_true))
	# print(y_true_flat.shape)

	# For Class 1: [0 0 1]
	true_positive1 = K.sum()
	
	

	return f1_score(y_true_flat, y_pred_flat, average='weighted')

def f_score_weighted_loss(y_true, y_pred):
	return -f_score_weighted(y_true, y_pred)