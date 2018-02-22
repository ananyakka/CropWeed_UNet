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
	intersection = K.sum(y_true * y_pred, axis=[-2,-3])
	# intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=[1,2]) # if the y values aren't between 0 and 1 already
	print(intersection.shape)
	print('intersection.shape')
	summation = K.sum(y_true, axis=[-2,-3]) + K.sum(y_pred, axis=[-2,-3])
	print(summation.shape)

	dice = K.mean((2.0 * intersection + smooth) / (summation + smooth))
	print(dice.shape)
	cce = categorical_crossentropy(y_true, y_pred)
	print('cce shape')
	print(cce.shape)
	return dice 

def dice_coef_loss(y_true, y_pred):
	return  -dice_coef(y_true, y_pred, smooth=0.0)

def weighted_dice_coef(y_true, y_pred, smooth=0.0):
	#Average dice coefficient per batch.

	num_samples = K.sum(K.round(y_true), axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels
	total_samples = K.sum(num_samples, axis=-1,keepdims=True)
	prop_samples=num_samples/total_samples
	print(prop_samples.shape)
	# weights = 1/prop_samples
	# weights /= K.sum(weights)
	weights = 1.0 - prop_samples

	# # Same weight for all images
	# weights = np.array([20,1,1])

	# intersection = K.sum(y_true * y_pred, axis=[-2,-3])
	# # intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=[1,2])# if the y values aren't between 0 and 1 already
	# summation = K.sum(y_true, axis=[-2,-3]) + K.sum(y_pred, axis=[-2,-3])

	# dice = K.mean((2.0 * intersection*weights + smooth) / (summation*weights + smooth))

	# Weighted by number of instances in each image
	# num_samples = K.sum(K.round(y_pred), axis=[1,2]) # number of samples = number of positives/ones in each layer in the true labels
	# total_samples = K.sum(num_samples, axis=-1,keepdims=True)
	intersection = K.sum(y_true * y_pred, axis=[-2,-3])
	# weights = num_samples/total_samples
	# weights=np.array([10,1,1])
	intersection_weighted = K.sum(intersection*weights)
	summation = K.sum(y_true, axis=[-2,-3]) + K.sum(y_pred, axis=[-2,-3])
	summation_weighted = K.sum(summation*weights)
	dice = K.mean((2.0 * intersection + smooth) / (summation+ smooth))


	return dice 

def weighted_dice_coef_loss(y_true, y_pred):
	return  -weighted_dice_coef(y_true, y_pred, smooth=0.0)

def jaccard_coef(y_true, y_pred, smooth=0.0):
	#Average jaccard coefficient per batch.
	intersection = K.sum(y_true * y_pred, axis=[-2,-3])
	union = K.sum(y_true, axis=[-2,-3]) + K.sum(y_pred, axis=[-2,-3]) - intersection
	return K.mean( (intersection + smooth) / (union + smooth))

def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

def jaccard_cross_entropy_loss(y_true, y_pred):
	return jaccard_coef_loss(y_true, y_pred)+ categorical_crossentropy(y_true, y_pred)

def dice_cross_entropy_loss(y_true, y_pred):
	return dice_coef_loss(y_true, y_pred) + categorical_crossentropy(y_true, y_pred)

# def mean_cross_entropy(y_true, y_pred): # doesn't work : ?? how to multiply weights into the losses
# 	#Source: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
# 	"""
# 	A weighted version of keras.objectives.categorical_crossentropy

# 	Variables:
# 	weights: numpy array of shape (C,) where C is the number of classes

# 	Usage:
# 	weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
# 	# loss = weighted_categorical_crossentropy(weights)
# 	# model.compile(loss=loss,optimizer='adam')
# 	"""
	
# 	# weights = np.array([0.5,2,10])
# 	# weights = K.variable(weights)


# 	num_samples = K.sum(y_pred, axis=[1,2]) # number of samples = number of positives/ones in each layer in the true labels
# 	print('num sampels')
# 	print(num_samples.shape)
# 	total_samples = K.sum(num_samples, axis=-1,keepdims=True)
# 	print('total samples')
# 	print(total_samples.shape)
# 	prop_samples=num_samples/total_samples
# 	weights = 1/prop_samples

# 	# weights0 = K.ones_like(y_pred[:,:,:,0]).*weights[:,0]
# 	weights0 = multiply((y_pred[:,:,:,0],weights[:,0]))
# 	weights1 = multiply((y_pred[:,:,:,1],weights[:,1]))
# 	weights2 = multiply((y_pred[:,:,:,2],weights[:,2]))
# 	# weights1 = K.ones_like(y_pred[:,:,:,1]).*weights[:,1]
# 	# weights2 = K.ones_like(y_pred[:,:,:,2]).*weights[:,2]
# 	weights = K.stack((weights0,weights1,weights2), axis=-1)
# 	# print(loss0.shape)
# 	# loss1 = loss[:,:,:,1]*weights[:,1]
# 	# loss2 = loss[:,:,:,2]*weights[:,2]
# 	# weights = K.variable(weights)
# 	print(type(weights))
# 	print('weight shape')
# 	print(weights.shape)

# 	# scale predictions so that the class probas of each sample sum to 1
# 	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
# 	# clip to prevent NaN's and Inf's
# 	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
# 	# calc
# 	loss = y_true * K.log(y_pred)

# 	# loss_weighted = loss0+loss1+loss2
# 	print("loss shape")
# 	print(loss.shape)
	
# 	# loss_weighted = K.dot(loss, weights)
# 	loss_weighted = weights* loss
# 	# print("loss shape")
# 	# print(loss_weighted.shape)
# 	# loss_weighted = -K.sum(loss, -1)
# 	print(loss_weighted.shape)
# 	return -loss_weighted
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
	# weights = np.array([20,1,1])

	# print('weight shape')
	# print(weights.shape)
	# num_samples = K.sum(y_pred, axis=[1,2]) # number of samples = number of positives/ones in each layer in the true labels
	# print(num_samples)
# 	print('num sampels')
# 	print(num_samples.shape)
# 	total_samples = K.sum(num_samples, axis=-1,keepdims=True)
# 	print('total samples')
# 	print(total_samples.shape)
# 	prop_samples=num_samples/total_samples
# 	weights = 1/prop_samples
	# weights = K.variable(weights)

	# scale predictions so that the class probas of each sample sum to 1
	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
	# clip to prevent NaN's and Inf's
	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
	# calc
	# # Same weight for all images
	# loss = y_true * K.log(y_pred) * weights
	# loss = -K.sum(loss, -1)

	# Weighted individually by number of instances
	num_samples = K.sum(y_pred, axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels
	# print(num_samples)
	# print('num sampels')
	# print(num_samples.shape)
	total_samples = K.sum(num_samples, axis=-1,keepdims=True)
	prop_samples = num_samples/total_samples
	weights = 1.0 - prop_samples
	# weights= K.ones([y_pred.shape[-3], y_pred.shape[-2], y_pred.shape[-1]])
	# weights = np.array([10,1,1])
	print(y_pred.shape)
	batch_size = 32
	loss = []
	for iImage in range(batch_size):
		loss_temp = y_true[iImage, :,:,:] * K.log(y_pred[iImage, :,:,:])*weights[iImage,:]
		loss_sum = -K.mean(loss_temp, -1)
		loss.append(loss_sum)
	# loss = -K.sum(loss, -1)
	loss = K.stack(loss)	



	return loss




def f_score_weighted(y_true, y_pred):
	precision = precision_score_class(y_true, y_pred)
	# print(precision.shape)
	recall = recall_score_class(y_true, y_pred)

	# num_samples = K.sum(K.round(y_pred), axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels
	# # print(num_samples.shape)
	# total_samples = K.sum(num_samples, axis=-1,keepdims=True)
	# prop_samples=num_samples/total_samples
	# weights = 1/prop_samples
	# # print(weights.shape)
	# weights /= K.sum(weights)	
	# # weights = np.array([20,1,1])

	# precision_weighted = precision*weights
	# precision_weighted/=K.sum(precision_weighted)
	# precision_weighted = K.clip(precision_weighted, K.epsilon(), 1 - K.epsilon())
	# # print(precision_weighted.shape)
	# recall_weighted = recall*weights
	# recall_weighted /= K.sum(recall_weighted)
	# recall_weighted = K.clip(recall_weighted, K.epsilon(), 1 - K.epsilon())

	# f1_score = 2*(precision_weighted*recall_weighted)/(precision_weighted+recall_weighted)
	# # print(f1_score.shape)

	# weighted by number of instances
	num_samples = K.sum(y_true, axis=[-2,-3]) # number of samples = number of positives/ones in each layer in the true labels
	print(num_samples)
	total_samples = K.sum(num_samples, axis=-1,keepdims=True)
	print(total_samples)
	prop_samples=num_samples/total_samples
	print(prop_samples)
	# weights=num_samples/total_samples
	weights = np.array([0.8,0.1,0.1])
	# weights = 1 - prop_samples
	precision_weighted = precision*weights
	recall_weighted =recall*weights

	f1_score = 2*(precision_weighted*recall_weighted)/(precision_weighted+recall_weighted) # this value is blowing up; why??
	# f1_score = recall_weighted
	# print(f1_score.shape)
	f1_score_weighted = K.mean(f1_score, axis=-1) # weighted f1 score
	# print(f1_score_weighted)
	# print(f1_score_weighted.shape)
	# a = np.array([1,2,3])
	
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

# def specificity(y_true, y_pred):
# 	true_positives = K.sum(K.round(y_true * y_pred), axis=[1,2])
# 	predicted_positives = K.sum(K.round(y_pred), axis=[1,2])
# 	possible_positives =K.sum(K.round(y_true), axis=[1,2])

# 	false_positive = predicted_positives - true_positives
# 	true_negative = 
# 	specificity = false_positive / (false_positive+true_negative)
# 	return specificity
