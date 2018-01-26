
from keras import backend as K

from keras.losses import categorical_crossentropy



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
	intersection = K.sum(y_true * y_pred, axis=[0,-1,-2])
	summation = K.sum(y_true, axis=[0,-1,-2]) + K.sum(y_pred, axis=[0,-1,-2])

	return K.mean((2.0 * intersection + smooth) / (summation + smooth))


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
	return dice_coef_loss(y_true, y_pred)+ categorical_crossentropy(y_true, y_pred)