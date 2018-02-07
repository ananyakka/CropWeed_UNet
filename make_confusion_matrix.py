'''
Program takes in the actual label and predicted label, converts them to class labels and outputs the confusion matrix
'''


import numpy as np 
from sklearn.metrics import confusion_matrix



def make_confusion_matrix(actual_label, predicted_label):

	if predicted_label.shape == actual_label.shape:

		total_pixels = actual_label.shape[0] *actual_label.shape[1]
		actual_label_flat = np.ndarray((total_pixels), dtype=np.uint8)
		predicted_label_flat = np.ndarray((total_pixels), dtype=np.uint8)
		
		for i in range(actual_label.shape[0]):
			for j in range(actual_label.shape[1]):
				pixel = actual_label[i,j]
				predicted_pixel = predicted_label[i,j]

				i_flat = i*actual_label.shape[0] + j

				# Make one hot labels to class labels

				if pixel.argmax(axis=-1) == 0:
					actual_label_flat[i_flat] = 1
				elif pixel.argmax(axis=-1) == 1:
					actual_label_flat[i_flat] = 2
				elif pixel.argmax(axis=-1) == 2:
					actual_label_flat[i_flat] = 3
		
				if predicted_pixel.argmax(axis=-1) == 0:
					predicted_label_flat[i_flat] = 1
				elif predicted_pixel.argmax(axis=-1) == 1:
					predicted_label_flat[i_flat] = 2
				elif predicted_pixel.argmax(axis=-1) == 2:
					predicted_label_flat[i_flat] = 3

	return confusion_matrix(actual_label_flat, predicted_label_flat)


