'''
Utility functions for UNet

'''
import numpy as np
from sklearn.metrics import confusion_matrix


def compare_image(actual_label, predicted_image):
	'''
	Takes in the actual label and predicted label, and outputs an image with blue pixels for correct prediction and red pixels for incorrect predictions
	Input: 	actual_label, array of shape (rows, cols, classes)
			predicted_label, array of shape (rows, cols, classes)
	Output: intersect_image, array of shape (rows, cols, classes)
	'''
	red = np.array([255, 0, 0])
	blue =np.array([0, 0, 255])

	#open cv manipulates images as BGR, not RGB
	cv_blue = np.array([255, 0, 0])
	cv_red =np.array([0, 0, 255])

	intersect_image = actual_label # copy label; we are changing only the green pixels

	if predicted_image.shape == actual_label.shape:

		for iPixel in range(actual_label.shape[0]):
			for jPixel in range(actual_label.shape[1]):
				pixel = actual_label[iPixel,jPixel]
				predicted_pixel = predicted_image[iPixel, jPixel]
				# print(pixel)
				if (pixel.argmax(axis=-1) == red.argmax(axis=-1)) or (pixel.argmax(axis=-1) == blue.argmax(axis=-1)):
					if pixel.argmax(axis=-1) == predicted_pixel.argmax(axis=-1):
						intersect_image[iPixel, jPixel, :] = cv_blue
						match+=1
						# print('yes')
					else:
						intersect_image[iPixel, jPixel, :] = cv_red
						wrong+=1
	return intersect_image, match, wrong

def make_confusion_matrix(actual_label, predicted_label):
	'''
	Takes in the actual label and predicted label, converts them to class labels and outputs the confusion matrix
	Input: 	actual_label, array of shape (rows, cols, classes)
			predicted_label, array of shape (rows, cols, classes)
	Output: confusion matrix, array of shape (classes, classes)

	'''

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

def make_grayscale_map(predicted_label):
	'''
	Takes in the actual label and predicted label, finds the highest probability and stores. 
	Returns a grayscale image of the intensities(probabilities)
	Input: 	predicted_label, array of shape (rows, cols, classes)
	Output: grayscale image, array of shape (rows, cols)
	'''

	predicted_label_grayscale = np.ndarray((predicted_label.shape[0], predicted_label.shape[1]), dtype=np.float32)

	for iPixel in range(predicted_label.shape[0]):
		for jPixel in range(predicted_label.shape[1]):
			predicted_pixel = predicted_label[iPixel,jPixel]

			if predicted_pixel.argmax(axis=-1) == 0:
				predicted_label_grayscale[iPixel,jPixel] = predicted_label[iPixel,jPixel, 0]
			elif predicted_pixel.argmax(axis=-1) == 1:
				predicted_label_grayscale[iPixel,jPixel] = predicted_label[iPixel,jPixel, 1]
			elif predicted_pixel.argmax(axis=-1) == 2:
				predicted_label_grayscale[iPixel,jPixel] = predicted_label[iPixel,jPixel, 2]

	return predicted_label_grayscale

