'''
Program for data generator; to generate batches of image to feed into the network.
Useful when you can't store all your data at once on memory


Source for structure: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
'''

import cv2
import numpy as np
import os
import PIL
from PIL import Image
from numpy import array
from glob import glob
import numpy as np
import math
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# filepath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/'
# labelpath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels/'

# batch_size = 32

# #Make a list of all files
# file_list = list(os.listdir(filepath))
# no_of_images = len(file_list)

# #Shuffle the indices of the file list
# indices_shuffled = np.random.choice(no_of_images, no_of_images, replace =False)

# #Split the indices into train, validation and test data
# x_train_size1 = int(no_of_images *0.7)
# x_train_size2 = int(no_of_images *0.6)

# x_val_size1 = int(no_of_images*0.9)
# x_val_size2 = int(no_of_images*0.8)


# x_train, x_val, x_test = np.split(indices_shuffled, [x_train_size1, x_val_size1])

# x_set_indices = [243,  54,  50 ,174 ,189, 327 ,187 ,169 , 58 , 48 ,235, 252 , 21 ,160, 276 ,191 ,293 ,257,
#  308 ,149, 130 ,151,  99 , 87 ,214 ,121 ,328,  20, 188  ,71 ,106, 270, 102]
def my_generator(filepath, labelpath, x_set_indices, batch_size):

	rows =200
	cols = 200
	file_list = list(os.listdir(filepath))
	label_list = list(os.listdir(labelpath))

	# Infinite loop
	while 1: 

		for index in x_set_indices:

			image_folder = os.listdir(os.path.join(filepath, file_list[index]))
			label_folder = os.listdir(os.path.join(labelpath, label_list[index]))
			# print(len(image_folder))
			nBatch_in_folder = int(len(image_folder)/batch_size)

			imgdata = np.ndarray((batch_size, rows,cols,3), dtype=np.uint8)
			imglabels = np.ndarray((batch_size,rows,cols,3), dtype=np.uint8)

			batch_start = 0
			batch_end = batch_size
			folder_total = len(image_folder)

			while batch_start< folder_total:

				limit = min(batch_end, folder_total)
				list_images = range(batch_start,batch_end)

				i=0
				for iImage in list_images:

					img = load_img(os.path.join(filepath, file_list[index])+'/'+ image_folder[iImage], grayscale = False)
					# print(os.path.join(filepath, file_list[index]+'/'+ image_folder[iImage]))
					# cv2.imshow('current image',img)
					# cv2.waitKey()	
					label = load_img(os.path.join(labelpath, label_list[index])+'/'+ label_folder[iImage], grayscale = False)
					# print(labelpath, label_list[index]+'/'+ label_folder[iImage])
					img = array(img)
					# img = img_to_array(img)
					# print(img.shape)
					# cv2.imshow('current image',img)
					# cv2.waitKey()
					# label = img_to_array(label)
					label = array(label)

					imgdata[i] = img

					imglabels[i] = label
					i+=1


					yield imgdata, imglabels
					batch_start += batch_size
					batch_end += batch_size


