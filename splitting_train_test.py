'''
Program to split images and labels into training dataset and test data 

'''

import cv2
import numpy as np
import os
import PIL
import pickle
from PIL import Image
from numpy import array
from glob import glob
import numpy as np
import pandas as pd
import math
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



filepath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/'

labelpath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels/'

rows =200
cols = 200

#Make a list of all files
file_list = list(os.listdir(filepath))
no_of_images = len(file_list)

#Shuffle the indices of the file list
indices_shuffled = np.random.choice(no_of_images, no_of_images, replace =False)

#Split the indices into train, validation and test data
x_train_size1 = int(no_of_images *0.7)
x_train_size2 = int(no_of_images *0.6)

x_val_size1 = int(no_of_images*0.9)
x_val_size2 = int(no_of_images*0.8)


x_train, x_val, x_test = np.split(indices_shuffled, [x_train_size1, x_val_size1])

i=0
for index in x_train:
	
	image_folder = os.listdir(os.path.join(filepath, file_list[index]))
	label_folder = os.listdir(os.path.join(labelpath, file_list[index]))
	# print(len(image_folder))
	imgdatas = np.ndarray((len(image_folder)*len(x_train), rows,cols,3), dtype=np.uint8)
	imglabels = np.ndarray((len(label_folder)*len(x_train),rows,cols,3), dtype=np.uint8)
	for (images, labels) in zip(image_folder, label_folder):
		# print(images[images.rindex("/")+1:])
		# midname = images[images.rindex("/")+1:]
		img = load_img(os.path.join(filepath, file_list[index])+'/'+ images, grayscale = False)
		label = load_img(os.path.join(filepath, file_list[index])+'/'+ labels, grayscale = False)
		img = img_to_array(img)
		label = img_to_array(label)		
		# print(img.shape)

		imgdatas[i] = img
		imglabels[i] = label
		if i % 100 == 0:
			print('Done: {0}/{1} images'.format(i, len(image_folder)*len(x_train)))
		i += 1
print('loading training data done')
np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data' + '/imgs_train702010.npy', imgdatas)
np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data' + '/imgs_mask_train702010.npy', imglabels)
print('Saving to .npy files done.')

i = 0
for index in x_val:
	
	image_folder = os.listdir(os.path.join(filepath, file_list[index]))
	label_folder = os.listdir(os.path.join(labelpath, file_list[index]))
	print(len(image_folder))
	print(image_folder[3])
	imgdatas = np.ndarray((len(image_folder)*len(x_val), rows,cols,3), dtype=np.uint8)
	imglabels = np.ndarray((len(label_folder)*len(x_val),rows,cols,3), dtype=np.uint8)
	for (images, labels) in zip(image_folder, label_folder):
		img = load_img(os.path.join(filepath, file_list[index])+'/'+ images, grayscale = False)
		label = load_img(os.path.join(filepath, file_list[index])+'/'+ labels, grayscale = False)
		img = img_to_array(img)
		label = img_to_array(label)		
		# print(img.shape)

		imgdatas[i] = img
		imglabels[i] = label
		# print(len(imgdatas))
		if i % 100 == 0:
			print('Done: {0}/{1} images'.format(i, len(image_folder)*len(x_val)))
		i += 1
print('loading validation data done')

np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data' + '/imgs_val702010.npy', imgdatas)
np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data' + '/imgs_mask_val702010.npy', imglabels)
print('Saving to .npy files done.')

i=0
for index in x_test:
	
	image_folder = os.listdir(os.path.join(filepath, file_list[index]))
	label_folder = os.listdir(os.path.join(labelpath, file_list[index]))
	# print(len(image_folder))
	imgdatas = np.ndarray((len(image_folder)*len(x_test), rows,cols,3), dtype=np.uint8)
	imglabels = np.ndarray((len(label_folder)*len(x_test),rows,cols,3), dtype=np.uint8)
	for (images, labels) in zip(image_folder, label_folder):
		img = load_img(os.path.join(filepath, file_list[index])+'/'+ images, grayscale = False)
		label = load_img(os.path.join(filepath, file_list[index])+'/'+ labels, grayscale = False)
		img = img_to_array(img)
		label = img_to_array(label)		
		# print(img.shape)

		imgdatas[i] = img
		imglabels[i] = label
		if i % 100 == 0:
			print('Done: {0}/{1} images'.format(i, len(image_folder)*len(x_test)))
		i += 1
print('loading testing data done')
np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data' + '/imgs_test702010.npy', imgdatas)
np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data' + '/imgs_mask_test702010.npy', imglabels)
print('Saving to .npy files done.')
# print(x_train)
# print(x_val_size)
    

# print(len(file_list))
# print(file_list[0])