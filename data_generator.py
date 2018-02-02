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




def my_generator(filepath, labelpath, x_set_indices, batch_size):

	rows =200
	cols = 200
	file_list = list(os.listdir(filepath))

	# Infinite loop
	while 1: 

		for index in x_set_indices:
			
			image_folder = os.listdir(os.path.join(filepath, file_list[index]))
			label_folder = os.listdir(os.path.join(labelpath, file_list[index]))
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
					label = load_img(os.path.join(filepath, file_list[index])+'/'+ label_folder[iImage], grayscale = False)
					img = img_to_array(img)
					label = img_to_array(label)

					imgdata[i] = img
					imglabels[i] = label
					i+=1

					yield imgdata, imglabels

					batch_start += batch_size
					batch_end += batch_size

