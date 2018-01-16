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

#glob_file_key contains the path to the directory containing the resizeable data(image/label)
glob_file_key = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Labelled Photos/*.png"
for f in enumerate(glob(glob_file_key)):
	# if f.endswith(".png"):
	name = f[1]
	base = os.path.basename(name)
	newfilename = os.path.splitext(base)[0]
	# print(newfilename)
	img = cv2.imread(name)

	#Resize the image to 200 by 200
	image_size =200
	resized_image = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST) 
	rows,cols, channels = resized_image.shape

	# cv2.imwrite(os.path.join('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Images 2/' ,name), resized_image)

	
	max_angle = 180
	incr_angle = (max_angle-0)/50
	# print(incr_angle)
	max_offset=20
	x_incr = 10
	y_incr = 10

	#Translating the image

	print('Translating the image')
	i = 0
	for x_offset in range(0, int(image_size/2), x_incr):
		for y_offset in range(0, int(image_size/2), y_incr):

			M = np.float32([[1, 0, x_offset],[0, 1, y_offset]])
			translated_image = cv2.warpAffine(resized_image, M, (cols, rows))
			i+=1
			fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Translated/"+newfilename +'_'+'translated'+str(i)+ ".png"
			cv2.imwrite(fullnewfilename, translated_image)

			M = np.float32([[1, 0, -x_offset],[0, 1, -y_offset]])
			translated_image = cv2.warpAffine(resized_image, M, (cols, rows))
			i+=1
			fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Translated/"+newfilename +'_'+'translated'+str(i)+ ".png"
			cv2.imwrite(fullnewfilename, translated_image)

	#Rotating the image
	print('Rotating the image')
	angles = np.arange(0, max_angle, incr_angle)
	i = 0 
	for ang in angles:
		M = cv2.getRotationMatrix2D((cols/2, rows/2), ang, 1)
		rotated_image = cv2.warpAffine(resized_image, M, (cols, rows))
		i+=1
		fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Rotated/"+newfilename +'_'+'rotated'+str(i)+ ".png"
		cv2.imwrite(fullnewfilename, rotated_image)

		M = cv2.getRotationMatrix2D((cols/2, rows/2), -ang, 1)
		rotated_image = cv2.warpAffine(resized_image, M, (cols, rows))
		i+=1
		fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Rotated/"+newfilename +'_'+'rotated'+str(i)+ ".png"
		cv2.imwrite(fullnewfilename, rotated_image)


	## Warping images 
	print('Distorting the image')
	# Source: OpenCV with Python By Example

	# Vertical wave

	vertical_wave_image = np.zeros(resized_image.shape, dtype=resized_image.dtype)

	for i in range(rows):
		for j in range(cols):
			offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
			offset_y = 0
			if j+offset_x < rows:
				vertical_wave_image[i,j] = resized_image[i,(j+offset_x)%cols]
			else:
				vertical_wave_image[i,j] = 0

	fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Distorted/"+newfilename +'_'+'vertical_wave'+ ".png"
	cv2.imwrite(fullnewfilename, vertical_wave_image)

	# Horizontal wave

	horizontal_wave_image = np.zeros(resized_image.shape, dtype=resized_image.dtype)

	for i in range(rows):
		for j in range(cols):
			offset_y = int(16.0 * math.sin(2 * 3.14 * i / 150))
			offset_x = 0
			if i+offset_y < rows:
				horizontal_wave_image[i,j] = resized_image[(i+offset_y)%rows, j]
			else:
				horizontal_wave_image[i,j] = 0

	fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Distorted/"+newfilename +'_'+'horizontal_wave'+ ".png"
	cv2.imwrite(fullnewfilename, horizontal_wave_image)
	
	# Vertical+horizontal wave

	verhor_wave_image = np.zeros(resized_image.shape, dtype=resized_image.dtype)

	for i in range(rows):
		for j in range(cols):
			offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
			offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
			if i+offset_y < rows and j+offset_x < cols:
				verhor_wave_image[i,j] = resized_image[(i+offset_y)%rows, (j+offset_x)%cols]
			else:
				verhor_wave_image[i,j] = 0

	fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Distorted/"+newfilename +'_'+'verhor_wave'+ ".png"
	cv2.imwrite(fullnewfilename, verhor_wave_image)

	# Concave effect

	concave_image = np.zeros(resized_image.shape, dtype=resized_image.dtype)

	for i in range(rows):
		for j in range(cols):
			offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2*cols)))
			offset_y = 0
			if j+offset_x < rows:
				concave_image[i,j] = resized_image[i,(j+offset_x)%cols]
			else:
				concave_image[i,j] = 0

	fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Labelled Distorted/"+newfilename +'_'+'concave'+ ".png"
	cv2.imwrite(fullnewfilename, concave_image)
