from __future__ import absolute_import
import keras.backend as K
import numpy as np
import random
import os
import cv2
import math
from keras.preprocessing.image import *
from pathlib import Path
import glob
# from keras.preprocessing.image import ImageDataGenerator

# import Augmentor
"""


"""
def rotatedRectWithMaxArea(w, h, angle):
	"""
	Credit : https://stackoverflow.com/a/16778797

	Given a rectangle of size wxh that has been rotated by 'angle' (in
	radians), computes the width and height of the largest possible
	axis-aligned rectangle (maximal area) within the rotated rectangle.
	"""
	if w <= 0 or h <= 0:
		return 0,0

	width_is_longer = w >= h
	side_long, side_short = (w,h) if width_is_longer else (h,w)

	# since the solutions for angle, -angle and 180-angle are all the same,
	# if suffices to look at the first quadrant and the absolute values of sin,cos:
	sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
	if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
		# half constrained case: two crop corners touch the longer side,
		#   the other two corners are on the mid-line parallel to the longer line
		x = 0.5*side_short
		wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
	else:
		# fully constrained case: crop touches all 4 sides
		cos_2a = cos_a*cos_a - sin_a*sin_a
		wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

	return wr,hr

def rotate_bound(image, angle):
	# CREDIT: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	return cv2.warpAffine(image, M, (nW, nH))


def rotate_max_area(image, angle):
	""" image: cv2 image matrix object
	angle: in degree
	"""
	wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
	rotated = rotate_bound(image, angle)
	h, w, _ = rotated.shape
	y1 = h//2 - int(hr/2)
	y2 = y1 + int(hr)
	x1 = w//2 - int(wr/2)
	x2 = x1 + int(wr)
	return rotated[y1:y2, x1:x2]

def add_salt_pepper_noise(image):
	# Source: https://github.com/Prasad9/ImageAugmentationTypes/blob/master/ImageAugmentation.ipynb
	# Need to produce a copy as to not modify the original image
	image_copy = image.copy()
	row, col, _ = image_copy.shape
	salt_vs_pepper = 0.2
	amount = 0.004
	num_salt = np.ceil(amount * image_copy.size * salt_vs_pepper)
	num_pepper = np.ceil(amount * image_copy.size * (1.0 - salt_vs_pepper))

	random.seed(a=100)
	# Add Salt noise
	coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_copy.shape]
	image_copy[coords[0], coords[1], :] = 1

	# Add Pepper noise
	coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_copy.shape]
	image_copy[coords[0], coords[1], :] = 0
	return image_copy

def add_gaussian_noise(image):
	# Source: https://github.com/Prasad9/ImageAugmentationTypes/blob/master/ImageAugmentation.ipynb
	# To vary lighting conditions
	# gaussian_noise_imgs = []
	row, col, _ = image.shape
	# Gaussian distribution parameters
	mean = 0
	var = 0.1
	sigma = var ** 0.5

	random.seed(a=100)

	gaussian = np.random.random((row, col, 1)).astype(np.float32)
	gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
	# gaussian = array_to_img(gaussian)
	image = img_to_array(image)
	gaussian_img = cv2.addWeighted(image, 0.75,  0.25*gaussian, 0.25, 0)
	# gaussian_noise_imgs.append(gaussian_img)
	gaussian_img = np.array(gaussian_img, dtype = np.float32)
	return gaussian_img



def get_mask_coord(imshape):
	# Source: https://github.com/Prasad9/ImageAugmentationTypes/blob/master/ImageAugmentation.ipynb
	vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]), 
						(0.43 * imshape[1], 0.32 * imshape[0]), 
						(0.56 * imshape[1], 0.32 * imshape[0]),
						(0.85 * imshape[1], 0.99 * imshape[0])]], dtype = np.int32)
	return vertices

def get_perspective_matrices(X_img):
	# Source: https://github.com/Prasad9/ImageAugmentationTypes/blob/master/ImageAugmentation.ipynb

	offset = 15
	img_size = (X_img.shape[1], X_img.shape[0])

	# Estimate the coordinates of object of interest inside the image.
	src = np.float32(get_mask_coord(X_img.shape))
	dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0], 
	                  [img_size[0] - offset, img_size[1]]])

	perspective_matrix = cv2.getPerspectiveTransform(src, dst)
	return perspective_matrix

def perspective_transform(X_img):
	# Source: https://github.com/Prasad9/ImageAugmentationTypes/blob/master/ImageAugmentation.ipynb
	# Doing only for one type of example
	perspective_matrix = get_perspective_matrices(X_img)
	warped_img = cv2.warpPerspective(X_img, perspective_matrix,
	                                 (X_img.shape[1], X_img.shape[0]),
	                                 flags = cv2.INTER_LINEAR)
	return warped_img

def random_erasing(image, rectangle_area):
	"""
	Params: 
	image: The image to add a random noise rectangle to.
	rectangle_area: The percentage are of the image to occlude, between 0.1 and 1.
	"""
	'''Source: https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py

	Performs Random Erasing, an augmentation technique described
	in `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
	by Zhong et al. To quote the authors, random erasing:
	"*... randomly selects a rectangle region in an image, and erases its
	pixels with random values.*"
	Random Erasing can make a trained neural network more robust to occlusion.

	Adds a random noise rectangle to a random area of the passed image,
	returning the original image with this rectangle superimposed.
    '''
	w, h, channels = image.shape
	
	w_occlusion_max = int(w * rectangle_area)
	h_occlusion_max = int(h * rectangle_area)

	w_occlusion_min = int(w * 0.1)
	h_occlusion_min = int(h * 0.1)

	random.seed(a=100)
	w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
	random.seed(a=200)
	h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

	if channels == 1:
		
		rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
	else:
		
		rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, channels) * 255))

	random.seed(a=300)
	random_position_x = random.randint(0, w - w_occlusion)
	random.seed(a=400)
	random_position_y = random.randint(0, h - h_occlusion)

	image.paste(rectangle, (random_position_x, random_position_y))

	return image_occluded


# def shift(image):
# 	height, width, channels = img.shape

# 	#Create border within which the image is to be shifted. This new image taken by excluding the border is smaller than the original image. 
# 	# This lets us shift the image without adding false pixels to fill up the shifted space
# 	# p.s.: Making it propotional to the respective dimensions
# 	border_height = int(height/100)
# 	border_width = int(width/100)

# 	# Set the shifting increments
# 	shift_width = 0:int(border_width/10):border_width
# 	shift_heigth = 0: int(border_height/10):border_height

# 	# The smaller image(original excluding borders) dimensions
# 	new_height = height - 2*border_height
# 	new_width = width - 2*border_width

# 	for shift_horizontal in shift_width:
# 		for shift_vertical in shift_heigth:

# 			#shifting up and right
# 			y1 = border_height +shift_vertical
# 			y2 = y1 + new_height

# 			x1 = border_width +shift_horizontal
# 			x2 = x1+ new_width

# 			img_shifted_positive = img[y1:y2, x1:x2, :]

# 			#shifting down and left
# 			y1 = border_height -shift_vertical
# 			y2 = y1 + new_height

# 			x1 = border_width -shift_horizontal
# 			x2 = x1+ new_width

# 			img_shifted_negative = img[y1:y2, x1:x2, :]

##-------

# img = cv2.imread('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Labelled Photos/newIMG_1394.png')

# angle = 60
# rows,cols, channels = img.shape

# img2 = rotate_max_area(img, angle)

# M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
# rotated_image = cv2.warpAffine(img, M, (cols, rows))

# cv2.imshow('original image', img)
# cv2.imshow('biggest rotated image', img2)
# cv2.imshow('basic rotated image', rotated_image)

# fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_original.png'
# cv2.imwrite(fullnewfilename, img)

# fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_biggest_rotated.png'
# cv2.imwrite(fullnewfilename, img2)

# fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_rotated.png'
# cv2.imwrite(fullnewfilename, rotated_image)

# resized_image = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST) 



## Flip Image ( Horizontal flip, Vertical Flip)

# data_gen_args = dict(horizontal_flip=True, vertical_flip=True)

# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)


# for images in image_datagen.flow_from_directory(image_directory, target_size=(row_size, col_size), batch_size=2 , seed = seed, save_to_dir = aug_image_directory):
# 	for i in range(0, batch_size):
# 		aug_img = images[i]
# 		fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_rotated.png'
# 		cv2.imwrite(fullnewfilename, aug_img)
							   

# # Add noise to the image
# salt_pepper_noise_imgs = add_salt_pepper_noise(X_imgs)
# # Change lighting condition
# gaussian_noise_imgs = add_gaussian_noise(X_imgs)
# Perspective transform
# perspective_img = perspective_transform(X_img) # need to know the position of object 

# Shear image
# random_shear(image, shear_angle) # have to add artificial pixel to points outside boundary
#https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py # Check here for a better implementation of shear
 
##----------

## Main 

#Target size
row_size = 200
col_size = 200
seed = 100
image_directory = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Labelled Photos'
labels_directory = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Three Class Labels'

max_angle = 180
incr_angle = (max_angle-0)/50
angles = np.arange(0, max_angle, incr_angle)

images = glob.glob(image_directory+"/*.png")
if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images'):
	os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images')

labels = glob.glob(labels_directory+"/*.png")
if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels'):
	os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels')

for img in images:

	base = os.path.basename(img)
	newfilename = os.path.splitext(base)[0]

	original_img = cv2.imread(img)
	nrows, ncols, nchannels = original_img.shape

	#Make a new for each image to store all its augmented images
	folder_path = os.path.join('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/', newfilename)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	#Rotation
	print('Rotating Image')
	iAngle = 0
	for angle in angles:
		rotated_image = rotate_max_area(original_img, angle)
		resized_rotated_image = cv2.resize(rotated_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
		iAngle+=1

		fullnewfilename = folder_path +'/'+newfilename +'_'+'rotated'+str(iAngle)+ ".png"
		cv2.imwrite(fullnewfilename, resized_rotated_image)

	# Flip Image ( Horizontal flip, Vertical Flip)
	#Horizontal
	hflipped_image = flip_axis(original_img, 1)
	resized_hflipped_image = cv2.resize(hflipped_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'hflipped.png'
	cv2.imwrite(fullnewfilename, resized_hflipped_image)

	#Vertical
	vflipped_image = flip_axis(original_img, 0)
	resized_vflipped_image = cv2.resize(vflipped_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'vflipped.png'
	cv2.imwrite(fullnewfilename, resized_vflipped_image)

	# Add noise to the image
	salt_pepper_noise_image = add_salt_pepper_noise(original_img)
	resized_spnoise_image = cv2.resize(salt_pepper_noise_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'spnoise.png'
	cv2.imwrite(fullnewfilename, resized_spnoise_image)

	# Change lighting condition
	gaussian_noise_image = add_gaussian_noise(original_img)
	resized_gnoise_image = cv2.resize(gaussian_noise_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'gnoise.png'
	cv2.imwrite(fullnewfilename, resized_gnoise_image)

	# #Random Erasing
	# occlusion_area = np.arange(0.1,0.6,0.1)

	# iErase = 0
	# random.seed(a=100)
	# for rectangle_area in occlusion_area:
	# 	ran_erased_image = random_erasing(img, rectangle_area)
	# 	resized_erased_image = cv2.resize(ran_erased_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	# 	iErase+=1
	# 	fullnewfilename = folder_path +'/'+newfilename +'_'+'erased'+str(iErase)+ ".png"
	# 	cv2.imwrite(fullnewfilename, resized_erased_image)


	#Image shift/translation
	height, width, channels = original_img.shape

	#Create border within which the image is to be shifted. This new image taken by excluding the border is smaller than the original image. 
	# This lets us shift the image without adding false pixels to fill up the shifted space
	# p.s.: Making it propotional to the respective dimensions
	border_height = int(height/10)
	border_width = int(width/10)

	# Set the shifting increments
	shift_width = np.arange(0, border_width, int(border_width/10))
	shift_heigth = np.arange(0, border_height, int(border_height/10))

	# The smaller image(original excluding borders) dimensions
	new_height = height - 2*border_height
	new_width = width - 2*border_width

	iShift = 0
	for shift_horizontal in shift_width:
		for shift_vertical in shift_heigth:

			#shifting up and right
			y1 = border_height +shift_vertical
			y2 = y1 + new_height

			x1 = border_width +shift_horizontal
			x2 = x1+ new_width

			img_shifted_positive = original_img[y1:y2, x1:x2, :]

			iShift+=1
			resized_pos_shifted_image = cv2.resize(img_shifted_positive, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
			fullnewfilename = folder_path +'/'+newfilename +'_'+'shifted'+str(iShift)+ ".png"
			cv2.imwrite(fullnewfilename, resized_pos_shifted_image)

			#shifting down and left
			y1 = border_height -shift_vertical
			y2 = y1 + new_height

			x1 = border_width -shift_horizontal
			x2 = x1+ new_width

			img_shifted_negative = original_img[y1:y2, x1:x2, :]

			iShift+=1
			resized_neg_shifted_image = cv2.resize(img_shifted_negative, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
			fullnewfilename = folder_path +'/'+newfilename +'_'+'shifted'+str(iShift)+ ".png"
			cv2.imwrite(fullnewfilename, resized_neg_shifted_image)


# Augmented image labels
for img in labels:

	base = os.path.basename(img)
	newfilename = os.path.splitext(base)[0]

	original_img = cv2.imread(img)
	nrows, ncols, nchannels = original_img.shape

	#Make a new for each image to store all its augmented images
	folder_path = os.path.join('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels/', newfilename)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	#Rotation
	print('Rotating Image')
	iAngle = 0
	for angle in angles:
		rotated_image = rotate_max_area(original_img, angle)
		resized_rotated_image = cv2.resize(rotated_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
		iAngle+=1

		fullnewfilename = folder_path +'/'+newfilename +'_'+'rotated'+str(iAngle)+ ".png"
		cv2.imwrite(fullnewfilename, resized_rotated_image)

	# Flip Image ( Horizontal flip, Vertical Flip)
	#Horizontal
	hflipped_image = flip_axis(original_img, 1)
	resized_hflipped_image = cv2.resize(hflipped_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'hflipped.png'
	cv2.imwrite(fullnewfilename, resized_hflipped_image)

	#Vertical
	vflipped_image = flip_axis(original_img, 0)
	resized_vflipped_image = cv2.resize(vflipped_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'vflipped.png'
	cv2.imwrite(fullnewfilename, resized_vflipped_image)

	# Add noise to the image
	# salt_pepper_noise_image = add_salt_pepper_noise(img)
	resized_spnoise_image = cv2.resize(original_img, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'spnoise.png'
	cv2.imwrite(fullnewfilename, resized_spnoise_image)

	# Change lighting condition
	# gaussian_noise_image = add_gaussian_noise(image)
	resized_gnoise_image = cv2.resize(original_img, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	fullnewfilename = folder_path +'/'+newfilename +'_'+'gnoise.png'
	cv2.imwrite(fullnewfilename, resized_gnoise_image)

	# #Random Erasing
	# occlusion_area = np.arange(0.1,0.6,0.1)

	# iErase = 0
	# random.seed(a=100)
	# for rectangle_area in occlusion_area:
	# 	ran_erased_image = random_erasing(img, rectangle_area)
	# 	resized_erased_image = cv2.resize(ran_erased_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
	# 	iErase+=1
	# 	fullnewfilename = folder_path +'/'+newfilename +'_'+'erased'+str(iErase)+ ".png"
	# 	cv2.imwrite(fullnewfilename, resized_erased_image)


	#Image shift/translation
	height, width, channels = original_img.shape

	#Create border within which the image is to be shifted. This new image taken by excluding the border is smaller than the original image. 
	# This lets us shift the image without adding false pixels to fill up the shifted space
	# p.s.: Making it propotional to the respective dimensions
	border_height = int(height/10)
	border_width = int(width/10)

	# Set the shifting increments
	shift_width = np.arange(0, border_width, int(border_width/10))
	shift_heigth = np.arange(0, border_height, int(border_height/10))

	# The smaller image(original excluding borders) dimensions
	new_height = height - 2*border_height
	new_width = width - 2*border_width

	iShift = 0
	for shift_horizontal in shift_width:
		for shift_vertical in shift_heigth:

			#shifting up and right
			y1 = border_height +shift_vertical
			y2 = y1 + new_height

			x1 = border_width +shift_horizontal
			x2 = x1+ new_width

			img_shifted_positive = original_img[y1:y2, x1:x2, :]

			iShift+=1
			resized_pos_shifted_image = cv2.resize(img_shifted_positive, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
			fullnewfilename = folder_path +'/'+newfilename +'_'+'shifted'+str(iShift)+ ".png"
			cv2.imwrite(fullnewfilename, resized_pos_shifted_image)

			#shifting down and left
			y1 = border_height -shift_vertical
			y2 = y1 + new_height

			x1 = border_width -shift_horizontal
			x2 = x1+ new_width

			img_shifted_negative = original_img[y1:y2, x1:x2, :]

			iShift+=1
			resized_neg_shifted_image = cv2.resize(img_shifted_negative, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
			fullnewfilename = folder_path +'/'+newfilename +'_'+'shifted'+str(iShift)+ ".png"
			cv2.imwrite(fullnewfilename, resized_neg_shifted_image)



