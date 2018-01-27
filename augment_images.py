from __future__ import absolute_import
import keras.backend as K
import numpy as np
import os
import cv2
import math
from keras.preprocessing.image import *
import Augmentor
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

def add_salt_pepper_noise(X_imgs):
	# Source: https://github.com/Prasad9/ImageAugmentationTypes/blob/master/ImageAugmentation.ipynb
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy

def add_gaussian_noise(X_imgs):
	# Source: https://github.com/Prasad9/ImageAugmentationTypes/blob/master/ImageAugmentation.ipynb
	# To vary lighting conditions
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs



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

	w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
	h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

	if channels == 1:
		rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
	else:
		rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, channels) * 255))

	random_position_x = random.randint(0, w - w_occlusion)
	random_position_y = random.randint(0, h - h_occlusion)

	image.paste(rectangle, (random_position_x, random_position_y))

	return image_occluded





# img = cv2.imread('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Labelled Photos/newIMG_1394.png')

# angle = 60
# rows,cols, channels = img.shape

img2 = rotate_max_area(img, angle)

# M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
# rotated_image = cv2.warpAffine(img, M, (cols, rows))

# cv2.imshow('original image', img)
# cv2.imshow('biggest rotated image', img2)
# cv2.imshow('basic rotated image', rotated_image)

# fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_original.png'
# cv2.imwrite(fullnewfilename, img)

# fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_biggest_rotated.png'
# cv2.imwrite(fullnewfilename, img2)

fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_rotated.png'
cv2.imwrite(fullnewfilename, rotated_image)

# resized_image = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST) 



# Flip Image ( Horizontal flip, Vertical Flip)

data_gen_args = dict(horizontal_flip=True, vertical_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


for images in image_datagen.flow_from_directory(image_directory, target_size=(row_size, col_size), batch_size=2 , seed = seed, save_to_dir = aug_image_directory):
	for i in range(0, batch_size):
		aug_img = images[i]
		fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/"+'newIMG_1394_rotated.png'
		cv2.imwrite(fullnewfilename, aug_img)
							   

# Add noise to the image
salt_pepper_noise_imgs = add_salt_pepper_noise(X_imgs)
# Change lighting condition
gaussian_noise_imgs = add_gaussian_noise(X_imgs)
# Perspective transform
# perspective_img = perspective_transform(X_img) # need to know the position of object 

# Shear image
# random_shear(image, shear_angle) # have to add artificial pixel to points outside boundary
#https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py # Check here for a better implementation of shear


## Main 

#Target size
row_size = 200
col_size = 200
seed = 1
image_directory = 
labels_directory = 

max_angle = 180
incr_angle = (max_angle-0)/50
angles = np.arange(0, max_angle, incr_angle)

images = glob.glob(image_directory+"/*."+png)
os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images')

labels = glob.glob(labels_directory+"/*."+png)
os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels')

for img in images:

	base = os.path.basename(img)
	newfilename = os.path.splitext(base)[0]

	original_img = cv2.imread(img)
	nrows, ncols, nchannels = img.shape

	#Make a new directory to store all the augmented images
	folder_path = os.path.join('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/', str(i)+'.jpg')
	if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/'):
		os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/')

	#Rotation
	print('Rotating Image')
	iAngle = 0
	for angle in angles:
		rotated_image = rotate_max_area(original_img, angle)
		resized_rotated_image = cv2.resize(rotated_image, (row_size, col_size), interpolation=cv2.INTER_NEAREST)
		i+=1

		fullnewfilename = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Resized Three Class Translated/"+newfilename +'_'+'rotated'+str(i)+ ".png"
		cv2.imwrite(fullnewfilename, resized_rotated_image)


# change folder names and filenames and paths
