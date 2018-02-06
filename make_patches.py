'''
Program to take an image and label, and make non overlapping patches of size 200 by 200
By Ananya, 5 Feb 2018
'''
from __future__ import absolute_import
import numpy as np
import random
import os
import cv2
import math
from pathlib import Path
import glob


if __name__ == '__main__':

	# Hard code some values for patch size
	patch_height = int(200)
	patch_width = int(200)

	seed = 100 # Set seed in case of randomization

	main_directory = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/'

	image_directory = os.path.join(main_directory, 'Labelled Photos')
	labels_directory = os.path.join(main_directory, 'Three Class Labels')

	images = glob.glob(image_directory+"/*.png")
	patch_image_folder = os.path.join(main_directory, 'Train_image_Patches')
	if not os.path.exists(patch_image_folder):
		os.makedirs(patch_image_folder)

	labels = glob.glob(labels_directory+"/*.png")
	patch_label_folder = os.path.join(main_directory, 'Train_label_patches')
	if not os.path.exists(patch_label_folder):
		os.makedirs(patch_label_folder)

	for img in images: # ? to zip or not to zip; no for now - to deal with name mismatch 

		print('Making image patches')

		base = os.path.basename(img)
		newfilename = os.path.splitext(base)[0]

		original_img = cv2.imread(img)
		nrows, ncols, nchannels = original_img.shape # Size of original image

		#Make a new folder for each image to store all its patches
		folder_path = os.path.join(patch_image_folder, newfilename)
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)

		# Number of patches possible along rows and columns
		row_increments = int(nrows/patch_height)
		# print(row_increments)
		col_increments = int(ncols/patch_width)
		# print(col_increments)

		iImage = 0

		for y_incr in range(row_increments):
			for x_incr in range(col_increments):

				row_start = y_incr*patch_height
				# print(row_start)
				row_end = row_start + patch_height
				# print(row_end)

				col_start = x_incr*patch_width
				# print(col_start)
				col_end = col_start + patch_width
				# print(col_end)

				if row_end > nrows: 
					# If row number of patch height end is greater than image height, make image height the patch end 
					row_end = nrows
					row_start = row_end - patch_height

				if col_end > ncols:
					# If row number of patch width end is greater than image width, make image width the patch end
					col_end = ncols
					col_start = col_end - patch_width 			

				img_patch = original_img[row_start:row_end, col_start:col_end, :] # copy only the necessary rows to make patch

				iImage+= 1
				fullnewfilename = folder_path +'/'+newfilename +'_'+'patch'+str(iImage)+ ".png"
				cv2.imwrite(fullnewfilename, img_patch)

	for img in labels:
		print('Making label patches')

		base = os.path.basename(img)
		newfilename = os.path.splitext(base)[0]

		original_img = cv2.imread(img)
		nrows, ncols, nchannels = original_img.shape # Size of original image

		#Make a new folder for each image to store all its patches
		folder_path = os.path.join(patch_label_folder, newfilename)
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)

		# Number of patches possible along rows and columns
		row_increments = int(nrows/patch_height)
		col_increments = int(ncols/patch_width)

		iImage = 0

		for y_incr in range(row_increments):
			for x_incr in range(col_increments):

				row_start = y_incr*patch_height
				row_end = row_start + patch_height

				col_start = x_incr*patch_width
				col_end = col_start + patch_width

				if row_end > nrows: 
					# If row number of patch height end is greater than image height, make image height the patch end 
					row_end = nrows
					row_start = row_end - patch_height

				if col_end > ncols:
					# If row number of patch width end is greater than image width, make image width the patch end
					col_end = ncols
					col_start = col_end - patch_width 			

				img_patch = original_img[row_start:row_end, col_start:col_end, :] # copy only the necessary rows to make patch

				iImage+= 1
				fullnewfilename = folder_path +'/'+newfilename +'_'+'patch'+str(iImage)+ ".png"
				cv2.imwrite(fullnewfilename, img_patch)
