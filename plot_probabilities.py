'''
Plot the histogram of probability values from an image prediction
By Ananya Feb 26, 2018
'''

import numpy as np
import os
import matplotlib 
import matplotlib.pyplot as plt 
plt.ion()

# load numpy array
npypath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/trial15/'
filepath = os.path.join(npypath, 'imgs_predicted_mask_test.npy')
filepath2 = os.path.join(npypath, 'imgs_true_mask_test.npy')

big_label_array = np.load(filepath)
big_true_label_array = np.load(filepath2)

# load one image, reshape into 40000 vector, plot histogram of each channel; press key to display next image; collect histogram values for all images in bins
fig = plt.figure()

if big_label_array.ndim == 4:
	no_images = big_label_array.shape[0]
	no_channels = big_label_array.shape[-1]

	for iImage in range(no_images):
		label_array = big_label_array[iImage]
		for iClass in range(no_channels):
			fig.clear()
			label_array_by_class = label_array[...,iClass]
			print('Channel')
			print(iClass)
			label_vector = label_array_by_class.flatten()


			plt.hist(label_vector, bins = np.arange(0,1,0.1))

			plt.xlabel('Probability')
			plt.ylabel('Number of Pixels')
			plt.title('Probability Distribution')
			plt.grid(True)
			plt.waitforbuttonpress()