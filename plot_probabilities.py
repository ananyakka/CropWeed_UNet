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
npypath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/trial20/'
predicted_mask_path = os.path.join(npypath, 'imgs_predicted_mask_test.npy')
true_mask_path = os.path.join(npypath, 'imgs_true_mask_test.npy')

big_label_array = np.load(predicted_mask_path)
big_true_label_array = np.load(true_mask_path)
print('Finished loading data')

bins = np.arange(0,1,0.1)
predicted_label_histogram = np.zeros((big_label_array.shape[0],big_label_array.shape[-1], (len(bins)-1)))
# print(predicted_label_histogram.shape)
true_label_histogram = np.zeros((big_true_label_array.shape[0],big_true_label_array.shape[-1], (len(bins)-1)))

# load one image, reshape into 40000 vector, plot histogram of each channel; press key to display next image; collect histogram values for all images in bins
fig = plt.figure()
plt.style.use('fivethirtyeight')


if big_label_array.shape[0] == big_true_label_array.shape[0]:

	if big_label_array.ndim == 4:
		no_images = big_label_array.shape[0]
		no_channels = big_label_array.shape[-1]

		for iImage in range(no_images):
			label_array = big_label_array[iImage]
			true_label_array = big_true_label_array[iImage]

			for iClass in range(no_channels):
				fig.clear()
				label_array_by_class = label_array[...,iClass]
				print(label_array_by_class)
				true_label_array_by_class = true_label_array[ ...,iClass]
				print(true_label_array_by_class)
				print('Channel')
				print(iClass)
				label_vector = label_array_by_class.flatten()
				true_label_vector = true_label_array_by_class.flatten()

				data = np.vstack([label_vector, true_label_vector]).T
				print(data.shape)
				# hist = np.histogram(label_vector, bins)[0]
				# print(hist.shape)
				predicted_label_histogram[iImage,iClass]= np.histogram(label_vector, bins)[0]
				true_label_histogram[iImage,iClass]= np.histogram(true_label_vector, bins)[0]


				plt.hist(data, bins, label=['predicted', 'true'])
				plt.legend(loc='upper right')
				# plt.hist(label_vector, bins)
				plt.xlabel('Probability')
				plt.ylabel('Number of Pixels')
				plt.title('Probability Distribution')
				plt.grid(True)
				plt.waitforbuttonpress()
else:
	print('Unequal number of actual labels and predicted labels')

predicted_prob_path =  os.path.join(npy_path, 'imgs_predicted_mask_test_probab.npy')
np.save(predicted_prob_path, predicted_label_histogram)
true_prob_path =  os.path.join(npy_path, 'imgs_true_mask_test_probab.npy')
np.save(true_prob_path, true_label_histogram)