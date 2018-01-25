from __future__ import absolute_import
import keras.backend as K
import numpy as np
import os

# not using this; added the necessary lines(31-42) in main unet.py file in save_imgs

def combine_images():
	print("array to image")
	imgs_mask = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/imgs_mask_test.npy')
	imgs_test = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/imgs_test.npy')

	if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results'):
		os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results')
	if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_test_images'):
		os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_test_images')			
	if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_combined'):
		os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_combined')


	for i in range(imgs_mask.shape[0]):
		print(i)
		img = imgs_mask[i]
		# print(img.size)
		img = array_to_img(img)
		img.save("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results/%d.jpg"%(i))
		img = imgs_test[i]	
		img = array_to_img(img)
		img.save("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_test_images/%d.jpg"%(i))

		# add translucent label(img1) to original image(img2)
		filepath1 = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results/", str(i)+'.jpg')
		img1 = cv2.imread(filepath1)
		img4 = img1[:,:,1]

		filepath2 = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_test_images/", str(i)+'.jpg')
		img2 = cv2.imread(filepath2)

		img3 = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)

		filepath3 = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_combined/", str(i)+'.jpg')
		cv2.imwrite(filepath3, img3)

	