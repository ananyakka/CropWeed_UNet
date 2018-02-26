import numpy as np
import os
import cv2
rows =200
cols = 200

filepath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/'
labelpath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels/'
x_set_indices= [ 53, 1,  49,  80, 205,  34, 263,  91,  52, 264, 241,  13,  88, 273, 166,  20, 134, 306,
 130, 243,  54,  50, 174, 189, 270, 187, 169,  58,  48, 235, 252,  21, 313]

file_list = list(os.listdir(filepath))
label_list = list(os.listdir(labelpath))
batch_size = 1

actual_label_array_big = np.empty([0, rows,cols,3])

npy_path = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/'


for index in x_set_indices:

	image_folder = os.listdir(os.path.join(filepath, file_list[index]))
	label_folder = os.listdir(os.path.join(labelpath, label_list[index]))
	# print(len(image_folder))
	nBatch_in_folder = int(len(image_folder)/batch_size)

	batch_start = 0
	batch_end = batch_size
	folder_total = len(image_folder)
	actual_label_array = np.ndarray((len(image_folder), rows,cols,3), dtype=np.float32)

	j=0 
	while batch_end< folder_total: #always ignores the last few in each; fix this

		limit = min(batch_end, folder_total)
		list_images = range(batch_start,limit)

		for iImage in list_images:
			
			filepath4 = os.path.join(labelpath, label_list[index]+'/'+ label_folder[iImage])
			actual_label = cv2.imread(filepath4)
			actual_label_array[j] = actual_label

			j+=1
			# print(j)
		batch_start += batch_size
		batch_end += batch_size
	print('appending')
	actual_label_array_big=np.append(actual_label_array_big, actual_label_array, axis=0)
	print(actual_label_array.shape)
	print(actual_label_array_big.shape)

filepath6 =  os.path.join(npy_path, 'imgs_true_mask_test.npy')
np.save(filepath6, actual_label_array_big)
print(actual_label_array_big.shape)