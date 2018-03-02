"""

"""
import os 
# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
os.environ['PYTHONHASHSEED'] = '0'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, Concatenate, Add, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Activation, Reshape
from keras.activations import softmax
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,ReduceLROnPlateau
from keras import backend as keras
from train_params import *
from PIL import Image
import cv2
import tensorflow as tf
import random as rn
import time
from sklearn.model_selection import StratifiedKFold, KFold
from make_confusion_matrix import make_confusion_matrix
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from keras.utils.generic_utils import get_custom_objects

from sklearn.metrics import f1_score
from train_params import f_score_weighted_loss, f_score_weighted , weighted_dice_coef_loss, weighted_dice_coef
from data_generator import my_generator #my_generator(filepath, labelpath, x_set_indices, batch_size)

class myUnet(object):

	def __init__(self, img_rows = 200, img_cols = 200, img_channels = 3):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.img_channels = img_channels

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols, self.img_channels))
		
		conv1 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(int(self.img_rows), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(int(self.img_rows), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(int(self.img_rows*2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(int(self.img_rows*2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(int(self.img_rows), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = Concatenate([drop4,up6], axis=-1)
		# merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(int(self.img_rows), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(int(self.img_rows), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(int(self.img_rows/2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		# up7 = Conv2D(int(self.img_rows/2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
		merge7 = Concatenate([conv3,up7], axis=-1)
		conv7 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(int(self.img_rows/4), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge7 = Concatenate([conv2,up8], axis=-1)
		conv8 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(int(self.img_rows/8), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge7 = Concatenate([conv1,up9], axis=-1)
		conv9 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

		out = Conv2D(3, 1, activation = 'softmax')(conv9)
		# out_flatten = Reshape((-1,3))(out) # Get a flatten output

		model = Model(input = inputs, output =out)
		print('Model Summary')
		model.summary()

		# model.compile(optimizer = Adam(lr = 1e-4), loss = f_score_weighted_loss, metrics = [f_score_weighted])
		# add custom_objects={"f_score_weighted_loss":f_score_weighted_loss, "f_score_weighted":f_score_weighted } to load_model
		
		# model.compile(optimizer = Adam(lr = 1e-4), loss = weighted_dice_coef_loss, metrics = [ weighted_dice_coef])
		# add custom_objects={"weighted_dice_coef_loss":weighted_dice_coef_loss, "weighted_dice_coef":weighted_dice_coef } to load_model

		model.compile(optimizer = Adam(lr = 1e-4), loss = mean_cross_entropy, metrics = [ 'accuracy'])
		# add custom_objects={"mean_cross_entropy":mean_cross_entropy} to load_model

		# model.compile(optimizer = Adam(lr = 1e-4), loss = jaccard_cross_entropy_loss, metrics = [ jaccard_coef])		
		# add custom_objects={"jaccard_cross_entropy_loss":jaccard_cross_entropy_loss, "jaccard_coef":jaccard_coef } to load_model

		return model


	def train(self):

		print("loading data")
		# imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test,imgs_mask_test = self.load_data()
		imgs_train, imgs_mask_train, imgs_test, imgs_mask_test = self.load_data()

		print("loading data done")
		# model = load_model('/extend_sda/Ananya_files/Weeding Bot Project/Codes/Keras TF/Segmentation/UNet/images_200by200/augmented/unet_trial3.hdf5') # load a trained model of unet

		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0, patience=5)
		adjust_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=50, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, early_stopping, adjust_learning_rate])

		print('predict test data')
		imgs_predicted_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

		print(imgs_predicted_mask_test.shape)
		np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/imgs_predicted_mask_test.npy', imgs_predicted_mask_test)

		print('evaluate test data')
		score = model.evaluate(x = imgs_test, y= imgs_mask_test)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])


	def train_kfold(self):

		print("loading data")
		# imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test,imgs_mask_test = self.load_data()
		imgs_train, imgs_mask_train, imgs_test = self.load_data()

		image_indices = np.arange(0,imgs_train.shape[0] , 1)

		# Source : https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
		# define 10-fold cross validation test harness
		n_splits=10
		# kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=seed)
		kfold = KFold(n_splits = n_splits, shuffle=True, random_state=seed)
		cvscores = []

		for train_index, test_index in kfold.split(image_indices):
		# for index, (train_indices, test_indices) in enumerate(kfold.split(imgs_train, imgs_mask_train)):
			print("TRAIN:", train_index, "TEST:", test_index)

			## Below steps are necessary to generate reproducible model runs
			# Necessary for starting Numpy generated random numbers in a well-defined initial state.
			np.random.seed(42)

			# Necessary for starting core Python generated random numbers in a well-defined state.
			rn.seed(12345)

			# Force TensorFlow to use single thread.
			# Multiple threads are a potential source of non-reproducible results.
			# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
			session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

			# The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
			# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
			tf.set_random_seed(1234)

			sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
			keras.set_session(sess)
			##
		
			print("loading data done")

			model = self.get_unet()
			print("got unet")

			model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
			print('Fitting model...')
			early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience=5)
			adjust_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
			model.fit(imgs_train[train_index], imgs_mask_train[train_index], batch_size=4, nb_epoch=500, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, early_stopping, adjust_learning_rate])

			print('evaluate test data')
			score = model.evaluate(x = imgs_train[test_index], y= imgs_mask_train[test_index])
			print('Test loss:', score[0])
			print('Test accuracy:', score[1])
			cvscores.append(score[1] * 100)

		print('Mean score with std deviation over {0} folds'.format(n_splits))
		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

	def train_batch(self):

		filepath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images/'
		labelpath = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels/'

		batch_size = 32

		#Make a list of all files
		file_list = list(os.listdir(filepath))
		no_of_images = len(file_list)

		#Shuffle the indices of the file list
		indices_shuffled = np.random.choice(no_of_images, no_of_images, replace =False)

		#Split the indices into train, validation and test data
		x_train_size1 = int(no_of_images *0.7)
		x_train_size2 = int(no_of_images *0.6)

		x_train, x_val, x_test = np.split(indices_shuffled, [x_train_size1, x_val_size1])
		print(x_train)
		print(x_val)
		print(x_test)

		# Generators
		training_generator = my_generator(filepath, labelpath, x_train, batch_size)
		validation_generator = my_generator(filepath, labelpath, x_val, batch_size)
		test_generator = my_generator(filepath, labelpath, x_test, 1)

		# model = self.get_unet()
		model = load_model('/extend_sda/Ananya_files/Weeding Bot Project/Codes/Keras TF/Segmentation/UNet/images_200by200/augmented/unet_trial15.hdf5', custom_objects={"mean_cross_entropy":mean_cross_entropy}) # load a trained model of unet
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience=5)
		adjust_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
		
		start_time = time.time()

		# model.fit_generator(training_generator, epochs=60, steps_per_epoch=2000, verbose=1, callbacks=[model_checkpoint], validation_data = validation_generator, validation_steps = 610 )

		model.fit_generator(training_generator, epochs=60, steps_per_epoch=2000, verbose=1, callbacks=[model_checkpoint, early_stopping, adjust_learning_rate], validation_data = validation_generator, validation_steps = 610 )
		##steps are found by dividing total images by batch size; (68080/32 ~= 2127), (19536/32 ~= 610)

		end_time = time.time()
		total_mins = (end_time - start_time)/60
		print("Training time: %0.2f min"% total_mins)

		print('predict test data')
		start_time = time.time()

		# imgs_predicted_mask_test = model.predict_generator(test_generator, steps=9768, max_queue_size=10, workers=1, verbose=1)
		# #steps: (9768/32 ~=305)
		# print(imgs_predicted_mask_test.shape)
		# np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/imgs_predicted_mask_test702010.npy', imgs_predicted_mask_test)

		predict_and_save(model,filepath, labelpath, x_test, save = True)

		end_time = time.time()
		total_mins = (end_time - start_time)/60
		print("Prediction time: %0.2f min"% total_mins)

		print('evaluate test data')
		score = model.evaluate_generator(test_generator, steps=9768, max_queue_size=10, workers=1)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

	def save_img(self):

		print("Array to image")

		#Path to read test images and predicted labels from
		npy_path = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/'

		imgs_mask = np.load(os.path.join(npy_path,'imgs_predicted_mask_test.npy'))
		print(imgs_mask.shape)

		imgs_test = np.load(os.path.join(npy_path,'imgs_test.npy'))
		print(imgs_test.shape)

		# Folders to save test images, predicted labels and their overlay
		results_path = os.path.join(npy_path,'results')
		if not os.path.exists(results_path):
			os.makedirs(results_path)
		results_test_images_path = os.path.join(npy_path,'results_test_images') 
		if not os.path.exists(results_test_images_path):
			os.makedirs(results_test_images_path)		
		results_combined_path = os.path.join(npy_path, 'results_combined')
		if not os.path.exists(results_combined_path):
			os.makedirs(results_combined_path)	


		for iMask in range(imgs_mask.shape[0]):
			# print(i)
			img_label = imgs_mask[iMask]
			# print(img.size)
			img_label = array_to_img(img_label)
			img_label.save(os.path.join(results_path+"/"+str(iMask)+".jpg"))
			img = imgs_test[iMask]	
			img = array_to_img(img)
			img.save(os.path.join(results_test_images_path+"/"+str(iMask)+".jpg"))

			# add translucent label(img1) to original image(img2)
			label_image_path = os.path.join(results_path+"/"+str(iMask)+".jpg")
			overlay_label = cv2.imread(label_image_path)

			saved_image_path = os.path.join(results_test_images_path+"/"+str(iMask)+".jpg")
			base_image = cv2.imread(saved_image_path)

			overlay_image = cv2.addWeighted(overlay_label, 0.4, base_image, 0.6, 0)

			overlay_path = os.path.join(results_combined_path+"/"+str(iMask)+".jpg")
			cv2.imwrite(overlay_path, overlay_image)

	def predict_and_save(model,filepath, labelpath, x_set_indices, save = True):

		rows = self.img_rows
		cols = self.img_cols
		classes = self.img_channels

		file_list = list(os.listdir(filepath))
		label_list = list(os.listdir(labelpath))
		batch_size = 1

		sum = 0
		image_count = 0
		total_confusion_mat = np.zeros((classes,classes))
		total_confusion_mat_percent = np.zeros((classes,classes))
		predicted_label_array_big = np.empty([0, rows,cols,classes])
		actual_label_array_big = np.empty([0, rows,cols,classes])
		# classwise_accuracy = np.zeros(classes)
		# precision_score= np.zeros(classes)
		# recall_score= np.zeros(classes)
		# sensitivity_score= np.zeros(classes)

		# Folders to save test images, predicted labels and their overlay
		npy_path = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/'

		for index in x_set_indices:

			image_folder = os.listdir(os.path.join(filepath, file_list[index]))
			label_folder = os.listdir(os.path.join(labelpath, label_list[index]))
			# print(len(image_folder))
			nBatch_in_folder = int(len(image_folder)/batch_size)

			batch_start = 0
			batch_end = batch_size
			folder_total = len(image_folder)
			predicted_label_array = np.ndarray((len(image_folder), rows,cols,3), dtype=np.float32)
			actual_label_array = np.ndarray((len(image_folder), rows,cols,3), dtype=np.float32)

			# Folders to save test images, predicted labels and their overlay
			results_path = os.path.join(npy_path,'results/'+file_list[index]+ '/')
			if not os.path.exists(results_path):
				os.makedirs(results_path)
			results_test_images_path = os.path.join(npy_path,'results_test_images/'+file_list[index]+ '/') 
			if not os.path.exists(results_test_images_path):
				os.makedirs(results_test_images_path)		
			results_combined_path = os.path.join(npy_path, 'results_combined/'+file_list[index]+ '/')
			if not os.path.exists(results_combined_path):
				os.makedirs(results_combined_path)	
			results_intersected_path = os.path.join(npy_path, 'results_intersected/'+file_list[index]+ '/')
			if not os.path.exists(results_intersected_path):
				os.makedirs(results_intersected_path)

			iAll_images_per_index=0 
			while batch_end< folder_total: #always ignores the last few in each; fix this

				limit = min(batch_end, folder_total)
				list_images = range(batch_start,limit)

				for iImage in list_images:
					
					match = 0
					wrong = 0

					
					# test_img = load_img(os.path.join(filepath, file_list[index])+'/'+ image_folder[iImage], grayscale = False)
					current_image_path = os.path.join(filepath, file_list[index]+'/'+ image_folder[iImage])
					test_img = cv2.imread(current_image_path)
					test_img_exp=np.expand_dims(test_img, axis=0) #expand to make it 4D array for prediction

					image_save_path = os.path.join(results_test_images_path, image_folder[iImage])
					# print(file_list[index])
					# print(image_folder[iImage])
					cv2.imwrite(image_save_path, test_img)

					predicted_label = model.predict(test_img_exp)
					predicted_label_array[iAll_images_per_index] = predicted_label
					print(predicted_label.shape)

					print(predicted_label[0].shape)
					# predicted_label = array_to_img(predicted_label)
					predicted_label = array_to_img(predicted_label[0])					
					predicted_label_path = os.path.join(results_path, image_folder[iImage])
					predicted_label.save(predicted_label_path)

					predicted_image = cv2.imread(path)
					# print(type(predicted_image))
					# print(predicted_image.shape)
					# print(type(test_img))
					# print(test_img.shape)
					combined_image = cv2.addWeighted(predicted_image, 0.4, test_img, 0.6, 0)
					overlay_path = os.path.join(results_combined_path, image_folder[iImage])
					cv2.imwrite(overlay_path, combined_image)
					
					# to compare predicted label and actual label, and display red for wrong prediction and blue for correct predictions, pixel-wise
					actual_label_path = os.path.join(labelpath, label_list[index]+'/'+ label_folder[iImage])
					actual_label = cv2.imread(actual_label_path)
					actual_label_array[iAll_images_per_index] = actual_label


					# print (match/(float)(match+wrong))
					intersect_image, match, wrong = compare_image(actual_label, predicted_image)
					sum+= match/(float)(match+wrong)
					image_count+=1

					# classwise_accuracy+= recall_score_class(actual_label, predicted_image).eval(session=sess)
					# precision_score+= precision_score_class(actual_label, predicted_image).eval(session=sess)
					# precision_score+= precision_score_class(actual_label, predicted_image).eval(session=sess)
					# recall_score+= recall_score_class(actual_label, predicted_image).eval(session=sess)
					# recall_score+= recall_score_class(actual_label, predicted_image).eval(session=sess)
					# sensitivity_score+= sensitivity(actual_label, predicted_image).eval(session=sess)

					intersect_image_path = os.path.join(results_intersected_path, image_folder[iImage])
					cv2.imwrite(intersect_image_path, intersect_image)

					confusion_mat = make_confusion_matrix(actual_label, predicted_image)
					# print(confusion_mat.shape)
					total_confusion_mat+= confusion_mat
					# print(total_confusion_mat)

					iAll_images_per_index+=1

				batch_start += batch_size
				batch_end += batch_size

			predicted_label_array_big=np.append(predicted_label_array_big, predicted_label_array, axis=0)
			actual_label_array_big=np.append(actual_label_array_big, actual_label_array, axis=0)

		for iRow in range(total_confusion_mat.shape[0]):
			total_confusion_mat_percent[iRow, :] = total_confusion_mat[iRow, :]/ np.sum(total_confusion_mat[iRow, :]) # divide each row by total number of 
			#pixels in that row

		
		# print(image_count)
		print ("Percentage: ", sum/(float)(image_count))
		print("Confusion Matrix", total_confusion_mat)
		print("Percentage Confusion Matrix", total_confusion_mat_percent)
		# print("Classwise accuracy", classwise_accuracy)
		# print("Precision",precision_score/(float)(image_count))
		# print("Recall", recall_score/(float)(image_count))
		# print("Sensitivity", sensitivity_score)
		predicted_mask_path =  os.path.join(npy_path, 'imgs_predicted_mask_test.npy')
		np.save(predicted_mask_path, predicted_label_array_big)
		true_mask_path =  os.path.join(npy_path, 'imgs_true_mask_test.npy')
		np.save(true_mask_path, actual_label_array_big)

if __name__ == '__main__':

	# Clear any interrupted sessions to free up memory

	if keras.backend() == 'tensorflow':
		keras.clear_session()

	## Below steps are necessary to generate reproducible model runs
	# Necessary for starting Numpy generated random numbers in a well-defined initial state.
	np.random.seed(42)

	# Necessary for starting core Python generated random numbers in a well-defined state.
	rn.seed(12345)

	# Force TensorFlow to use single thread.
	# Multiple threads are a potential source of non-reproducible results.
	# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

	# The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
	# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
	tf.set_random_seed(1234)

	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	keras.set_session(sess)
	##

	# Load UNet model
	myunet = myUnet()
	model = myunet.get_unet()

	#Train model 

	# myunet.train() # comment for kfold cross-validation
	# myunet.train_kfold()# uncomment for kfold cross-validation
	myunet.train_batch()
	# myunet.save_img()# comment for kfold cross-validation