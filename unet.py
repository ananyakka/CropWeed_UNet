#Skeleton source: https://github.com/zhixuhao/unet


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
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Activation, Reshape
from keras.activations import softmax
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,ReduceLROnPlateau
from keras import backend as keras
from data import *
from train_params import *
from PIL import Image
import cv2
import tensorflow as tf
import random as rn
import time
from sklearn.model_selection import StratifiedKFold, KFold
from make_confusion_matrix import make_confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import f1_score
from train_params import f_score_weighted_loss, f_score_weighted, weighted_dice_coef_loss, weighted_dice_coef
# import keras.backend.tensorflow_backend

from data_generator import my_generator #my_generator(filepath, labelpath, x_set_indices, batch_size)

class myUnet(object):

	def __init__(self, img_rows = 200, img_cols = 200):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols, 0.8)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test, imgs_mask_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test
		'''
		#can't run because data doesn't fit in memory
		npy_path = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data'
		imgs_train = np.load(npy_path+"/imgs_train702010.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_train /= 255
		imgs_mask_train = np.load(npy_path+"/imgs_mask_train702010.npy")
		imgs_mask_train = imgs_mask_train.astype('float32')

		imgs_val = np.load(npy_path+"/imgs_val702010.npy")
		imgs_val = imgs_val.astype('float32')
		imgs_val /= 255
		imgs_mask_val = np.load(npy_path+"/imgs_mask_val702010.npy")
		imgs_mask_val = imgs_mask_val.astype('float32')

		imgs_test = np.load(npy_path+"/imgs_test702010.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		imgs_mask_test = np.load(npy_path+"/imgs_mask_test702010.npy")
		imgs_mask_test = imgs_mask_test.astype('float32')


		return imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test,imgs_mask_test
		'''

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,3))
		
		conv1 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		# print "conv1 shape:",conv1.shape
		conv1 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		# print "conv1 shape:",conv1.shape
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		# print "pool1 shape:",pool1.shape

		conv2 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		# print "conv2 shape:",conv2.shape
		conv2 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		# print "conv2 shape:",conv2.shape
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		# print "pool2 shape:",pool2.shape

		conv3 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		# print "conv3 shape:",conv3.shape
		conv3 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		# print "conv3 shape:",conv3.shape
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		# print "pool3 shape:",pool3.shape

		conv4 = Conv2D(int(self.img_rows), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(int(self.img_rows), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		# print(drop4.get_shape())
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
		# print(pool4.get_shape())

		# conv5 = Conv2D(400, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		# print(conv5.get_shape())

		# conv5 = Conv2D(400, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		# drop5 = Dropout(0.5)(conv5)
		# print(conv5.get_shape())

		# print(drop5.get_shape())



		# up6 = Conv2D(200, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		# print(up6.get_shape())
		# merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		# conv6 = Conv2D(200, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		# conv6 = Conv2D(200, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		# up7 = Conv2D(100, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		up7 = Conv2D(int(self.img_rows/2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))

		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(int(self.img_rows/2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(int(self.img_rows/4), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(int(self.img_rows/4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(int(self.img_rows/8), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(int(self.img_rows/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		# conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		# conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)
		out = Conv2D(3, 1, activation = 'softmax')(conv9)
		out_flatten = Reshape((-1,3))(out)
		# conv11 = Conv2D(3, 1, activation = 'softmax')(conv10)

		# out = Activation('softmax')(conv10)
		# print(out_flatten.get_shape())

		model = Model(input = inputs, output =out)
		model.summary()

		# model.compile(optimizer = Adam(lr = 1e-4), loss = f_score_weighted_loss, metrics = [f_score_weighted])
		model.compile(optimizer = Adam(lr = 1e-4), loss = weighted_dice_coef_loss, metrics = [ weighted_dice_coef])
		# model.compile(optimizer = Adam(lr = 1e-4), loss = mean_cross_entropy, metrics = [ 'accuracy'])

		# model.compile(optimizer = Adam(lr = 1e-4), loss = jaccard_cross_entropy_loss, metrics = [ jaccard_coef])		

		return model


	def train(self):

		print("loading data")
		# imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test,imgs_mask_test = self.load_data()
		imgs_train, imgs_mask_train, imgs_test, imgs_mask_test = self.load_data()

		# # The below is necessary for starting Numpy generated random numbers
		# # in a well-defined initial state.

		# np.random.seed(42)

		# # The below is necessary for starting core Python generated random numbers
		# # in a well-defined state.

		# rn.seed(12345)

		# # Force TensorFlow to use single thread.
		# # Multiple threads are a potential source of
		# # non-reproducible results.
		# # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

		# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

		# # The below tf.set_random_seed() will make random number generation
		# # in the TensorFlow backend have a well-defined initial state.
		# # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

		# tf.set_random_seed(1234)

		# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
		# keras.set_session(sess)



		print(imgs_train.shape)
		print(imgs_mask_train.shape)
		print(imgs_test.shape)

		print("loading data done")
		# model = load_model('/extend_sda/Ananya_files/Weeding Bot Project/Codes/Keras TF/Segmentation/UNet/images_200by200/augmented/unet_trial3.hdf5') # load a trained model of unet

		# model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0, patience=5)
		adjust_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
		model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=50, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, early_stopping, adjust_learning_rate])

		print('predict test data')
		imgs_predicted_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		# print(imgs_predicted_mask_test.shape)

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
		# print(imgs_train.shape[0])

		X = np.arange(0,imgs_train.shape[0] , 1)
		# fix random seed for reproducibility
		seed = 7
		np.random.seed(seed)
		# Source : https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
		# define 10-fold cross validation test harness
		n_splits=10
		# kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=seed)
		kfold = KFold(n_splits = n_splits, shuffle=True, random_state=seed)
		cvscores = []

		print(imgs_train.shape)
		print(imgs_mask_train.shape)
		print(imgs_test.shape)
		for train_index, test_index in kfold.split(X):
			print("TRAIN:", train_index, "TEST:", test_index)

		# for index, (train_indices, test_indices) in enumerate(kfold.split(imgs_train, imgs_mask_train)):
			print("loading data done")
			# The below is necessary for starting Numpy generated random numbers
			# in a well-defined initial state.

			np.random.seed(42)

			# The below is necessary for starting core Python generated random numbers
			# in a well-defined state.

			rn.seed(12345)

			# Force TensorFlow to use single thread.
			# Multiple threads are a potential source of
			# non-reproducible results.
			# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

			session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

			# The below tf.set_random_seed() will make random number generation
			# in the TensorFlow backend have a well-defined initial state.
			# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

			tf.set_random_seed(1234)

			sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
			keras.set_session(sess)
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

		x_val_size1 = int(no_of_images*0.9)
		x_val_size2 = int(no_of_images*0.8)


		x_train, x_val, x_test = np.split(indices_shuffled, [x_train_size1, x_val_size1])
		print(x_train)
		print(x_val)
		print(x_test)
		print(x_test.shape)

		# Generators
		training_generator = my_generator(filepath, labelpath, x_train, batch_size)
		validation_generator = my_generator(filepath, labelpath, x_val, batch_size)
		test_generator = my_generator(filepath, labelpath, x_test, 1)

		# ## Find class weights for an imbalanced dataset
		# # Convert on one hot labels to class labels
		# #https://stackoverflow.com/a/44855957

		# # Create a pd.series that represents the categorical class of each one-hot encoded row
		# label_sample = np.identity(3)
		# df_y = pd.DataFrame(label_sample)
		# y_classes = df_y.idxmax(1, skipna=False)
		# print('y_classes')
		# print(y_classes)


		# # Instantiate the label encoder
		# le = LabelEncoder()

		# # Fit the label encoder to our label series
		# le.fit(list(y_classes))

		# # Create integer based labels Series
		# y_integers = le.transform(list(y_classes))

		# # Create dict of labels : integer representation
		# labels_and_integers = dict(zip(y_classes, y_integers))

		# class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
		# sample_weights = compute_sample_weight('balanced', y_integers)

		# class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))
		# print('class_weights')
		# print(class_weights_dict)


		
		model = self.get_unet()
		# model = load_model('/extend_sda/Ananya_files/Weeding Bot Project/Codes/Keras TF/Segmentation/UNet/unet.hdf5') # load a trained model of unet
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience=5)
		adjust_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
		start_time = time.time()
		# model.fit_generator(training_generator, epochs=60, steps_per_epoch=2000, verbose=1, callbacks=[model_checkpoint], validation_data = validation_generator, validation_steps = 610 )
		# train_steps_per_epoch = np.math.ceil(training_generator.samples / training_generator.batch_size)
		# print(train_steps_per_epoch)
		# print(val_steps_per_epoch)
		# val_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)
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

		print("array to image")
		#Path to read test images and predicted labels from
		# imgs_mask = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/imgs_predicted_mask_test702010.npy')
		# imgs_test = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/imgs_test702010.npy')

		# # Folders to save test images, predicted labels and their overlay
		# if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results'):
		# 	os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results')
		# if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results_test_images'):
		# 	os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results_test_images')			
		# if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results_combined'):
		# 	os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results_combined')		


		#Path to read test images and predicted labels from
		imgs_mask = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/imgs_predicted_mask_test.npy')
		print(imgs_mask.shape)
		imgs_test = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/imgs_test.npy')
		print(imgs_test.shape)

		# Folders to save test images, predicted labels and their overlay
		if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results'):
			os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results')
		if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results_test_images'):
			os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results_test_images')			
		if not os.path.exists('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results_combined'):
			os.makedirs('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results_combined')	


		for i in range(imgs_mask.shape[0]):
			# print(i)
			img = imgs_mask[i]
			# print(img.size)
			img = array_to_img(img)
			img.save("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results/%d.jpg"%(i))
			img = imgs_test[i]	
			img = array_to_img(img)
			img.save("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results_test_images/%d.jpg"%(i))

			# add translucent label(img1) to original image(img2)
			filepath1 = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results/", str(i)+'.jpg')
			img1 = cv2.imread(filepath1)
			# img4 = img1[:,:,1]

			filepath2 = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results_test_images/", str(i)+'.jpg')
			img2 = cv2.imread(filepath2)

			img3 = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)

			filepath3 = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/results_combined/", str(i)+'.jpg')
			cv2.imwrite(filepath3, img3)


			# print(type(img1))
			# print(img2.size)
			# print(type(img2))
			
			# img2 = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
			# filepath = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_combined/", str(i)+'.jpg')
			# print(type(img2))
			# img2 = array_to_img(img2)
			# cv2.imwrite(filepath, img2)

def predict_and_save(model,filepath, labelpath, x_set_indices, save = True):

	rows =200
	cols = 200
	file_list = list(os.listdir(filepath))
	label_list = list(os.listdir(labelpath))
	batch_size = 1

	sum = 0
	count = 0
	total_confusion_mat = np.zeros((3,3))
	total_confusion_mat_percent = np.zeros((3,3))
	classwise_accuracy = np.zeros(3)
	precision_score= np.zeros(3)
	recall_score= np.zeros(3)
	sensitivity_score= np.zeros(3)

	# Folders to save test images, predicted labels and their overlay
	npy_path = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/'
	# if not os.path.exists(os.path.join(npy_path, 'results')):
	# 	os.makedirs(os.path.join(npy_path, 'results'))
	# if not os.path.exists(os.path.join(npy_path, 'results_test_images')):
	# 	os.makedirs(os.path.join(npy_path, 'results_test_images'))			
	# if not os.path.exists(os.path.join(npy_path, 'results_combined')):
	# 	os.makedirs(os.path.join(npy_path, 'results_combined'))

	for index in x_set_indices:

		image_folder = os.listdir(os.path.join(filepath, file_list[index]))
		label_folder = os.listdir(os.path.join(labelpath, label_list[index]))
		# print(len(image_folder))
		nBatch_in_folder = int(len(image_folder)/batch_size)

		batch_start = 0
		batch_end = batch_size
		folder_total = len(image_folder)

		if not os.path.exists(os.path.join(npy_path, 'results/'+file_list[index]+ '/')):
			os.makedirs(os.path.join(npy_path, 'results/'+file_list[index]+ '/'))
		if not os.path.exists(os.path.join(npy_path, 'results_test_images/'+file_list[index]+ '/')):
			os.makedirs(os.path.join(npy_path, 'results_test_images/'+file_list[index]+ '/'))			
		if not os.path.exists(os.path.join(npy_path, 'results_combined/'+file_list[index]+ '/')):
			os.makedirs(os.path.join(npy_path, 'results_combined/'+file_list[index]+ '/'))
		if not os.path.exists(os.path.join(npy_path, 'results_intersected/'+file_list[index]+ '/')):
			os.makedirs(os.path.join(npy_path, 'results_intersected/'+file_list[index]+ '/'))

		while batch_end< folder_total:

			limit = min(batch_end, folder_total)
			list_images = range(batch_start,batch_end)

			i=0

			for iImage in list_images:
				
				match = 0
				wrong = 0

				# '''
				# test_img = load_img(os.path.join(filepath, file_list[index])+'/'+ image_folder[iImage], grayscale = False)
				filepath1 = os.path.join(filepath, file_list[index]+'/'+ image_folder[iImage])
				test_img = cv2.imread(filepath1)
				test_img_exp=np.expand_dims(test_img, axis=0)

				filepath2 = os.path.join(npy_path, 'results_test_images/'+file_list[index]+ '/'+image_folder[iImage])
				# print(file_list[index])
				# print(image_folder[iImage])
				# print(filepath2)
				cv2.imwrite(filepath2, test_img)

				predicted_label = model.predict(test_img_exp)
				# print(predicted_label[0].shape)
				# predicted_label = array_to_img(predicted_label)
				predicted_label = array_to_img(predicted_label[0])
				# '''
				path = os.path.join(npy_path, 'results/'+file_list[index]+'/'+image_folder[iImage])
				predicted_label.save(path)
				# predicted_label.save(path, 'jpg')
				# cv2.imwrite(path, predicted_label[0])

				predicted_image = cv2.imread(path)


				# print(type(predicted_image))
				# print(predicted_image.shape)
				# print(type(test_img))
				# print(test_img.shape)

				# '''
				combined_image = cv2.addWeighted(predicted_image, 0.4, test_img, 0.6, 0)

				filepath3 = os.path.join(npy_path, 'results_combined/'+file_list[index]+'/'+ image_folder[iImage])
				cv2.imwrite(filepath3, combined_image)
				# '''

				# to compare predicted label and actual label, and display red for wrong prediction and blue for correct predictions, pixel-wise
				filepath4 = os.path.join(labelpath, label_list[index]+'/'+ label_folder[iImage])
				actual_label = cv2.imread(filepath4)

				red = np.array([255, 0, 0])
				blue =np.array([0, 0, 255])

				#open cv manipulates images as BGR, not RGB
				cv_blue = np.array([255, 0, 0])
				cv_red =np.array([0, 0, 255])

				intersect_image = actual_label

				if predicted_image.shape == actual_label.shape:

					for i in range(actual_label.shape[0]):
						for j in range(actual_label.shape[1]):
							pixel = actual_label[i,j]
							predicted_pixel = predicted_image[i, j]
							# print(pixel)
							if (pixel.argmax(axis=-1) == red.argmax(axis=-1)) or (pixel.argmax(axis=-1) == blue.argmax(axis=-1)):
								if pixel.argmax(axis=-1) == predicted_pixel.argmax(axis=-1):
									intersect_image[i, j, :] = cv_blue
									match+=1
									# print('yes')
								else:
									intersect_image[i, j, :] = cv_red
									wrong+=1
					# print (match/(float)(match+wrong))
					sum+= match/(float)(match+wrong)
					count+=1

				classwise_accuracy+= recall_score_class(actual_label, predicted_image).eval()
				precision_score+= precision_score_class(actual_label, predicted_image).eval()
				recall_score+= recall_score_class(actual_label, predicted_image).eval()
				sensitivity_score+= sensitivity(actual_label, predicted_image).eval()

				filepath5 = os.path.join(npy_path, 'results_intersected/'+file_list[index]+'/'+ image_folder[iImage])
				cv2.imwrite(filepath5, intersect_image)

				confusion_mat = make_confusion_matrix(actual_label, predicted_image)
				# print(confusion_mat.shape)
				total_confusion_mat+= confusion_mat
				# print(total_confusion_mat)


				i+=1
			batch_start += batch_size
			batch_end += batch_size

	for iRow in range(total_confusion_mat.shape[0]):
		total_confusion_mat_percent[iRow, :] = total_confusion_mat[iRow, :]/ np.sum(total_confusion_mat[iRow, :])


	# print(count)
	print ("Percentage: ", sum/(float)(count))
	print("Confusion Matrix", total_confusion_mat)
	print("Percentage Confusion Matrix", total_confusion_mat_percent)
	print("Classwise accuracy", classwise_accuracy)
	print("Precision",precision_score)
	print("Recall", recall_score)
	print("Sensitivity", sensitivity_score)


if __name__ == '__main__':

	if keras.backend() == 'tensorflow':
		keras.clear_session()
	# The below is necessary for starting Numpy generated random numbers
	# in a well-defined initial state.

	np.random.seed(42)

	# The below is necessary for starting core Python generated random numbers
	# in a well-defined state.

	rn.seed(12345)

	# Force TensorFlow to use single thread.
	# Multiple threads are a potential source of
	# non-reproducible results.
	# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

	# The below tf.set_random_seed() will make random number generation
	# in the TensorFlow backend have a well-defined initial state.
	# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

	tf.set_random_seed(1234)

	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	keras.set_session(sess)


	myunet = myUnet()
	model = myunet.get_unet()


	# myunet.train() # comment for kfold cross-validation
	# myunet.train_kfold()# uncomment for kfold cross-validation
	myunet.train_batch()
	# myunet.save_img()# comment for kfold cross-validation








