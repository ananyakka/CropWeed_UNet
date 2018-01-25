#Source: https://github.com/zhixuhao/unet


import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Activation
from keras.activations import softmax
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
from data import *
from PIL import Image
import cv2

class myUnet(object):

	def __init__(self, img_rows = 200, img_cols = 200):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols, 0.8)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,3))
		
		'''
		unet with crop(because padding = valid) 

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
		print "conv1 shape:",conv1.shape
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
		print "conv1 shape:",conv1.shape
		crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
		print "crop1 shape:",crop1.shape
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print "pool1 shape:",pool1.shape

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
		print "conv2 shape:",conv2.shape
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
		print "conv2 shape:",conv2.shape
		crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
		print "crop2 shape:",crop2.shape
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print "pool2 shape:",pool2.shape

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
		print "conv3 shape:",conv3.shape
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
		print "conv3 shape:",conv3.shape
		crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
		print "crop3 shape:",crop3.shape
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print "pool3 shape:",pool3.shape

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
		'''

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
		conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
		# conv11 = Conv2D(3, 1, activation = 'softmax')(conv10)

		out = Activation('softmax')(conv10)
		print(out.get_shape())

		model = Model(input = inputs, output =out)
		model.summary()

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

		return model


	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()

		print(imgs_train.shape)
		print(imgs_mask_train.shape)
		print(imgs_test.shape)

		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0, patience=5)
		model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=500, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, early_stopping])

		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		print(imgs_mask_test.shape)
		np.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):

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


			# print(type(img1))
			# print(img2.size)
			# print(type(img2))
			
			# img2 = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
			# filepath = os.path.join("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/results_combined/", str(i)+'.jpg')
			# print(type(img2))
			# img2 = array_to_img(img2)
			# cv2.imwrite(filepath, img2)

if __name__ == '__main__':
	myunet = myUnet()
	# model = myunet.get_unet()
	myunet.train()
	myunet.save_img()








