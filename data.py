from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import random
#import cv2




class dataProcess(object):

	def __init__(self, out_rows, out_cols, train_test_split, data_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Training Set/Training original", 
		label_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Training Set/Training ground truth", 
		test_data_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Test Set/images", 
		test_label_path ='/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Test Set/Testing ground truth', 
		npy_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata", 
		img_type = "png"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.train_test_split = train_test_split
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_data_path = test_data_path
		self.test_label_path = test_label_path
		self.npy_path = npy_path

	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*.JPG")
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex(".")]
			# print(midname)
			img = load_img(self.data_path + "/" + midname+".JPG", grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
			label = load_img(self.label_path + "/" + midname+".png", grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			#label = np.array([label])
			print(img.shape)
	
			imgdatas[i] = img
			# print(imgdatas.shape)
			imglabels[i] = label
			# print(imglabels.shape)
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):

		# For predicting on all the unlabelled images
		# i = 0
		# print('-'*30)
		# print('Creating test images...')
		# print('-'*30)
		# imgs = glob.glob(self.test_path+"/*.JPG")
		# imgs2 = glob.glob(self.test_path+"/*.jpg")
		# print(len(imgs))
		# imgdatas = np.ndarray(((len(imgs)+len(imgs2)),self.out_rows,self.out_cols,3), dtype=np.uint8)
		# for imgname in imgs:
		# 	midname = imgname[imgname.rindex("/")+1:]
		# 	img = load_img(self.test_path + "/" + midname,grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
		# 	img = img_to_array(img)
		# 	#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
		# 	#img = np.array([img])
		# 	imgdatas[i] = img
		# 	i += 1
		# print('loading JPG done')

		
		# print(len(imgs2))
		# for imgname in imgs2:
		# 	midname = imgname[imgname.rindex("/")+1:]
		# 	img = load_img(self.test_path + "/" + midname,grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
		# 	img = img_to_array(img)
		# 	#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
		# 	#img = np.array([img])
		# 	imgdatas[i] = img
		# 	i += 1
		# print('loading jpg done')

		# np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		# print('Saving to imgs_test.npy files done.')


		#For comparison data
		iImage = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.test_data_path+"/*.JPG")
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		red = np.array([1, 0, 0])
		blue = np.array([0, 0, 1])
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex(".")]
			img = load_img(self.test_data_path + "/" + midname+".JPG", grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
			label = load_img(self.test_label_path + "/" + midname+".png", grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
			img = img_to_array(img)
			# label = img_to_array(label)
			label = np.array(label)

			for i in range(label.shape[0]):
				for j in range(label.shape[1]):
					pixel = label[i,j]
					if (pixel.argmax(axis=-1) == red.argmax(axis=-1)) or (pixel.argmax(axis=-1) == blue.argmax(axis=-1)):
						print('yes, i have red or blue pixels')

			# img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			# label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			# img = np.array([img])
			# label = np.array([label])
			print(img.shape)
	
			imgdatas[iImage] = img
			imglabels[iImage] = label
			if iImage % 100 == 0:
				print('Done: {0}/{1} images'.format(iImage, len(imgs)))
			iImage += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_test.npy', imglabels)
		print('Saving to .npy files done.')



	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		#mean = imgs_train.mean(axis = 0)
		#imgs_train -= mean	
		# imgs_mask_train /= 255
		# imgs_mask_train[imgs_mask_train > 0.5] = 1
		# imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		imgs_mask_test = np.load(self.npy_path+"/imgs_mask_test.npy")
		imgs_mask_test = imgs_mask_test.astype('float32')
		imgs_mask_test /= 255
		#mean = imgs_test.mean(axis = 0)
		#imgs_test -= mean	
		return imgs_test, imgs_mask_test

if __name__ == "__main__":


	mydata = dataProcess(200,200, 0.8)
	# mydata.create_train_data()
	mydata.create_test_data()
	#imgs_train,imgs_mask_train = mydata.load_train_data()
