from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import shutil
import fnmatch
import random
#import cv2
#from libtiff import TIFF

class myAugmentation(object):
	
	"""
	A class used to augmentate image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""

	def __init__(self, train_path="train", label_path="label", merge_path="merge", aug_merge_path="aug_merge", aug_train_path="aug_train", aug_label_path="aug_label", img_type="tif"):
		
		"""
		Using glob to get all .img_type form path
		"""

		self.train_imgs = glob.glob(train_path+"/*."+img_type)
		self.label_imgs = glob.glob(label_path+"/*."+img_type)
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.slices = len(self.train_imgs)
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):

		"""
		Start augmentation.....
		"""
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print ("trains can't match labels")
			return 0
		for i in range(len(trains)):
			img_t = load_img(path_train+"/"+str(i)+"."+imgtype)
			img_l = load_img(path_label+"/"+str(i)+"."+imgtype)
			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)
			img = x_t
			img = img.reshape((1,) + img.shape)
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))


	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):

		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
		    i += 1
		    if i > imgnum:
		        break

	def splitMerge(self):

		"""
		split merged image apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path
		for i in range(self.slices):
			path = path_merge + "/" + str(i)
			train_imgs = glob.glob(path+"/*."+self.img_type)
			savedir = path_train + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			savedir = path_label + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			for imgname in train_imgs:
				midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
				img = cv2.imread(imgname)
				img_train = img[:,:,2]#cv2 read image rgb->bgr
				img_label = img[:,:,0]
				cv2.imwrite(path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type,img_train)
				cv2.imwrite(path_label+"/"+str(i)+"/"+midname+"_label"+"."+self.img_type,img_label)

	def splitTransform(self):

		"""
		split perspective transform images
		"""
		#path_merge = "transform"
		#path_train = "transform/data/"
		#path_label = "transform/label/"
		path_merge = "deform/deform_norm2"
		path_train = "deform/train/"
		path_label = "deform/label/"
		train_imgs = glob.glob(path_merge+"/*."+self.img_type)
		for imgname in train_imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
			cv2.imwrite(path_label+midname+"."+self.img_type,img_label)



class dataProcess(object):

	# def __init__(self, out_rows, out_cols, train_test_split, data_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_images", 
	# 	label_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels", 
	# 	test_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/All Images", 
	# 	train_data_path = '/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Shuffled Data/Labelled Images Train and Val',
	# 	 test_data_path='/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Shuffled Data/Labelled Images Test', 
	# 	 train_label_path ='/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Shuffled Data/Labels Train and Val', 
	# 	 test_label_path ='/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Shuffled Data/Labels Test', 
	# 	 npy_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata", 
	# 	 img_type = "png"):
	def __init__(self, out_rows, out_cols, train_test_split, data_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Training Set/Training original", 
		label_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Training Set/Training ground truth", 
		test_data_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Test Set/images", 
		test_label_path ='/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/Test Set/Testing ground truth', 
		npy_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata", 
		img_type = "png"):
		"""

		test_path = "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Test Data"
		"""
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.train_test_split = train_test_split
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_data_path = test_data_path
		self.test_label_path = test_label_path
		self.npy_path = npy_path

	# def move_images_train_test(self): # doesn't work
	# 	labelled_images = glob.glob(self.data_path+"/*."+self.img_type)
	# 	labels = glob.glob(self.label_path+"/*."+self.img_type)
	# 	# no_of_images = len(fnmatch.filter(os.listdir(self.data_path), '*.png'))
	# 	# x_test_size = int(no_of_images *(1-self.train_test_split))
	# 	# print(x_test_size)
	# 	# # indices_shuffled = np.random.choice(no_of_images, no_of_images, replace =False)

	# 	# i_test_images = 0
	# 	# for f_img in labelled_images:
	# 	# 	if np.random.randint(0,10,1)<4:
	# 	# 		shutil.move(f_img, self.train_data_path)
	# 	# 		# shutil.copy(f_lab, self.train_label_path)

	# 	# 	elif i_test_images<=x_test_size:
	# 	# 		shutil.move(f_img, self.test_data_path)		
	# 	# 		i_test_images+=1
	# 	# 		print(i_test_images)

	# 	# 		# shutil.copy(f_lab, self.test_label_path)	

	# 	# labelled_images = os.listdir(os.path.join(self.data_path, '/*.'+self.img_type))
	# 	# labels = os.listdir(os.path.join(self.label_path, '/*.'+self.img_type))
	# 	no_of_images = len(fnmatch.filter(os.listdir(self.data_path), '*.png'))
	# 	x_train_size = int(no_of_images *self.train_test_split)

	# 	np.random.seed(2)
	# 	# for f_img, f_lab in zip(*random.sample(list(zip(labelled_images, labels)), 5)):
	# 	# 	shutil.copy(f_img, self.train_data_path)
	# 	# # for f_lab in random.Random(500).sample(labels, x_train_size):
	# 	# 	shutil.copy(f_lab, self.train_label_path)

	# 	f_img, f_lab = zip(*random.sample(list(zip(labelled_images, labels)), 5))

	# 	print(f_img)
	# 	print(f_lab)



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
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.test_data_path+"/*.JPG")
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex(".")]
			img = load_img(self.test_data_path + "/" + midname+".JPG", grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
			label = load_img(self.test_label_path + "/" + midname+".png", grayscale = False, target_size=(self.out_rows,self.out_cols),interpolation='nearest')
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			#label = np.array([label])
			print(img.shape)
	
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
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

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess(200,200, 0.8)
	# mydata.move_images_train_test()
	mydata.create_train_data()
	mydata.create_test_data()
	#imgs_train,imgs_mask_train = mydata.load_train_data()
#print imgs_train.shape,imgs_mask_train.shape