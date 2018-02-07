import numpy as np
import os
import cv2

sum = 0
count = 0

# for filename in os.listdir("All_segmented"):
#     result = cv2.imread("All_segmented/"+filename)
#     gt = cv2.imread("GT/"+filename)
#     match = 0
#     wrong = 0
#     for i in xrange(gt.shape[0]):
# 	for j in xrange(gt.shape[1]):
# 	    pixel = gt[i,j]
# 	    if np.array_equal(pixel, [255,0,0]) or np.array_equal(pixel, [0,0,255]):
# 		if np.array_equal(result[i,j], pixel):
# 		    match+=1
# 		else:
# 		   wrong+=1
#     print match/(float)(match+wrong)
#     sum+= match/(float)(match+wrong)
#     count+=1

# print "Percentage: ", sum/(float)(count)


imgs_mask = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/imgs_mask_test.npy')
# imgs_test = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/imgs_test.npy')
imgs_predicted_mask = np.load('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/CompareStudy/npydata/imgs_predicted_mask_test.npy')
print(imgs_mask.shape)
print(imgs_predicted_mask.shape)

for mask_number in range(imgs_mask.shape[0]):
    match = 0
    wrong = 0
    red = np.array([1, 0, 0])
    blue =np.array([0, 0, 1])
    mask = imgs_mask[mask_number]
    print(mask.shape)
    mask_predicted = imgs_predicted_mask[mask_number]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            pixel = mask[i,j]
            predicted_pixel = mask_predicted[i, j]
            # print(pixel.argmax(axis=-1))
            if (pixel.argmax(axis=-1) == red.argmax(axis=-1)) or (pixel.argmax(axis=-1) == blue.argmax(axis=-1)):
                # print('yes')
                if pixel.argmax(axis=-1) == predicted_pixel.argmax(axis=-1):
                    match+=1
                    # print('yes')
                else:
                   wrong+=1
    print (match/(float)(match+wrong))
    sum+= match/(float)(match+wrong)
    # print(match)
    # print(wrong)
    count+=1
print(count)
print ("Percentage: ", sum/(float)(count))

