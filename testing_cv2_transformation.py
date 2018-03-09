import cv2
import numpy as np

label = cv2.imread('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results/newIMG_1865/newIMG_1865_rotated10.png')


print(label[84,84, :])
print(label[140,30,:])
print(label[30,140,:])
cv2.imshow('label', label)
cv2.waitKey(60000)
cv2.destroyAllWindows()