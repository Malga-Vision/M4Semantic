from PIL import Image
import cv2
import numpy as np
import scipy
import os
from sklearn import metrics
import matplotlib.pyplot as plt


def map_wuav(a):
    #print(a)
    if (a == [230,175,127]).all():
        return 0
    elif (a == [185,163,75]).all():
        return 1
    elif (a == [0,128,50]).all():
        return 2
    elif (a == [69,58,105]).all():
        return 3
    elif (a == [59,94,53]).all():
        return 4
    elif (a == [97,97,117]).all():
        return 5
    elif (a == [128,64,128]).all():
        return 6
    elif (a == [128,64,64]).all():
        return 7
    elif (a == [64,64,128]).all():
        return 8

def map_aeroscapes(a):
    #print(a)
    if (a == [0, 0, 0]).all():
        return 0
    elif (a == [192, 128, 128]).all():
        return 1
    elif (a == [0, 128, 0]).all():
        return 2
    elif (a == [128, 128, 128]).all():
        return 3
    elif (a == [128, 0, 0]).all():
        return 4
    elif (a == [0, 0, 128]).all():
        return 5
    elif (a == [192, 0, 128]).all():
        return 6
    elif (a == [192, 0, 0]).all():
        return 7
    elif (a == [192, 128, 0]).all():
        return 8
    elif (a == [0, 64, 0]).all():
        return 9
    elif (a == [128, 128, 0]).all():
        return 10
    elif (a == [0, 128, 128]).all():
        return 11

#inp_path1 = "/media/DATA_4TB/Yara/results_wuav_joint01/gt_sem"
#inp_path2 = "/media/DATA_4TB/Yara/results_wuav_joint01/est_sem"
inp_path1 = "/media/DATA_4TB/Yara/results_aeroscapes/gt_ind"
inp_path2 = "/media/DATA_4TB/Yara/results_aeroscapes/est_ind"
imgs = os.listdir(inp_path1)
#imgs = os.listdir(inp_path2)
actual = np.array([])
predicted = np.array([])

#FRAME_SKIP = 50
FRAME_SKIP = 2
idx = 0
for im in imgs:
    if idx % FRAME_SKIP == 0:
        print(im)
        im1_path = os.path.join(*[inp_path1, im])
        im1 = cv2.imread(im1_path)
        im2_path = os.path.join(*[inp_path2, im])
        im2 = cv2.imread(im2_path)
        #im_upd1 = np.apply_along_axis(map_wuav, -1,im1)
        #im_upd2 = np.apply_along_axis(map_wuav, -1,im2)
        #actual = np.append(actual,im_upd1.flatten())
        #predicted = np.append(predicted,im_upd2.flatten())
        
        actual = np.append(actual,im1[:,:,0].flatten())
        predicted = np.append(predicted,im2[:,:,0].flatten())
        '''
        print("ACTUAL = ", im1)
        print("SHAPE = ", np.shape(im1))
        im1 = np.apply_along_axis(map_aeroscapes, -1,im1)
        im2 = np.apply_along_axis(map_aeroscapes, -1,im2)
        actual = np.append(actual,im1[:,:,0].flatten())
        predicted = np.append(predicted,im2[:,:,0].flatten())
        '''
    idx = idx+1
    
print("FINISHED")
  

confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display= metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
#cm_display= metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Sky","Water","Trees","Dirt Ground","Vegetation","Rocks","Road","Others"])
#cm_display= metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Sky","Water","Trees","Land","Vehicles","Rocks","Road","Construction","Others"])

cm_display.plot()
plt.show()
