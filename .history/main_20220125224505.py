
from pyexpat import model
from yolov4 import Yolov4, do_detect, models
import yolov5
import cv2

import sys

sys.path.append('yolov4')
sys.path.append('yolov4/tools')



imgfile='image/15.jpg'
class_names='/home/user/Mask_detection/yolov4/data/Mask.names' 
sized=608

img = cv2.imread(imgfile)

'''
yolov4 bbox
'''
model = Yolov4(yolov4conv137weight=None, n_classes=3, inference=True)
for i in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
    boxes = do_detect(model, sized, 0.4, 0.6)

bboxes_v4 = models.plot_boxes_cv2(img, boxes[0], './predictions.jpg', class_names)








# bboxes_v5 = yolov5.detect