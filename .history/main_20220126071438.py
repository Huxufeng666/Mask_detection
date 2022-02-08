import sys
from turtle import width
# sys.path.append('yolov4')
from yolov4 import models
from pyexpat import model
import yolov5
import cv2
import torch



imgfile='image/15.jpg'
class_names='Mask_detection/yolov4/data/Mask.names' 
weightfile = 'yolov4/Yolov4_epoch300.pth'
width=608
height=680
img = cv2.imread(imgfile)
sized = cv2.resize(img, (width, height))
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

'''
yolov4 bbox
'''
model = models.Yolov4(yolov4conv137weight=None, n_classes=3, inference=True)
pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
model.load_state_dict(pretrained_dict)


for i in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
    boxes = models.do_detect(model, sized, 0.4, 0.6)

bboxes_v4 = models.plot_boxes_cv2(img, boxes[0], './predictions.jpg', class_names)








# bboxes_v5 = yolov5.detect