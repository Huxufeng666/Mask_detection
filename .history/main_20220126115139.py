import sys
from turtle import width

from matplotlib import image
from numpy import source
# sys.path.append('yolov4')
from yolov4 import models
from pyexpat import model
import yolov5
import cv2
import torch
import ensemble_boxes 


source='image/15.jpg'
namesfile_v4='Mask_detection/yolov4/data/Mask.names' 
weight_v4 = 'yolov4/Yolov4_epoch300.pth'
n_classes=3
weight_v5 ='yolov5/runs/train/exp9/weights/best.pt'
imgsz = 608,608

# '''
# yolov4 bbox
# '''
# model = models.Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=False)
# pretrained_dict = torch.load(weight_v4, map_location=torch.device('cuda'))
# model.load_state_dict(pretrained_dict)
# use_cuda = True
# if use_cuda:
#     model.cuda()

# img = cv2.imread(source)
# sized = cv2.resize(img, imgsz)
# sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)


# for i in range(2):  # This 'for' loop is for speed check
#                     # Because the first iteration is usually longer
#     boxes = models.do_detect(model, sized, 0.4, 0.6,use_cuda)
# class_names = models.load_class_names(namesfile_v4)
# bbox_yolov4 = models.plot_boxes_cv2(img, boxes[0], class_names)


# '''
# yolov5 bbox
# '''
device = yolov5.select_device(0) # cuda device, i.e. 0 or 0,1,2,3 or cpu
bboxes_v5 = yolov5.detect.run(imgsz,source,weight_v5, device=device, dnn=False)