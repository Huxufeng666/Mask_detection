from turtle import Turtle
from numpy import source
import torch
import numpy
from torch import nn
import torch.nn.functional as F
import sys
import cv2
# sys.path.append("yolov4")
from yolov4.models import Yolov4
from yolov4.tool.torch_utils import *
from yolov4.tool.yolo_layer import YoloLayer
from yolov5.detect import *
from yolov5.utils.general import *
from yolov4.tool.utils import load_class_names, plot_boxes_cv2
from yolov4.tool.torch_utils import do_detect, get_region_boxes
import argparse
import os
from pathlib import Path
import torch.backends.cudnn as cudnn

'''
yolov4
'''
namesfile = 'yolov4/data/Mask.names'
n_classes = 3
weights_v4 = 'yolov4/Yolov4_Mask_detection.pth'
weights_v5 = 'yolov5/runs/train/exp9/weights/last.pt'
source = 'image/17.jpg'
imgsz = (608,608)
model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
pretrained_dict = torch.load(weights_v4, map_location=torch.device('cuda'))
model.load_state_dict(pretrained_dict)
use_cuda = True
device = ''

if use_cuda:
    model.cuda()
img = cv2.imread(source)
sized = cv2.resize(img, imgsz)
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

'''
yolov5
'''

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


if __name__ == "__main__":
   

# yolov4 bbox

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        bbox = do_detect(model, sized, 0.4, 0.6, use_cuda)

    class_names = load_class_names(namesfile)
    boxes =  plot_boxes_cv2(img, bbox[0], None, None)

    bbox_v4 =np.array(boxes)
    # region_yolov4 = get_region_boxes()   #坐标
    # b=[x[0] for x in bbox]
    # print(bbox_v4)
    
# yolov5 bbox

    bbox = run(weights_v5=weights_v5,  # model.pt path(s)
        source=source,  # file/dir/URL/glob, 0 for webcam
        imgsz=imgsz,  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/',  # save results to project/name
        name='yolov5',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        )
    bbox_v5 = bbox.cpu().detach().numpy()
    # bbox_v5 = np.array(bbox)
    # print(bbox_v5)

# merged


    # bbox_merged =  np.append(bbox_v4, bbox_v5, axis= 0)   
    # bbox_merged = bbox_merged.tolist()
    # b_v4=torch.from_numpy(bbox_v4)[0:1]
    b_v4=[x[0:3]  for x in torch.from_numpy(bbox_v4)]
    # bbox_merged = bbox_ious(b_v4,b_v5)             
    print(b_v4)


# #绘图
# img = cv2.imread(str(source))
# h, w = img.shape[:2]
# for _, x in enumerate(bbox_merged):

#     cv2.rectangle(img,(int(x[0]),int(x[1])),(int(x[2]),int(x[3])),(0, 255, 0) )
#     cv2.putText(img,None,(int(x[0]), int(x[1] - 2)),fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0, 0, 255),thickness=1)

# cv2.imwrite('./17.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
