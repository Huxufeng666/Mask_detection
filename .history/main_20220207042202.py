from turtle import Turtle
from webbrowser import Elinks
from numpy import source
import torch
import numpy
from torch import nn
import torch.nn.functional as F
import sys
import cv2
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

from ensemble_boxes import *


'''
yolov4
'''
namesfile = 'yolov4/data/Mask.names'
n_classes = 3
weights_v4 = 'yolov4/Yolov4_Mask_detection.pth'
weights_v5 = 'yolov5/runs/train/exp8/weights/last.pt'
source = 'image/56.jpg'
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

def non_max_suppress(predicts_dict, threshold=0.2):
    """
    implement non-maximum supression on predict bounding boxes.
    Args:
        predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
        threshhold: iou threshold
    Return:
        predicts_dict processed by non-maximum suppression
    """
    for bbox,object_name, in predicts_dict.items():   #对每一个类别的目标分别进行NMS
        bbox_array = np.array(bbox, dtype=np.float)
 
        ## 获取当前目标类别下所有矩形框（bounding box,下面简称bbx）的坐标和confidence,并计算所有bbx的面积
        x1, y1, x2, y2, scores = bbox_array[:,0], bbox_array[:,1], bbox_array[:,2], bbox_array[:,3], bbox_array[:,4]
        areas = (x2-x1+1) * (y2-y1+1)
        #print("areas shape = ", areas.shape)
 
        ## 对当前类别下所有的bbx的confidence进行从高到低排序（order保存索引信息）
        order = scores.argsort()[::-1]
        print("order = ", order)
        keep = [] #用来存放最终保留的bbx的索引信息
 
        ## 依次从按confidence从高到低遍历bbx，移除所有与该矩形框的IOU值大于threshold的矩形框
        while order.size > 0:
            i = order[0]
            keep.append(i) #保留当前最大confidence对应的bbx索引
 
            ## 获取所有与当前bbx的交集对应的左上角和右下角坐标，并计算IOU（注意这里是同时计算一个bbx与其他所有bbx的IOU）
            xx1 = np.maximum(x1[i], x1[order[1:]]) #当order.size=1时，下面的计算结果都为np.array([]),不影响最终结果
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2-xx1+1) * np.maximum(0.0, yy2-yy1+1)
            iou = inter/(areas[i]+areas[order[1:]]-inter)
            print("iou =", iou)
 
            print(np.where(iou<=threshold)) #输出没有被移除的bbx索引（相对于iou向量的索引）
            indexs = np.where(iou<=threshold)[0] + 1 #获取保留下来的索引(因为没有计算与自身的IOU，所以索引相差１，需要加上)
            print("indexs = ", type(indexs))
            order = order[indexs] #更新保留下来的索引
            print("order = ", order)
        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()
        predicts_dict = predicts_dict
    return predicts_dict





if __name__ == "__main__":
   

# yolov4 bbox

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        bbox = do_detect(model, sized, 0.4, 0.6, use_cuda)

    class_names = load_class_names(namesfile)
    boxes =  plot_boxes_cv2(img, bbox[0], 'runs/56.jpg',None)


    
    # bbox_v4 = np.array(bbox).reshape(len(boxes),1,6)
    bbox_v4 = np.array(bbox)
    b_4=[i[:6] for item in bbox_v4 for i in item]
    s_4=[i[4:5] for item in bbox_v4 for i in item] 
    l_4=[i[5:6] for item in bbox_v4 for i in item] 



    print(s_4)
    
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
    # bbox_v5 = bbox.cpu().detach().numpy()
    bbox_v5 =np.array(bbox)
    b_5=[i[:6] for item in bbox_v5 for i in item]
    s_5=[i[4:5] for item in bbox_v5 for i in item] 
    l_5=[i[5:6] for item in bbox_v5 for i in item] 


    



# merged

    bbox_merged_2 =  np.append(b_4, b_5, axis= 0)   
    boxes_list= np.append(b_4, b_5, axis= 0)
    scores_list=np.append(s_4, s_5, axis= 0) 
    labels_list=np.append(l_4, l_5, axis= 0) 

    # bbox_merged_2 =  bbox_v4.append([bbox_v5])   
  
    # b_v4=[]
    # b_v5=[]
    # bbox_merged = [] 
    # for x in torch.from_numpy(bbox_v4):
       
    #     b_v4=x[0],x[1],x[2],x[3]
     
    #     mul_v4 = (x[2]-x[0]) * (x[3]-x[1])  

    # for y in torch.from_numpy(bbox_v5):

    #     b_v5=y[0],y[1],y[2],y[3]  
    
    #     mul_v5 = (y[2]-y[0]) * (y[3]-y[1])

    # if mul_v4 > mul_v5:
    #    bbox_merged.append([y[0],y[1],y[2],y[3]])
    # else:
    #     bbox_merged.append([x[0],x[1],x[2],x[3]])
    
    # if len(bbox_v5)> len(bbox_v4):
    #    bbox_merged=bbox_v5 
    
    # elif mul_v4 > mul_v5:
    #    bbox_merged=bbox_v5
    # else:
    #     bbox_merged=bbox_v4
    
    
    # bbox_merged = [] 
    # for x in bbox_merged_2:
    #     # if not x in bbox_merged :
    #     if  x.all() != bbox_merged.all() :

    #         bbox_merged.append(x)
 

    # print(scores_list)


# ensenle_boxes


                                            

boxes, scores= non_max_suppress(boxes_list)

print(boxes)


# box to image

# img = cv2.imread(str(source))
# h, w = img.shape[:2]
# for _, x in enumerate(bbox_v4):

#     cv2.rectangle(img,(int(x[0]),int(x[1])),(int(x[2]),int(x[3])),(0,255,0),2 )
#     cv2.putText(img,None,(int(x[0]), int(x[1] - 2)),fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0, 0),thickness=1)

# cv2.imwrite('./text1.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
