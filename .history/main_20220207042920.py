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
def NMS(dects,threshhold):
    """
    detcs:二维数组(n_samples,5)
    5列：x1,y1,x2,y2,score
    threshhold: IOU阈值
    """
    x1=dects[:,0]
    y1=dects[:,1]
    x2=dects[:,2]
    y2=dects[:,3]
    score=dects[:,4]
    ndects=dects.shape[0]#box的数量
    area=(x2-x1+1)*(y2-y1+1)
    order=score.argsort()[::-1] #score从大到小排列的indexs,一维数组
    keep=[] #保存符合条件的index
    suppressed=np.array([0]*ndects) #初始化为0，若大于threshhold,变为1，表示被抑制
    
    for _i in range(ndects):
        i=order[_i]  #从得分最高的开始遍历
        if suppressed[i]==1:
            continue
        keep.append(i) 
        for _j in range(i+1,ndects):
            j=order[_j]
            if suppressed[j]==1: #若已经被抑制，跳过
                continue
            xx1=np.max(x1[i],x1[j])#求两个box的交集面积interface
            yy1=np.max(y1[i],y1j])
            xx2=np.min(x2[i],x2[j])
            yy2=np.min(y2[i],y2[j])
            w=np.max(0,xx2-xx1+1)
            h=np.max(0,yy2-yy1+1)
            interface=w*h
            overlap=interface/(area[i]+area[j]-interface) #计算IOU（交/并）
            
            if overlap>=threshhold:#IOU若大于阈值，则抑制
                suppressed[j]=1
    return keep

if __name__ == "__main__":
   

# yolov4 bbox

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        bbox = do_detect(model, sized, 0.4, 0.6, use_cuda)

    class_names = load_class_names(namesfile)
    boxes =  plot_boxes_cv2(img, bbox[0], 'runs/56.jpg',None)


    
    # bbox_v4 = np.array(bbox).reshape(len(boxes),1,6)
    bbox_v4 = np.array(bbox)
    b_4=[i[:5] for item in bbox_v4 for i in item]
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
    b_5=[i[:5] for item in bbox_v5 for i in item]
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


                                            

boxes, scores= NMS(boxes_list,0.9)

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
