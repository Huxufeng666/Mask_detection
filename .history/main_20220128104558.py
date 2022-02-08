import sys
from turtle import width
import cv2
import torch
import ensemble_boxes 
from matplotlib import image
from numpy import source
# sys.path.append('yolov4')
from yolov4 import models
from pyexpat import model
# import yolov5
# from yolov5.models import common
# from yolov5 import detect
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.general import clip_coords,scale_coords,increment_path,non_max_suppression



source='yolov4/Mdata/train/images/7.jpg'
namesfile_v4='Mask_detection/yolov4/data/Mask.names' 
weight_v4 = 'yolov4/Yolov4_Mask_detection.pth'
n_classes=3
weight_v5 ='yolov5/runs/train/exp9/weights/best.pt'
imgsz = 608,608
onf_thres=''
iou_thres=''
classes=''
agnostic_nms=''

'''
 yolov4 bbox
'''
model = models.Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=False)
pretrained_dict = torch.load(weight_v4, map_location=torch.device('cuda'))
model.load_state_dict(pretrained_dict)
use_cuda = True
if use_cuda:
    model.cuda()

img = cv2.imread(source)
sized = cv2.resize(img,imgsz )
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)


for i in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
    boxes = models.do_detect(model, sized, 0.4, 0.6,use_cuda)
class_names = models.load_class_names(namesfile_v4)
bbox_yolov4 = models.plot_boxes_cv2(img, boxes[0],'./predictions.jpg', class_names)


#  '''
# yolov5 bbox
# '''

# # Inference
# visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
# pred = model(im, augment=augment, visualize=visualize)
# # t3 = time_sync()
# # dt[1] += t3 - t2

# # NMS
# dt, seen = [0.0, 0.0, 0.0], 0
# pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=3, agnostic_nms=False, max_det=1000)    
# dt[2] += time_sync() - t3

# # Second-stage classifier (optional)
# # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

# # Process predictions
# for i, det in enumerate(pred):  # per image
#     seen += 1
#     if webcam:  # batch_size >= 1
#         p, im0, frame = path[i], im0s[i].copy(), dataset.count
#         s += f'{i}: '
#     else:
#         p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

#     p = Path(p)  # to Path
#     save_path = str(save_dir / p.name)  # im.jpg
#     txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
#     s += '%gx%g ' % im.shape[2:]  # print string
#     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#     imc = im0.copy() if save_crop else im0  # for save_crop
#     annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#     if len(det):
#         # Rescale boxes from img_size to im0 size
#         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()



# device = select_device(0) # cuda device, i.e. 0 or 0,1,2,3 or cpu
# model = DetectMultiBackend(weights, device=device, dnn=dnn)
# bboxes_v5 = detect.run(weight_v5,source,imgsz, device=device, dnn=False)




# # if len(det):
# #         # Rescale boxes from img_size to im0 size
# #         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()


#  det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
