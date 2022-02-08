from turtle import Turtle
import torch
from torch import nn
import torch.nn.functional as F
import sys
# sys.path.append("yolov4")
from yolov4.models import Yolov4
from yolov4.tool.torch_utils import *
from yolov4.tool.yolo_layer import YoloLayer
from yolov5.detect import run
from yolov5.utils.general import check_requirement





def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    # import sys
    # import cv2

    # namesfile = 'yolov4/data/Mask.names'

    # n_classes = 3
    # weightfile = 'yolov4/Yolov4_Mask_detection.pth'
    # imgfile = 'image/17.jpg'
    # height = 608
    # width = 608

    # model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    # pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    # model.load_state_dict(pretrained_dict)

    # use_cuda = True
    # if use_cuda:
    #     model.cuda()

    # img = cv2.imread(imgfile)
    # sized = cv2.resize(img, (width, height))
    # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    # from yolov4.tool.utils import load_class_names, plot_boxes_cv2
    # from yolov4.tool.torch_utils import do_detect

    # for i in range(2):  # This 'for' loop is for speed check
    #                     # Because the first iteration is usually longer
    #     boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

    # class_names = load_class_names(namesfile)
    # bbox_v4 =  plot_boxes_cv2(img, boxes[0], './17.jpg', class_names)

    bbox_v5 = parse_opt()
    opt = parse_opt()
    main(opt)