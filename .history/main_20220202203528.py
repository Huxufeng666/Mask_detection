# from turtle import Turtle
# import torch
# from torch import nn
# import torch.nn.functional as F
# import sys
# from yolov4.tool.torch_utils import *
# from yolov4.tool.yolo_layer import YoloLayer


# class Mish(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x = x * (torch.tanh(torch.nn.functional.softplus(x)))
#         return x


# class Upsample(nn.Module):
#     def __init__(self):
#         super(Upsample, self).__init__()

#     def forward(self, x, target_size, inference=False):
#         assert (x.data.dim() == 4)
#         # _, _, tH, tW = target_size

#         if inference:

#             #B = x.data.size(0)
#             #C = x.data.size(1)
#             #H = x.data.size(2)
#             #W = x.data.size(3)

#             return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
#                     expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
#                     contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
#         else:
#             return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


# class Conv_Bn_Activation(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
#         super().__init__()
#         pad = (kernel_size - 1) // 2

#         self.conv = nn.ModuleList()
#         if bias:
#             self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
#         else:
#             self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
#         if bn:
#             self.conv.append(nn.BatchNorm2d(out_channels))
#         if activation == "mish":
#             self.conv.append(Mish())
#         elif activation == "relu":
#             self.conv.append(nn.ReLU(inplace=False))
#         elif activation == "leaky":
#             self.conv.append(nn.LeakyReLU(0.1, inplace=False))
#         elif activation == "linear":
#             pass
#         else:
#             print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
#                                                        sys._getframe().f_code.co_name, sys._getframe().f_lineno))

#     def forward(self, x):
#         for l in self.conv:
#             x = l(x)
#         return x


# class ResBlock(nn.Module):
#     """
#     Sequential residual blocks each of which consists of \
#     two convolution layers.
#     Args:
#         ch (int): number of input and output channels.
#         nblocks (int): number of residual blocks.
#         shortcut (bool): if True, residual tensor addition is enabled.
#     """

#     def __init__(self, ch, nblocks=1, shortcut=True):
#         super().__init__()
#         self.shortcut = shortcut
#         self.module_list = nn.ModuleList()
#         for i in range(nblocks):
#             resblock_one = nn.ModuleList()
#             resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
#             resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
#             self.module_list.append(resblock_one)

#     def forward(self, x):
#         for module in self.module_list:
#             h = x
#             for res in module:
#                 h = res(h)
#             x = x + h if self.shortcut else h
#         return x


# class DownSample1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

#         self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
#         self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
#         # [route]
#         # layers = -2
#         self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

#         self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
#         self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
#         # [shortcut]
#         # from=-3
#         # activation = linear

#         self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
#         # [route]
#         # layers = -1, -7
#         self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         # route -2
#         x4 = self.conv4(x2)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         # shortcut -3
#         x6 = x6 + x4

#         x7 = self.conv7(x6)
#         # [route]
#         # layers = -1, -7
#         x7 = torch.cat([x7, x3], dim=1)
#         x8 = self.conv8(x7)
#         return x8


# class DownSample2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
#         self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
#         # r -2
#         self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

#         self.resblock = ResBlock(ch=64, nblocks=2)

#         # s -3
#         self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
#         # r -1 -10
#         self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x1)

#         r = self.resblock(x3)
#         x4 = self.conv4(r)

#         x4 = torch.cat([x4, x2], dim=1)
#         x5 = self.conv5(x4)
#         return x5


# class DownSample3(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
#         self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
#         self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

#         self.resblock = ResBlock(ch=128, nblocks=8)
#         self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
#         self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x1)

#         r = self.resblock(x3)
#         x4 = self.conv4(r)

#         x4 = torch.cat([x4, x2], dim=1)
#         x5 = self.conv5(x4)
#         return x5


# class DownSample4(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
#         self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
#         self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

#         self.resblock = ResBlock(ch=256, nblocks=8)
#         self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
#         self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x1)

#         r = self.resblock(x3)
#         x4 = self.conv4(r)

#         x4 = torch.cat([x4, x2], dim=1)
#         x5 = self.conv5(x4)
#         return x5


# class DownSample5(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
#         self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
#         self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

#         self.resblock = ResBlock(ch=512, nblocks=4)
#         self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
#         self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

#     def forward(self, input):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x1)

#         r = self.resblock(x3)
#         x4 = self.conv4(r)

#         x4 = torch.cat([x4, x2], dim=1)
#         x5 = self.conv5(x4)
#         return x5


# class Neck(nn.Module):
#     def __init__(self, inference=False):
#         super().__init__()
#         self.inference = inference

#         self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
#         self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
#         self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
#         # SPP
#         self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

#         # R -1 -3 -5 -6
#         # SPP
#         self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
#         self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
#         self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
#         self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         # UP
#         self.upsample1 = Upsample()
#         # R 85
#         self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         # R -1 -3
#         self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
#         self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
#         self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
#         # UP
#         self.upsample2 = Upsample()
#         # R 54
#         self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
#         # R -1 -3
#         self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
#         self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
#         self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
#         self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
#         self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

#     def forward(self, input, downsample4, downsample3, inference=False):
#         x1 = self.conv1(input)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         # SPP
#         m1 = self.maxpool1(x3)
#         m2 = self.maxpool2(x3)
#         m3 = self.maxpool3(x3)
#         spp = torch.cat([m3, m2, m1, x3], dim=1)
#         # SPP end
#         x4 = self.conv4(spp)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x7 = self.conv7(x6)
#         # UP
#         up = self.upsample1(x7, downsample4.size(), self.inference)
#         # R 85
#         x8 = self.conv8(downsample4)
#         # R -1 -3
#         x8 = torch.cat([x8, up], dim=1)

#         x9 = self.conv9(x8)
#         x10 = self.conv10(x9)
#         x11 = self.conv11(x10)
#         x12 = self.conv12(x11)
#         x13 = self.conv13(x12)
#         x14 = self.conv14(x13)

#         # UP
#         up = self.upsample2(x14, downsample3.size(), self.inference)
#         # R 54
#         x15 = self.conv15(downsample3)
#         # R -1 -3
#         x15 = torch.cat([x15, up], dim=1)

#         x16 = self.conv16(x15)
#         x17 = self.conv17(x16)
#         x18 = self.conv18(x17)
#         x19 = self.conv19(x18)
#         x20 = self.conv20(x19)
#         return x20, x13, x6


# class Yolov4Head(nn.Module):
#     def __init__(self, output_ch, n_classes, inference=False):
#         super().__init__()
#         self.inference = inference

#         self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
#         self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

#         self.yolo1 = YoloLayer(
#                                 anchor_mask=[0, 1, 2], num_classes=n_classes,
#                                 anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
#                                 num_anchors=9, stride=8)

#         # R -4
#         self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

#         # R -1 -16
#         self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
#         self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
#         self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
#         self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
#         self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
#         self.yolo2 = YoloLayer(
#                                 anchor_mask=[3, 4, 5], num_classes=n_classes,
#                                 anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
#                                 num_anchors=9, stride=16)

#         # R -4
#         self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

#         # R -1 -37
#         self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
#         self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
#         self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
#         self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
#         self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
#         self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
#         self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
#         self.yolo3 = YoloLayer(
#                                 anchor_mask=[6, 7, 8], num_classes=n_classes,
#                                 anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
#                                 num_anchors=9, stride=32)

#     def forward(self, input1, input2, input3):
#         x1 = self.conv1(input1)
#         x2 = self.conv2(x1)

#         x3 = self.conv3(input1)
#         # R -1 -16
#         x3 = torch.cat([x3, input2], dim=1)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x7 = self.conv7(x6)
#         x8 = self.conv8(x7)
#         x9 = self.conv9(x8)
#         x10 = self.conv10(x9)

#         # R -4
#         x11 = self.conv11(x8)
#         # R -1 -37
#         x11 = torch.cat([x11, input3], dim=1)

#         x12 = self.conv12(x11)
#         x13 = self.conv13(x12)
#         x14 = self.conv14(x13)
#         x15 = self.conv15(x14)
#         x16 = self.conv16(x15)
#         x17 = self.conv17(x16)
#         x18 = self.conv18(x17)
        
#         if self.inference:
#             y1 = self.yolo1(x2)
#             y2 = self.yolo2(x10)
#             y3 = self.yolo3(x18)

#             return get_region_boxes([y1, y2, y3])
        
#         else:
#             return [x2, x10, x18]


# class Yolov4(nn.Module):
#     def __init__(self, yolov4conv137weight=None, n_classes=3, inference=False):
#         super().__init__()

#         output_ch = (4 + 1 + n_classes) * 3

#         # backbone
#         self.down1 = DownSample1()
#         self.down2 = DownSample2()
#         self.down3 = DownSample3()
#         self.down4 = DownSample4()
#         self.down5 = DownSample5()
#         # neck
#         self.neck = Neck(inference)
#         # yolov4conv137
#         if yolov4conv137weight:
#             _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
#             pretrained_dict = torch.load(yolov4conv137weight)

#             model_dict = _model.state_dict()
#             # 1. filter out unnecessary keys
#             pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
#             # 2. overwrite entries in the existing state dict
#             model_dict.update(pretrained_dict)
#             _model.load_state_dict(model_dict)
        
#         # head
#         self.head = Yolov4Head(output_ch, n_classes, inference)


#     def forward(self, input):
#         d1 = self.down1(input)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         d5 = self.down5(d4)

#         x20, x13, x6 = self.neck(d5, d4, d3)

#         output = self.head(x20, x13, x6)
#         return output


# if __name__ == "__main__":
#     import sys
#     import cv2

#     namesfile = 'yolov4/data/Mask.names'

#     n_classes = 3
#     weightfile = 'yolov4/Yolov4_Mask_detection.pth'
#     imgfile = 'image/17.jpg'
#     height = 608
#     width = 608


#     model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

#     pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
#     model.load_state_dict(pretrained_dict)

#     use_cuda = True
#     if use_cuda:
#         model.cuda()

#     img = cv2.imread(imgfile)
#     sized = cv2.resize(img, (width, height))
#     sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

#     from yolov4.tool.utils import load_class_names, plot_boxes_cv2
#     from yolov4.tool.torch_utils import do_detect

#     for i in range(2):  # This 'for' loop is for speed check
#                         # Because the first iteration is usually longer
#         boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

#     class_names = load_class_names(namesfile)
#     bbox_v4 =  plot_boxes_cv2(img, boxes[0], './17.jpg', class_names)



"""
yolov5 bbox 

"""


import argparse
import os
import sys
sys.path.append('yolov5')
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

from ensemble_boxes import *
import numpy as np

@torch.no_grad()
def run(weights=ROOT / './runs/train/exp8/weights/last.pt',  # model.pt path(s)
        source=ROOT / './data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
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
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # save_dir = 'text.txt'
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)    
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                        # with open('text.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5/runs/train/exp9/weights/last.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'image/14.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[608], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', default='True', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', default='', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', default='', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default='', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":

    opt = parse_opt()
    main(opt)
