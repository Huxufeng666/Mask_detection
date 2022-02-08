from turtle import Turtle
import torch
from torch import nn
import torch.nn.functional as F
import sys
# sys.path.append("yolov4")
from yolov4.models import Yolov4
from yolov4.tool.torch_utils import *
from yolov4.tool.yolo_layer import YoloLayer
from yolov5.detect import *
from yolov5.utils.general import *





def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    import sys
    import cv2

    namesfile = 'yolov4/data/Mask.names'

    n_classes = 3
    weightfile = 'yolov4/Yolov4_Mask_detection.pth'
    imgfile = 'image/17.jpg'
    height = 608
    width = 608

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from yolov4.tool.utils import load_class_names, plot_boxes_cv2
    from yolov4.tool.torch_utils import do_detect

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        bbox_v4 = do_detect(model, sized, 0.4, 0.6, use_cuda)

    class_names = load_class_names(namesfile)
    boxes_v4 =  plot_boxes_cv2(img, bbox_v4[0], None, class_names)


    get_boxv4= []
    for i in bbox_v4:
        get_boxv4 = i[0:4,:]

    print(get_boxv4)

    


# yolov5 bbox

    # bbox_v5 = parse_opt()
    # opt = parse_opt()
    # main(opt)
