from turtle import Turtle
from numpy import source
import torch
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
from yolov4.tool.torch_utils import do_detect
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
weights_v5 = '/home/user/Mask_detection/yolov5/runs/train/exp9/weights/last.pt'
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

@torch.no_grad()
def run(weights_v5=ROOT / 'yolov5/runs/train/exp9/weights/last.pt',  # model.pt path(s)
        source=ROOT / 'image/17.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=(608, 608),  # inference size (height, width)
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
    model = DetectMultiBackend(weights_v5, device=device, dnn=dnn)
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
        strip_optimizer(weights_v5)  # update model (to fix SourceChangeWarning)

    return det[:, :4]

# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights_v5', nargs='+', type=str, default=ROOT / '/yolov5/runs/train/exp9/weights/last.pt', help='model path(s)')
#     parser.add_argument('--source', type=str, default=ROOT / 'image/17.jpg', help='file/dir/URL/glob, 0 for webcam')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[608], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', default='True', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', default='', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', default='', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', default='', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(FILE.stem, opt)
#     return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))




if __name__ == "__main__":
   
    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        bbox_v4 = do_detect(model, sized, 0.4, 0.6, use_cuda)

    class_names = load_class_names(namesfile)
    boxes_v4 =  plot_boxes_cv2(img, bbox_v4[0], None, class_names)

    print(boxes_v4)

    
# yolov5 bbox

    opt = parse_opt()
    main(parse_opt)

    bbox_v5 = run()
 
    print(bbox_v5)