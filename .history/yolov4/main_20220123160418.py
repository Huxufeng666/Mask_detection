from demo import  detect_cv2




cfgfile='/home/user/Mask_detection/pytorch-YOLOv4/cfg/yolov4_copy.cfg'
weightfile='pytorch-YOLOv4/yolov4-p6.weights'
imgfile='pytorch-YOLOv4/data/dog.jpg'




bboxes_v4  =  detect_cv2(cfgfile, weightfile, imgfile)




if __name__ == '__main__':
    pass