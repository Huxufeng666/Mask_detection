from demo import  detect_cv2




cfgfile='yolov4/cfg/yolov4.cfg'
weightfile='yolov4/Yolov4_epoch300.pth'
imgfile='yolov4/Mdata/train/images/10.jpg'




bboxes_v4  =  detect_cv2(cfgfile, weightfile, imgfile)




if __name__ == '__main__':
    pass