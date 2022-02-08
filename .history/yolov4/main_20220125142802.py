from demo import  detect_cv2




cfgfile='./cfg/yolov4_copy.cfg'
weightfile='./yolov4/yolov4-p6.weights'
imgfile='./yolov4/data/dog.jpg'




bboxes_v4  =  detect_cv2(cfgfile, weightfile, imgfile)




if __name__ == '__main__':
    pass