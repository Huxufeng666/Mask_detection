
import yolov4
import yolov5


cfgfile=''
weightfile=''
imgfile=''



bboxes_v4 = yolov4.models.detect_cv2(cfgfile, weightfile, imgfile)
bboxes_v5 = yolov5.detect