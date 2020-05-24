# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

# from scipy.misc import imread
import cv2

import sys, os
sys.path.append('/home/nms/darknet/python/')

import darknet as dn

def generate_boxes_confidences_classids(outs, tconf):
    boxes = []
    confidences = []
    classids = []
    for detection in outs:
        # print(detection)
        confidence = detection[1]
        classid = detection[0]
        # confidence = scores[classid]
        if confidence > tconf:
            if classid != b'bicycle':
               continue

            box = detection[2][0:4]
            centerX, centerY, bwidth, bheight = box
            x = int(centerX - (bwidth / 2))
            y = int(centerY - (bheight / 2))
            boxes.append([x, y, int(bwidth), int(bheight)])
            confidences.append(float(confidence))
            classids.append(classid)
    return boxes, confidences, classids






# Darknet
net = dn.load_net(b"yolo/yolov2.cfg", b"yolo/yolov2.weights", 0)
# print("hi1")
meta = dn.load_meta(b"yolo/coco.data")
# print("hi2")
r = dn.detect(net, meta, b"yolo/dog.jpg")
# print("hi3")
# print (r)

# scipy
# arr= imread('data/dog.jpg')
# im = array_to_image(arr)
# r = detect2(net, meta, im)
# print (r)

# OpenCV
arr = cv2.imread('yolo/dog.jpg')

r = dn.detect(net, meta, cv2.imread('yolo/dog.jpg'))
# print (r)
boxes, confidences, classids = generate_boxes_confidences_classids(r, 0.5)
print(boxes)
print(confidences)
print(classids)

if len(boxes)>0:
    for i in range(len(boxes)):
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        cv2.rectangle(arr, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('arr', arr)
cv2.waitKey(0)