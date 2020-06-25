# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import requests
import os
import math
import shutil

# remove and create contacts
shutil.rmtree('contacts/', ignore_errors=True)
os.mkdir('contacts/')

import sys, os
sys.path.append('../darknet_old/python/')
import darknet as dn

# construct the argument parse and parse the arguments
# '''
ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, default="videos/sample1.mp4",
 	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
# ap.add_argument("-c", "--confidence", type=float, default=0.4,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-s", "--skip-frames", type=int, default=30,
# 	help="# of skip frames between detections")
args = vars(ap.parse_args())
# '''
model_cfg=b'yolo/yolov3.cfg'
model_wgt=b'yolo/yolov3.weights'
#model_cfg='/home/ncair/Downloads/people-counting-opencv/yolov3-coco/frozen_tiny_yolo_v3.xml'
#model_wgt='/home/ncair/Downloads/people-counting-opencv/yolov3-coco/frozen_tiny_yolo_v3.bin'

#input='/home/ncair/Desktop/Counter/people.mp4'
input="rtsp://admin:transit@123@10.185.151.213/"
input="/home/nms/people-counting/videos/sample1.mp4"
input=args["input"]
thresh=0.5
skip_frames=5
nms_thresh = 0.3
max_width_resize = 600
max_disappeared = 30
max_distance = 40
contact_limit = 100

def euclid_dist(centroid1, centroid2):
	return(math.sqrt((centroid1[0]-centroid2[0])**2 + (centroid1[1]-centroid2[1])**2))


# load our serialized model from disk
print("[INFO] loading model...")

net = dn.load_net(model_cfg, model_wgt, 0)
meta = dn.load_meta(b"yolo/coco.data")


# if a video path was not supplied, grab a reference to the webcam

vs = cv2.VideoCapture(input)

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared = max_disappeared, maxDistance = max_distance)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

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
            if classid != b'person':
               continue

            box = detection[2][0:4]
            centerX, centerY, bwidth, bheight = box
            x = int(centerX - (bwidth / 2))
            y = int(centerY - (bheight / 2))
            boxes.append([x, y, int(bwidth), int(bheight)])
            confidences.append(float(confidence))
            classids.append(classid)
    return boxes, confidences, classids
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
no_frame_count = 0

while True:
	totalFrames += 1
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	grab,frame = vs.read()
	if grab == False:
		no_frame_count += 1
		if no_frame_count > 10:
			break

	if grab == True:
		frame = frame #if args.get("input", False) else frame
		#frame = frame[100:500, 250:704]
		#print(frame.shape[1])
		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video


		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib

		frame = imutils.resize(frame, width = max_width_resize)

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		height, width = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)
		
		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % skip_frames == 0 :
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to required size and pass through the
			# network and obtain the detections
			outs = dn.detect(net, meta, frame, thresh, 0.5, nms_thresh)

			boxes = []
			confidences =[]
			classids =[]
			boxes, confidences, classids = generate_boxes_confidences_classids(outs, 0.5)
			
			#idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
			
			if len(boxes)>0:
				# for i in idxs.flatten():
				for i in range(len(boxes)):
					x, y = boxes[i][0], boxes[i][1]
					w, h = boxes[i][2], boxes[i][3]


					cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(x,y,x+w,y+w)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))			

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects,frame)

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		# cv2.line(frame, (0, int(H // 4)), (W, int(H // 4)), (0, 255, 255), 1)
		# cv2.line(frame, (0, int(H // 1.25)), (W, int(H // 1.25)), (0, 0, 255), 1)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():

			# contact tracing
			for (objectID_2, centroid_2) in objects.items():
				dist = euclid_dist(centroid, centroid_2)
				if dist < contact_limit and objectID_2 != objectID:
					cv2.line(frame, (centroid[0], centroid[1]), (centroid_2[0], centroid_2[1]), (255, 0, 0), 2)
					f=open("contacts/" + str(objectID) + "_contact.txt", "a+")
					f.write("Contact object ID: " + str(objectID_2) + " Frame no: " + str(totalFrames) + "\r\n")


			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is above the center
					# line, count the object
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						#data="http://18.191.211.159/counter.php?Store_Id="+str(2)+"&upcount="+str(totalUp)+"&downcount="+str(totalDown)
						#r = requests.get(url = data)
						#print(r)
						to.counted = True

					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, count the object
					elif direction > 0 and centroid[1] > H // 2:
							totalDown += 1
							#data="http://18.191.211.159/counter.php?Store_Id="+str(2)+"&upcount="+str(totalUp)+"&downcount="+str(totalDown)
							#r = requests.get(url = data)
							#print(r)
							to.counted = True

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		# construct a tuple of information we will be displaying on the
		# frame
		info = [
			#("Up", totalUp),
			#("Down", totalDown),
			("Status", status),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)
		
		# show the output frame
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		#	 break

		# increment the total number of frames processed thus far and
		# then update the FPS counter

		fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

vs.release()
cv2.destroyAllWindows()
