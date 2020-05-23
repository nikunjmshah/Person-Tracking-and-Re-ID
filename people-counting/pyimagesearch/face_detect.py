import cv2 as cv
import math
import time
import argparse
import glob



def face(frame):
    net = cv.dnn.readNet('/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml', '/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin')

    # Specify target device
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    none_type=None

    def predict(frame, net):
        # Prepare input blob and perform an inference
        blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
        net.setInput(blob)
        out = net.forward()

        predictions = []

        # Draw detected faces on the frame
        for detection in out.reshape(-1, 7):
            conf = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if conf > 0.5:
                pred_boxpts = ((xmin, ymin), (xmax, ymax))

                # create prediciton tuple and append the prediction to the
                # predictions list
                prediction = (conf, pred_boxpts)
                predictions.append(prediction)

        # return the list of predictions to the calling function
        return predictions
    predictions = predict(frame, net)
    if len(predictions)!=0:
        #print(frame.shape)
        #rect=[]
        for (i, pred) in enumerate(predictions):
            # extract prediction data for readability
            (pred_conf, pred_boxpts) = pred
            (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
            rect=[ptA[0], ptA[1],ptB[0], ptB[1]]
        return rect
    else:
        return 0
