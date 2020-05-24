import cv2 as cv
import math
import time
import argparse
import glob
def gender_predict(frame,id):
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


        parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
        parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

        args = parser.parse_args()



        #ageProto = "/home/ncair/Downloads/people-counting-opencv/pyimagesearch/age_deploy.prototxt"
        #ageModel = "/home/ncair/Downloads/people-counting-opencv/pyimagesearch/age_net.caffemodel"

        #genderProto = "/home/ncair/Downloads/people-counting-opencv/pyimagesearch/gender_deploy.prototxt"
        #genderModel = "/home/ncair/Downloads/people-counting-opencv/pyimagesearch/gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(18-25)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']

        # Load network
        #ageNet = cv.dnn.readNet(ageModel, ageProto)
        #genderNet = cv.dnn.readNet(genderModel, genderProto)




        # Load the model
        #net = cv.dnn.readNet('/home/yogesh/Desktop/movidius-rpi-master/models/face-detection-adas-0001.xml', '/home/yogesh/Desktop/movidius-rpi-master/models/face-detection-adas-0001.bin')

        net = cv.dnn.readNet('/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml', '/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin')

        # Specify target device
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        if len(frame)!=0:
            #print(frame.shape)
            predictions = predict(frame, net)

            if len(predictions)!=0:
                return 0
            else:
                return 1
