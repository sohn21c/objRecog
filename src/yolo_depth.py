#!usr/bin/env python3

"""
yolo_depth.py
Author: James Sohn
"""

# importing the necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import imutils
import time
import matplotlib.pyplot as plt 
import os

# global variable declarations

# class definitions

# function definitions

# main 

# initiating the depth camera
pipeline = rs.pipeline()
config = rs.config()
# configure depth and color stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# start streaming D435 RGB-D camera
pipeline.start(config)
# pause for 2 seconds for warmup
time.sleep(2)
print ("[INFO] Camera streaming started...")

# argument parsing for weight and model
# default confidence for detection is set to 0,5 and threshold for detection is defaulted to be 0.3. Both can be reconfigured at runtime
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO dir")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppresion")
args = vars(ap.parse_args())

# load the label of trained model from MS-COCO data set.
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
copyLABELS = LABELS.copy()
copyLABELS.sort()
print ("[INFO] Object categories model is trained on: \n", copyLABELS)

# use numpy random seed to generate same set of colors for labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# pahts to the pretrained YOLO model weights and configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load pretrained YOLO model on MS-COCO data and feed into opencv DNN module
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("[INFO] YOLO model loaded")

# process the frame and perform detection
while True:
    
    # time step for FPS calculation
    start = time.time()

    # wait for depth and image frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # convert the frame into blob for YOLO
    (h, w) = color_image.shape[:2]
    blob = cv2.dnn.blobFromImage(color_image, 1 / 255.0, (416,416), swapRB=True, crop=False)

    # obtain detection
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over the detection
        for detection in output:
            # extract the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability is greater than min. probability
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppressions to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():

            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(color_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # find the center of the box
            center = np.array([(x + (x + w))/2, (y + (y + h))/2])
            (centerX, centerY) = center.astype("int")
            coord = "{:.2f}, {:.2f}".format(centerX, centerY)

            # find the depth at the center of the detection
            depth = depth_image[centerY, centerX]
            depth = depth.astype("int")
            depth_meter = "{:.2f}m".format(depth/1000)
            cv2.putText(color_image, depth_meter, (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    
    # press 'q' to kill the window
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    # time step for FPS calculation    
    end = time.time()
    print ("Video frame rate is {:.2f}".format(1/(end - start)))

pipeline.stop()