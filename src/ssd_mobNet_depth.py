#!usr/bin/env python3

"""
ssd_mobNet_depth.py
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

# global variable declarations

# class definitions

# function definitions

# main 


# initiating the depth camera
pipeline = rs.pipeline()
config = rs.config()
# configure depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# start streaming
pipeline.start(config)

# argument parsing for weight and model
# default confidence for detection is set to 0,5 and threshold for detection is defaulted to be 0.3. Both can be reconfigured at runtime
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the name of classes for classification and the colors for bounding boxes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "monitor"]
COLORS = np.random.uniform(0,255, size=(len(CLASSES), 3))

# loading the SSD model via opencv DNN module with Caffee backend
print("[INFO] loading model... ")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] Caffe model loaded... ")

# process the frames and perform detection
while True:
    
    # wait for depth and image rames
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

    # convert the frame into blob for SSD
    (h, w) = color_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300,300)), 0.00743, (300,300), 127.5)

    # obtain detection
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        # extract the class ID and confidence of the current object detection
        confidence = detections[0,0,i,2]
        # filter out weak predictions by ensuring the detected probability is greater than minimum value
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the predictions on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(color_image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY > 30 else startY + 15
            cv2.putText(color_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2) 

            # find the center of the box
            center = np.array([(startX + endX)/2, (startY + endY)/2])
            (centerX, centerY) = center.astype("int")
            coord = "{:.2f}, {:.2f}".format(centerX, centerY)
            # cv2.putText(color_image, coord, (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # find the depth at the center of the detection
            depth = depth_image[centerY, centerX]
            depth = depth.astype("int")
            depth_meter = "{:.2f}m".format(depth/1000)
            cv2.putText(color_image, depth_meter, (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            

    # stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)

    # press 'q' to kill the window
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

pipeline.stop()