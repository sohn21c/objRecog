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
from imutils.video import VideoStream
from imutils.video import FPS

# global variable declarations

# class definitions

# function definitions

# main 


# initiating the depth camera
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# argument parsing for weight and model
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0,255, size=(len(CLASSES), 3))

# load the serialized model from disk
print("[INFO] loading model... ")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] Caffe model loaded... ")


# try:
while True:
    
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # convert the frame into blob
    (h, w) = color_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300,300)), 0.00743, (300,300), 127.5)

    # obtain detection
    net.setInput(blob)
    detections = net.forward()

    # loop over detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0,0,i,2]
        # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the predictions on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(color_image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY > 30 else startY + 15
            cv2.putText(color_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2) 

    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    key = cv2.waitKey(1)

    # kill the windown when 'q' is pressed
    if key == ord("q"):
        break

# finally:

    # Stop streamin
pipeline.stop()