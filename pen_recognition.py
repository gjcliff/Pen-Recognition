#!/usr/bin/python

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

def nothing(x):
    pass

#opencv
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('H','image',270,360, nothing)
cv2.createTrackbar('S','image',255,255, nothing)
cv2.createTrackbar('V','image',123,255, nothing)
cv2.createTrackbar('delta','image',10,50, nothing)

# Streaming loop
try:
    while True:
        # get current positions of four trackbars
        h = cv2.getTrackbarPos('H','image') / 2
        s = cv2.getTrackbarPos('S','image')
        v = cv2.getTrackbarPos('V','image')
        d = cv2.getTrackbarPos('delta','image')

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # convert the BGR image to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([h-d,s,v])
        upper_bound = np.array([h+d,s,v])

        print(f"lower bound: {lower_bound}, upper bound: {upper_bound}")

        # threshold the HSV image to get purple colors within our range
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # bitwise-AND mask and original image
        masked_image = cv2.bitwise_and(color_image, color_image, mask= mask)

        cv2.imshow('original', color_image)
        cv2.imshow('mask', mask)
        cv2.imshow('image', masked_image)
        key = cv2.waitKey(1) & 0xFF
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        

finally:
    pipeline.stop()
    cv2.destroyAllWindows()