import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Create align object
align = rs.align(rs.stream.color)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth and color frames
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Convert depth and color frames to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        lower_blue = np.array([70, 0, 0])
        upper_blue = np.array([240, 255, 255])
        depth_image = np.asanyarray(depth_image)
        # depth_image = cv2.resize(np.asanyarray(self.frame.get_data()),(960,540))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.18), cv2.COLORMAP_JET)
        hsv2 = cv2.cvtColor(depth_colormap,cv2.COLOR_BGR2HSV)
        mask_red = cv2.medianBlur(cv2.inRange(hsv2, lower_blue, upper_blue),1)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("mask",mask_red)
        # Display the frames
        cv2.imshow('Aligned Depth Frame', depth_image)
        cv2.imshow('Aligned Color Frame', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close all OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()