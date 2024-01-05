import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
pipeline = rs.pipeline()
distance = 0
frame_count = 0
start_time = time.time()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1)  # Enable infrared stream if needed
config.enable_stream(rs.stream.infrared, 2)  # Enable infrared stream if needed
# Start streaming
pipeline.start(config)
# Access the depth sensor options
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
# Increase Laser Power
depth_sensor.set_option(rs.option.laser_power, 100)  # Adjust laser power value as needed
# Access the color sensor options
color_sensor = pipeline.get_active_profile().get_device().first_color_sensor()
def find_Depth(depth_frame):
        lower_blue = np.array([70, 100, 100])
        upper_blue = np.array([115, 255, 255])
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.18), cv2.COLORMAP_JET)
        hsv2 = cv2.cvtColor(depth_colormap,cv2.COLOR_BGR2HSV)
        mask_red = cv2.medianBlur(cv2.inRange(hsv2, lower_blue, upper_blue),7)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours_red
def find_Edge(color_image):
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #gray_blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
        edges = cv2.Canny(gray_image, 400,110,5)
        return edges
        
try:
    while True:
        frame_count += 1
        vertical_offset = 10
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        align = rs.align(rs.stream.depth)
        aligned_frames = align.process(frames) 
        # Extract the aligned RGB and Depth frames from the aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
          # Convert RealSense color frame to OpenCV format
        thread1 = threading.Thread(target=find_Depth(depth_frame))
        thread2 = threading.Thread(target=find_Edge(color_image))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        edges = find_Edge(color_image)
        contours_red = find_Depth(depth_frame)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=200, maxLineGap=5)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=70, maxRadius=130)
        # Create a black image for each frame
        black_image = np.zeros_like(frames)
        # Convert RealSense color frame to OpenCV format
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # cv2.circle(black_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw circles on the original frame
                for contour in contours_red:
                    result = cv2.pointPolygonTest(contour,(i[0], i[1]), False)
                    if result > 0:
                        cv2.circle(color_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                        cv2.circle(color_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
                        print(i[0], i[1])
                # Draw circles on the black image
                # cv2.circle(black_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                # cv2.circle(black_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         if abs(x1 - x2) < vertical_offset:
        #             cv2.line(black_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #combined_image = np.hstack((color_image, black_image))
        #cv2.imshow('RealSense Camera and Circles', combined_image)
        # cv2.drawContours(color_image,contours_red,-1,(255,0,0),2)
        # cv2.imshow('Webcam Circles Detection', color_image)
        # cv2.imshow('gray', edges)
        # cv2.imshow('Depth', depth_colormap)
        #cv2.imshow("contour_red",mask_red)
        # Display the image using OpenCV
        cv2.imshow('RealSense Camera', color_image)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    print(f"Frames Per Second (FPS): {fps}")
    pipeline.stop()
    cv2.destroyAllWindows()