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
# Start streaming
pipeline.start(config)
def is_point_inside_circle(point, circle_center, circle_radius):
    """
    Check if a point is inside a circle.

    Parameters:
    - point: Tuple (x, y) representing the coordinates of the point.
    - circle_center: Tuple (x, y) representing the center of the circle.
    - circle_radius: Radius of the circle.

    Returns:
    - True if the point is inside the circle, False otherwise.
    """
    distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)
    return distance <= circle_radius
class Detection:
    def __init__(self, frame):
        self.frame = frame
    def find_Depth(self):
        lower_blue = np.array([70, 0, 0])
        upper_blue = np.array([240, 255, 255])
        # depth_image = np.asanyarray(self.frame.get_data())
        depth_image = cv2.resize(np.asanyarray(self.frame.get_data()),(960,540))
        depth_image = depth_image[70:,:]
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.18), cv2.COLORMAP_JET)
        hsv2 = cv2.cvtColor(depth_colormap,cv2.COLOR_BGR2HSV)
        mask_red = cv2.medianBlur(cv2.inRange(hsv2, lower_blue, upper_blue),1)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("mask",mask_red)

        return contours_red,depth_colormap
    def find_Edge(self):
        color_image = np.asanyarray(self.frame.get_data())
        color_image = cv2.resize(color_image,(960,540))
        #color_image = self.frame
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
        edges = cv2.Canny(gray_blurred, 40,50)
        cv2.imshow("aa",color_image)
        return edges,color_image
        
try:
    while True:
        frame_count += 1
        vertical_offset = 10
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        white_frame = np.ones((540, 960, 3), dtype=np.uint8) * 255
        align = rs.align(rs.stream.depth)
        aligned_frames = align.process(frames) 
        # Extract the aligned RGB and Depth frames from the aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        DD = Detection(depth_frame)
        EE = Detection(color_frame)
        # EE = Detection(DD.find_Depth()[1])
          # Convert RealSense color frame to OpenCV format
        # thread1 = threading.Thread(target=DD.find_Depth())
        # thread2 = threading.Thread(target=EE.find_Edge())
        # thread1.start()
        # thread2.start()
        # thread1.join()
        # thread2.join()
        edges = EE.find_Edge()[0]
        contours_red = DD.find_Depth()[0]
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=200, maxLineGap=5)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=5, maxRadius=40)
        # Create a black image for each frame
        black_image = np.zeros_like(frames)
        # Convert RealSense color frame to OpenCV format
        cv2.circle(white_frame, (480, 0), 125, (0, 255, 0), 2)
        cv2.circle(white_frame, (480, 0), 190, (0, 255, 0), 2)
        cv2.circle(white_frame, (480, 0), 250, (0, 255, 0), 2)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # cv2.circle(black_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw circles on the original frame
                # cv2.circle(EE.find_Edge()[0], (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                # cv2.circle(EE.find_Edge()[0], (i[0], i[1]), 2, (0, 0, 255), 3)  # center
                # cv2.circle(white_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                # cv2.circle(white_frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
                for contour in contours_red:
                    result = cv2.pointPolygonTest(contour,(i[0], i[1]), False)
                    if result > 0:
                        cv2.circle(EE.find_Edge()[0], (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                        cv2.circle(EE.find_Edge()[0], (i[0], i[1]), 2, (0, 0, 255), 3)  # center
                        cv2.circle(white_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                        cv2.circle(white_frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
                        if is_point_inside_circle((i[0], i[1]),(480,0), 125):
                            print("1",i[0], i[1])
                        elif is_point_inside_circle((i[0], i[1]),(480,0), 190):
                            print("2",i[0], i[1])
                        elif is_point_inside_circle((i[0], i[1]),(480,0), 250):
                            print("3",i[0], i[1])

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
        cv2.drawContours(white_frame,contours_red,-1,(255,0,0),2)
        # cv2.imshow('Webcam Circles Detection', mask)
                        
        cv2.imshow('gray', edges)
        cv2.imshow('white', white_frame)
        #cv2.imshow('Depth', depth_colormap)
        cv2.imshow("contour_red", DD.find_Depth()[1])
        # Display the image using OpenCV
        # cv2.putText(EE.find_Edge()[1],"ASDASKHDWEHGERHRGOUHUF",(10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        cv2.imshow('RealSense Camera', EE.find_Edge()[1])
        
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