import numpy as np
import pyrealsense2 as rs
import cv2
import time

pipeline = rs.pipeline()
config = rs.config()
frame_count = 0
x_sum = 0
y_sum = 0
x3 = 0
y3 = 0
h3 = 0
w3 = 0
x = 0
y = 0
h = 0
w = 0
min_distance = 0  # in meters
max_distance = 0.30  # in meters
min_distance2 = 0.5  # in meters
max_distance2 = 0.65  # in meters
scale = ((690*np.sin(np.radians(34.5)))*2.0)/960.0
mid_pixel = (480,270)
gap = 130
state_Gap1 = 10 #10cm
state_Gap2 = 50 #15cm
state_Gap3 = 100 #20cm
rect_list = []
highlight1 = (255,0,0)
highlight2 = (255,0,0)
highlight3 = (255,0,0)
ellipse_axes_lengths = (int(90/scale),int(30/scale))
ellipse_axes_lengths2 = (int(140/scale),int(90/scale))
ellipse_axes_lengths3 = (int(200/scale),int(150/scale))
start_time = time.time()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
def align(pipeline):
        # Wait for a new frame
    frames = pipeline.wait_for_frames()
    # align = rs.align(rs.stream.color)
    # aligned_frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    depth_data = cv2.resize(np.asanyarray(depth_frame.get_data()),(960,540))
    # color_data = cv2.resize(np.asanyarray(color_frame.get_data()),(960,540))
    return depth_data
def pixel_convert(mid_pixel,pixel):
    x = pixel[0]-mid_pixel[0]
    y = pixel[1]-mid_pixel[1]
    return x*scale,y*scale
def create_hatty(mask):
    kernel = np.ones((15,15),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.erode(mask,kernel,iterations = 1)
    return mask
def is_point_inside_circle(point, circle_center, circle_radius):
    distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)
    return distance <= circle_radius
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.filtered_value = None

    def update(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value

        return self.filtered_value
lowpass_filter_x = LowPassFilter(alpha=0.5)
lowpass_filter_y = LowPassFilter(alpha=0.5)
class Detection:
    def __init__(self, frame):
        self.frame = frame
    def find_Depth(self):
        depth_roi_mask = np.logical_and(self.frame >= min_distance * 1000, depth_data <= max_distance * 1000)
        # depth_roi_mask2 = np.logical_and(self.frame >= min_distance2 * 1000, depth_data <= max_distance2 * 1000)
        # Apply the mask to the depth data
        depth_roi = np.where(depth_roi_mask, self.frame, 0)
        # depth_roi2 = np.where(depth_roi_mask2, self.frame, 0)
        # Create a grayscale image from the ROI data
        depth_roi_image = np.uint8(255-(depth_roi / np.max(depth_roi) * 255))
        # depth_roi_image2 = np.uint8((depth_roi2 / np.max(depth_roi2) * 255))
        _, binary_image = cv2.threshold(depth_roi_image, 128, 255, cv2.THRESH_BINARY)
        # binary_image = create_hatty(binary_image)
        contours_black, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # _, binary_image2 = cv2.threshold(depth_roi_image2, 128, 255, cv2.THRESH_BINARY)
        # binary_image2 = create_hatty(binary_image2)
        # contours_white, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("asd",depth_roi_image2)
        return depth_roi_image,contours_black
    def find_Edge(self):
        color_image = self.frame
        # cv2.imshow("RGB Frame with ROI2", color_image)
        # cv2.imshow("RGB Frame with ROI", color_data)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray_image, (9, 9), 10)
        edges = cv2.Canny(gray_blurred, 150,100)
        contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours_edge, -1, (255), thickness=cv2.FILLED)
        return color_image,edges,contours_edge
    def find_flag(self):
        color_image = self.frame
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray_image, (11, 11), 20)
        edges = cv2.Canny(gray_blurred, 50,70)
        return edges,gray_blurred
while True:
    frame_count += 1
    # Wait for a new frame
    frames = pipeline.wait_for_frames()
    # Align the depth frame with the color frame
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    # Access depth data as a numpy array
    depth_data = cv2.resize(np.asanyarray(depth_frame.get_data()),(960,540))
    color_data = cv2.resize(np.asanyarray(color_frame.get_data()),(960,540))

    # Access color data as a numpy array
    depth1 = Detection(depth_data)
    # Define the distance range for your ROI
    # Create a mask for the ROI
    # depth_roi_mask = np.logical_and(depth_data >= min_distance * 1000, depth_data <= max_distance * 1000)
    # # Apply the mask to the depth data
    # depth_roi = np.where(depth_roi_mask, depth_data, 0)
    # # Create a grayscale image from the ROI data
    # depth_roi_image = np.uint8(255-(depth_roi / np.max(depth_roi) * 255))
    # depth_roi_image2 = np.uint8((depth_roi / np.max(depth_roi) * 255))
    # _, binary_image = cv2.threshold(depth_roi_image, 128, 255, cv2.THRESH_BINARY)
    # contours_black, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, binary_image2 = cv2.threshold(depth_roi_image2, 128, 255, cv2.THRESH_BINARY)
    # contours_white, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Display the RGB frame with the ROI
    # color_image = np.asanyarray(color_frame.get_data())
    # gray_image = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
    # gray_blurred = cv2.GaussianBlur(gray_image, (11, 11), 10)
    # edges = cv2.Canny(gray_blurred, 50,100)
    # contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(edges, contours_edge, -1, (255), thickness=cv2.FILLED)
    # gray_image2 = cv2.cvtColor(depth_roi, cv2.COLOR_BGR2GRAY)
    # gray_blurred2 = cv2.GaussianBlur(gray_image2, (9, 9), 2)
    # edges2 = cv2.Canny(depth_roi_image, 50,100)
    # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
    #                            param1=50, param2=30, minRadius=1, maxRadius=40)
    # circles2 = cv2.HoughCircles(edges2, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
    #                            param1=50, param2=30, minRadius=200, maxRadius=400)
    # Create a black image for each frame
    # Convert RealSense color frame to OpenCV format
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         for contour in contours_white:
    #             result = cv2.pointPolygonTest(contour,(i[0], i[1]), False)
    #             if result > 0:
    #                 cv2.circle(color_data, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #                 cv2.circle(color_data, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #                 if is_point_inside_circle((i[0], i[1]),(480,0), 125):
    #                     print("1",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 190):
    #                     print("2",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 250):
    #                     print("3",i[0], i[1])
    # if circles2 is not None:
    #     circles = np.uint16(np.around(circles2))
    #     for i in circles[0, :]:
    #         cv2.circle(depth_roi_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #         cv2.circle(depth_roi_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #         for contour in contours_white:
    #             result = cv2.pointPolygonTest(contour,(i[0], i[1]), False)
    #             if result > 0:
    #                 cv2.circle(depth_roi_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #                 cv2.circle(depth_roi_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #                 cv2.circle(white_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #                 cv2.circle(white_frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #                 if is_point_inside_circle((i[0], i[1]),(480,0), 125):
    #                     print("1",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 190):
    #                     print("2",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 250):
    #                     print("3",i[0], i[1])
    # for cnt in depth1.find_Depth()[2]:
    #     contour_area = cv2.contourArea(cnt)
    #     if contour_area > 1500:#limit lower BB
    #         x3, y3, w3, h3 = cv2.boundingRect(cnt)
    #         center = int(x3+(w3/2)), int(h3-(w3/2)) #center of place (000,000)
    #         # cv2.rectangle(depth_roi_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         cv2.circle(color_data, center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
    #         cv2.circle(color_data, center, 1, (0, 255, 0), 5)
    #         if center[1] not in range(0,30):
    #             None
    #             # if center[1] >= 0:
    #             #     # print("move to left",0-center[1])
    #             # else:
    #             #     # print("move to right",0-center[1])
    #         else:
    #             break
    for cnt in depth1.find_Depth()[1]:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 600 and contour_area < 5000:#limit lower BB
            x, y, w, h = cv2.boundingRect(cnt) # พื้นที่ของแท่งวางธงที่สามารถอยู่ได้ x = 000 , y = 000 , w = 000 , h = 000
            cv2.rectangle(color_data, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(color_data,(int(x+w/2),int(y+h/2)), 1, (0, 255, 255), 5)
   
    # cv2.drawContours(edges,contours_red,-1,(255,0,0),2)
    # cv2.drawContours(color_data,depth1.find_Depth()[2],-1,(255,0,0),2)       
    cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
    cv2.imshow("RGB Frame with ROI", color_data)
    # print(scale)
    cv2.imshow("ROI Frame", depth1.find_Depth()[0])
    # print(100/scale,150/scale,200/scale)
    # Wait for a key press, and exit the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break