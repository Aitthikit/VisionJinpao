import numpy as np
import pyrealsense2 as rs
import cv2
import time
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
lowpass_filter_x = LowPassFilter(alpha=0.6)
lowpass_filter_y = LowPassFilter(alpha=0.7)

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
min_distance = 0.59  # in meters
max_distance = 0.7  # in meters
scale = ((640*np.sin(np.radians(34.5)))*2.0)/960.0
mid_pixel = (480,270)
gap = 130
state_Gap1 = -70
state_Gap2 = -130
state_Gap3 = -190
rect_list = []
start_time = time.time()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
def pixel_convert(mid_pixel,pixel):
    x = pixel[0]-mid_pixel[0]
    y = pixel[1]-mid_pixel[1]
    return x*scale,y*scale
def create_hatty(mask):
    kernel = np.ones((10,10),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.erode(mask,kernel,iterations = 1)
    return mask
def is_point_inside_circle(point, circle_center, circle_radius):
    distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)
    return distance <= circle_radius
class Detection:
    def __init__(self, frame):
        self.frame = frame
    def find_Depth(self):
        depth_roi_mask = np.logical_and(self.frame >= min_distance * 1000, depth_data <= max_distance * 1000)
        # Apply the mask to the depth data
        depth_roi = np.where(depth_roi_mask, self.frame, 0)
        # Create a grayscale image from the ROI data
        depth_roi_image = np.uint8(255-(depth_roi / np.max(depth_roi) * 255))
        depth_roi_image2 = np.uint8((depth_roi / np.max(depth_roi) * 255))
        _, binary_image = cv2.threshold(depth_roi_image, 128, 255, cv2.THRESH_BINARY)
        binary_image = create_hatty(binary_image)
        contours_black, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, binary_image2 = cv2.threshold(depth_roi_image2, 128, 255, cv2.THRESH_BINARY)
        binary_image2 = create_hatty(binary_image2)
        contours_white, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return depth_roi_image,contours_black,contours_white
    def find_Edge(self):
        color_image = self.frame
        # cv2.imshow("RGB Frame with ROI2", color_image)
        # cv2.imshow("RGB Frame with ROI", color_data)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray_image, (11, 11), 20)
        edges = cv2.Canny(gray_blurred, 50,70)
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
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    # Access depth data as a numpy array
    depth_data = cv2.resize(np.asanyarray(depth_frame.get_data()),(960,540))
    # Access color data as a numpy array
    
    color_data = cv2.resize(np.asanyarray(color_frame.get_data()),(960,540))
    depth1 = Detection(depth_data)
    edges1 = Detection(color_data)
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
    for cnt in depth1.find_Depth()[2]:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 100:#limit lower BB
            x3, y3, w3, h3 = cv2.boundingRect(cnt)
            center = int(x3+(w3/2)), int(h3-(w3/2)) #center of place (000,000)
            # cv2.rectangle(depth_roi_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(color_data, center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
            cv2.circle(color_data, center, 1, (0, 255, 0), 5)
            if center[1] not in range(0,30):
                None
                # if center[1] >= 0:
                #     # print("move to left",0-center[1])
                # else:
                #     # print("move to right",0-center[1])
            else:
                break
    cv2.line(color_data, (center[0]+gap,0), (center[0]+gap, 540), (0, 255, 0), 3)
    cv2.line(color_data, (center[0],0), (center[0], 540), (0, 255, 0), 3)
    for cnt in depth1.find_Depth()[1]:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 400 and contour_area < 5000:#limit lower BB
            x, y, w, h = cv2.boundingRect(cnt) # พื้นที่ของแท่งวางธงที่สามารถอยู่ได้ x = 000 , y = 000 , w = 000 , h = 000
            cv2.rectangle(color_data, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for cnt in edges1.find_Edge()[2]:
        contour_area = cv2.contourArea(cnt)
        if contour_area < 500:#limit lower BB
            x2, y2, w2, h2 = cv2.boundingRect(cnt)
            if x2 in range(x,x+w) and y2 in range(y,y+h):
                # cv2.rectangle(color_data, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                rect_list.append((x2,y2,w2,h2))
                min_x = min(rect[0] for rect in rect_list)
                min_y = min(rect[1] for rect in rect_list)
                max_x = max(rect[0] + rect[2] for rect in rect_list)
                max_y = max(rect[1] + rect[3] for rect in rect_list)
                mid_x = int(max_x-((max_x-min_x)/2)) # ตำแหน่งของจุดศูนย์กลางรูในแกน X
                mid_y = int(max_y-((max_y-min_y)/2)) # ตำแหน่งของจุดศูนย์กลางรูในแกน Y
                mid_x = int(lowpass_filter_x.update(mid_x))
                mid_y = int(lowpass_filter_y.update(mid_y))
                if len(rect_list) > 2:
                    rect_list = rect_list[-2:-1]
                    distance = cv2.pointPolygonTest((np.array([center], dtype=np.int32)), (mid_x, mid_y), True)
                    if mid_x in range(x,x+w) and mid_y in range(y,y+h) and (mid_y-y)<17:
                        cv2.circle(color_data, (mid_x, mid_y), 1, (0, 255, 255), 5)
                        print(pixel_convert(mid_pixel,(mid_x,mid_y)),distance)
                        if mid_x in range(center[0]-gap,center[0]+gap):
                            if distance > state_Gap1:
                                print("1")
                            elif distance <= state_Gap1 and distance > state_Gap2:
                                print("2")
                            elif distance <= state_Gap2 and distance > state_Gap3:
                                print("3")
    circles = cv2.HoughCircles(
            edges1.find_flag()[1],
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=7,
            param2=25,
            minRadius=10,
            maxRadius=27
        )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw circles on the original frame
            cv2.circle(color_data, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
            cv2.circle(color_data, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
            print(i[0], i[1])
            # Draw circles on the black image
          # center
    # cv2.drawContours(edges,contours_red,-1,(255,0,0),2)
    # cv2.drawContours(color_data,depth1.find_Depth()[2],-1,(255,0,0),2)       
    cv2.imshow('gray', edges1.find_Edge()[1])
    # # cv2.imshow('gray_bb', gray_blurred)
    cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
    cv2.imshow("RGB Frame with ROI", color_data)
    # print(scale)
    cv2.imshow("ROI Frame", depth1.find_Depth()[0])
    # Wait for a key press, and exit the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# except KeyboardInterrupt:
#     pass
# finally:
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     fps = frame_count / elapsed_time
#     print(f"Frames Per Second (FPS): {fps}")
#     pipeline.stop()
#     cv2.destroyAllWindows()