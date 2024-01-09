import numpy as np
import pyrealsense2 as rs
import math
import cv2
from itertools import permutations
known_length = 110
known_height = 110
# focal_length = 650
# def find_Distance(pixel_lenght) :
#     distance = (known_length * focal_length) / pixel_lenght
#     return round(distance,2)

def find_pos(image,w_pix,x,y) :
    # pixel_size_mm = w_h / known_height
    pix = known_height / w_pix
    horizontal_mm, verticle_mm = find_res(image)[0] * pix ,find_res(image)[1] * pix
    x_ori = horizontal_mm / 2                               # Origin of frame
    y_ori = verticle_mm / 2 
    x_fromOriginal = round(((x * pix) - x_ori ),2)           # Reorigin to center of frame
    y_fromOriginal = round(y_ori  - (y * pix),2)
    # width = known_length * pixel_size_mm
    return [x_fromOriginal,y_fromOriginal]

def find_res(image) :
    height, width, channels = image.shape
    return [width,height]

def cameraFrame2cameraOrigin(position,theta):
    theta = -np.deg2rad(theta) #reverse direction
    R = [[np.cos(theta), 0, np.sin(theta)]
         ,[0, 1, 0]
         ,[-np.sin(theta), 0, np.cos(theta)]]
    return np.matmul(R,position)

def align(pipeline):
    
        # Wait for a new frame
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # if not depth_frame or not color_frame:
    #     continue

    depth_data = np.asanyarray(depth_frame.get_data())
    color_data = np.asanyarray(color_frame.get_data())
    return depth_data, color_data
    
def find_table(data, ans):
    temp = [["","",""],["","",""],["","",""]]
    check = []
    min_x = min_x = min(point[0] for point in data)
    max_x = max(point[0] for point in data)
    min_y = min(point[1] for point in data)
    max_y = max(point[1] for point in data)

    grid_x = ((max_x-min_x) / 3)+1
    grid_y = ((max_y-min_y) / 3)+1
    for i in data:
        column = int((i[0]-min_x) // grid_x)
        roll = int((i[1]-min_y) // grid_y)
        # print(str(i[3]) , (i[0] - min_x)/grid_x)
        temp[roll][column] = [i[3] , i[2]]
        ans[roll][column] = [str(i[3]),[i[4][0],i[4][1],i[2]]]
        # print(temp)

    for i in range(len(temp)):
        check.append("" not in temp[i])
        
    if all(check) == True:
        delta_x = max_x - min_x
        d_left = (temp[0][0][1] + temp[1][0][1] + temp[2][0][1]) / 3
        d_right = (temp[0][2][1] + temp[1][2][1] + temp[2][2][1]) / 3
        delta_depth = d_right - d_left
        angle_radians = math.atan2(delta_depth, delta_x)
        angle_degrees = round(math.degrees(angle_radians),4)
        return ans, angle_degrees
    else :
        return ans, 99999
    
def create_ROI(min_distance, max_distance, color_data, depth_data):
    depth_roi_mask = np.logical_and(depth_data >= min_distance * 1000, depth_data <= max_distance * 1000)
    depth_roi = np.where(depth_roi_mask, depth_data, 0)
    depth_roi_image = cv2.medianBlur(np.uint8(depth_roi / np.max(depth_roi) * 255),1)
    contours, _ = cv2.findContours(depth_roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_frame = np.zeros_like(color_data)
    cv2.drawContours(new_frame, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    contour_area = cv2.bitwise_and(color_data, color_data, mask=new_frame[:, :, 0])
    return contour_area

def cost_function(color, position, robot_pos):
        Vx = 1 # velocity of x in box/sec
        Vy = 5 # velocity of y in box/sec

        if color == "red":
            offset = -1
        elif color == "blue":
            offset = 1
        else:
            offset = 0

        cost = np.abs(position[0] - robot_pos[0]) * Vx + np.abs(position[1] + offset - robot_pos[1]) * Vy
        return cost, [position[0], position[1] + offset]

def BoxPath(robot_init ,BoxColor):
    # Init min Cost
    min_cost = float('inf')

    # create box and position of box in index
    Box = [[BoxColor[i][j][0], [i, j]] for i in range(3) for j in range(3)]
    print(Box)
    # create permutation of possible way to pick box
    color_permutations = permutations(Box,3)

    for perm in color_permutations:
        # use only without duplicate color
        if all(perm[i][0] != perm[j][0] for i in range(2) for j in range(i+1, 3)):

            # Init path cost
            RobotPath = [robot_init]
            Cost = []

            # loop for 3 box picking
            for i in range(3):
                cost, robot_pos = cost_function(perm[i][0], perm[i][1], RobotPath[i])
                RobotPath.append(robot_pos)
                Cost.append(cost)
                
   
            # Goal state check
            if sum(Cost) < min_cost:
                min_cost = sum(Cost)
                CostA = Cost
                min_path = RobotPath[1:]
                color = perm[0][0],perm[1][0],perm[2][0]

    return min_path, color, min_cost, CostA

def separate_hatty(contour,color):
    points_array = np.array(contour)[:, 0]
    
    x_coordinates = points_array[:, 0]
    y_coordinates = points_array[:, 1]
    
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)
    
    width = x_max-x_min
    height = y_max-y_min
    x_threshold = 0.15
    y_threshold = 0.1
    
    filtered_points1 = points_array[points_array[:, 0] < x_min+(width*x_threshold)]
    filtered_points2 = points_array[points_array[:, 0] > x_max-(width*x_threshold)]
    filtered_points3 = points_array[points_array[:, 1] < y_min+(height*y_threshold)]

    x_min_ROI = min(filtered_points3[:,0])
    x_max_ROI = max(filtered_points3[:,0])
    
    y_min1 = min(filtered_points1[:,1])
    y_min2 = min(filtered_points2[:,1])
    
    x_box = x_min_ROI
    y_box = y_min
    h_box = (y_min1+y_min2)//2
    w_box = x_max_ROI - x_min_ROI
    
    x_strip = x_min
    y_strip = (y_min1+y_min2)//2
    w_strip = x_max
    h_strip = y_max
    # cv2.rectangle(contour_area_image, (x_min, (y_min1+y_min2)//2), (x_max, y_max), color, 2)  # Draw strip
    # cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)  # Draw color box
    # cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3)
    # cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
    # cv2.circle(contour_area_image,((x_min + x_max)//2 ,((y_min1+y_min2)//2+y_max)//2 ),2,(255,255,255),3)
    # cv2.putText(contour_area_image, "x: " + str((x_min + x_max)//2) + "y: " + str(((y_min1+y_min2)//2+y_max)//2 ), ((x_min + x_max)//2, (y_min1+y_min2)//2 +30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
    # cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x_min, (y_min1+y_min2)//2 +10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
    
    return x_box,y_box,w_box,h_box ,x_strip,y_strip,w_strip,h_strip

def create_hatty(mask):
    kernel = np.ones((10,10),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.erode(mask,kernel,iterations = 1)
    return mask
