import numpy as np
import pyrealsense2 as rs
import math
import cv2
known_length = 130
known_height = 120
# focal_length = 650
# def find_Distance(pixel_lenght) :
#     distance = (known_length * focal_length) / pixel_lenght
#     return round(distance,2)

def find_pos(image,w_h,x,y) :
    pixel_size_mm = w_h / known_height
    pix = known_height / w_h
    horizontal_mm,verticle_mm = find_res(image)[0] * pix ,find_res(image)[1] * pix
    x_ori = horizontal_mm / 2                               # Origin of frame
    y_ori = verticle_mm / 2 
    x_fromOriginal = ((x * pix) - x_ori )           # Reorigin to center of frame
    y_fromOriginal = y_ori  - (y * pix)
    width = known_length * pixel_size_mm
    return [x_fromOriginal,y_fromOriginal,x_ori,y_ori,horizontal_mm,verticle_mm,width]

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
    align = rs.align(rs.stream.depth)
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
    # output = [["","",""],["","",""],["","",""]]
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
        temp[roll][column] = [i[3] , i[2]]
        ans[roll][column] = str(i[3])

    for i in range(len(temp)):
        check.append("" not in temp[i])
        # print("" in temp[i])
    # print(check)
    if all(check) == True:
        # print(555) 
        delta_x = max_x - min_x
        d_left = (temp[0][0][1] + temp[1][0][1] + temp[2][0][1]) / 3
        d_right = (temp[0][2][1] + temp[1][2][1] + temp[2][2][1]) / 3
        delta_depth = d_right - d_left
        angle_radians = math.atan2(delta_depth, delta_x)
        angle_degrees = round(math.degrees(angle_radians),4)
        return ans, angle_degrees
    else :
        # print(1230129370)
        # return [["","",""],["","",""],["","",""]],0
        # print(temp)
        return ans, 99999
    
def create_ROI(min_distance, max_distance, color_data, depth_data):
    # min_distance = 0.5  # in meters
    # max_distance = 1.25  # in meters
    depth_roi_mask = np.logical_and(depth_data >= min_distance * 1000, depth_data <= max_distance * 1000)
    depth_roi = np.where(depth_roi_mask, depth_data, 0)
    depth_roi_image = cv2.medianBlur(np.uint8(depth_roi / np.max(depth_roi) * 255),1)
    contours, _ = cv2.findContours(depth_roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_frame = np.zeros_like(color_data)
    cv2.drawContours(new_frame, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    contour_area = cv2.bitwise_and(color_data, color_data, mask=new_frame[:, :, 0])
    return contour_area

# roll, pitch, yaw = 0.0, 0.0, 0.0
# def gyro_data_to_euler(gyro_data, dt):
#     global roll, pitch, yaw
#     gx, gy, gz = gyro_data
#     roll += gx * dt
#     pitch += gy * dt
#     yaw += gz * dt
#     return roll, pitch, yaw