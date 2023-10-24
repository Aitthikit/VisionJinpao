import numpy as np
import pyrealsense2 as rs
known_length = 130
known_height = 120
focal_length = 650
def find_Distance(pixel_lenght) :
    distance = (known_length * focal_length) / pixel_lenght
    return round(distance,2)

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
