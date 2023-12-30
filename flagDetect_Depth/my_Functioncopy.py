import cv2
import numpy as np
import pyrealsense2 as rs
known_length = 25
focal_length = 650
def config(width,height):
    # Configure depth and color streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, width,height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width,height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1)  # Enable infrared stream if needed
    config.enable_stream(rs.stream.infrared, 2)  # Enable infrared stream if needed
    return config
def find_Distance(pixel_lenght) :
    distance = (known_length * focal_length) / pixel_lenght
    return round(distance,2)

def find_focal_length(pixel_length,distance):
    focal_length = (pixel_length * distance)/known_length
    print("Focal length : ",focal_length)
    
def find_pixel_length(dis):
    pixel_length = known_length*focal_length/dis
    print(pixel_length)
    return pixel_length
#
def find_pos(image,width,x,y) :
    pixel_size_mm = known_length / width
    horizontal_mm,verticle_mm = find_res(image)[0] * pixel_size_mm ,find_res(image)[1] * pixel_size_mm
    x_ori = horizontal_mm / 2                      # Distance from Center frame
    y_ori = verticle_mm / 2 
    x_mm = (x * pixel_size_mm)                      # Distance from 
    y_mm = (y * pixel_size_mm)
    x_fromOriginal = (x * pixel_size_mm) - x_ori
    y_fromOriginal = y_ori  -(y * pixel_size_mm)
    
    return [x_fromOriginal,y_fromOriginal,x_ori,y_ori,horizontal_mm,verticle_mm]

def find_res(image) :
    height, width, channels = image.shape
    resolution = (width, height)
    # print("Resolution" , resolution)
    return [width,height]



