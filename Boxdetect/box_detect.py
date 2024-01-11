import numpy as np
import pyrealsense2 as rs
import cv2
import time
import my_Function as ff
import math
from my_Function import BOXDETECTION

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

target_image = cv2.imread('target.jpg')
Position = []
timestamp = 0

BoxDetect = BOXDETECTION()

time.sleep(2)

state = 'Idle'
frame_count = 0
start_time = time.time()
def update_mask(x):
    global min_h
    global max_h
    min_h = cv2.getTrackbarPos('min','HueR')
    max_h = cv2.getTrackbarPos('max','HueR')

cv2.namedWindow("HueR")
min_h = 109
max_h = 255
# i=0

cv2.createTrackbar('min','HueR',min_h,255,update_mask)
cv2.createTrackbar('max','HueR',max_h,255,update_mask)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_550mm_place2.avi', fourcc, 20.0, (1280, 720))
stack_weight=0.1
depth_data, color_data = ff.align(pipeline)
result = np.float32(color_data)
try:

    while True:
        # BoxDetect.lower_red =  np.array([min_h, 110, 0])
        # BoxDetect.upper_red = np.array([max_h, 255, 255])

        # fps count
        frame_count += 2
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Real sense Align
        depth_data, color_data = ff.align(pipeline)
        

        cv2.imshow('rgb',color_data)
        # matched_image = ff.histogram_matching(color_data,target_image)
        # cv2.imshow("Matched Image",matched_image)
        # color_data = ff.auto_exposure_adjustment(color_data)
        # cv2.imshow("Exp",color_data)
        # Convert frame to float32
        frame_float32 = np.float32(color_data)

        # Update the stack with a weighted average
        # result = cv2.addWeighted(result, 1 - stack_weight, frame_float32, stack_weight, 0)

        # Convert result to uint8 for display
        # color_data = np.uint8(result)

        # Display the result
        # cv2.imshow("Real-Time Stacking", color_data)
        # Create ROI from depth cameara
        contour_area = ff.create_ROI(0.7,1.2,color_data, depth_data)
        # out.write(contour_area)
        cv2.imshow("Roi",contour_area)
        # find box
        Box_Pos, Box_Color = BoxDetect.HSV_filtering(contour_area, depth_data)
        # print(Box_Pos)
        # print(Box_Color)
        # BoxDetect.mask_show()
        key = cv2.waitKey(1) & 0xFF
        # press q for exit
        if key == ord('q'):
            break
        
        # press T for capture sensor

        # # init Cap
        if key == ord('t'):
            Color = []
            Position = []
            print("Cap jaa")
            # i +=1 
            # cv2.imwrite(f'captured_frame_550mm_Place_{i}.jpg', contour_area)

            t_cap = 5
            timestamp = time.time() + t_cap
            state = 'Capture'

        # finish Cap
        if time.time() > timestamp and state == 'Capture':
            Color = [element for sublist in Color for element in sublist]
            # print(Color)
            Position = [element for sublist in Position for element in sublist]
            # print(Position)
            # print(Color)
            position, color = ff.positionFilter(Position,Color)
            # print(position)
            # print(color)

            print(ff.BoxPath([2,1], color))
    
            state = 'Idle'

        elif state == 'Capture':
            Color.append(Box_Color)
            Position.append(Box_Pos)

        # anglemode
        if key == ord('a'):
            state = 'Angle'

        if state == 'Angle':
            z_plot = []
            x_plot = []
            for x in range(1280):
                for z in depth_data[:,x]:
                    z_plot.append(z)
                    x_plot.append(x)
            import matplotlib.pyplot as plt
            # print(z_plot)
            plt.scatter(z_plot,x_plot, color='blue', marker='o', label='Points')
            plt.show()
            state = 'Idle'
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()