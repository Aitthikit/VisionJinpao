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


Position = []
timestamp = 0

BoxDetect = BOXDETECTION()

time.sleep(2)

capture = 0
frame_count = 0
start_time = time.time()


try:

    while True:
        # fps count
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Real sense Align
        depth_data, color_data = ff.align(pipeline)
        
        # Create ROI from depth cameara
        contour_area = ff.create_ROI(0.5,1.25,color_data, depth_data)
        
        # find box
        box_list = BoxDetect.HSV_filtering(contour_area, depth_data)
        
    
        key = cv2.waitKey(1) & 0xFF
        # press q for exit
        if key == ord('q'):
            break
        
        # press T for capture sensor

        # init Cap
        if key == ord('t'):
            print("Cap jaa")
            t_cap = 3
            timestamp = time.time() + t_cap
            capture = 1

        # finish Cap
        if time.time() > timestamp and capture:
            try:
                Position = np.array(Position).tolist()
                print(ff.BoxPath([2,1],ff.positionFilter(Position)))
                Position = []
                capture = 0
            except :
                print("error repeat")
                timestamp = time.time() + t_cap
                capture = 1
            # print(Position)
        
        # while cap
        elif capture:
            # Capture
            if len(box_list)>0:
                #find table
                ans, angle = BoxDetect.pose_calculate(box_list)
                ans = np.array(ans,dtype='object')
                Position.append(ans.copy())
        # break
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()