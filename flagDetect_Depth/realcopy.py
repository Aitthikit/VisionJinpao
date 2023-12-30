import pyrealsense2 as rs
import numpy as np
import cv2
import my_Functioncopy as ff
import time
pipeline = rs.pipeline()
distance = 0
dx = []
dy = []
dz = []
Position = []
frame_count = 0
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
try:
    while True:
        start_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Align the RGB and Depth frames
        align = rs.align(rs.stream.depth)
        aligned_frames = align.process(frames) 
        # Extract the aligned RGB and Depth frames from the aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # dept = ff.Depth
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)
        originX = ff.find_res(color_image)[0]//2
        originY = ff.find_res(color_image)[1]//2
        # depth_width, depth_height = depth_image.shape[:2]
        # color_image = color_image[100:depth_width-120, 190:depth_height-210]
        # lower_red = np.array([0, 70, 60])
        # upper_red = np.array([10, 255, 255])
        lower_blue = np.array([70, 100, 100])
        upper_blue = np.array([115, 255, 255])
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        hsv2 = cv2.cvtColor(depth_colormap,cv2.COLOR_BGR2HSV)
        # create a mask for red color
        mask_red = cv2.medianBlur(cv2.inRange(hsv2, lower_blue, upper_blue),7)
        # print(mask_red.shape)
        black_portion = np.zeros_like(mask_red[:, :1100])
        # Concatenate the black portion with the lower part of mask_red
        mask_red = cv2.hconcat([black_portion, mask_red[:, 1100:]])
        # mask_red = mask_red[600:,:]
        # find contours in the red mask
        #contours_red, _ = cv2.findContours(cv2.medianBlur((mask_red),5), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("contour_red",mask_red)
        cv2.drawContours(color_image,contours_red,-1,(255,0,0),2)
        cv2.circle(color_image,(ff.find_res(color_image)[0]//2,ff.find_res(color_image)[1]//2),2,(0,0,255),5)         # Create a Origin circle
        for cnt in contours_red:
            contour_area = cv2.contourArea(cnt)
            if contour_area > 100:#limit lower BB
                x, y, w, h = cv2.boundingRect(cnt)
                print(x,y+h//2,x+w,y+h ,end='\r')
                if w<200 and h<500:
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    pos = ff.find_pos(color_image,w,x+int(w)//2 ,y+int(h)//2)
                    distance = depth_image[int(y+h//2),int(x+w//2)]
                    #distance = depth_image[ff.find_res(color_image)[1]//2,ff.find_res(color_image)[0]//2]
                    if len(Position) < 10:
                        Position.append([pos[0],pos[1],distance])
                    else :
                        Position.pop(0)
                        Position.append([pos[0],pos[1],distance])
                    column_averages = np.mean(Position, axis=0)
                    # print(Position)
                    cv2.circle(color_image,(int(x+w//2),int(y+h//2)),2,(0,255,0),5)
                    cv2.circle(depth_colormap,(int(x+w//2),int(y+h//2)),2,(0,0,0),5)
                    cv2.putText(color_image, "Flag", (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    cv2.putText(color_image, str(column_averages[0]), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    cv2.putText(color_image, str(column_averages[1]), (x, y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    cv2.putText(color_image, str(column_averages[2]), (x, y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    dx.append(column_averages[0])
                    dy.append(column_averages[1])
                    dz.append(column_averages[2])   
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        # Calculate frame rate (FPS)
        fps = 1/ elapsed_time
        print(fps)
        cv2.imshow('RealSense', color_image)
        cv2.imshow('Depth', depth_colormap)
        cv2.waitKey(1)
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    print((max(dx)-min(dx))/2,(max(dy)-min(dy))/2,(max(dz)-min(dz))/2)