import numpy as np
import pyrealsense2 as rs
import cv2
import time
import my_Function as ff
import math

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

box_list, LL = [], []
ans =[["","",""],["","",""],["","",""]]
angle = 0
scale_dis = 2
# time.sleep(2)

lower_green = np.array([70, 55, 60]) 
upper_green = np.array([85, 255, 255])

lower_red = np.array([170, 55, 50])
upper_red = np.array([180, 255, 255])

lower_blue = np.array([105, 80, 70])
upper_blue = np.array([110, 255, 255])



try:
    frame_count = 0
    start_time = time.time()

    while True:
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time > 0:  # Avoid division by zero
            fps = frame_count / elapsed_time
            # print(f"FPS: {fps}", end='\r')
            
        if elapsed_time > 1:
            frame_count = 0
            start_time = time.time()
            box_list = []

        # Align depth + rgb frame
        depth_data, color_data = ff.align(pipeline)
        
        # Create ROI from range of distance 
        contour_area = ff.create_ROI(0.5,1.25,color_data, depth_data)
        
        hsv = cv2.cvtColor(contour_area,cv2.COLOR_BGR2HSV)
        
        # create a mask 
        mask_red = cv2.medianBlur(cv2.inRange(hsv, lower_red, upper_red),7)
        mask_green = cv2.medianBlur(cv2.inRange(hsv, lower_green, upper_green),7)
        mask_blue = cv2.medianBlur(cv2.inRange(hsv, lower_blue, upper_blue),7)
        
        mask_red = ff.create_hatty(mask_red)
        mask_green = ff.create_hatty(mask_green)
        mask_blue = ff.create_hatty(mask_blue)

        # find contours 
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       

        contour_area_image = contour_area.copy()
        plot = np.zeros_like(contour_area_image)
        cv2.circle(contour_area_image,(contour_area_image.shape[1]//2 ,contour_area_image.shape[0]//2 ),2,(200,200,200),3)      # Create Camera Frame
        
        for contour in contours_red:
            color = (0,0,255)
            contour_area = cv2.contourArea(contour)
            
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)    
                if (w/h < 1.5) & (h/w < 1.5):
                    pos = ff.find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                    depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                    box_list.append([x,y,depth_value,"red",pos])
                    cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(plot, "x: " + str(x) + "y: " + str(y), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3)  
                    cv2.putText(contour_area_image, "x: " + str(pos) , (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                elif(w/h>4.5) & (w/h<13):
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    
                elif(w/h>1.5) & (w/h<2.5) :
                    x1,y1,w1,h1, x_s,y_s,w_s,h_s = ff.separate_hatty(contour,color)
                    pos = ff.find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                    depth_value = depth_data[y1+int(h1)//2,x1+int(w1)//2]
                    box_list.append([x,y,depth_value,"red",pos])
                    cv2.circle(contour_area_image,(x1+int(w1)//2 ,y1+int(h1)//2),2,(255,255,255),10) # Center of circle
                    
                else :                    
                    pass

        for contour in contours_green:
            color = (0,255,0)
            contour_area = cv2.contourArea(contour)
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)                  
                if (w/h < 1.5) & (h/w < 1.5):
                    pos = ff.find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                    depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                    box_list.append([x,y,depth_value,"green",pos])
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3) 
                    cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(plot, "x: " + str(x) + "y: " + str(y), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                elif(w/h>1.5) & (w/h<2.5) :
                    x1,y1,w1,h1, x_s,y_s,w_s,h_s = ff.separate_hatty(contour,color)
                    pos = ff.find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                    depth_value = depth_data[y1+int(h1)//2,x1+int(w1)//2]
                    box_list.append([x,y,depth_value,"green",pos])
                    cv2.circle(contour_area_image,(x1+int(w1)//2 ,y1+int(h1)//2),2,(255,255,255),10) # Center of circle
                    
                elif(w/h>4.5) & (w/h<13):
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                else :                    
                    pass

        for contour in contours_blue:
            color = (255,0,0)
            contour_area = cv2.contourArea(contour)
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if (w/h < 1.5) & (h/w < 1.5):
                    pos = ff.find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                    depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                    box_list.append([x,y,depth_value,"blue",pos])
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3) 
                    cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(contour_area_image, "x: " + str(pos) , (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                elif(w/h>1.5) & (w/h<2.5) :
                    x1,y1,w1,h1, x_s,y_s,w_s,h_s = ff.separate_hatty(contour,color)
                    pos = ff.find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                    depth_value = depth_data[y1+int(h)//2,x1+int(w)//2]
                    box_list.append([x,y,depth_value,"blue",pos])
                    cv2.circle(contour_area_image,(x1+int(w1)//2 ,y1+int(h1)//2),2,(255,255,255),10) # Center of circle
                                     
                elif(w/h>4.5) & (w/h<13):
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                else :                    
                    pass
                
        if len(box_list)>0:
            ans,angle = ff.find_table(box_list,ans)
            ans = np.array(ans,dtype='object')
            LL.append(ans.copy())

        cv2.putText(contour_area_image, "Angle : " + str(angle),(1000,650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("RGB Frame", contour_area_image)
        cv2.imshow("Plot ", cv2.resize(plot,(plot.shape[1]//scale_dis,plot.shape[0]//scale_dis)))
        # cv2.imshow("contour_gree",cv2.resize(mask_green,(mask_green.shape[1]//scale_dis,mask_green.shape[0]//scale_dis)))
        # cv2.imshow("contour_red",cv2.resize(mask_red,(mask_red.shape[1]//scale_dis,mask_red.shape[0]//scale_dis)))
        # cv2.imshow("contour_blue",cv2.resize(mask_blue,(mask_blue.shape[1]//scale_dis,mask_blue.shape[0]//scale_dis)))
        
        # print(ans,angle)
        # Wait for a key press, and exit the loop if 'q' is pressed
        
        # print(ans,end='\r')
        # print(LL)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # break
except KeyboardInterrupt:
    pass
finally:
    # import BoxPath
    
    LL = np.array(LL)
    # print(LL.tolist())
    print(ff.BoxPath([2,1],ans))
    pipeline.stop()
    cv2.destroyAllWindows()