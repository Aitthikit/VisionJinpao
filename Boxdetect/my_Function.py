import numpy as np
import pyrealsense2 as rs
import math
import cv2
from itertools import permutations
import seaborn as sns  
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from statistics import mode



known_height = 110
# focal_length = 650
# def find_Distance(pixel_lenght) :
#     distance = (known_length * focal_length) / pixel_lenght
#     return round(distance,2)
def auto_exposure_adjustment(image, target_mean=150):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the mean intensity of the image
    current_mean = np.mean(gray_image)
    
    # Calculate the exposure adjustment factor
    exposure_adjustment = target_mean / current_mean
    
    # Apply the exposure adjustment to the image
    adjusted_image = cv2.convertScaleAbs(image, alpha=exposure_adjustment, beta=0)

    return adjusted_image

def cumulative_distribution(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum() / hist.sum()
    return cdf, bins

def getCDF(image):
    cdf, bins = cumulative_distribution(image)
    cdf = np.insert(cdf, 0, [0] * int(bins[0]))
    cdf = np.append(cdf, [1] * int(255 - bins[-1]))
    return cdf

def histMatch(cdfInput, cdfTemplate, imageInput):
    pixels = np.arange(256)
    new_pixels = np.interp(cdfInput, cdfTemplate, pixels)
    imageMatch = (np.reshape(new_pixels[imageInput.ravel()], imageInput.shape)).astype(np.uint8)
    return imageMatch

def histogram_matching(image, imageTemplate):

    imageResult = np.zeros(image.shape, dtype=np.uint8)

    # CDF and histogram
    for channel in range(3):
        cdfInput = getCDF(image[:, :, channel])
        cdfTemplate = getCDF(imageTemplate[:, :, channel])
        imageResult[:, :, channel] = histMatch(cdfInput, cdfTemplate, image[:, :, channel])

    return imageResult


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

def create_ROI(min_distance, max_distance, color_data, depth_data):
    depth_roi_mask = np.logical_and(depth_data >= min_distance * 1000, depth_data <= max_distance * 1000)
    depth_roi = np.where(depth_roi_mask, depth_data, 0)
    depth_roi_image = cv2.medianBlur(np.uint8(depth_roi / np.max(depth_roi) * 255),1)
    contours, _ = cv2.findContours(depth_roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_frame = np.zeros_like(color_data)
    cv2.drawContours(new_frame, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    contour_area = cv2.bitwise_and(color_data, color_data, mask=new_frame[:, :, 0])
    return depth_roi_mask, contour_area

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

def BoxPath(robot_init, Color):
    # Init min Cost
    min_cost = float('inf')
    position = [[0,0],[0,1],[0,2],
                [1,0],[1,1],[1,2],
                [2,0],[2,1],[2,2]]
    # create permutation of possible way to pick box
    color_permutations = permutations(range(len(Color)),3)
    # print(color_permutations)
    # print(color)
    for idx in color_permutations:
        # print(idx)
        # print(color[idx[0]],color[idx[1]],color[idx[2]])
        # use only without duplicate color
        if Color[idx[0]] != Color[idx[1]] and Color[idx[0]] != Color[idx[2]] and Color[idx[1]] != Color[idx[2]]:
            
            # print(position[id1],position[id2],position[id3])

            # Init path cost
            RobotPath = [robot_init]
            Cost = []

            #  3 box picking
            for i in range(3):
                cost, robot_pos = cost_function(Color[idx[i]], position[idx[i]], RobotPath[i])
                RobotPath.append(robot_pos)
                Cost.append(cost)
                
   
            # Goal state check
            if sum(Cost) < min_cost:
                min_cost = sum(Cost)
                CostA = Cost
                min_path = RobotPath[1:]
                color = Color[idx[0]],Color[idx[1]],Color[idx[2]]
    # print (min_path, color, min_cost, CostA)
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
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    kernel = np.ones((10,10),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)

    return mask

def flatten_data(data):
    color = []
    for sublist in data:
        for subsublist in sublist:
            # print(subsublist)
            # print(np.array(subsublist, dtype = 'object')[:,0])
            color.append(np.array(subsublist, dtype = 'object')[:,0].tolist())
    
    return np.array([point[1] for sublist in data for subsublist in sublist for point in subsublist if isinstance(point[1], list)]), np.array(color).reshape(-1)

def plot_3d_scatter(data, x_label='X Axis', y_label='Y Axis', z_label='Z Axis', title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting all data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', label='Raw Data')

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Add a legend
    ax.legend()

    # Set plot title if provided
    if title:
        plt.title(title)

    # Show the plot
    plt.show()
# Flattening the data to get only the position coordinates
def filterDensity(kde_values, threshold_percentage = 0.7):
    # Get the maximum density value
    max_density = max(kde_values[1])
    print(max_density)

    # Set the threshold as a percentage of the maximum density
    
    threshold_value = max_density * threshold_percentage
    
    # Filter data based on the threshold
    return [(kde_values[0])[kde_values[1] > threshold_value],(kde_values[1])[kde_values[1] > threshold_value]]
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))



def positionFilter(Position,Color):

    Position = np.array(Position)
    
    
    # print(Position[0])
    # print(Position[:,1])
    # print(Position[:,2])

    # Calculate KDE
    kde_values_x = sns.kdeplot(Position[:,0]).get_lines()[0].get_data()
    # filtered_kde_values_x = filterDensity(kde_values_x)

    kde_values_y = sns.kdeplot(Position[:,1]).get_lines()[1].get_data()
    # filtered_kde_values_y = filterDensity(kde_values_y)

    kde_values_z = sns.kdeplot(Position[:,2]).get_lines()[2].get_data()
    # filtered_kde_values_z = filterDensity(kde_values_z)
    
    # print(kde_values_x[0])
    # print(kde_values_y[0])
    # print(kde_values_z[0])
    
    # Plot the filtered KDE
    # plt.show()
    # kde_values_x = sns.kdeplot(kde_values_x[0]).get_lines()[0].get_data()
    # kde_values_y = sns.kdeplot(kde_values_y[0]).get_lines()[1].get_data()
    # kde_values_z = sns.kdeplot(kde_values_z[0]).get_lines()[2].get_data()

    # plt.plot(filtered_kde_values_x[0], filtered_kde_values_x[1], label='X')
    # plt.plot(filtered_kde_values_y[0], filtered_kde_values_y[1], label='Y')
    # plt.plot(filtered_kde_values_z[0], filtered_kde_values_z[1], label='Z')

    # print(filtered_kde_values_x[0])
    # print(filtered_kde_values_y[0])
    # print(filtered_kde_values_z[0])




    # # Find multiple peaks_x using scipy's find_peaks_x
    peaks_x, _ = find_peaks(kde_values_x[1], height=0)  # Adjust the height threshold as needed
    peaks_y, _ = find_peaks(kde_values_y[1], height=0)  # Adjust the height threshold as needed
    peaks_z, _ = find_peaks(kde_values_z[1], height=0)  # Adjust the height threshold as needed


    peak_density_x = (kde_values_x[1][peaks_x])
    peak_density_y = (kde_values_y[1][peaks_y])
    peak_density_z = (kde_values_z[1][peaks_z])

    # print(filtered_kde_values_x[0][peaks_x[np.argsort(peak_density_x)][::-1][:3]])
    # print(filtered_kde_values_y[0][peaks_y[np.argsort(peak_density_y)][::-1][:3]])
    # print(filtered_kde_values_z[0][peaks_z[np.argsort(peak_density_z)][::-1][0]])
    peak_density_idx = np.argsort(peak_density_x)
    peak_density_idy = np.argsort(peak_density_y)
    peak_density_idz = np.argsort(peak_density_z)
    peak_position_x = sorted(kde_values_x[0][peaks_x[peak_density_idx][::-1][:3]])
    peak_position_y = sorted(kde_values_y[0][peaks_y[peak_density_idy][::-1][:3]], reverse=True)
    peak_position_z = kde_values_z[0][peaks_z[peak_density_idz][::-1][0]]
    plt.show()

    position = []
    for i in range(3):
        for j in range(3):
            position.append([peak_position_x[j],peak_position_y[i],peak_position_z])

    distance = 40
    color = []
    for c in range(9):
        temp = []
        for i in range(len((Position))):
            if euclidean_distance([Position[:,0][i],Position[:,1][i],Position[:,2][i]],position[c]) <= distance:
                temp.append(Color[i])
        color.append(mode(temp))
    # print(Position[0])

    
    
    return position, color

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
    h_box = (y_min1+y_min2)//2 - y_min
    w_box = x_max_ROI - x_min_ROI
    
    x_strip = x_min
    y_strip = (y_min1+y_min2)//2
    w_strip = x_max - x_min
    h_strip = y_max - (y_min1+y_min2)//2
    # cv2.rectangle(contour_area_image, (x_strip, y_strip), (x_strip + w_strip, y_strip+  h_strip), color, 2)  # Draw strip
    # cv2.rectangle(contour_area_image, (x_box, y_box), (x_box + w_box, y_min + h_box),color, 2)  # Draw color box
    # cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3)
    # cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
    # cv2.circle(contour_area_image,((x_min + x_max)//2 ,((y_min1+y_min2)//2+y_max)//2 ),2,(255,255,255),3)
    # cv2.putText(contour_area_image, "x: " + str((x_min + x_max)//2) + "y: " + str(((y_min1+y_min2)//2+y_max)//2 ), ((x_min + x_max)//2, (y_min1+y_min2)//2 +30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
    # cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x_min, (y_min1+y_min2)//2 +10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
    
    return x_box,y_box,w_box,h_box ,x_strip,y_strip,w_strip,h_strip









class BOXDETECTION:
    def __init__(self):
        self.box_list = []
        self.ans =[[[],[],[]],[[],[],[]],[[],[],[]]]
        # self.lower_saturation = 0
        # self.upper_saturation = 255


        # foo
        # self.lower_green = np.array([70, 55, 60]) 
        # self.upper_green = np.array([85, 255, 255])

        # self.lower_red = np.array([170, 55, 50])
        # self.upper_red = np.array([180, 255, 255])

        # self.lower_blue = np.array([105, 80, 70])
        # self.upper_blue = np.array([110, 255, 255])

        # jp
        # self.lower_green = np.array([40, 100, 0]) 
        # self.upper_green = np.array([100, 255, 255])

        # self.lower_red = np.array([109, 110, 0])
        # self.upper_red = np.array([255, 255, 255]) # 120 255

        # self.lower_blue = np.array([95, 110, 0])
        # self.upper_blue = np.array([110, 255, 255])


        self.lower_green = np.array([40, 70, 10]) 
        self.upper_green = np.array([100, 255, 255])

        self.lower_red = np.array([156, 100, 10])
        self.upper_red = np.array([179, 255, 255])

        self.lower_blue = np.array([98, 100, 10])
        self.upper_blue = np.array([110, 255, 255])

        # Debug Var
        self.mask_blue = None
        self.mask_red = None
        self.mask_green = None

    def HSV_filtering(self, contour_area, depth_data):
        # bgr2hsv
        hsv = cv2.cvtColor(contour_area,cv2.COLOR_BGR2HSV)
        
        # create a mask 
        mask_red = cv2.medianBlur(cv2.inRange(hsv, self.lower_red, self.upper_red),7)
        mask_green = cv2.medianBlur(cv2.inRange(hsv, self.lower_green, self.upper_green),7)
        mask_blue = cv2.medianBlur(cv2.inRange(hsv,self. lower_blue, self.upper_blue),7)
        
        self.mask_red = create_hatty(mask_red)
        self.mask_green = create_hatty(mask_green)
        self.mask_blue = create_hatty(mask_blue)

        # find contours 
        contours_red, _ = cv2.findContours(self.mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(self.mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(self.mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       

        contour_area_image = contour_area.copy()
        plot = np.zeros_like(contour_area_image)
        cv2.circle(contour_area_image,(contour_area_image.shape[1]//2 ,contour_area_image.shape[0]//2 ),2,(200,200,200),3)      # Create Camera Frame
        Box_Color = []
        Box_Pos = []
        for contour in contours_red:
            color = (0,0,255)
            contour_area = cv2.contourArea(contour)
            
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)    
                if (w/h < 1.5) & (h/w < 1.5):
                    pos = find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                    depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                    Box_Color.append("red")
                    Box_Pos.append([pos[0], pos[1], depth_value])
                    cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(plot, "x: " + str(x) + "y: " + str(y), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3)  
                    cv2.putText(contour_area_image, "x: " + str(pos) , (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                # elif(w/h>1.5) & (w/h<2.5) :
                    
                #     x1,y1,w1,h1, x_s,y_s,w_s,h_s = separate_hatty(contour,color)
                #     pos = find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                #     depth_value = depth_data[y1+(h1//2),x1+(w1//2)]
                #     Box_Color.append("red")
                #     Box_Pos.append([pos[0], pos[1], depth_value])
                #     cv2.circle(contour_area_image,(x1+int(w1)//2 ,y1+int(h1)//2),2,(255,255,255),10) # Center of circle
                #     cv2.circle(plot,(x1+int(w1)//2 ,y1+int(h1)//2),2,color,3) # Center of circle
                    
                # else :
                #     cv2.putText(plot, "w: " + str(w) + "h: " + str(h), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # elif(w/h>4.5) & (w/h<13):
                #     cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                #     cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                #     cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
            

        for contour in contours_green:
            color = (0,255,0)
            contour_area = cv2.contourArea(contour)
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)                  
                if (w/h < 1.5) & (h/w < 1.5):
                    pos = find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                    depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                    Box_Color.append("green")
                    Box_Pos.append([pos[0], pos[1], depth_value])
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3) 
                    cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(plot, "x: " + str(x) + "y: " + str(y), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                # elif(w/h>1.5) & (w/h<2.5) :
                #     x1,y1,w1,h1, x_s,y_s,w_s,h_s = separate_hatty(contour,color)
                #     pos = find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                #     depth_value = depth_data[y1+int(h1)//2,x1+int(w1)//2]
                #     Box_Color.append("green")
                #     Box_Pos.append([pos[0], pos[1], depth_value])
                #     cv2.circle(contour_area_image,(x1+(w1//2) ,y1+(h1//2)),2,(255,255,255),10) # Center of circle
                #     cv2.circle(plot,(x1+int(w1)//2 ,y1+int(h1)//2),2,color,3) # Center of circle
                    
                    
                # elif(w/h>4.5) & (w/h<13):
                #     cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                #     cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                #     cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)

        for contour in contours_blue:
            color = (255,0,0)
            contour_area = cv2.contourArea(contour)
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if (w/h < 1.5) & (h/w < 1.5):
                    pos = find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                    depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                    Box_Color.append("blue")
                    Box_Pos.append([pos[0], pos[1], depth_value])
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3) 
                    cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(contour_area_image, "x: " + str(pos) , (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # elif(w/h>1.5) & (w/h<2.5) :
                #     x1,y1,w1,h1, x_s,y_s,w_s,h_s = separate_hatty(contour,color)
                #     pos = find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                #     depth_value = depth_data[y1+int(h1)//2,x1+int(w1)//2]
                #     Box_Color.append("blue")
                #     Box_Pos.append([pos[0], pos[1], depth_value])
                #     cv2.circle(contour_area_image,(x1+(w1//2) ,y1+(h1//2)),2,(255,255,255),10) # Center of circle
                    # cv2.circle(plot,(x1+int(w1)//2 ,y1+int(h1)//2),2,color,3) # Center of circle
                    
                # elif(w/h>4.5) & (w/h<13):
                #     cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                #     cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                #     cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
            
        # cv2.putText(contour_area_image, "Angle : " + str(angle),(1000,650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("RGB Frame", contour_area_image)
        # cv2.imshow("Plot ", cv2.resize(plot,(plot.shape[1]//scale_dis,plot.shape[0]//scale_dis)))

        return Box_Pos, Box_Color
    
        
    def pose_calculate(self,data):
        temp = [["","",""],["","",""],["","",""]]
        

        check = []

        min_x = min_x = min(point[0] for point in data)
        max_x = max(point[0] for point in data)
        min_y = min(point[1] for point in data)
        max_y = max(point[1] for point in data)

        grid_x = ((max_x-min_x) / 3)+1
        grid_y = ((max_y-min_y) / 3)+1
        print(data)
        for i in data:
            column = int((i[0]-min_x) // grid_x)
            roll = int((i[1]-min_y) // grid_y)
            # print(str(i[3]) , (i[0] - min_x)/grid_x)
            temp[roll][column] = [i[3] , i[2]]
            self.ans[roll][column] = [str(i[3]),[i[4][0],i[4][1],i[2]]]
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
            return self.ans, angle_degrees
        else :
            return self.ans, 99999
    
    def mask_show(self):
        cv2.imshow("mask blue", self.mask_blue)
        cv2.imshow("mask red", self.mask_red)
        cv2.imshow("mask green", self.mask_green)


class STRIPDETECTION:
    def __init__(self):
        self.box_list = []
        self.ans =[[[],[],[]],[[],[],[]],[[],[],[]]]
        # self.lower_saturation = 0
        # self.upper_saturation = 255


        # foo
        # self.lower_green = np.array([70, 55, 60]) 
        # self.upper_green = np.array([85, 255, 255])

        # self.lower_red = np.array([170, 55, 50])
        # self.upper_red = np.array([180, 255, 255])

        # self.lower_blue = np.array([105, 80, 70])
        # self.upper_blue = np.array([110, 255, 255])

        # jp
        # self.lower_green = np.array([40, 100, 0]) 
        # self.upper_green = np.array([100, 255, 255])

        # self.lower_red = np.array([109, 110, 0])
        # self.upper_red = np.array([255, 255, 255]) # 120 255

        # self.lower_blue = np.array([95, 110, 0])
        # self.upper_blue = np.array([110, 255, 255])


        self.lower_green = np.array([40, 70, 10]) 
        self.upper_green = np.array([100, 255, 255])

        self.lower_red = np.array([156, 100, 10])
        self.upper_red = np.array([179, 255, 255])

        self.lower_blue = np.array([98, 100, 10])
        self.upper_blue = np.array([110, 255, 255])

        # Debug Var
        self.mask_blue = None
        self.mask_red = None
        self.mask_green = None

    def HSV_filtering(self, contour_area, depth_data):
        # bgr2hsv
        hsv = cv2.cvtColor(contour_area,cv2.COLOR_BGR2HSV)
        
        # create a mask 
        mask_red = cv2.medianBlur(cv2.inRange(hsv, self.lower_red, self.upper_red),7)
        mask_green = cv2.medianBlur(cv2.inRange(hsv, self.lower_green, self.upper_green),7)
        mask_blue = cv2.medianBlur(cv2.inRange(hsv,self. lower_blue, self.upper_blue),7)
        
        self.mask_red = create_hatty(mask_red)
        self.mask_green = create_hatty(mask_green)
        self.mask_blue = create_hatty(mask_blue)

        # find contours 
        contours_red, _ = cv2.findContours(self.mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(self.mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(self.mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       

        contour_area_image = contour_area.copy()
        plot = np.zeros_like(contour_area_image)
        cv2.circle(contour_area_image,(contour_area_image.shape[1]//2 ,contour_area_image.shape[0]//2 ),2,(200,200,200),3)      # Create Camera Frame
        Box_Color = []
        Box_Pos = []
        for contour in contours_red:
            color = (0,0,255)
            contour_area = cv2.contourArea(contour)
            
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)    
            #     if (w/h < 1.5) & (h/w < 1.5):
            #         pos = find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
            #         depth_value = depth_data[y+int(h)//2,x+int(w)//2]
            #         Box_Color.append("red")
            #         Box_Pos.append([pos[0], pos[1], depth_value])
            #         cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #         cv2.putText(plot, "x: " + str(x) + "y: " + str(y), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #         cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
            #         cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
            #         cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3)  
            #         cv2.putText(contour_area_image, "x: " + str(pos) , (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                # if(w/h>1.5) & (w/h<2.5) :
                    
                #     x1,y1,w1,h1, x_s,y_s,w_s,h_s = separate_hatty(contour,color)
                #     pos = find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                #     depth_value = depth_data[y1+(h1//2),x1+(w1//2)]
                #     Box_Color.append("red")
                #     Box_Pos.append([pos[0], pos[1], depth_value])
                #     cv2.circle(contour_area_image,(x1+int(w1)//2 ,y1+int(h1)//2),2,(255,255,255),10) # Center of circle
                #     cv2.circle(plot,(x1+int(w1)//2 ,y1+int(h1)//2),2,color,3) # Center of circle
                    
                # else :
                #     cv2.putText(plot, "w: " + str(w) + "h: " + str(h), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if(w/h>4.5) & (w/h<13):
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
            

        for contour in contours_green:
            color = (0,255,0)
            contour_area = cv2.contourArea(contour)
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)                  
                # if (w/h < 1.5) & (h/w < 1.5):
                #     pos = find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                #     depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                #     Box_Color.append("green")
                #     Box_Pos.append([pos[0], pos[1], depth_value])
                #     cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                #     cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                #     cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3) 
                #     cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                #     cv2.putText(plot, "x: " + str(x) + "y: " + str(y), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                # if(w/h>1.5) & (w/h<2.5) :
                #     x1,y1,w1,h1, x_s,y_s,w_s,h_s = separate_hatty(contour,color)
                #     pos = find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                #     depth_value = depth_data[y1+int(h1)//2,x1+int(w1)//2]
                #     Box_Color.append("green")
                #     Box_Pos.append([pos[0], pos[1], depth_value])
                #     cv2.circle(contour_area_image,(x1+(w1//2) ,y1+(h1//2)),2,(255,255,255),10) # Center of circle
                #     cv2.circle(plot,(x1+int(w1)//2 ,y1+int(h1)//2),2,color,3) # Center of circle
                    
                    
                if(w/h>4.5) & (w/h<13):
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)

        for contour in contours_blue:
            color = (255,0,0)
            contour_area = cv2.contourArea(contour)
            if contour_area >= 1000:
                x, y, w, h = cv2.boundingRect(contour)
                # if (w/h < 1.5) & (h/w < 1.5):
                    # pos = find_pos(contour_area_image,w,x+int(w)//2 ,y+int(h)//2)
                    # depth_value = depth_data[y+int(h)//2,x+int(w)//2]
                    # Box_Color.append("blue")
                    # Box_Pos.append([pos[0], pos[1], depth_value])
                    # cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    # cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
                    # cv2.circle(plot,(x+int(w)//2 ,y+int(h)//2 ),2,color,3) 
                    # cv2.putText(plot, "Depth: " + str(depth_value), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(contour_area_image, "x: " + str(pos) , (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # if(w/h>1.5) & (w/h<2.5) :
                #     x1,y1,w1,h1, x_s,y_s,w_s,h_s = separate_hatty(contour,color)
                #     pos = find_pos(contour_area_image,w1,x1+int(w1)//2 ,y1+int(h1)//2)
                #     depth_value = depth_data[y1+int(h1)//2,x1+int(w1)//2]
                #     Box_Color.append("blue")
                #     Box_Pos.append([pos[0], pos[1], depth_value])
                #     cv2.circle(contour_area_image,(x1+(w1//2) ,y1+(h1//2)),2,(255,255,255),10) # Center of circle
                #     cv2.circle(plot,(x1+int(w1)//2 ,y1+int(h1)//2),2,color,3) # Center of circle
                    
                if(w/h>4.5) & (w/h<13):
                    cv2.rectangle(contour_area_image, (x, y), (x + w, y + h),color, 2)
                    cv2.putText(contour_area_image, "Strip " + str(w) + "  " + str(h), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
                    cv2.circle(contour_area_image,(x+int(w)//2 ,y+int(h)//2 ),2,(255,255,255),3)
            
        # cv2.putText(contour_area_image, "Angle : " + str(angle),(1000,650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("RGB Frame", contour_area_image)
        # cv2.imshow("Plot ", cv2.resize(plot,(plot.shape[1]//scale_dis,plot.shape[0]//scale_dis)))

        return Box_Pos, Box_Color
    
        
    def pose_calculate(self,data):
        temp = [["","",""],["","",""],["","",""]]
        

        check = []

        min_x = min_x = min(point[0] for point in data)
        max_x = max(point[0] for point in data)
        min_y = min(point[1] for point in data)
        max_y = max(point[1] for point in data)

        grid_x = ((max_x-min_x) / 3)+1
        grid_y = ((max_y-min_y) / 3)+1
        print(data)
        for i in data:
            column = int((i[0]-min_x) // grid_x)
            roll = int((i[1]-min_y) // grid_y)
            # print(str(i[3]) , (i[0] - min_x)/grid_x)
            temp[roll][column] = [i[3] , i[2]]
            self.ans[roll][column] = [str(i[3]),[i[4][0],i[4][1],i[2]]]
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
            return self.ans, angle_degrees
        else :
            return self.ans, 99999
    
    def mask_show(self):
        cv2.imshow("mask blue", self.mask_blue)
        cv2.imshow("mask red", self.mask_red)
        cv2.imshow("mask green", self.mask_green)