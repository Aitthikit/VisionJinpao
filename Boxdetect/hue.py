import cv2
import numpy as np
import pyrealsense2 as rs
points = []
points_r = (0,0)

def get_hsv_range(frame, point1, point2):
    # Extract the region of interest (ROI) around the points
    roi = frame[min(point1[1], point2[1]):max(point1[1], point2[1]),
                min(point1[0], point2[0]):max(point1[0], point2[0])]

    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate the min and max HSV values
    min_hue = np.min(hsv_roi[:, :, 0])
    max_hue = np.max(hsv_roi[:, :, 0])

    return min_hue, max_hue

def on_mouse_click(event, x, y, flags, param):
    global points
    global points_r
    

    
    if event == cv2.EVENT_LBUTTONUP:
        points_r = (x, y)
        points.append((x, y))
        
        # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # cv2.imshow("Frame", frame)

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Create a window and set the mouse callback function
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", on_mouse_click)

    while True:
        # Read a frame from the camera
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())

        # Display the frame
        cv2.circle(frame, points_r, 1,(0, 255, 0))
        cv2.imshow("Frame", frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # Check if the 'c' key is pressed
        if key == ord('c'):
            if len(points) >= 2:
                # Get the min and max hue values in the specified area
                min_hue, max_hue = get_hsv_range(frame, points[0],points[1])
                
                # Print the results
                print(f"Min Hue: {min_hue}, Max Hue: {max_hue}")
                # Clear the points list
                
                points.clear()
        
        # Check if the 'q' key is pressed
        elif key == ord('q'):
            break

        

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
