import cv2
import numpy as np
import pyrealsense2 as rs

import torch

# model = torch.hub.load('ultralytics/yolov5', 'custom', 'static/best_1_openvino_model')
model = torch.hub.load('WongKinYiu/yolov7','custom','epoch_024.pt')
model.eval()  # Set the model to evaluation mode
def adjust_contrast(image, alpha):
    """
    Adjusts the contrast of an image.

    Parameters:
    - image: Input image (numpy array).
    - alpha: Contrast adjustment factor (float).

    Returns:
    - adjusted_image: Image with adjusted contrast.
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image

def adjust_exposure(image, gamma):
    """
    Adjusts the exposure of an image.

    Parameters:
    - image: Input image (numpy array).
    - gamma: Exposure adjustment factor (float).

    Returns:
    - adjusted_image: Image with adjusted exposure.
    """
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted_image = cv2.LUT(image, table)
    return adjusted_image
def main():
    BoxClass = ['red_box', 'red_strip', 'green_box', 'green_strip', 'blue_box', 'blue_strip']
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
    }
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # Access the color sensor
    color_sensor = profile.get_device().query_sensors()[1]

    # Disable auto exposure and auto white balance
    color_sensor.set_option(rs.option.enable_auto_exposure, False)
    color_sensor.set_option(rs.option.enable_auto_white_balance, True)
    # color_sensor.set_option(rs.option.enable_auto_color_balance, False)

    # Set manual exposure and white balance values

    color_sensor.set_option(rs.option.exposure,156)  # Adjus                                          t the exposure value as needed
    # color_sensor.set_option(rs.option.white_balance, 4800)  # Adjust the white balance value as needed
    # color_sensor.set_option(rs.option.tint, 10)
    
    # Open the default camera (index 0)



    while True:
        # Read a frame from the camera
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())



        #  # Manually adjust tint using cv2.addWeighted
        # tint_value = 20
        # frame = cv2.addWeighted(frame, 1, np.zeros_like(frame), 0, tint_value)

        
        # Adjust contrast
        contrast_factor = 1.4  # You can adjust this value accordingly
        image_with_adjusted_contrast = adjust_contrast(frame, contrast_factor)
        # cv2.imshow('con', image_with_adjusted_contrast)
        # Adjust exposure
        exposure_factor = 2  # You can adjust this value accordingly
        image_with_adjusted_exposure = adjust_exposure(image_with_adjusted_contrast, exposure_factor)


        # Display the original and darkened images
        cv2.imshow('Original Image', frame)
        cv2.imshow('exp Image', image_with_adjusted_exposure)

        pred = model(image_with_adjusted_exposure)
        # print(pred.xyxy)
        pred_list = np.array(pred.xyxy[0][:].tolist())

        # print(color_sensor.get_option(rs.option.exposure))
        if len(pred_list > 0):
            for row in pred_list:
                x1, y1, x2, y2, conf, class_label = row
                if conf >= 0:
                    # Convert float to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw the bounding box
                    BoxType = BoxClass[int(class_label)].split('_')
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[BoxType[0]], 2)

                    # Display class label
                    label = f"Class: {BoxType} : {int(conf*100)}"

                    # Put text on the image
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF


        
        # Check if the 'q' key is pressed
        if key == ord('q'):
            break

        

    # Release the camera and close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
