import cv2
import pyrealsense2 as rs
import time
import numpy as np

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outputCFlag2_video.avi', fourcc, 30.0, (1280, 720))
out2 = cv2.VideoWriter('outputDFFF_video.avi', fourcc, 30.0, (1280, 720))
# Set the recording duration to 1 minute (60 seconds)
record_duration = 60
start_time = time.time()

try:
    while time.time() - start_time < record_duration:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Get the color frame
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            continue

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # Write the frame to the video file
        out.write(color_image)
        out2.write(depth_image)

        # Display the frame (optional)
        cv2.imshow('Video', color_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the VideoWriter and RealSense pipeline
    out.release()
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()