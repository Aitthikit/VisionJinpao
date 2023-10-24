import pyrealsense2 as rs
import numpy as np
import time
# Initialize variables
roll, pitch, yaw = 0.0, 0.0, 0.0

# Configure the pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.gyro)
# Start streaming
profile = pipeline.start(config)
start_time = time.time()



try:
    prev_time = time.time()
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Get gyro frame
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            # Get gyroscope data
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            # Calculate time delta
            current_time = time.time()
            elapsed_time = current_time - start_time
            dt = current_time - prev_time
            prev_time = current_time
            # Calculate roll, pitch, and yaw
            roll, pitch, yaw = gyro_data_to_euler([gyro_data.x, gyro_data.y, gyro_data.z], dt)
            c_r = -8.63/60 +0.28/60
            c_p = 1.73/60 -0.64/60
            c_y = -1.74/60 + 0.07/60
            print("Roll: {:.2f}, Pitch: {:.2f}, Yaw: {:.2f}".format(np.degrees(roll) - (c_r*elapsed_time),np.degrees(pitch) - (c_p*elapsed_time),np.degrees(yaw) - (c_y*elapsed_time)))

         
finally:
    # Stop streaming
    pipeline.stop()
    
    