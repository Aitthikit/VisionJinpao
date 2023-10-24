import pyrealsense2 as rs
import math

#Configure the streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.gyro)

#Start the stream
pipeline.start(config)

try:
    last_timestamp = None

    while True:
        # Wait for gyro frame
        frames = pipeline.wait_for_frames()
        gyro = frames.first_or_default(rs.stream.gyro)

        if not gyro:
            continue

        gyro_data = gyro.as_motion_frame().get_motion_data()

        if last_timestamp:
            dt = gyro.get_timestamp() - last_timestamp  # time difference in milliseconds
            dt /= 1000.0  # convert to seconds

            # Calculate rotation magnitudes (in degrees) for each axis
            rotation_x = math.degrees(gyro_data.x * dt)
            rotation_y = math.degrees(gyro_data.y * dt)
            rotation_z = math.degrees(gyro_data.z * dt)
            print(f"Total Rotation: {rotation_x:.2f}" , f"Total Rotation: {rotation_y:.2f}", f"Total Rotation: {rotation_z:.2f}")
            #total_rotation = math.sqrt(rotation_x**2 + rotation_y**2 + rotation_z**2)

            #print(f"Total Rotation: {total_rotation:.2f}°")

        last_timestamp = gyro.get_timestamp()

except KeyboardInterrupt:
    print("Exiting...")

finally:
    pipeline.stop()