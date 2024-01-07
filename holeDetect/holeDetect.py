import cv2
import numpy as np
import time
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\\Users\AAA\Desktop\\Jinpao\\VisionJinpao\\holeDetect\\Assemm.mp4")
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    

while True:
    vertical_offset = 10
    ret, frame = cap.read()
    # frame = frame[:475,500:1500]
    frame = cv2.resize(frame,(600,600))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=7,
            param2=25,
            minRadius=10,
            maxRadius=27
        )
        edges = cv2.Canny(gray, 50, 50,5)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=200, maxLineGap=5)
        

        # Create a black image for each frame
        black_image = np.zeros_like(frame)

        # If circles are found, draw them on the frame and the black image
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw circles on the original frame
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
                print(i[0], i[1])
                # Draw circles on the black image
                cv2.circle(black_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
                cv2.circle(black_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # center

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < vertical_offset:
                    cv2.line(black_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Webcam Circles Detection', frame)
        cv2.imshow('gray', edges)
        # Display the video frame
        resized_frame = cv2.resize(frame, (960, 540))
        resized_black_image = cv2.resize(black_image, (960, 540))
        
        cv2.imshow('Detected Circles and Lines', black_image)

        if cv2.waitKey(1) == 27:
            break
    else:
        print("Error: Failed to capture frame.")
        break

cap.release()
cv2.destroyAllWindows()
