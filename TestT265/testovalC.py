import cv2
import numpy as np
scale = ((690*np.sin(np.radians(34.5)))*2.0)/960.0
# Function to get the contour of the ellipse area
def get_ellipse_contour(center, axes_lengths, angle):
    # Create a black image
    mask = np.zeros((540, 960), dtype=np.uint8)

    # Draw the ellipse on the mask
    cv2.ellipse(mask, center, axes_lengths, angle, 0, 360, 255, thickness=cv2.FILLED)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# Example ellipse parameters
ellipse_center = (480, 0)
ellipse_axes_lengths = (int(90/scale),int(30/scale))
ellipse_axes_lengths2 = (int(140/scale),int(90/scale))
ellipse_axes_lengths3 = (int(200/scale),int(150/scale))
ellipse_angle = 0
point = (400,10)
# Get the contour of the ellipse area
contours = get_ellipse_contour(ellipse_center, ellipse_axes_lengths, ellipse_angle)
contours2 = get_ellipse_contour(ellipse_center, ellipse_axes_lengths2, ellipse_angle)
contours3 = get_ellipse_contour(ellipse_center, ellipse_axes_lengths3, ellipse_angle)
# Draw the contours on a white image
contour_image = np.ones((540, 960, 3), dtype=np.uint8) * 255
cv2.drawContours(contour_image, contours3, -1, (255, 0, 0), -1)
cv2.drawContours(contour_image, contours2, -1, (0, 255, 0), -1)
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), -1)
distance = cv2.pointPolygonTest(contours[0], point, measureDist=True)
distance2 = cv2.pointPolygonTest(contours2[0], point, measureDist=True)
distance3 = cv2.pointPolygonTest(contours3[0], point, measureDist=True)

if distance >= 0:
    print("Point is inside 1")
elif distance < 0 and distance2 >= 0:
    print("Point is inside 2")
elif distance2 < 0 and distance3 >= 0:
    print("Point is inside 3") 
else:
    print("Point is outside")

cv2.circle(contour_image,point,1,(255,0,0),3)
# Display the contour image
cv2.imshow('Ellipse Contour', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()