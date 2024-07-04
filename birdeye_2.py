import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Bilder/calibration.jpg')

# Define the range for white color in RGB space
# Adjust the lower and upper bounds if necessary
lower_white = np.array([210, 210, 210])  # Lower bound for white
upper_white = np.array([255, 255, 255])  # Upper bound for white

# Create a binary mask where white colors are in the range
mask = cv2.inRange(image, lower_white, upper_white)

# Apply the mask to get the white regions in the original image
white_regions = cv2.bitwise_and(image, image, mask=mask)

# Convert the masked image to grayscale
gray_image = cv2.cvtColor(white_regions, cv2.COLOR_BGR2GRAY)

min_circularity = 0.305
circularity_increment = 0.0025

while True:

    print(min_circularity)
    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area
    params.filterByArea = True
    params.minArea = 200

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = min_circularity





    '''  convexity und inertia machen ned wiakli an unterschied ... ka was de mochn'''
    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Filter by inertia ratio
    params.filterByInertia = True
    params.minInertiaRatio = 0.01



    # Create a SimpleBlobDetector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    gray_image = np.invert(gray_image)


    # Detect blobs in the grayscale image
    keypoints = detector.detect(gray_image)

    # Draw detected blobs as red circles
    blobs = cv2.drawKeypoints(gray_image, keypoints, np.zeros((1, 1)), (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display number of blobs detected
    number_of_blobs = len(keypoints)
    text = f"Number: {number_of_blobs}"
    cv2.putText(blobs, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Extrahiere die Mittelpunkte der erkannten Blobs
    blob_centers = [(int(keypoint.pt[0]), int(keypoint.pt[1])) for keypoint in keypoints]

    # Gib die Mittelpunkte aus
    print("Mittelpunkte der erkannten Blobs:")
    for center in blob_centers:
        print(center)
    
    if len(keypoints) == 4:
        break
    elif len(keypoints) < 4:
        min_circularity -= circularity_increment
    else:
        min_circularity += circularity_increment
    
dst_points = [[200+297,0+300,0], [0+297,-200+300,0], [0+297,200+300,0], [-200+297,0+300,0]]

# Berechne die Homographiematrix
H, _ = cv2.findHomography(np.array(blob_centers), np.array(dst_points), method=cv2.RANSAC, ransacReprojThreshold=3.0)

# Wende die Homographie auf das Bild an
warped_image = cv2.warpPerspective(image, H, (1000, 1000))  # Ersetze original_image, width und height entsprechend

cv2.imwrite('warped_image.jpg', warped_image)
cv2.imshow('Red Circle Detection', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zeichne erkannte Blobs als grüne Kreise im Originalbild
for center in keypoints:
    cv2.circle(image, (int(center.pt[0]), int(center.pt[1])), 5, (0, 255, 0), -1)


# Show blobs using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("White Blobs")
plt.axis('off')
plt.show()