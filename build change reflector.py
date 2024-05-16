import cv2
import numpy as np

# Load the before and after images
before_image = cv2.imread("/home/krakenmare/Documents/Machine Learning/Resources/Land Classifier Testdata/sentinel-2/2019-07-16-Sentinel-2_L1C_Mesero.jpg")
after_image = cv2.imread("/home/krakenmare/Documents/Machine Learning/Resources/Land Classifier Testdata/sentinel-2/2021-08-14-Sentinel-2_L1C_Mesero.jpg")

if before_image.shape[:2] != after_image.shape[:2]:
    before_image = cv2.resize(before_image, (after_image.shape[1], after_image.shape[0]))


# Convert images to grayscale
before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference between the two images
difference = cv2.absdiff(before_gray, after_gray)

# Apply a threshold to the difference image
_, thresholded_diff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded difference image
contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

after_image = cv2.resize(after_image, None, fx=5, fy=5)
before_image = cv2.resize(before_image, None, fx=5, fy=5)

# Iterate through the contours and draw rectangles around changes
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Filter out small changes
        x, y, w, h = cv2.boundingRect(contour)
        x *= 5  # Scale the coordinates if the image was resized
        y *= 5
        w *= 5
        h *= 5
        x-=int(0.2*x)
        if x<0:
            x=0
        y-=int(0.2*y)
        if y<0:
            y=0
        w+=int(0.4*w)
        if w>before_image.shape[0]:
            w=before_image.shape[0]
        h+=int(0.4*h)
        if h>before_image.shape[1]:
            h=before_image.shape[1]
        cv2.rectangle(after_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle
        cv2.rectangle(before_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Display the image with changes highlighted
cv2.imshow("After Image", after_image)
cv2.imshow("Before Image", before_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
