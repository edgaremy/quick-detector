import cv2
import sys

# This script can be use to find easily the left and right coordinates
# in order to perform a prliminary horizontal crop of the images.

DISPLAY_SIZE = (1000, 800)
click_counter = [0]

def click_event(event, x, y, flags, param):
    # global click_counter
    click_counter = param['click_counter']

    # Calculate the scaling factor
    x_scaling = image.shape[1] / DISPLAY_SIZE[0]
    y_scaling = image.shape[0] / DISPLAY_SIZE[1]

    # print(image.shape)
    if event == cv2.EVENT_LBUTTONDOWN:
        click_counter[0] += 1
        # Calculate the relative coordinates
        relative_x = int(x * x_scaling)
        relative_y = int(y * y_scaling)
        print("Point #{} Coordinates: ({}, {})".format(click_counter[0], relative_x, relative_y))
    if click_counter[0] == 2:
        sys.exit(0) # Close the window and exit the program


# Read the image
image = cv2.imread("examples/Entomoscope/frame_2022-04-28T174238.328898p0000_BloomLive359232710006228_cam1.jpeg")

# Create a window to display the image
cv2.namedWindow("Image")

# Set the callback function for mouse events
cv2.setMouseCallback("Image", click_event, {'click_counter': click_counter})

# Display the image
cv2.imshow("Image", cv2.resize(image, DISPLAY_SIZE))
if click_counter[0] == 2:
    cv2.destroyAllWindows()
# Wait for the user to close the window
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()