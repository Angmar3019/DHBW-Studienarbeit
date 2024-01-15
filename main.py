import numpy as np
import cv2 as cv
import argparse

from loguru import logger
logger.add("main.log")

from modules.hough_transformation import hough_transformation
from modules.tensorflow import tensorflow

camera_device = 0

# Erstellen des ArgumentParsers
arg_desc = '''\
    Coin detection with different methods
    '''

parser = argparse.ArgumentParser(description=arg_desc)

#Erstellen einer gegenseitig auschließenend Gruppe
group = parser.add_mutually_exclusive_group(required=True)

# Hinzufügen der Argumente zur Grippe
group.add_argument("-ht", "--houghtransformation", action="store_true", help="use the hough transformation to detect the coins")
group.add_argument("-t", "--tensorflow", action="store_true", help="use the tensorflow model to detect the coins")

args = vars(parser.parse_args())

cap = cv.VideoCapture(camera_device)

# Check if Camera can be opend
if not cap.isOpened():
    logger.error(f"Cannot open camera /dev/video{camera_device}")
    exit()

while True:
    ret, frame = cap.read()

    # Check if a video stream is resieved
    if not ret:
        logger.error("Error opening the steam")
        break

    # Check which detection method should be used
    if args["houghtransformation"]:
        window_name = "Hough Transformation"
        value = hough_transformation(logger, cv, frame)

    elif args["tensorflow"]:
        window_name = "Tensorflow"
        value = tensorflow(logger, cv, frame)
    else:
        logger.error("Invalid flag")

    # Print current coin value in the frame
    count_display = "Wert: " + str(value)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, count_display, (10,10), font, 1, (0,255,255), 2, cv.LINE_4)

    # Display current image in the window
    cv.imshow(window_name, frame)

    # Check if close button or "q" is pressed and close abort the loop
    if cv.waitKey(1) & 0xFF == ord("q") or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv.destroyAllWindows()