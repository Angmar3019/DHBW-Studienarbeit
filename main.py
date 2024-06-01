""" DHBW-Studienarbeit - Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
    DHBW-Study paper  -  Development of a prototype for coin counting using image processing and machine learning
    
    author:  Angmar3019
    date:    01.06.2024
    version: 3.0.0
    licence: GNU General Public License v3.0 
"""

import numpy as np
import cv2 as cv
import argparse
import os

from libcamera import controls
from picamera2 import Picamera2
from loguru import logger

from modules.hough_transformation import hough_transformation
from modules.tensorflow import tensorflow

# If the program is executed remotely with ssh, the OpenCV window always opens on the first connected screen
os.environ["DISPLAY"] = ":0" 

cv.startWindowThread()
logger.add("main.log")



def create_args():
    """Create arguments
    - Allows you to pass arguments when executing the pyhton file
    - Allows you to add the model and the labels

    Test:
        - Does "--help" displays the help?
        - Can you only use either "-ht" or "t"?
    """

    arg_desc = '''\
        DHBW-Studienarbeit - Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
        DHBW-Study paper  -  Development of a prototype for coin counting using image processing and machine learning

        Use "-ht" or "-t" to select the option with which method you want to recognize the coins.
        For example, if you want to use Tensorflow, the command would look like this:

        python main.py -t model.tflite -l labels.txt
        '''

    parser = argparse.ArgumentParser(description=arg_desc, formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-ht", "--houghtransformation", action="store_true", help="use the Hough-Transformation to detect the coins")
    parser.add_argument("-c", "--calibrate", action="store_true", help="for calibrating the millimetre-pixel ratio with a 20 cent coin")

    group.add_argument("-t", "--tensorflow", type=str ,help="use the tensorflow model to detect the coins. Specify a tflite model that should be used")
    parser.add_argument("-l", "--labels", type=str, help="specify a txt with the labels if you are using Tensorflow")

    args = vars(parser.parse_args())

    if args["tensorflow"] and not args["labels"]:
        parser.error("A txt with the labels used in the model is required")

    if args["calibrate"] and args["houghtransformation"]:
        window_name = "Hough Transformation calibration"
        calibrate = True
        option = hough_transformation(logger, calibrate)
        return option, calibrate, window_name

    if args["houghtransformation"]:
        window_name = "Hough Transformation"
        calibrate = False
        option = hough_transformation(logger, calibrate)
        return option, calibrate, window_name

    elif args["tensorflow"]:
        window_name = "Tensorflow"
        calibrate = False
        option = tensorflow(logger, args["tensorflow"], args["labels"])
        return option, calibrate, window_name
    
    else:
        logger.error("Invalid flag")



def initialize_camera():
    """Initialize camera
    - Initializes the pyhton webcam, which is connected via the csi connector
    - Sets the color space and allows the resolution to be adjusted
    - Activates the autofocus of the camra, if available

    Test:
        - Is the live image of the camera displayed?
        - Does the camera focus on close objects, e.g. when you hold your hand up to it
    """

    camera = Picamera2()
    camera.configure(
        camera.create_preview_configuration(main={
            "format": "YUV420",
            "size": (2304, 1296)
        }))

    camera.set_controls({"FrameRate": 10})
    camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    camera.start()

    return camera



option, calibrate, window_name = create_args()
camera = initialize_camera()

while True:
    frame = camera.capture_array("main")
    frame = cv.cvtColor(frame, cv.COLOR_YUV420p2RGB)

    if not frame.any():
        logger.error("Error opening the steam")
        break
    
    if calibrate:
        frame =  option.calibrate(frame)
    else:
        frame = option.detect(frame)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    cv.imshow(window_name, frame)
    
cv.destroyAllWindows()