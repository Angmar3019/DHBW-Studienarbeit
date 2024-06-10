""" DHBW-Studienarbeit - Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
    DHBW-Study paper  -  Development of a prototype for coin counting using image processing and machine learning
    
    author:  Angmar3019
    date:    01.06.2024
    version: 3.0.0
    licence: GNU General Public License v3.0 
"""

import sys
import cv2 as cv
import numpy as np
import modules.gui as gui
import time
from RPLCD.i2c import CharLCD



class hough_transformation:
    def __init__(self, logger, calibrate):
        """Initialisation of Hough-Transformation detection
        - If not in calibration mode, loads the millimetre-pixel ratio
        - Sets the radii of the coins used
        - Initilaize variables for the calibration

        Args:
            - logger:               Contains the global logger variable
            - frame (array):        Contains the image of the webcam as an array for opencv
            - calibrate(boolean):   Specifies whether the calibration is executed in this program run   
        """

        self.cv = cv
        self.logger = logger
        self.logger.info("Hough-Transformation is used as the detection method")
        
        if not calibrate:
            try:
                with open("mm_per_pixel.txt", 'r') as f:
                    self.mm_per_pixel = float(f.readline())
                self.logger.info(f"Loaded millimetre-pixel ratio: {self.mm_per_pixel}")
                
            except FileNotFoundError:
                self.logger.error("mm_per_pixel.txt not found. Calibration needed")
                cv.destroyAllWindows()
                sys.exit(0)

        self.tolerance = 2  # Millimetre tolerance for the smallest coin downwards and largest coin upwards

        self.radii = {
            1: 16.25,   # 1 Cent
            2: 18.75,   # 2 Cent
            5: 21.25,   # 5 Cent
            10: 19.75,  # 10 Cent
            20: 22.25,  # 20 Cent
            50: 24.25,  # 50 Cent
            100: 23.25, # 1 Euro
            200: 25.75  # 2 Euro
        }

        # Required for calibration
        self.radii_pixels = []
        self.is_calibrating = False
        self.calibration_start_time = None



    def preprocess(self, frame):
        """Preprocessing of theframe for the recognition methodon method
        - Brighten the frame and grey scale it
        - Blur the image and enhance the edges
        - Fill in the gaps at the recognised edges

        Args:
            - frame (array):    Contains the image of the webcam as an array for opencv

        Test:
            - Output the frame directly, are changes then recognisable?
            - Comment out various steps and see how the frame changes
        """

        frame = cv.convertScaleAbs(frame, alpha=1, beta=25)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        frame = cv.GaussianBlur(frame, (9, 9), 2)
        frame = cv.Canny(frame, 50, 150)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        frame = cv.dilate(frame, kernel, iterations=1)

        return frame



    def radius_to_value(self, radius_pixel):
        """Assign the radius to the value of a coin
        - Check whether the radius matches the standard radius for the coin
        - For the tolerance, the radius is given up to the centre of the next smaller/larger coin
        - For the bottom and top coin, this is defined by a parameter

        Args:
            - frame (array):    Contains the image of the webcam as an array for opencv

        Test:
            - Check whether the correct coin matches the radii shown on the coins
            - If the value is outside the limits, the programme should terminate
        """

        radius_mm = radius_pixel * self.mm_per_pixel
        sorted_keys = sorted(self.radii.keys())
        
        for i in range(len(sorted_keys)):
            value = sorted_keys[i]
            if i == 0:
                upper_bound = (self.radii[sorted_keys[i]] + self.radii[sorted_keys[i+1]]) / 2
                if self.radii[sorted_keys[i]] - self.tolerance <= radius_mm <= upper_bound:
                    return value
            elif i == len(sorted_keys) - 1:
                if radius_mm >= (self.radii[sorted_keys[i-1]] + self.radii[sorted_keys[i]]) / 2:
                    return value
            else:
                lower_bound = (self.radii[sorted_keys[i-1]] + self.radii[sorted_keys[i]]) / 2
                upper_bound = (self.radii[sorted_keys[i]] + self.radii[sorted_keys[i+1]]) / 2
                if lower_bound <= radius_mm <= upper_bound:
                    return value

        self.logger.warning("Unable to assign circle diameter to any coin")
        return None



    def detect(self, frame):
        """Detect the coins
        - Uses OpenCV's Hough Circle Detection to recognise the circle shape of the coins
        - The frame is preprocessed to make the detection more accurate
        - The min and max radius and the distance between the coins is created based on the used coins
        - The radii in pixels are converted to mm using the millimetre-pixel ratio

        Args:
            - frame (array):    Contains the image of the webcam as an array for opencv

        Test:
            - Is the correct value of the coins displayed at the top left of the image?
            - Is there a rectangular box around the recognized coins?
            - Is there a circle around recognized coins?
        """

        value = 0
        amount=0

        frame_processed = self.preprocess(frame)

        circles = cv.HoughCircles(frame_processed, cv.HOUGH_GRADIENT, dp=1,
                                  minDist=int((self.radii[1] - self.tolerance) / self.mm_per_pixel * 2),
                                  param1=50, param2=30,
                                  minRadius=int((self.radii[1] - self.tolerance) / self.mm_per_pixel),
                                  maxRadius=int((self.radii[200] + self.tolerance) / self.mm_per_pixel))
        
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]

                cv.circle(frame, center, radius, (0, 255, 0), 2)
                cv.circle(frame, center, 2, (0, 0, 255), 3)

                center_x = int(center[0])
                center_y = int(center[1])
                radius = int(radius)

                left = max(center_x - radius, 0)
                right = min(center_x + radius, frame.shape[1])
                top = max(center_y - radius, 0)
                bottom = min(center_y + radius, frame.shape[0])

                label = self.radius_to_value(radius)

                gui.draw_rectangle(self.logger, frame, label, left, right, top, bottom)

                value = value + int(label)
                amount = amount +1

        gui.display_value(frame, value, amount)

        return frame



    def calibrate(self, frame):
        """Calibrate the millimetre-pixel ratio
        - Use a 20 cent coin to determine the millimetre-pixel ratio
        - As soon as the coin is in the square shown, a 5-second timer is started
        - Then 10 radii are recorded and the average is calculated
        - This is then saved to mm_per_pixel.txt

        Args:
            - frame (array):    Contains the image of the webcam as an array for opencv

        Test:
            - Check whether the detection radius only starts when the coin is within the area
            - Check whether the average millimetre-pixel ratio is written to mm_per_pixel.txt
        """
        
        lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
        
        if not self.is_calibrating:
            lcd.clear()
            lcd.write_string("Place a 20 cent coin in the area")
        elif self.is_calibrating and not (time.time() - self.calibration_start_time >= 5):
            lcd.clear()
            lcd.write_string("5. Second timer Please wait")

        frame_processed = self.preprocess(frame)
        circles = cv.HoughCircles(frame_processed, cv.HOUGH_GRADIENT, dp=1,
                                  minDist=1000, param1=50, param2=30, minRadius=10, maxRadius=1000)

        height, width = frame.shape[:2]
        rectangle_center = (width // 2, height // 2)
        rectangle_size = (300, 300)  
        left_top = (rectangle_center[0] - rectangle_size[0] // 2, rectangle_center[1] - rectangle_size[1] // 2)
        right_bottom = (rectangle_center[0] + rectangle_size[0] // 2, rectangle_center[1] + rectangle_size[1] // 2)

        self.cv.rectangle(frame, left_top, right_bottom, (255, 0, 0), 2)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                x, y, radius = circle
                if left_top[0] <= x - radius and x + radius <= right_bottom[0] and left_top[1] <= y - radius and y + radius <= right_bottom[1]:
                    self.cv.circle(frame, (x, y), radius, (0, 255, 0), 2)
                    self.cv.circle(frame, (x, y), 2, (0, 0, 255), 3)

                    if not self.is_calibrating:
                        self.is_calibrating = True
                        self.calibration_start_time = time.time()
                        self.logger.info("20 cent coin fully within the area. Waiting 5 seconds before starting radius measurements")

                        lcd.clear()
                        lcd.write_string("Coin detected   Starting Timer")

        if self.is_calibrating and (time.time() - self.calibration_start_time >= 5):
            if len(self.radii_pixels) < 10:
                self.radii_pixels.append(radius)

                lcd.clear()
                lcd.write_string(f"{len(self.radii_pixels)}. captured")
                lcd.cursor_pos = (1, 0)
                lcd.write_string(f"radius: {radius} px")

                self.logger.info(f"{len(self.radii_pixels)}. captured radius: {radius} pixels.")

            if len(self.radii_pixels) == 10:
                average_radius = sum(self.radii_pixels) / len(self.radii_pixels)
                mm_per_pixel = self.radii[20] / average_radius

                try:
                    with open("mm_per_pixel.txt", 'w') as f:
                        f.write(str(mm_per_pixel))
                        
                except FileNotFoundError:
                    self.logger.error(f"Error while writing mm_per_pixel.txt")
                    cv.destroyAllWindows()
                    sys.exit(0)

                lcd.clear()
                lcd.write_string(f"Average mm-pixelvalue: {round(mm_per_pixel, 2)}")

                self.logger.info(f"The calculated average mm per pixel value is {mm_per_pixel} and was written to mm_per_pixel.txt")
                cv.destroyAllWindows()
                sys.exit(0)

        return frame