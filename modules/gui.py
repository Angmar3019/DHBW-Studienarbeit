""" DHBW-Studienarbeit - Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
    DHBW-Study paper  -  Development of a prototype for coin counting using image processing and machine learning
    
    author:  Angmar3019
    date:    01.06.2024
    version: 1.0.1
    licence: GNU General Public License v3.0 
"""

import cv2 as cv
font = cv.FONT_HERSHEY_SIMPLEX



def display_value(frame, value):
    """Display value
    - Writes the total recognized value of the coins in the image
    - Converts from cents to full euros

    Args:
        - frame (array):    Contains the image of the webcam as an array for opencv
        - value (int):      Contains the total value of the recognized coins in cents

    Test:
        - Is the value of the coins displayed at the top left of the image?
        - Is the value displayed in whole euros and not in cents?
    """

    value = "Wert: " + str(value / 100) + " Euro"
    cv.putText(frame, value, (50,50), font , 1, (0,255,0), 2, cv.LINE_4)



def draw_rectangle(logger, frame, label, left, right, top, bottom):
    """Draw rectangle
    - Draws a rectangle around the recognized coins
    - Writes the recognized coin as text over the rectangle

    Args:
        - logger:           Contains the global logger variable
        - frame (array):    Contains the image of the webcam as an array for opencv
        - label (str):      Contains the label text of the recognized class
        - left (int):       Coordinates for the rectangle
        - right (int):      Coordinates for the rectangle
        - top (int):        Coordinates for the rectangle
        - bottom (int):     Coordinates for the rectangle

    Test:
        - Is there a rectangular box around the recognized coins?
        - Is there a text above the box that contains the value of the coin?
    """

    if int(label) <= 50:
        label = str(label) + " Cent"
    elif 100 <= int(label) <= 200:
        label = str(int(int(label) / 100)) + " Euro"
    else:
        logger.error("Received a label that is in the wrong format or not supported")
        exit()

    cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    cv.putText(frame, label, (int(left), int(top) - 10), font, 0.5, (255, 255, 255), 2)