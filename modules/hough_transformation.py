""" DHBW-Studienarbeit - Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
    DHBW-Study paper  -  Development of a prototype for coin counting using image processing and machine learning
    
    author:  Angmar3019
    date:    07.02.2023
    version: 1.0.0
    licence: GNU General Public License v3.0 
"""

import cv2 as cv
import numpy as np
import modules.gui as gui



class hough_transformation:
    def __init__(self, logger):
        self.cv = cv
        self.logger = logger

        self.loggerinfo("Hough-Transformation is used as the detection method")
    
    def detect(self, frame):
        return frame