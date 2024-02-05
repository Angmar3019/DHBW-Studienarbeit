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

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter



class tensorflow:
    def __init__(self, logger, model_path, labels_path):
        """Display value
        - Writes the total recognized value of the coins in the image
        - Converts from cents to full euros

        Args:
            - logger:           Contains the global logger variable
            - frame (array):    Contains the image of the webcam as an array for opencv
            - model_path:       The path to the model that should be used
            - labels_path:      The path to the labels that should be used
        """

        self.cv = cv
        self.logger = logger

        self.logger.info("Tensorflow is used as the detection method with EdgeTPU support")

        self.interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()       

        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]



    def preprocess(self, frame):
        """Preprocess the image
        - Changes the size of the image to that of the model
        - Converts the image into the int8 format required by the EdgeTPU and the model

        Args:
            - frame (array):    Contains the image of the webcam as an array for opencv

        Test:
            - If the interpreter does not error due to the incorrect format, then this is the correct format
        """

        input_shape = self.input_details[0]['shape']
        frame_resized = self.cv.resize(frame, (input_shape[1], input_shape[2]))
        input_data = frame_resized.astype(np.int8)
        return np.expand_dims(input_data, axis=0)



    def detect(self, frame):
        """Detect the coins
        - Receives the recognized boxes, their classes and the matching score
        - If the score is above the threshold, it draws a box around it and adds the value of the coin to the total value

        Args:
            - frame (array):    Contains the image of the webcam as an array for opencv

        Test:
            - Is the value of the coins displayed at the top left of the image?
            - Is there a rectangular box around the recognized coins?
        """

        input_data = self.preprocess(frame)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
            
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        #num = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        detection_threshold = 0.2
        value = 0
        
        for i in range(len(scores)):
            if scores[i] > detection_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                left = xmin * frame.shape[1]
                right = xmax * frame.shape[1]
                top = ymin * frame.shape[0]
                bottom = ymax * frame.shape[0]
                                
                class_id = int(classes[i])
                label = self.labels[class_id]

                gui.draw_rectangle(self.logger, frame, label, left, right, top, bottom)
                
                value = value + int(label)

        gui.display_value(frame, value)
        
        return frame