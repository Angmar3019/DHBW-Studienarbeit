""" DHBW-Studienarbeit - Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
    DHBW-Study paper  -  Development of a prototype for coin counting using image processing and machine learning
    
    author:  Angmar3019
    date:    01.06.2024
    version: 1.1.0
    licence: GNU General Public License v3.0 
"""

import sys
import cv2 as cv
import numpy as np
import modules.gui as gui

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter



class tensorflow:
    def __init__(self, logger, model_path, labels_path):
        """Initialisation of tensorflow detection
        - Initilaize EdgeTPU library
        - Configures the EdgeTPU interpreter
        - Imports the label file with the labels used in the model

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

        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
                self.logger.info(f"Imported labels {self.labels} from {labels_path}")

        except FileNotFoundError:
            self.logger.error(f"{labels_path} not found")
            cv.destroyAllWindows()
            sys.exit(0)



    def preprocess(self, frame):
        """Preprocessing of theframe for the recognition methodon method
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
            - Is the correct value of the coins displayed at the top left of the image?
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