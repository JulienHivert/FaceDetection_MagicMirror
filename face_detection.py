#! /usr/bin/env python3

import cv2
import pickle
import os, sys, gc, re
import time
from clock import Clock

USAGE = """Usage: {filename} trainer_file
        
with:
    trainer_file: The trainer file obtained using face_trainer.py, in YML format.""".format(filename=sys.argv[0])

class Capture:
    # Macro definition
    FAILED = -1
    CONTINUE = 1
    MIN_PREDICT_PERCENT = 85
    DEFAULT_NAME = "Unknown"
    DELTA_TIME_SCREENSHOT = 3
    DELTA_TIME_TEXT = 1

    # ARGS:
    #   trainer: The trainer file name (dynamically made by 'face_trainer.py')
    def __init__(self, trainer):
        # VARIABLES INIT
        #   Camera and face recognition
        self.faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.trainer = trainer
        self.frames = {
            'full': [],
            'grayscale': [],
            'roi': [],
            'roiGrayscale': []
        }
        self.rect = {
            'color': (255, 0, 0), #BGR
            'stroke': 2
        }
        self.text = {
            'style': {
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'color': (255, 255, 0), #BGR
                'scale': 1,
                'stroke': 1
            },
            'strings': { 
                'pname': "Unknown",
                'confidence': 0
            }
        }
        
        #   Internal algorithm
        self.paths = {
            'origin': os.getcwd() + '/',
            'images': os.getcwd() + '/images/'
        }
        self.title = "Magic Mirror - Face Detection"
        self.extension = ".png"

        #   Generate different clocks and their delta time for update
        self.clocks = {
            'screenshot': Clock(self.DELTA_TIME_SCREENSHOT),
            'text': Clock(self.DELTA_TIME_TEXT)
        }
        self.__errors = [
            "Can't find trainer file with name '%s'.",
            "Can't open webcam. Check your USB connection, if the camera is enabled and if video0 exists in /dev.",
            "Error while reading next camera frame. The camera may have been disconnected.",
            "A face has been recognize, but the label doesn't exist somehow.",
            "Repository with path '%s' doesn't exist.",
            "Can't copy frame in folder '%s'."
        ]

        # OPENCV INIT
        #   Get trained values
        try:
            self.recognizer.read(self.trainer)
        except:
            self.printError(0, (self.trainer))
            raise ValueError()
        
        with open('label.pickle', "rb") as f:
            og_labels = pickle.load(f)
            self.labels = { v:k for k,v in og_labels.items() }
        self.camera = cv2.VideoCapture(0)

    def release(self):
        print("Exiting the Magic Mirror Face Detection. Goodbye!")
        self.camera.release()
        cv2.destroyAllWindows()

    def detection(self):
        if self.camera.isOpened() == False:
            self.printError(1)
            return self.FAILED

        ret, self.frames['full'] = self.camera.read()
        
        if self.frames['full'] == []:
            self.printError(2)
            return self.FAILED

        # Convert the captured frame in grayscale
        self.frames['grayscale'] = cv2.cvtColor(self.frames['full'], cv2.COLOR_BGR2GRAY)
        self.faces = self.faceCascade.detectMultiScale(self.frames['grayscale'], 1.3, 4)

        # Display a blue rectangle at face's position
        # Execute the algorithm next if a face is detected
        for (x, y, width, height) in self.faces:
            cv2.rectangle(self.frames['full'], (x, y), (x + width, y + height), self.rect['color'], self.rect['stroke'])
            min = { 
                'x': x + self.rect['stroke'],
                'y': y + self.rect['stroke']
            }
            max = { 
                'x': x + width - self.rect['stroke'],
                'y': y + height - self.rect['stroke']
            }

            # R.O.I = Region Of Interest
            # Recognize a face according to the captured grayscaled frame
            # Get a preempted label and confidence percentage depending on how much the algorithm recognize the face
            self.frames['roi'] = self.frames['full'][min['y']:max['y'], min['x']:max['x']]
            self.frames['roiGrayscale'] = self.frames['grayscale'][min['y']:max['y'], min['x']:max['x']]
            plabel, confidence = self.recognizer.predict(self.frames['roiGrayscale'])

            # Get the name of the recognized face. If none, get 'Unknown' instead.
            if confidence >= self.MIN_PREDICT_PERCENT:
                if plabel in self.labels:
                    pname = self.labels[plabel]
                else:
                    self.printError(3)
                    return self.FAILED
            else:
                pname = self.DEFAULT_NAME

            # Store info for display purpose
            # _d is for "display"
            if self.clocks['text'].timeElapsed():
                self.text['strings']['pname'] = pname
                self.text['strings']['confidence'] = confidence
                self.clocks['text'].setTime(time.time())

            pfolder = self.paths['images'] + pname + '/'

            # Process these lines each DELTA_TIME_SCREENSHOT seconds
            # Check if the folder exist and save a copy of the frame inside the folder
            # Create a new folder otherwise
            # Always keep your AI learns by itself :)
            if self.clocks['screenshot'].timeElapsed():
                self.clocks['screenshot'].setTime(time.time())
                if not os.path.isdir(pfolder):
                    os.mkdir(pfolder)
                if self.copyFrameToFolder(self.frames['roi'], pfolder, pname) == self.FAILED:
                    self.printError(4, (pfolder))

        return self.CONTINUE

    # Write new image in the correct folder when a face is detected
    def copyFrameToFolder(self, frame, folder = ".", name = ""):
        if frame == []:
            return self.FAILED
        
        filename = name + str(len(os.listdir(folder)) + 1) + self.extension
        cv2.imwrite(folder + filename, frame)

    def display(self):
        if self.frames['full'] == []:
            self.printError(2)
            return self.FAILED

        # Draw text with style on the top left corner of the rectangle each delta time
        for (x, y, width, height) in self.faces:
            strings = self.text['strings']
            style = self.text['style']

            cv2.putText(self.frames['full'], strings['pname'], (x, y),
                style['font'], style['scale'], style['color'], style['stroke'], cv2.LINE_AA)

            cv2.putText(self.frames['full'], "Confidence: " + '%.2f' % strings['confidence'] + "%", (x, y + height),
                style['font'], style['scale'] / 1.5, style['color'], style['stroke'], cv2.LINE_AA)

        cv2.imshow(self.title, self.frames['full'])
        return self.CONTINUE

    def printError(self, errno, *args):
        print("[ERROR] " + self.__errors[errno] % args, file=sys.stderr)


# Argument 1 is the trainer yml file
if len(sys.argv) < 2:
    print(USAGE, file=sys.stderr)
    sys.exit()

# Miscellaneous variable initialization
try:
    capture = Capture(sys.argv[1])
except ValueError:
    sys.exit()

firstLaunch = True

# Wait until 'Q' is pressed to quit or an action failed
while capture.detection() != capture.FAILED:
    if firstLaunch:
        print("Welcome to the Magic Mirror face detection. Hit 'Q' if you want to quit.")
        firstLaunch = False
    
    # Garbage collect in __del__ and break the loop
    if ord('q') == (cv2.waitKey(1) & 0xFF):
        break
    
    capture.display()

capture.release()