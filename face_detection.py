#! /usr/bin/env python3

import cv2
import pickle
import os, sys, gc, re
import time
from PIL import Image
from clock import Clock

# Macro definition
FAILED = -1
CONTINUE = 1
USAGE = """Usage: {filename} trainer_file [haar_file]
        
with:
    trainer_file: Path to the trainer file obtained using face_trainer.py, in YML format.
    haar_file (optional): Path to the HAAR file used for training.""".format(filename=sys.argv[0])

class Capture:
    # Macro definition
    ROOT_DIR = os.getcwd() + '/'
    MIN_PREDICT_PERCENT = 50
    DEFAULT_NAME = "Unknown"
    DELTA_TIME_SCREENSHOT = 0.1
    DELTA_TIME_TEXT = 1
    # Not used yet
    # Number of photos saved per person recognized or unknown
    # Too much datas will fill the Pi's space
    MAX_PHOTOS_PER_PERSON = 500
    MAX_PHOTOS_PER_UNKNOWN = 1000
    IMAGE_SIZE = (300, 300)

    # ARGS:
    #   trainer: The trainer file name (dynamically made by 'face_trainer.py')
    def __init__(self, trainer, haar = "haar/haarcascade_frontalface_default.xml"):
        #   Internal algorithm
        self.paths = {
            'root': self.ROOT_DIR,
            'images': self.ROOT_DIR + 'images/',
            'haar': self.ROOT_DIR + 'haar/'
        }
        self.title = "Magic Mirror - Face Detection"
        self.extension = ".png"

        #   Generate different clocks and their delta time for update
        self.clocks = {
            'screenshot': Clock(self.DELTA_TIME_SCREENSHOT),
            'text': Clock(self.DELTA_TIME_TEXT)
        }
        self.__errors = [
            "Can't find trainer file with path '%s'.",
            "Can't open webcam. Check your USB connection, if the camera is enabled and if video0 exists in /dev.",
            "Error while reading next camera frame. The camera may have been disconnected.",
            "A face has been recognize, but the label doesn't exist somehow.",
            "Repository with path '%s' doesn't exist.",
            "Can't copy frame in folder '%s'.",
            "Can't load HAAR file with path '%s'." 
        ]
        
        # VARIABLES INIT
        #   Camera and face recognition
        self.haar = haar
        self.faceCascade = cv2.CascadeClassifier(self.haar)

        if self.faceCascade.empty():
            self.printError(6, (self.haar))
            raise ValueError()

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
            # 'predicted' is an array with dict containing every values for each faces detected
            # Each value in the array is composed using the format { 'pname': "", 'confidence': 0 }
            'predicted': []
        }

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
            return FAILED

        ret, self.frames['full'] = self.camera.read()
        
        if self.frames['full'] == []:
            self.printError(2)
            return FAILED

        # Convert the captured frame in grayscale
        self.frames['grayscale'] = cv2.cvtColor(self.frames['full'], cv2.COLOR_BGR2GRAY)
        self.faces = self.faceCascade.detectMultiScale(self.frames['grayscale'])

        # On timeout, flush old predicted infos
        if self.clocks['text'].timeElapsed():
            self.text['predicted'] = []

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

            # /!\ 0.0 confidence == 100% face matching !!!
            confidence = 100 - confidence if 100 - confidence > 0 else 0

            # Get the name of the recognized face. If none, get 'Unknown' instead.
            if confidence >= self.MIN_PREDICT_PERCENT:
                if plabel in self.labels:
                    pname = self.labels[plabel]
                else:
                    self.printError(3)
                    return FAILED
            else:
                pname = self.DEFAULT_NAME

            # Store info for display purpose
            if self.clocks['text'].timeElapsed():
                self.text['predicted'].append({
                    'pname': pname,
                    'confidence': confidence
                })

            pfolder = self.paths['images'] + pname + '/'

            # Process these lines each DELTA_TIME_SCREENSHOT seconds
            # Check if the folder exist and save a copy of the frame inside the folder
            # Create a new folder otherwise
            # Always keep your AI learns by itself :)
            if self.clocks['screenshot'].timeElapsed():
                if not os.path.isdir(pfolder):
                    os.mkdir(pfolder)
                if self.copyFrameToFolder(self.frames['roi'], pfolder, pname) == FAILED:
                    self.printError(4, (pfolder))
                    return FAILED

        # Reset each clock after execution
        for clock in self.clocks:
            if self.clocks[clock].timeElapsed():
                self.clocks[clock].setTime(time.time())

        return CONTINUE

    # Write new image in the correct folder when a face is detected
    def copyFrameToFolder(self, frame, folder = ".", name = ""):
        if frame == []:
            return FAILED
        
        filename = name + str(len(os.listdir(folder)) + 1) + self.extension

        # Resize the image to keep aspect ratio on every photos
        frame = cv2.resize(frame, self.IMAGE_SIZE)
        cv2.imwrite(folder + filename, frame)

    def display(self):
        if self.frames['full'] == []:
            self.printError(2)
            return FAILED

        id = 0

        # Get values and display them correctly for each faces
        for (x, y, width, height) in self.faces:
            predicted = self.text['predicted']
            style = self.text['style']

            if len(predicted) > id:
                cv2.putText(self.frames['full'], predicted[id]['pname'], (x, y - 10),
                    style['font'], style['scale'], style['color'], style['stroke'], cv2.LINE_AA)

                cv2.putText(self.frames['full'], "Confidence: " + '%.2f' % predicted[id]['confidence'] + "%", (x, y + height + 20),
                    style['font'], style['scale'] / 1.5, style['color'], style['stroke'], cv2.LINE_AA)

            id += 1

        cv2.imshow(self.title, self.frames['full'])
        return CONTINUE

    def printError(self, errno, *args):
        print("[ERROR] " + self.__errors[errno] % args, file=sys.stderr)


# Argument 1 is the trainer yml file
if len(sys.argv) < 2:
    print(USAGE, file=sys.stderr)
    sys.exit()

# Initialize capture
try:
    # Add haar to the parameters
    if len(sys.argv) > 2:
        capture = Capture(sys.argv[1], sys.argv[2])
    else:
        capture = Capture(sys.argv[1])
except ValueError:
    sys.exit()

firstLaunch = True

# Wait until 'Q' is pressed to quit or an action failed
while capture.detection() != FAILED:
    if firstLaunch:
        print("Welcome to the Magic Mirror face detection. Hit 'Q' if you want to quit.")
        firstLaunch = False
    
    # Break the loop on 'q' pressed
    if ord('q') == (cv2.waitKey(1) & 0xFF):
        break
    
    capture.display()

capture.release()