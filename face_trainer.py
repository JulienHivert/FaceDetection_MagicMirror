#! /usr/bin/env python3

# os pour pourvoir choper les pahts des dossiers / fichiers
#cv2 pour l'utilisation des haar
#numpy pour faire des calculs matriciel
#pickle pour save les objets
# PIL pour faire du traitemenent d'image

import os 
import cv2
from PIL import Image
import numpy as np 
import pickle

#On localise le dossiers contenant les images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#On utilise l'haar visage 
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')

#eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
#print(help(cv2.LBPHFaceRecognizer))
#On crée la fonction qui va nous servir à reconnaitre les visages en fonction des images
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

#On parcours les fichiers, on souhaite travailler qu'avec ceux qui on pour extension : png, jpg
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ","-").lower()
            print(label,path)
            if label in label_ids:
                pass
            else:
                label_ids[label] =  current_id
                current_id += 1
                id_ =  label_ids[label]
                print(label_ids)
            #y_labels.append(label) #Some number
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array)

            for (x,y,w,h) in faces:
                roi = image_array [y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                cv2.imshow("Trainer", roi)
                cv2.waitKey(0)

#print(y_labels)
#print(x_train)

with open('label.pickle', "wb") as f:
    pickle.dump(label_ids, f)
#On entraine le modele et on le sauvegarde 
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
