#! /usr/bin/env python3

import cv2, os, time, sys
import numpy as np 
import pickle
import shutil 

#cv2 pour l'utilisation de la webcam
#os pour pouvoir save les images tests
#numpy pour faire des calculs matriciels
#time pour gerer le temps et l'ajouter dans le nom pour le trainning
#shutil pour deplacer les photos dans un repertoire

#Utilsation des patterns pour les haars
face_cascade =  cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#On recupere le temps courant 
localtime = time.asctime((time.localtime(time.time())))
final_path = "/home/julien/python/OpenCV/Script/Mirroir_Connecté/images/"

#On lit le modèle entrainé
recognizer.read("trainner.yml")
labels = {"person_name": 1}

with open('label.pickle', "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#initialisation de la vidéo
cap = cv2.VideoCapture(0)
#Ouverture de la vidéo 

while (cap.isOpened()):
    #decoupage frame par frame
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        #Affiche les postion x,y,w,h du visage en fonction du frontalface pattern    
        #print(x,y,w,h)
        rectangle_color = (255,0,0)
        stroke = 2
        cv2.rectangle(img, (x,y), (x+w, y+h), rectangle_color, stroke)
        roi_gray =  gray[y:y+h, x:x+y]
        #cv2.imwrite("hello.jpg",roi_gray)
        roi_color = img[y:y+h, x:x+y]
        id_, conf = recognizer.predict(roi_gray)

        if conf>=70 and conf <=85:
            #print(conf)
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 0)
            strole = 2
            #print(name)
            #print(i)
            counter = 1

            #Affiche le texte au dessus de l'haar
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            for i in range (0,2):
                print(name)
                img_item = name+".png"
                print(img_item)
                cv2.imwrite(img_item, img)
                cv2.imwrite(img_item, roi_color)

                origin_path = "/home/julien/python/OpenCV/Script/Mirroir_Connecté/"+img_item
                print(origin_path)
                folder_list = os.listdir(final_path)
                #Pour les noms de fichier dans le dossier "python/OpenCV/Script/Mirroir_Connecté/images/"
                for folder_name in folder_list:
                    #On affiche la liste des dossiers
                    print(folder_name)
                    #Si l'un des dossiers se nomme "michel" alors on dit bonjour, on se deplace dans le dossier 
                    if folder_name == "michel":
                        print("bonjour")
                        #os.changedir()
                        os.chdir("/home/julien/python/OpenCV/Script/Mirroir_Connecté/images")
                        #os.getCurrentWorkingDirectory
                        print(os.getcwd())
                        #sys.exit()
                
            #Affichage du resultat
            cv2.imwrite(img_item, roi_color)

    cv2.imshow("frame", img)
    if cv2.waitKey(30) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


                # for folder_name in folder_list:
                #     print(folder_name)
                #     if img_item == "Julien_hivert" and folder_name == "Julien_hivert":
                #         #Mettre ça dans un tableau et à chaque nouvelle personne on l'ajoute dans le tableau
                #         shutil.move("/home/julien/python/OpenCV/Script/Mirroir_Connecté/"+img_item, "/home/julien/python/OpenCV/Script/Mirroir_Connecté/images/"+x+"/")
                #         print("bonjour julien ")
                #     elif folder_name == "eva-green":
                #         print("Hello")
                #         shutil.move("/home/julien/python/OpenCV/Script/Mirroir_Connecté/"+img_item, "/home/julien/python/OpenCV/Script/Mirroir_Connecté/images/"+x+"/")

                #     elif folder_name == "michel":
                #         print("Hallo")
                #         shutil.move("/home/julien/python/OpenCV/Script/Mirroir_Connecté/"+img_item, "/home/julien/python/OpenCV/Script/Mirroir_Connecté/images/"+x+"/")

                #    # elif x == "emilia_clarcke":
                #     #    print("true")
                #      #   shutil.move(origin_path, "/home/julien/python/OpenCV/Script/Mirroir_Connecté/images"+x+"/")
                #     else : 
                #         print("false")
            #     shutil.move(origin_path+img_item, final_path)
            #     sys.exit()
            # else :
            #     print("salut")
            #     img_item =  name_1+".png"
